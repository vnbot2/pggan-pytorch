import dataloader as DL
from config import config
import network as net
from math import floor, ceil
import os, sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
from tqdm import tqdm
import tf_recorder as tensorboard
import utils as utils
import numpy as np
# import tensorflow as tf

class trainer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.use_cuda = False
            torch.set_default_tensor_type('torch.FloatTensor')
        
        self.nz = config.nz
        self.optimizer = config.optimizer

        self.resolution = 2           # we start from 2^2 = 4
        self.lr = config.lr
        self.eps_drift = config.eps_drift
        self.smoothing = config.smoothing
        self.max_resolution = config.max_resolution
        self.transition_tick = config.transition_tick
        self.stablize_tick = config.stablize_tick
        self.TICK = config.TICK
        self.globalIter = 0
        self.globalTick = 0
        self.kimgs = 0
        self.stack = 0
        self.epoch = 0
        self.fadein = {'gen':None, 'dis':None}
        self.complete = {'gen':0, 'dis':0}
        self.phase = 'init'
        self.flag_flush_gen = False
        self.flag_flush_dis = False
        self.flag_add_noise = self.config.flag_add_noise
        self.flag_add_drift = self.config.flag_add_drift
        
        # network and cirterion
        self.G = nn.DataParallel(net.Generator(config))
        self.D = nn.DataParallel(net.Discriminator(config))
        print ('Generator structure: ')
        print(self.G.module.model)
        print ('Discriminator structure: ')
        print(self.D.module.model)
        self.mse = torch.nn.MSELoss()
        # if self.use_cuda:
        #     self.mse = self.mse.cuda()
        #     torch.cuda.manual_seed(config.random_seed)
        #     if torch.cuda.device_count()==1:
        #         self.G = torch.nn.DataParallel(self.G).cuda(device=0)
        #         self.D = torch.nn.DataParallel(self.D).cuda(device=0)
        #     else:
        #         gpus = []
        #         for i  in range(config.n_gpu):
        #             gpus.append(i)

        
        # define tensors, ship model to cuda, and get dataloader.
        self.renew_everything()
        
        # tensorboard
        self.use_tb = config.use_tb
        if self.use_tb:
            self.tb = tensorboard.tf_recorder()
        

    def resolution_scheduler(self):
        '''
        this function will schedule image resolution(self.resolution) progressively.
        it should be called every iteration to ensure resolution value is updated properly.
        step 1. (transition_tick) --> transition in generator.
        step 2. (stablize_tick) --> stablize.
        step 3. (transition_tick) --> transition in discriminator.
        step 4. (stablize_tick) --> stablize.
        '''
        if floor(self.resolution) != 2 :
            self.transition_tick = self.config.transition_tick
            self.stablize_tick = self.config.stablize_tick
        
        self.batchsize = self.loader.batchsize
        
        total_tick = 2*(self.transition_tick+self.stablize_tick)

        delta = 1.0/total_tick
        self.d_alpha = 1.0*self.batchsize/self.transition_tick/self.TICK

        # update alpha if fade-in layer exist.
        if self.fadein['gen'] is not None:
            if self.resolution%1.0 < (self.transition_tick)*delta:
                self.fadein['gen'].update_alpha(self.d_alpha)
                self.complete['gen'] = self.fadein['gen'].alpha*100
                self.phase = 'gtransition'
            elif self.resolution%1.0 >= (self.transition_tick)*delta and self.resolution%1.0 < (self.transition_tick+self.stablize_tick)*delta:
                self.phase = 'gstablize'
        if self.fadein['dis'] is not None:
            if self.resolution % 1.0 >= 0.5 and self.resolution%1.0 < (self.stablize_tick + self.transition_tick*2)*delta:
                self.fadein['dis'].update_alpha(self.d_alpha)
                self.complete['dis'] = self.fadein['dis'].alpha*100
                self.phase = 'dtransition'
            elif self.resolution%1.0 >= (self.stablize_tick + self.transition_tick*2)*delta and self.phase!='final':
                self.phase = 'dstablize'
            
        prev_kimgs = self.kimgs
        self.kimgs = self.kimgs + self.batchsize
        ticked = '-'
        if floor(prev_kimgs/self.TICK) < floor(self.kimgs/self.TICK) :
            ticked = '+'
            self.globalTick = self.globalTick + 1
            # increase linearly every tick, and grow network structure.
            prev_resolution = floor(self.resolution)
            self.resolution = self.resolution + delta
            self.resolution = max(2, min(10.5, self.resolution))        # clamping, range: 4 ~ 1024

            # flush network.
            if self.flag_flush_gen and self.resolution%1.0 >= (self.transition_tick+self.stablize_tick)*delta and prev_resolution!=2:
                if self.fadein['gen'] is not None:
                    self.fadein['gen'].update_alpha(self.d_alpha)
                    self.complete['gen'] = self.fadein['gen'].alpha*100
                self.flag_flush_gen = False
                self.G.module.flush_network()   # flush G
                self.fadein['gen'] = None
                self.complete['gen'] = 0.0
                self.phase = 'dtransition'
            elif self.flag_flush_dis and floor(self.resolution) != prev_resolution and prev_resolution!=2:
                if self.fadein['dis'] is not None:
                    self.fadein['dis'].update_alpha(self.d_alpha)
                    self.complete['dis'] = self.fadein['dis'].alpha*100
                self.flag_flush_dis = False
                self.D.module.flush_network()   # flush and,
                self.fadein['dis'] = None
                self.complete['dis'] = 0.0
                if floor(self.resolution) < self.max_resolution and self.phase != 'final':
                    self.phase = 'gtransition'

            # grow network.
            if floor(self.resolution) != prev_resolution and floor(self.resolution)<self.max_resolution+1:
                self.lr = self.lr * float(self.config.lr_decay)
                self.G.module.grow_network(floor(self.resolution))
                self.D.module.grow_network(floor(self.resolution))
                self.renew_everything()
                self.fadein['gen'] = dict(self.G.module.model.named_children())['fadein_block']
                self.fadein['dis'] = dict(self.D.module.model.named_children())['fadein_block']
                self.flag_flush_gen = True # next iter will flush
                self.flag_flush_dis = True

            if floor(self.resolution) >= self.max_resolution and self.resolution%1.0 >= (self.stablize_tick + self.transition_tick*2)*delta:
                self.phase = 'final'
                self.resolution = self.max_resolution + (self.stablize_tick + self.transition_tick*2)*delta

        return {'ticked':ticked}
            
    def renew_everything(self):
        # renew dataloader.
        self.loader = DL.CustomDataloader(config)
        self.loader.renew(min(floor(self.resolution), self.max_resolution))
        
        # define tensors
        self.z = torch.FloatTensor(self.loader.batchsize, self.nz)
        self.x = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.x_tilde = torch.FloatTensor(self.loader.batchsize, 3, self.loader.imsize, self.loader.imsize)
        self.real_label = torch.FloatTensor(self.loader.batchsize).fill_(1)
        self.fake_label = torch.FloatTensor(self.loader.batchsize).fill_(0)
        
        # enable cuda
        if self.use_cuda:
            self.z = self.z.cuda()
            self.x = self.x.cuda()
            self.x_tilde = self.x.cuda()
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()
            torch.cuda.manual_seed(config.random_seed)

        # wrapping autograd Variable.
        self.x = Variable(self.x)
        self.x_tilde = Variable(self.x_tilde)
        self.z = Variable(self.z)
        self.real_label = Variable(self.real_label)
        self.fake_label = Variable(self.fake_label)
        
        # ship new model to cuda.
        if self.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
        
        # optimizer
        betas = (self.config.beta1, self.config.beta2)
        if self.optimizer == 'adam':
            self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
            self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.lr, betas=betas, weight_decay=0.0)
        

    def feed_interpolated_input(self, x):
        if self.phase == 'gtransition' and floor(self.resolution)>2 and floor(self.resolution)<=self.max_resolution:
            alpha = self.complete['gen']/100.0
            # transform = transforms.Compose( [   transforms.ToPILImage(),
            #                                     transforms.Scale(size=int(pow(2,floor(self.resolution)-1)), interpolation=0),      # 0: nearest
            #                                     transforms.Scale(size=int(pow(2,floor(self.resolution))), interpolation=0),      # 0: nearest
            #                                     transforms.ToTensor(),
            #                                 ] )
            # x_low = x.clone().add(1).mul(0.5)
            # for i in range(x_low.size(0)):
            #     x_low[i] = transform(x_low[i]).mul(2).add(-1)
            x_low = x.clone()
            x_low = nn.functional.interpolate(x_low, size=int(pow(2,floor(self.resolution)-1)), mode='nearest')
            x_low = nn.functional.interpolate(x_low, size=int(pow(2,floor(self.resolution))), mode='nearest')
            x = torch.add(x.mul(alpha), x_low.mul(1-alpha)) # interpolated_x

        if self.use_cuda:
            return x.cuda()
        else:
            return x

    def add_noise(self, x):
        # TODO: support more method of adding noise.
        if self.flag_add_noise==False:
            return x

        if hasattr(self, '_d_'):
            self._d_ = self._d_ * 0.9 + torch.mean(self.fx_tilde).item() * 0.1
        else:
            self._d_ = 0.0
        strength = 0.2 * max(0, self._d_ - 0.5)**2
        z = np.random.randn(*x.size()).astype(np.float32) * strength
        z = Variable(torch.from_numpy(z)).cuda() if self.use_cuda else Variable(torch.from_numpy(z))
        return x + z

    def train(self):
        # noise for test.
        self.z_test = torch.FloatTensor(self.loader.batchsize, self.nz)
        if self.use_cuda:
            self.z_test = self.z_test.cuda()
        self.z_test = Variable(self.z_test, volatile=True)
        self.z_test.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
        
        # for step in range(2, self.max_resolution+1+5):
        # for iter in range(0,(self.transition_tick*2+self.stablize_tick*2)*self.TICK, self.loader.batchsize):
        final_step = 0
        while True:
            self.globalIter = self.globalIter+1
            self.stack = self.stack + self.loader.batchsize
            if self.stack > ceil(len(self.loader.dataset)):
                self.epoch = self.epoch + 1
                self.stack = int(self.stack%(ceil(len(self.loader.dataset))))

            # resolutionolution scheduler.
            sched_results = self.resolution_scheduler()
            
            # zero gradients.
            self.G.zero_grad()
            self.D.zero_grad()

            # update discriminator.
            self.x.data = self.feed_interpolated_input(self.loader.get_batch())
            if self.flag_add_noise:
                self.x = self.add_noise(self.x)
            self.z.data.resize_(self.loader.batchsize, self.nz).normal_(0.0, 1.0)
            self.x_tilde = self.G(self.z)
            
            self.fx = self.D(self.x)
            self.fx_tilde = self.D(self.x_tilde.detach())
            
            loss_d = self.mse(self.fx.squeeze(), self.real_label) + \
                                self.mse(self.fx_tilde, self.fake_label)
            loss_d.backward()
            self.opt_d.step()

            # update generator.
            fx_tilde = self.D(self.x_tilde)
            loss_g = self.mse(fx_tilde.squeeze(), self.real_label.detach())
            loss_g.backward()
            self.opt_g.step()
            
            # logging.
            log_msg = sched_results['ticked'] + ' [E:{0}][T:{1}]  errD: {4:.4f} | errG: {5:.4f} | [lr:{11:.5f}][cur:{6:.3f}][resolution:{7:4}][{8}]'.format(self.epoch, self.globalTick, self.stack, len(self.loader.dataset), loss_d.item(), loss_g.item(), self.resolution, int(pow(2,floor(self.resolution))), self.phase, self.complete['gen'], self.complete['dis'], self.lr)
            if hasattr(self, 'fadein') and self.fadein['dis'] is not None:
                log_msg += '|D-Alpha:{:0.2f}'.format(self.fadein['dis'].alpha)
                
            if hasattr(self, 'fadein') and self.fadein['gen'] is not None:
                log_msg += '|G-Alpha:{:0.2f}'.format(self.fadein['gen'].alpha)

            print(log_msg)
            if self.phase == 'final':
                final_step+=1
                if final_step > self.config.final_steps:
                    self.snapshot('repo/model')
                    break
            # tqdm.write(log_msg)

            # save model.
            self.snapshot('repo/model')

            # save image grid.
            if self.globalIter%self.config.save_img_every == 0:
                with torch.no_grad():
                    x_test = self.G(self.z_test)
                utils.mkdir('repo/save/grid')
                utils.save_image_grid(x_test.data, 'repo/save/grid/{}_{}_G{}_D{}.jpg'.format(int(self.globalIter/self.config.save_img_every), self.phase, self.complete['gen'], self.complete['dis']))
                utils.mkdir('repo/save/resolution_{}'.format(int(floor(self.resolution))))
                utils.save_image_single(x_test.data, 'repo/save/resolution_{}/{}_{}_G{}_D{}.jpg'.format(int(floor(self.resolution)),int(self.globalIter/self.config.save_img_every), self.phase, self.complete['gen'], self.complete['dis']))

            # tensorboard visualization.
            if self.use_tb:
                with torch.no_grad():
                    x_test = self.G(self.z_test)
                self.tb.add_scalar('data/loss_g', loss_g.item(), self.globalIter)
                self.tb.add_scalar('data/loss_d', loss_d.item(), self.globalIter)
                self.tb.add_scalar('tick/lr', self.lr, self.globalIter)
                self.tb.add_scalar('tick/cur_resolution', int(pow(2,floor(self.resolution))), self.globalIter)
                '''IMAGE GRID
                self.tb.add_image_grid('grid/x_test', 4, utils.adjust_dyn_range(x_test.data.float(), [-1,1], [0,1]), self.globalIter)
                self.tb.add_image_grid('grid/x_tilde', 4, utils.adjust_dyn_range(self.x_tilde.data.float(), [-1,1], [0,1]), self.globalIter)
                self.tb.add_image_grid('grid/x_intp', 4, utils.adjust_dyn_range(self.x.data.float(), [-1,1], [0,1]), self.globalIter)
                '''

    def get_state(self, target):
        if target == 'gen':
            state = {
                'resolution' : self.resolution,
                'state_dict' : self.G.module.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resolution' : self.resolution,
                'state_dict' : self.D.module.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
            }
            return state


    def get_state(self, target):
        if target == 'gen':
            state = {
                'resolution' : self.resolution,
                'state_dict' : self.G.module.state_dict(),
                'optimizer' : self.opt_g.state_dict(),
            }
            return state
        elif target == 'dis':
            state = {
                'resolution' : self.resolution,
                'state_dict' : self.D.module.state_dict(),
                'optimizer' : self.opt_d.state_dict(),
            }
            return state


    def snapshot(self, path):
        if not os.path.exists(path):
            if os.name == 'nt':
                os.system('mkdir {}'.format(path.replace('/', '\\')))
            else:
                os.system('mkdir -p {}'.format(path))
        # save every 100 tick if the network is in stablize phase.
        ndis = 'dis_R{}_T{}.pth.tar'.format(int(floor(self.resolution)), self.globalTick)
        ngen = 'gen_R{}_T{}.pth.tar'.format(int(floor(self.resolution)), self.globalTick)
        if self.globalTick%50==0:
            if self.phase == 'gstablize' or self.phase =='dstablize' or self.phase == 'final':
                save_path = os.path.join(path, ndis)
                if not os.path.exists(save_path):
                    torch.save(self.get_state('dis'), save_path)
                    save_path = os.path.join(path, ngen)
                    torch.save(self.get_state('gen'), save_path)
                    print('[snapshot] model saved @ {}'.format(path))

if __name__ == '__main__':
    ## perform training.
    print('----------------- configuration -----------------')
    for k, v in vars(config).items():
        print('  {}: {}'.format(k, v))
    print('-------------------------------------------------')
    torch.backends.cudnn.benchmark = True           # boost speed.
    trainer = trainer(config)
    trainer.train()

