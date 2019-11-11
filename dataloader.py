import os
import torch as torch
import numpy as np
from io import BytesIO
import scipy.misc
#import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image
from common import *
# from torch.multiprocessing import Pool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

def detect_background_color(img):
    h, w = img.shape[:2]
    # if w > h:
    region = img[:10]
    return int(region.mean()> 128) *255    

def pad_if_needed(img, values=255):
    h, w = img.shape[:2]
    s = max(h, w)
    pad = np.ones([s,s, 3], dtype=img.dtype)*values
    start_h = (s-h)//2
    start_w = (s-w)//2
    pad[start_h:start_h+h, start_w:start_w+w] = img
    return pad


def data_preprocessing(input):
    img_path, size = input
    img = pv.imread(img_path)
    bg_color = detect_background_color(img)
    img = pad_if_needed(img, bg_color)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img / 127.5 - 1
    img = img.transpose([2,0,1])
    return img.astype(np.float32) 



class MotoDataset(Dataset):
    def __init__(self, root, size):
        self.size = size
        self.img_list  = glob.glob(os.path.join(root, '*', '*.png'))
        self.img_list  += glob.glob(os.path.join(root, '*', '*.jpg'))
        assert len(self.img_list) > 0, os.path.join(root, '*', '*.png')
        # self.load_in_mem = load_in_mem
        # if self.load_in_mem:
        print('Load in mem...', 'size:', self.size)
        self.data = self._loaddata()
    
    @memoize
    def _loaddata(self):
        return multi_thread(data_preprocessing, zip(self.img_list, [self.size]*len(self.img_list)), verbose=1, max_workers=32)

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self,idx):
        # if self.load_in_mem:
        img = self.data[idx]
        # else:
        #     full_img_path = self.img_list[idx]
            # img = data_preprocessing(full_img_path)
        return img, 0
        # return {'img':img, 'label':0}


class CustomDataloader:
    def __init__(self, config):
        self.root = config.train_data_root
        self.batch_table = {4:512, 8:, 16:256, 32:128, 64:64, 128:64}
        self.batchsize = int(self.batch_table[pow(2,2)])        # we start from 2^2=4
        self.imsize = int(pow(2,2))
        self.num_workers = 4
        
    def renew(self, resolution):
        print('[*] Renew dataloader configuration, load data from {}.'.format(self.root))
        
        self.batchsize = int(self.batch_table[pow(2,resolution)])
        self.imsize = int(pow(2,resolution))
        self.dataset = MotoDataset(self.root, size=(self.imsize,self.imsize))
        # self.dataset = ImageFolder(
        #             root=self.root,
        #             transform=transforms.Compose(   [
        #                                             transforms.Resize(size=(self.imsize,self.imsize), interpolation=Image.NEAREST),
        #                                             transforms.ToTensor(),
        #                                             ]))

        self.dl = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchsize,
            shuffle=True,) 
            # num_workers=8*torch.cuda.device_count())
        # for data in self.dl:break
        # import ipdb; ipdb.set_trace()

    def __iter__(self):
        return iter(self.dl)
    
    def __next__(self):
        return next(self.dl)

    def __len__(self):
        return len(self.dl.dataset)

       
    def get_batch(self):
        dataIter = iter(self.dl)
        return next(dataIter)[0].mul(2).add(-1)         # pixel range [-1, 1]


        








