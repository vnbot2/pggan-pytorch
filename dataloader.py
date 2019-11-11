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
    def __init__(self, root, size, num_samples=None):
        self.size = size
        self.img_list  = glob.glob(os.path.join(root, '*', '*.png'))
        self.img_list  += glob.glob(os.path.join(root, '*', '*.jpg'))
        # self.img_list  = self.img_list[:100]
        assert len(self.img_list) > 0, os.path.join(root, '*', '*.png')
        print('Load in mem...', 'size:', self.size)
        self.data = self._loaddata()
        self.num_samples = num_samples
    
    @memoize
    def _loaddata(self):
        return multi_thread(data_preprocessing, zip(self.img_list, [self.size]*len(self.img_list)), verbose=1, max_workers=32)

    def __len__(self):
        if self.num_samples is None:
            num_samples = len(self.img_list)
        else:
            num_samples = self.num_samples
        return num_samples

    def __getitem__(self,idx):
        img = self.data[idx%len(self.data)]
        return img, 0


def get_batch_size(size, bz_128=4):
    num_pix = 128*128*bz_128
    rt = int(num_pix)//(size*size)
    print('size: {} -> Batchsize: {}'.format(size, rt))
    return rt

class CustomDataloader:
    def __init__(self, config):
        self.root = config.train_data_root
        sizes = [4, 8, 16, 32, 64, 128]
        self.batch_table = {size: get_batch_size(size) for size in sizes}
        self.batchsize = int(self.batch_table[pow(2,2)])*torch.cuda.device_count()        # we start from 2^2=4
        self.imsize = int(pow(2,2))
        self.num_workers = 4
        self.config = config
        
    def renew(self, resolution):
        
        self.batchsize = int(self.batch_table[pow(2,resolution)])
        self.imsize = int(pow(2,resolution))
        _n_stick = (self.config.transition_tick*2+self.config.stablize_tick*2)
        _num_samples = self.batchsize * _n_stick
        print('[*] Renew dataloader configuration, load data from {}.'.format(self.root), 'Num of sample:', _num_samples, '\tImsize:', self.imsize, '\t Batchsize:', self.batchsize)
        self.dataset = MotoDataset(self.root, size=(self.imsize,self.imsize), num_samples=_num_samples)
        self.dl = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchsize,
            shuffle=True,) 

    def __iter__(self):
        return iter(self.dl)
    
    def __next__(self):
        return next(self.dl)

    def __len__(self):
        return len(self.dl.dataset)

       
    def get_batch(self):
        dataIter = iter(self.dl)
        return next(dataIter)[0].mul(2).add(-1)         # pixel range [-1, 1]


        








