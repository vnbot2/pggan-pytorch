# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import torchvision.utils as vutils

import glob
import os
import random
import shutil
import time
import xml.etree.ElementTree as ET  # for parsing XML

import albumentations as A
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mmcv
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensor
from PIL import Image, ImageEnhance, ImageOps
from torch import autograd, nn, optim
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from pyson.utils import multi_thread, memoize
from pyson.vision import plot_images
import pyson.vision as pv
import pyson.utils as pu
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
IMG_SIZE = 128
IMG_SIZE_2 = IMG_SIZE * 2