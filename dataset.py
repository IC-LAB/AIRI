import torch
import os
import glob
import numpy as np
import scipy
from scipy.misc import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

from .utils import generateMask

class EdgeDataset(Dataset):
    def __init__(self, fpath):
        self.data = self.loadFlist(fpath)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            img, img_gray, edge, mask = self.loadItem(idx)
        except:
            print('load error: ' + self.data[idx])
            img, img_gray, edge, mask = self.loadItem(0)
        return img, img_gray, edge, mask
    
    def loadFlist(self, fpath):
        if isinstance(fpath, str):
            if os.path.isdir(fpath):
                flist = list(glob.glob(fpath+'/*jpg')) + \
                        list(glob.glob(fpath+'/*.JPG')) + \
                        list(glob.glob(fpath+'/*.png')) + \
                        list(glob.glob(fpath+'/*.bmp')) + \
                        list(glob.glob(fpath+'/[a-z]/[a-z]*/*.jpg')) + \
                        list(glob.glob(fpath+'/[a-z]/[a-z]*/*.png')) + \
                        list(glob.glob(fpath+'/[a-z]/[a-z]*/*.bmp'))
                flist.sort()
                return flist
            else:
                return []
        else:
            return []
    
    def loadItem(self, idx):
        size = 256

        #load image (channel=3, uint8)
        img = imread(self.data[idx])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)
        
        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image (channel=1, uint8)
        img_gray = rgb2gray(img)

        # generate random mask (channel=1, float)
        mask = generateMask(size, size)
        mask = np.asarray(mask>0.5, np.float_)

        # extract edge (channel=1, float) 
        edge = canny(img_gray, 2).astype(np.float)

        # translate to Tensor
        img_t = self.to_tensor(img)
        img_gray_t = self.to_tensor(img_gray)
        edge_t = self.to_tensor(edge)
        mask_t = self.to_tensor(mask) 

        return img_t, img_gray_t, edge_t, mask_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]
        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh-side) // 2
            i = (imgw-side) // 2
            img = img[j:j+side, i:i+side, ...]
        img = scipy.misc.imresize(img, [height, width])
        return img
    
    def to_tensor(self, img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
            img = img * 255.0
        # img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
    
    def createIterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

class AdaDataset(Dataset):
    def __init__(self, fpath):
        self.data = self.loadFlist(fpath)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            img, img_gray, edge = self.loadItem(idx)
        except:
            print('load error: ' + self.data[idx])
            img, img_gray, edge = self.loadItem(0)
        return img, img_gray, edge
    
    def loadFlist(self, fpath):
        if isinstance(fpath, str):
            if os.path.isdir(fpath):
                flist = list(glob.glob(fpath+'/*jpg')) + \
                        list(glob.glob(fpath+'/*.JPG')) + \
                        list(glob.glob(fpath+'/*.png')) + \
                        list(glob.glob(fpath+'/*.bmp')) + \
                        list(glob.glob(fpath+'/[a-z]/[a-z]*/*jpg')) + \
                        list(glob.glob(fpath+'/[a-z]/[a-z]*/*.png')) + \
                        list(glob.glob(fpath+'/[a-z]/[a-z]*/*.bmp'))
                flist.sort()
                return flist
            else:
                return []
        else:
            return []
    
    def loadItem(self, idx):
        size = 256

        #load image (channel=3, uint8)
        img = imread(self.data[idx])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)
        
        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # create grayscale image (channel=1, uint8)
        img_gray = rgb2gray(img)

        # extract edge (channel=1, float) 
        edge = canny(img_gray, 2).astype(np.float)

        # translate to Tensor
        img_t = self.to_tensor(img)
        img_gray_t = self.to_tensor(img_gray)
        edge_t = self.to_tensor(edge)

        return img_t, img_gray_t, edge_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]
        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh-side) // 2
            i = (imgw-side) // 2
            img = img[j:j+side, i:i+side, ...]
        img = scipy.misc.imresize(img, [height, width])
        return img
    
    def to_tensor(self, img):
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
            img = img * 255.0
        img_t = F.to_tensor(img).float()
        return img_t
    
    def createIterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item