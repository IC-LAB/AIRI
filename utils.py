import numpy as np
import cv2
from random import randint, seed
from copy import deepcopy
import torch
from skimage.color import gray2rgb

import torch

def generateMask(height, width, rand_seed=None):
    """Generates a random irregular mask with lines, circles and elipses"""
    img = np.zeros((height, width))

    if rand_seed:
        seed(rand_seed)

    # Set size scale
    size = int((width + height) * 0.03)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")
    
    # Draw random lines
    for _ in range(randint(1, 20)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(3, size)
        cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
        
    # Draw random circles
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, size)
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
        
    # Draw random ellipses
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
    
    return img

def postprocess(img):
    if(img.device!='cpu'):
        img = img.cpu()
    img = deepcopy(img.numpy())
    img = np.transpose(img, (0, 2, 3, 1))
    if(img.shape[-1]==1):
        img = np.tile(img, (1,1,1,3))
    return img

def expandMask(mask, expandD):
    return mask.repeat(1,expandD,1,1)

def intervalMaximizeSaturation(x, alpha=100, delta=32/255):
    # Tensor: torch.tensor [Batch_size, C, H, W] [0,1]
    if(torch.is_tensor(x)):
        if(len(x.size())!=4):
            print('ERROR: Input dim should be 4.')
            return None
        c = x.size(1)
        x_max = torch.max(x, 1)[0].unsqueeze(1)
        x_min = torch.min(x, 1)[0].unsqueeze(1)
        x_max = x_max.repeat(1,c,1,1)
        x_min = x_min.repeat(1,c,1,1)
        L = (torch.floor((x-x_min)/delta) + torch.min(torch.ceil((x-x_min)/delta),(x_max-x_min)/delta))*delta/2 + x_min
        max_saturation_x = x + (x-L)*alpha
        result = torch.clamp(max_saturation_x, 0.0, 1.0)

    # Image: numpy.ndarray [H, W, C]/[H, W] [0,1]/[0,255]
    elif(isinstance(x, np.ndarray)):
        # gray to rgb
        if len(x.shape) < 3:
            x = gray2rgb(x)
        # uint8 to float
        if(np.issubdtype(x.dtype, np.uint8)):
            x = np.asarray(x, dtype=np.float) / 255.0
        x_max = np.expand_dims(np.max(x, axis=-1), axis=-1)
        x_min = np.expand_dims(np.min(x, axis=-1), axis=-1)
        x_max = np.tile(x_max, (1,1,3))
        x_min = np.tile(x_min, (1,1,3))
        L = (np.floor((x-x_min)/delta) + np.minimum(np.ceil((x-x_min)/delta),(x_max-x_min)/delta))*delta/2 + x_min
        max_saturation_x = x + (x-L)*alpha
        result = np.clip(max_saturation_x, 0.0, 1.0)
        # float to uint8
        result = np.asarray(result*255, dtype=np.uint8)
    
    # Others
    else:
        print('Error: Input should be torch.tensor or numpy.ndarray, but get ' + str(type(x)) + '.')
        return None

    return result


def draw(img, width, height, nums):
    # Set size scale
    size = int((width + height) * 0.03)

    # Draw random lines
    for _ in range(randint(1, nums)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(3, size)
        cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
        
    # Draw random circles
    for _ in range(randint(1, nums)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, size)
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
        
    # Draw random ellipses
    for _ in range(randint(1, nums)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
    
def getRatio(img):
    width, height = img.shape
    return np.sum(img)/(width*height)

def genRatioMask(height, width, start_ratio, end_ratio):
    while(True):
        img = np.zeros((width, height))
        draw(img, width, height, 10)
        if(getRatio(img)>=end_ratio):
            continue
        elif(getRatio(img)>start_ratio):
            break
        else:
            label = 0
            while(True):
                draw(img, width, height, 5)
                if(getRatio(img)>=end_ratio):
                    break
                elif(getRatio(img)>start_ratio):
                    label = 1
                    break
                else:
                    continue
            if(label==1):
                break
            else:
                continue
    mask = np.asarray(img>0.5, np.float32)
    return mask

def batchRatioMask(height, width, start_ratio, end_ratio, batch_size):
    batch_mask = np.zeros((batch_size, 1, height, width), dtype=np.float32)
    for i in range(batch_size):
        batch_mask[i,0] = genRatioMask(height, width, start_ratio, end_ratio)
    batch_mask_tensor = torch.from_numpy(batch_mask)
    return batch_mask_tensor
    