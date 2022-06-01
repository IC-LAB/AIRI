# coding: utf-8

import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import DataLoader

from model import AIRI
from dataset import AdaDataset, EdgeDataset
from utils import postprocess, expandMask, batchRatioMask

GPUs = [0,1,2,3]
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUs)[1:-1].replace(' ', '')

if torch.cuda.is_available():
    gpu_device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    gpu_device = torch.device("cpu")
print('[TORCH DEVICE]: ' + str(gpu_device))

# Training settings
batch_size = 8
checkpoint_path = './checkpoint'
sample_path = './sample'
log_path = './log'
model_name = 'airi'
if not os.path.exists(os.path.join(checkpoint_path,model_name)):
    os.mkdir(os.path.join(checkpoint_path,model_name))
if not os.path.exists(os.path.join(sample_path,model_name)):
    os.mkdir(os.path.join(sample_path,model_name))

# Dataset
data_path = './dataset/place365'
fpath = os.path.join(data_path, 'data_large')
train_dataset = AdaDataset(fpath)
vpath = os.path.join(data_path, 'val_large')
value_dataset = EdgeDataset(vpath)
# Data loader
train_loader = DataLoader(
    dataset= train_dataset,
    batch_size= batch_size,
    num_workers= 8,
    drop_last= True,
    shuffle= True
)
value_sample_iterator = value_dataset.createIterator(4)

# model
model = AIRI(GPUs).to(gpu_device)

def sample(epoch, batch_idx):
    if(len(value_dataset)==0):
        return
    
    model.eval()

    image, image_gray, edge, mask = next(value_sample_iterator)
    image = image.to(gpu_device)
    image_gray = image_gray.to(gpu_device)
    edge = edge.to(gpu_device)
    mask = mask.to(gpu_device)
    rgb_mask = expandMask(mask, 3)

    rgb_masked = image * (1-rgb_mask) + rgb_mask
    outputs = model(image, image_gray, edge, mask)

    gen_imgs = np.concatenate([postprocess(rgb_masked), postprocess(outputs.detach()), postprocess(image)])
    titles = ['Input', 'Output', 'GT']
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    cnt = 0
    for i in range(3):
        for j in range(4):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[i])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(sample_path,model_name,"%d_%d.png" % (epoch, batch_idx)))
    plt.close()

def train(epoch):
    total = len(train_dataset)
    if(total==0):
        print('No training data was provided!')
        return
    
    for batch_idx, (image, image_gray, edge) in enumerate(train_loader):
        model.train()

        rand_ratio = random.uniform(0, 6)
        start_ratio = np.floor(rand_ratio) / 10
        end_ratio = np.ceil(rand_ratio) / 10
        mask = batchRatioMask(image.size(2), image.size(3), start_ratio, end_ratio, image.size(0))

        image = image.to(gpu_device)
        image_gray = image_gray.to(gpu_device)
        edge = edge.to(gpu_device)
        mask = mask.to(gpu_device)

        outputs, dis_loss, gen_loss, log = model.process(image, image_gray, edge, mask)
        model.backward(dis_loss, gen_loss)

        if(batch_idx%10==0):
            disp_str = 'Epoch:{} [{}/{}], D_loss:{:.6f} G_Loss:{:.6f} [0.1*{:.6f}+{:.6f}+0.1*{:.6f}+0.5*{:.6f}]'\
                .format(epoch, batch_idx*batch_size, total, log['dis_loss'], log['gen_loss'],\
                    log['gen_adversarial_loss'], log['gen_l1_loss'], log['gen_mhs_loss'], log['gen_ims_loss'])
            with open(os.path.join(log_path,model_name+'.txt'), 'a') as f:
                f.write('%s\n'% disp_str)
            print(disp_str)    

        if(batch_idx%100==0):
            print('>>Saving ...')
            model.save(os.path.join(checkpoint_path, model_name))
            print('>>Sampling ...')
            sample(epoch, batch_idx)

for epoch in range(1,6):
    train(epoch)