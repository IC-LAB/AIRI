import os

import torch
import torch.nn as nn
import torch.optim as optim
from networks import Generator, Discriminator
from loss import AdversarialLoss, MultiHiddenSpaceLoss, IntervalMaxSaturationLoss
from utils import postprocess, expandMask

class AIRI(nn.Module):
    def __init__(self, GPUs):
        super(AIRI, self).__init__()

        # init models
        edge_generator = torch.load('./checkpoint/edge/edge_gen.pth')
        generator = Generator(5, 3, 8)
        discriminator = Discriminator(4)
        if(len(GPUs)>1):
            edge_generator = nn.DataParallel(edge_generator)
            generator = nn.DataParallel(generator)
            discriminator = nn.DataParallel(discriminator)
        self.add_module('edge_generator', edge_generator)
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        # init losses
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss()
        mhs_loss = MultiHiddenSpaceLoss()
        ims_loss = IntervalMaxSaturationLoss()
        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('mhs_loss', mhs_loss)
        self.add_module('ims_loss', ims_loss)

        # init optimizer
        self.gen_optimizer = optim.Adam(params=generator.parameters(), lr=0.0001, betas=(0.0,0.9))
        self.dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=0.0001, betas=(0.0,0.9))
    
    def process(self, image, image_gray, edge, mask):
        rgb_mask = expandMask(mask, 3)

        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # edge inpainting
        edge_masked = edge * (1-mask)
        image_gray_masked = image_gray * (1-mask) + mask
        
        edge_inputs = torch.cat((image_gray_masked, edge_masked, mask), dim=1) 
        edge_inpaintinged = self.edge_generator(edge_inputs)
        if True or np.random.binomial(1,0.5)>0:
            edge_outputs = edge_inpaintinged*mask + edge*(1-mask)
            edge_outputs = edge_outputs.detach()
        else:
            edge_outputs = edge

        # rgb inpainting
        rgb_masked = image * (1-rgb_mask) + rgb_mask
        inputs = torch.cat((rgb_masked, edge_outputs, mask), dim=1) 
        outputs = self.generator(inputs)

        # discriminator loss
        dis_input_real = torch.cat((image, mask), dim=1) 
        dis_input_fake = torch.cat((outputs.detach(), mask), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss = (dis_real_loss+dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = torch.cat((outputs, mask), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)
        gen_adversarial_loss = self.adversarial_loss(gen_fake, True, False)

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, image) / torch.mean(rgb_mask)

        # generator multihiddenspace perceptual loss
        gen_mhs_loss = self.mhs_loss(outputs, image)

        # generator internalMaximizeSaturation loss
        gen_ims_loss = self.ims_loss(outputs, image)

        # generator loss
        gen_loss = 0.1*gen_adversarial_loss + gen_l1_loss + 0.1*gen_mhs_loss + 0.5*gen_ims_loss

        logs = {
            'dis_loss': dis_loss.item(),
            'gen_loss': gen_loss.item(),
            'gen_adversarial_loss': gen_adversarial_loss.item(),
            'gen_l1_loss': gen_l1_loss.item(),
            'gen_mhs_loss': gen_mhs_loss.item(),
            'gen_ims_loss': gen_ims_loss.item()
        }

        return outputs, dis_loss, gen_loss, logs
    
    def forward(self, image, image_gray, edge, mask):
        rgb_mask = expandMask(mask, 3)
        # edge inpainting
        edge_masked = edge * (1-mask)
        image_gray_masked = image_gray * (1-mask) + mask
        
        edge_inputs = torch.cat((image_gray_masked, edge_masked, mask), dim=1) 
        edge_inpaintinged = self.edge_generator(edge_inputs)
        edge_outputs = edge_inpaintinged*mask + edge*(1-mask)
        edge_outputs = edge_outputs.detach()

        # rgb inpainting
        rgb_masked = image * (1-rgb_mask) + rgb_mask
        inputs = torch.cat((rgb_masked, edge_outputs, mask), dim=1) 
        outputs = self.generator(inputs)
        outputs = outputs.detach()

        return outputs
    
    def backward(self, dis_loss=None, gen_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()
        gen_loss.backward()
        self.gen_optimizer.step()
    
    def load(self, checkpoint_path):
        self.generator = torch.load(os.path.join(checkpoint_path, 'gen.pth'))
        self.discriminator = torch.load(os.path.join(checkpoint_path, 'dis.pth'))
    
    def save(self, checkpoint_path):
        torch.save(self.generator, os.path.join(checkpoint_path, 'gen.pth'))
        torch.save(self.discriminator, os.path.join(checkpoint_path, 'dis.pth'))
