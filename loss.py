import os

import torch
import torch.nn as nn
import torchvision.models as models

from utils import intervalMaximizeSaturation

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs).cuda()
            loss = self.criterion(outputs, labels)
            return loss

class MultiHiddenSpaceLoss(nn.Module):
    def __init__(self):
        super(MultiHiddenSpaceLoss, self).__init__()
        self.add_module('hs_model_1', HiddenSpace('./checkpoint/four_autoencoder/ae_1.pth').cuda())
        self.add_module('hs_model_2', HiddenSpace('./checkpoint/four_autoencoder/ae_2.pth').cuda())
        self.add_module('hs_model_3', HiddenSpace('./checkpoint/four_autoencoder/ae_3.pth').cuda())
        self.add_module('hs_model_4', HiddenSpace('./checkpoint/four_autoencoder/ae_4.pth').cuda())
        self.criterion = torch.nn.L1Loss()
    
    def __call__(self, x, y):
        x_hs_1 = self.hs_model_1(x)
        y_hs_1 = self.hs_model_1(y)
        x_hs_2 = self.hs_model_2(x)
        y_hs_2 = self.hs_model_2(y)
        x_hs_3 = self.hs_model_3(x)
        y_hs_3 = self.hs_model_3(y)
        x_hs_4 = self.hs_model_4(x)
        y_hs_4 = self.hs_model_4(y)

        mhs_loss = 0.0
        mhs_loss += self.criterion(x_hs_1['conv1'], y_hs_1['conv1'])
        mhs_loss += self.criterion(x_hs_1['conv2'], y_hs_1['conv2'])
        mhs_loss += self.criterion(x_hs_1['conv3'], y_hs_1['conv3'])
        mhs_loss += self.criterion(x_hs_1['code'], y_hs_1['code'])
        mhs_loss += self.criterion(x_hs_2['conv1'], y_hs_2['conv1'])
        mhs_loss += self.criterion(x_hs_2['conv2'], y_hs_2['conv2'])
        mhs_loss += self.criterion(x_hs_2['conv3'], y_hs_2['conv3'])
        mhs_loss += self.criterion(x_hs_2['code'], y_hs_2['code'])
        mhs_loss += self.criterion(x_hs_3['conv1'], y_hs_3['conv1'])
        mhs_loss += self.criterion(x_hs_3['conv2'], y_hs_3['conv2'])
        mhs_loss += self.criterion(x_hs_3['conv3'], y_hs_3['conv3'])
        mhs_loss += self.criterion(x_hs_3['code'], y_hs_3['code'])
        mhs_loss += self.criterion(x_hs_4['conv1'], y_hs_4['conv1'])
        mhs_loss += self.criterion(x_hs_4['conv2'], y_hs_4['conv2'])
        mhs_loss += self.criterion(x_hs_4['conv3'], y_hs_4['conv3'])
        mhs_loss += self.criterion(x_hs_4['code'], y_hs_4['code'])

        return mhs_loss

class HiddenSpace(torch.nn.Module):
    def __init__(self, model_path):
        super(HiddenSpace, self).__init__()
        model = torch.load(model_path)

        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.conv3 = model.conv3
        self.middle = model.middle

        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        code = self.middle(conv3)

        out = {
            'conv1': conv1,
            'conv2': conv2,
            'conv3': conv3,
            'code': code
        }

        return out

class IntervalMaxSaturationLoss(nn.Module):
    def __init__(self):
        super(IntervalMaxSaturationLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
    
    def forward(self, x, y):
        ims_x = intervalMaximizeSaturation(x)
        ims_y = intervalMaximizeSaturation(y)
        ims_loss = self.criterion(ims_x, ims_y)
        return ims_loss