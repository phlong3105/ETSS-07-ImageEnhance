# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the ICCV 2021 paper:
"Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement": https://arxiv.org/abs/2110.00984

Please cite the paper if you use this code

@InProceedings{Zheng_2021_ICCV,
    author    = {Zheng, Chuanjun and Shi, Daming and Shi, Wentian},
    title     = {Adaptive Unfolding Total Variation Network for Low-Light Image Enhancement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4439-4448}
}

Tested with Pytorch 1.7.1, Python 3.6

Author: Chuanjun Zheng (chuanjunzhengcs@gmail.com)

'''
import torch
import torch.fft
import torch.nn as nn
from models import basicblock as B, lc_model, nli_model, ns_model, utv_model


class UTVNet(nn.Module):
    
    def __init__(self):
        super(UTVNet, self).__init__()
        self.a = utv_model.ADMM(1, 8, 1)
        self.noiselevel = nli_model.IRCNN(3, 24, 32)
        self.denoise = ns_model.UNet()
        self.LIGHT = lc_model.LIRCNN(3, 3, 48)
        self.device = torch.device('cuda')
        self.hyp = B.HyPaNet()

    def forward(self, x):
        levelr, levelg, levelb, level = self.noiselevel(x)
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        level1 = levelr
        level2 = levelg
        level3 = levelb
        level_rgb = level
        smoothr = self.a(r, level1.squeeze(0)).unsqueeze(0).unsqueeze(0)
        smoothg = self.a(g, level2.squeeze(0)).unsqueeze(0).unsqueeze(0)
        smoothb = self.a(b, level3.squeeze(0)).unsqueeze(0).unsqueeze(0)
        smooth_rgb = torch.cat((smoothr, smoothg, smoothb), 1)
        denoise = self.denoise(x - smooth_rgb, level_rgb)
        smooth = self.LIGHT(smooth_rgb)
        out = denoise + smooth
        return out


def make_model(args, parent=False):
    return UTVNet()
