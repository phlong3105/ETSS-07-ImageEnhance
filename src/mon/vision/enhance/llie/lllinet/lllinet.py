#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LLLInet.

Low-Light Learnable Instance Normalization Network.
"""

from __future__ import annotations

__all__ = [
    "LLLINet",
    "LLLINetHVI",
]

from typing import Any, Literal

import torch
from torchvision.models import vgg

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Loss

class Loss(nn.Loss):
    """
    λ1, λ2, λ3, λ4 = {0.40, 0.05, 0.15, 0.40} for the LOL dataset.
    λ1, λ2, λ3, λ4 = {0.35, 0.10, 0.25, 0.30} for the VE-LOL dataset.
    """
    
    def __init__(
        self,
        str_weight: float = 0.35,
        tv_weight : float = 0.10,
        reg_weight: float = 0.25,
        per_weight: float = 0.30,
        reduction : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs, reduction=reduction)
        self.str_weight = str_weight
        self.tv_weight  = tv_weight
        self.reg_weight = reg_weight
        self.per_weight = per_weight
        
        self.ms_ssim_loss = nn.MSSSIMLoss(data_range=1.0, reduction=reduction)
        self.ssim_loss    = nn.SSIMLoss(data_range=1.0, non_negative_ssim=True, reduction=reduction)
        # self.psnr_loss    = nn.PSNRLoss(reduction=reduction)
        self.per_loss     = nn.PerceptualLoss(
            net        = vgg.vgg19(weights=vgg.VGG19_Weights.IMAGENET1K_V1).features,
            layers     = ["26"],
            preprocess = True,
            reduction  = reduction,
        )
        self.tv_loss = nn.TotalVariationLoss(reduction=reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, *_) -> torch.Tensor:
        str_loss  = self.ms_ssim_loss(input, target) + self.ssim_loss(input, target)
        # psnr_loss = self.psnr_loss(input, target)
        # str_loss  = 0.6 * ssim_loss + 0.4 * psnr_loss
        per_loss  = self.per_loss(input, target)
        reg_loss  = self.region_loss(input, target)
        tv_loss   = self.tv_loss(input)
        loss      = (
              self.str_weight * str_loss
            + self.tv_weight  * tv_loss
            + self.reg_weight * reg_loss
            + self.per_weight * per_loss
        )
        return loss
    
    @staticmethod
    def region_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gray     = (
              0.30 * target[:, 0, :, :]
            + 0.59 * target[:, 1, :, :]
            + 0.11 * target[:, 2, :, :]
        )
        gray     = gray.view(-1)
        value    = -torch.topk(-gray, int(gray.shape[0] * 0.4))[0][0]
        weight   = 1 * (target > value) + 4 * (target <= value)
        abs_diff = torch.abs(input - target)
        return torch.mean(weight * abs_diff)
    
# endregion


# region Module

class UNetConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        relu_slope  : float = 0.2,
        use_in      : bool  = True,
    ):
        super().__init__()
        self.use_in  = use_in
        self.conv1   = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True)
        if use_in:
            self.norm1 = nn.LearnableInstanceNorm2d(in_channels, r=0.5, affine=True)
        else:
            self.norm1 = nn.Identity()
            # self.in_ratio = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.relu1   = nn.LeakyReLU(relu_slope, inplace=False)
        self.simam   = nn.SimAM()
        
        self.conv2   = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True)
        
        self.conv3   = nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=True)
        self.relu3   = nn.LeakyReLU(relu_slope, inplace=False)
        
        self.conv4   = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.relu4   = nn.LeakyReLU(relu_slope, inplace=False)
        
        self.conv1_3 = nn.Conv2d(in_channels,     in_channels,  1, 1, 0)
        self.conv3_4 = nn.Conv2d(in_channels * 2, out_channels, 1, 1, 0)
    
    @property
    def in_ratio(self):
        return self.norm1.r if self.use_in else 0
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        #
        x1   = self.conv1(x)
        x1   = self.norm1(x1)
        x1   = self.relu1(x1)
        x1   = self.simam(x1)
        #
        x2   = self.conv2(x1)
        #
        x3   = torch.cat([x2, self.conv1_3(x)], dim=1)
        x3_4 = self.conv3_4(x3)
        #
        x3   = self.conv3(x3)
        x3   = self.relu3(x3)
        x4   = self.conv4(x3)
        x4   = self.relu4(x4)
        #
        x4  += x3_4
        #
        return x4

# endregion


# region Model

@MODELS.register(name="lllinet", arch="lllinet")
class LLLINet(base.ImageEnhancementModel):
    """Low-Light Learnable Instance Normalization Network."""
    
    model_dir: core.Path    = current_dir
    arch     : str          = "lllinet"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def __init__(
        self,
        in_channels : int = 3,
        out_channels: int = 3,
        weights     : Any = None,
        loss_weights: list[float] = [0.35, 0.10, 0.25, 0.30],
        *args, **kwargs
    ):
        super().__init__(
            name         = "lllinet",
            in_channels  = in_channels,
            out_channels = out_channels,
            weights      = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels",  in_channels)
            out_channels = self.weights.get("out_channels", out_channels)
        self.in_channels  = in_channels or self.in_channels
        self.out_channels = out_channels or self.out_channels
        
        # Construct model
        nb_filter    = [32, 64, 128, 256, 512]
        #
        self.pool    = nn.MaxPool2d(2, 2)
        self.up      = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #
        self.conv0_0 = UNetConvBlock(self.in_channels, nb_filter[0], use_in=False)
        self.conv1_0 = UNetConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = UNetConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = UNetConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = UNetConvBlock(nb_filter[3], nb_filter[4])
        #
        self.conv0_1 = UNetConvBlock(nb_filter[0] + nb_filter[1] + nb_filter[1], nb_filter[0])
        self.conv1_1 = UNetConvBlock(nb_filter[1] + nb_filter[2] + nb_filter[2], nb_filter[1])
        self.conv2_1 = UNetConvBlock(nb_filter[2] + nb_filter[3] + nb_filter[3], nb_filter[2])
        self.conv3_1 = UNetConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        #
        self.conv0_2 = UNetConvBlock(nb_filter[0] * 2 + nb_filter[1] + nb_filter[1], nb_filter[0])
        self.conv1_2 = UNetConvBlock(nb_filter[1] * 2 + nb_filter[2] + nb_filter[2], nb_filter[1])
        self.conv2_2 = UNetConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        #
        self.conv0_3 = UNetConvBlock(nb_filter[0] * 3 + nb_filter[1] + nb_filter[1], nb_filter[0])
        self.conv1_3 = UNetConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        #
        self.conv0_4 = UNetConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        #
        self.final   = nn.Conv2d(nb_filter[0], self.out_channels, kernel_size=1)
        
        # Loss
        self.loss = Loss(*loss_weights, reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x    = datapoint.get("image")
        #
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        #
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.up(x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0), self.up(x1_1)], 1))
        #
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.up(x2_2)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1), self.up(x1_2)], 1))
        #
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2), self.up(x1_3)], 1))
        #
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        #
        y    = self.final(x0_4)
        y    = torch.clamp(y, 0, 1)
        #
        return {"enhanced": y}
    

@MODELS.register(name="lllinet_hvi", arch="lllinet")
class LLLINetHVI(LLLINet):
    """Low-Light Learnable Instance Normalization Network."""
    
    zoo: dict = {}
    
    def __init__(
        self,
        in_channels : int         = 3,
        out_channels: int         = 3,
        weights     : Any         = None,
        hvi_weight  : float       = 1.0,
        loss_weights: list[float] = [0.35, 0.10, 0.25, 0.30],
        *args, **kwargs
    ):
        super().__init__(
            name         = "lllinet_hvi",
            arch         = "lllinet",
            in_channels  = in_channels,
            out_channels = out_channels,
            weights      = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            out_channels = self.weights.get("out_channels", out_channels)
            hvi_weight   = self.weights.get("hvi_weight"  , hvi_weight)
            loss_weights = self.weights.get("loss_weights", loss_weights)
        self.in_channels  = in_channels  or self.in_channels
        self.out_channels = out_channels or self.out_channels
        self.hvi_weight   = hvi_weight
        self.loss_weights = loss_weights
        
        # Construct model
        nb_filter    = [32, 64, 128, 256, 512]
        #
        self.pool    = nn.MaxPool2d(2, 2)
        self.up      = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        #
        self.conv0_0 = UNetConvBlock(self.in_channels, nb_filter[0], use_in=False)
        self.conv1_0 = UNetConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = UNetConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = UNetConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = UNetConvBlock(nb_filter[3], nb_filter[4])
        #
        self.conv0_1 = UNetConvBlock(nb_filter[0] + nb_filter[1] + nb_filter[1], nb_filter[0])
        self.conv1_1 = UNetConvBlock(nb_filter[1] + nb_filter[2] + nb_filter[2], nb_filter[1])
        self.conv2_1 = UNetConvBlock(nb_filter[2] + nb_filter[3] + nb_filter[3], nb_filter[2])
        self.conv3_1 = UNetConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])
        #
        self.conv0_2 = UNetConvBlock(nb_filter[0] * 2 + nb_filter[1] + nb_filter[1], nb_filter[0])
        self.conv1_2 = UNetConvBlock(nb_filter[1] * 2 + nb_filter[2] + nb_filter[2], nb_filter[1])
        self.conv2_2 = UNetConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        #
        self.conv0_3 = UNetConvBlock(nb_filter[0] * 3 + nb_filter[1] + nb_filter[1], nb_filter[0])
        self.conv1_3 = UNetConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        #
        self.conv0_4 = UNetConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        #
        self.final   = nn.Conv2d(nb_filter[0], self.out_channels, kernel_size=1)
        #
        self.trans   = core.RGBToHVI()
        
        # Loss
        # from mon.vision.enhance.llie.hvi_cidnet import Loss as CIDNetLoss
        self.loss = Loss(*self.loss_weights, reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        pred_rgb   = outputs.get("enhanced")
        pred_hvi   = self.rgb_to_hvi(pred_rgb)
        target_rgb = datapoint.get("ref_image")
        target_hvi = self.rgb_to_hvi(target_rgb)
        loss_rgb   = self.loss(pred_rgb, target_rgb)
        loss_hvi   = self.loss(pred_hvi, target_hvi)
        loss       = loss_rgb + self.hvi_weight * loss_hvi
        # Return
        return {
            "enhanced": pred_rgb,
            "hvi_k"   : float(self.trans.density_k.item()),
            "loss"    : loss,
        }
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x    = datapoint.get("image")
        hvi  = self.trans.rgb_to_hvi(x)
        #
        x0_0 = self.conv0_0(hvi)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        #
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.up(x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.up(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0), self.up(x1_1)], 1))
        #
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.up(x2_2)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1), self.up(x1_2)], 1))
        #
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2), self.up(x1_3)], 1))
        #
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        #
        y    = self.final(x0_4)
        y    = self.trans.hvi_to_rgb(y)
        y    = torch.clamp(y, 0, 1)
        return {"enhanced": y}
    
    def rgb_to_hvi(self, input: torch.Tensor) -> torch.Tensor:
        return self.trans.rgb_to_hvi(input)
    
# endregion
