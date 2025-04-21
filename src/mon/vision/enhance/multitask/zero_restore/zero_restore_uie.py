#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-shot Single Image Restoration through Controlled
Perturbation of Koschmieder's Model," CVPR 2021.

References:
	- https://github.com/aupendu/zero-restore
"""

__all__ = [
    "ZeroRestoreUIE",
]

import random

import torch

from mon import core, nn
from mon.constants import MLType, MODELS, Task
from mon.vision import geometry, types
from mon.vision.enhance import base

torch.autograd.set_detect_anomaly(True)

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Module -----
class DoubleConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class InDoubleConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 9, stride=4, padding=4, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class InConv(nn.Module):
    
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride=4, padding=3, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        self.convf = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=False)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        R    = x[:, 0:1, :, :]
        G    = x[:, 1:2, :, :]
        B    = x[:, 2:3, :, :]
        xR   = torch.unsqueeze(self.conv(R), 1)
        xG   = torch.unsqueeze(self.conv(G), 1)
        xB   = torch.unsqueeze(self.conv(B), 1)
        x    = torch.cat([xR, xG, xB], 1)
        x, _ = torch.min(x, dim=1)
        return self.convf(x)


class SKConv(nn.Module):
    
    def __init__(
        self,
        in_channels : int = 1,
        out_channels: int = 64,
        M           : int = 4,
        L           : int = 32
    ):
        super().__init__()
        self.M     = M
        self.convs = nn.ModuleList([])
        in_conv    = InConv(in_channels, out_channels)
        for i in range(M):
            if i == 0:
                self.convs.append(in_conv)
            else:
                self.convs.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=1 / (2 ** i), mode="bilinear", align_corners=True),
                        in_conv,
                        nn.Upsample(scale_factor=2 ** i, mode="bilinear", align_corners=True)
                    )
                )
        self.fc  = nn.Linear(out_channels, L)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(L, out_channels))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            fea = torch.unsqueeze(fea, 1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_u = torch.sum(feas, dim=1)
        fea_s = fea_u.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z)
            vector = torch.unsqueeze(vector, 1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = torch.unsqueeze(attention_vectors, -1)
        attention_vectors = torch.unsqueeze(attention_vectors, -1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class Estimation(nn.Module):
    
    def __init__(self, num_channels: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.in_conv      = SKConv(1, num_channels, 3, 32)
        # Transmission Map
        self.conv_t1  = DoubleConv(num_channels, num_channels)
        self.conv_t2  = nn.Conv2d(num_channels, 3, 3, padding=1, stride=1, bias=False, padding_mode="reflect")
        self.up       = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        # Atmospheric Light
        self.conv_a1  = InDoubleConv(3, num_channels)
        self.conv_a2  = DoubleConv(num_channels, num_channels)
        self.maxpool  = nn.MaxPool2d(15, 7)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.dense    = nn.Linear(num_channels, 3, bias=False)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x     = input
        x_min = self.in_conv(x)
        trans = self.conv_t2(self.up(self.conv_t1(x_min)))
        trans = torch.sigmoid(trans) + 1e-12
        atm   = self.conv_a1(x)
        atm   = torch.mul(atm, x_min)
        atm   = self.pool(self.conv_a2(self.maxpool(atm)))
        atm   = atm.view(-1, self.num_channels)
        atm   = torch.sigmoid(self.dense(atm))
        return trans, atm


# ----- Model -----
@MODELS.register(name="zero_restore_uie", arch="zero_restore")
class ZeroRestoreUIE(base.ImageEnhancementModel):
    """Zero-Restore model for underwater image enhancement.
    
    References:
	    - https://github.com/aupendu/zero-restore
    """
    
    arch     : str          = "zero_restore"
    name     : str          = "zero_restore_uie"
    tasks    : list[Task]   = [Task.UIE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        in_channels : int = 3,
        num_channels: int = 64,
        iters       : int = 10000,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.in_channels  = in_channels
        self.num_channels = num_channels
        self.iters        = iters
        
        # Network
        self.estimation = Estimation(self.num_channels)
        
        # Optimizer
        self.configure_optimizers()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    # ----- Initialize -----s
    def init_weights(self, m: nn.Module):
        """Initializes the model's weights.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:  # 0.02
            m.weight.data.normal_(0.0, 0.001)
        if classname.find("Linear") != -1:  # 0.02
            m.weight.data.normal_(0.0, 0.001)
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"enhanced"`` keys.
        """
        # Forward 1
        outputs     = self.forward(datapoint=datapoint, *args, **kwargs)
        image       = datapoint["image"]
        trans_map   = outputs["trans"]
        atm_map     = outputs["atm"]
        enhanced    = outputs["enhanced"]
        # Forward 2
        p_x         = 0.9
        image_x     = image * p_x + (1 - p_x) * atm_map
        outputs_x   = self.forward(datapoint={"image": image_x}, *args, **kwargs)
        trans_map_x = outputs_x["trans"]
        atm_map_x   = outputs_x["atm"]
        enhanced_x  = outputs_x["enhanced"]
        
        # Loss
        o_tensor = torch.ones(enhanced.shape).to(self.device)
        z_tensor = torch.zeros(enhanced.shape).to(self.device)
        loss_t   = torch.sum((trans_map_x - p_x * trans_map) ** 2)
        loss_a   = torch.sum((atm_map - atm_map_x) ** 2)
        loss_mx  =   torch.sum(torch.max(enhanced, o_tensor)) + torch.sum(torch.max(enhanced_x, o_tensor)) - 2 * torch.sum(o_tensor)
        loss_mn  = - torch.sum(torch.min(enhanced, z_tensor)) - torch.sum(torch.min(enhanced_x, z_tensor))
        loss_col = nn.ColorConstancyLoss()(enhanced)
        loss_tv  = nn.TotalVariationLoss()(enhanced)
        loss     = 0.001 * loss_tv + loss_t + loss_a + 0.001 * loss_mx + 0.001 * loss_mn + 1000 * loss_col
        
        return outputs | {
			"loss": loss,
		}
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass of the model.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"enhanced"`` keys.
        """
        image      = datapoint["image"]
        trans, atm = self.estimation(image)
        atm        = torch.unsqueeze(torch.unsqueeze(atm, 2), 2)
        atm        = atm.expand_as(image)
        trans      = trans.expand_as(image)
        enhanced   = (image - (1 - trans.clone()) * atm) / trans
        
        return {
            "trans"   : trans,
            "atm"     : atm,
            "enhanced": enhanced,
        }
    
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        it = random.randint(0, 7)
        if it == 1:
            image = image.rot90(1, [2, 3])
        if it == 2:
            image = image.rot90(2, [2, 3])
        if it == 3:
            image = image.rot90(3, [2, 3])
        if it == 4:
            image = image.flip(2).rot90(1, [2, 3])
        if it == 5:
            image = image.flip(3).rot90(1, [2, 3])
        if it == 6:
            image = image.flip(2)
        if it == 7:
            image = image.flip(3)
        return image
    
    # ----- Predict -----
    def infer(self, datapoint: dict, reset_weights: bool = True, *args, **kwargs) -> dict:
        """Infers model output with optional processing.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
            reset_weights: Whether to reset the weights before training. Default is ``True``.
            
        Returns:
            ``dict`` of model predictions.
    
        Notes:
            Override for custom pre/post-processing; defaults to ``self.forward()``.
        """
        # Initialize training components
        if reset_weights:
            self.load_state_dict(self.initial_state_dict, strict=False)
        optimizer = self.optimizer.get("optimizer", None)
        optimizer = optimizer or nn.Adam(self, lr=1e-3, weight_decay=1e-2)
        
        # Input
        image  = datapoint["image"].to(self.device)
        h0, w0 = types.image_size(image)
        image  = geometry.resize(image, divisible_by=32)
        
        # Optimize
        timer = core.Timer()
        timer.tick()
        self.train()
        for _ in range(self.iters):
            image_  = self.augment(image)
            outputs = self.forward_loss(datapoint={"image": image_})
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
        self.eval()
        outputs = self.forward(datapoint={"image": image})
        timer.tock()
        
        # Post-processing
        enhanced = outputs["enhanced"]
        enhanced = geometry.resize(enhanced, (h0, w0))
        enhanced = torch.clamp(enhanced, 0, 1)
        
        # Return
        return outputs | {
            "enhanced": enhanced,
            "time"    : timer.avg_time,
        }
