#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements RegNet models."""

__all__ = [
    "RegNetX_32GF",
    "RegNet_X_16GF",
    "RegNet_X_1_6GF",
    "RegNet_X_3_2GF",
    "RegNet_X_400MF",
    "RegNet_X_800MF",
    "RegNet_X_8GF",
    "RegNet_Y_128GF",
    "RegNet_Y_16GF",
    "RegNet_Y_1_6GF",
    "RegNet_Y_32GF",
    "RegNet_Y_3_2GF",
    "RegNet_Y_400MF",
    "RegNet_Y_800MF",
    "RegNet_Y_8GF",
]

from abc import ABC
from typing import Any

from torchvision.models import (
    regnet_x_16gf, regnet_x_1_6gf, regnet_x_32gf, regnet_x_3_2gf,
    regnet_x_400mf, regnet_x_800mf, regnet_x_8gf, regnet_y_128gf,
    regnet_y_16gf, regnet_y_1_6gf, regnet_y_32gf, regnet_y_3_2gf,
    regnet_y_400mf, regnet_y_800mf, regnet_y_8gf,
)

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class RegNet(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """RegNet model for image classification.

    References:
        - https://arxiv.org/abs/2003.13678
    """
    
    arch     : str          = "regnet"
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        """Initializes weights for the model.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        pass
    
    # ----- Forward Pass -----
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass on the model.
    
        Args:
            datapoint: ``dict`` with image data.
    
        Returns:
            ``dict`` of predictions with ``"logits"`` keys.
        """
        x = datapoint["image"]
        y = self.model(x)
        return {"logits": y}
    

@MODELS.register(name="regnet_y_400mf", arch="regnet")
class RegNet_Y_400MF(RegNet):
    """RegNet-Y-400MF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_y_400mf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_400mf/imagenet1k_v1/regnet_y_400mf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_400mf-e6988f5f.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_400mf/imagenet1k_v2/regnet_y_400mf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_y_400mf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="regnet_y_800mf", arch="regnet")
class RegNet_Y_800MF(RegNet):
    """RegNet-Y-800MF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_y_800mf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_800mf/imagenet1k_v1/regnet_y_800mf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_800mf-58fc7688.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_800mf/imagenet1k_v2/regnet_y_800mf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_y_800mf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    

@MODELS.register(name="regnet_y_1_6gf", arch="regnet")
class RegNet_Y_1_6GF(RegNet):
    """RegNet-Y-1.6GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_y_1_6gf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_1_6gf/imagenet1k_v1/regnet_y_1_6gf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_1_6gf-0d7bc02a.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_1_6gf/imagenet1k_v2/regnet_y_1_6gf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_y_1_6gf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="regnet_y_3_2gf", arch="regnet")
class RegNet_Y_3_2GF(RegNet):
    """RegNet-Y-3.2GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_y_3_2gf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_3_2gf/imagenet1k_v1/regnet_y_3_2gf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_3_2gf-9180c971.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_3_2gf/imagenet1k_v2/regnet_y_3_2gf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_y_3_2gf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="regnet_y_8gf", arch="regnet")
class RegNet_Y_8GF(RegNet):
    """RegNet-Y-8GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_y_8gf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_8gf/imagenet1k_v1/regnet_y_8gf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_8gf-dc2b1b54.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_8gf/imagenet1k_v2/regnet_y_8gf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_y_8gf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="regnet_y_16gf", arch="regnet")
class RegNet_Y_16GF(RegNet):
    """RegNet-Y-16GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_y_16gf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_16gf/imagenet1k_v1/regnet_y_16gf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_16gf-3e4a00f9.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_16gf/imagenet1k_v2/regnet_y_16gf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_16gf_swag-43afe44d.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_16gf_swag_e2e/imagenet1k_v1/regnet_y_16gf_swag_e2e_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_lc_v1" : {
            "url"        : "https://download.pytorch.org/models/regnet_y_16gf_lc_swag-f3ec0043.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_16gf_lc_swag/imagenet1k_v1/regnet_y_16gf_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_y_16gf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
     

@MODELS.register(name="regnet_y_32gf", arch="regnet")
class RegNet_Y_32GF(RegNet):
    """RegNet-Y-32GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_y_32gf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_32gf/imagenet1k_v1/regnet_y_32gf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_y_32gf-8db6d4b5.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_32gf/imagenet1k_v2/regnet_y_32gf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_32gf_swag-04fdfa75.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_32gf_swag/imagenet1k_v1/regnet_y_32gf_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_lc_v1" : {
            "url"        : "https://download.pytorch.org/models/regnet_y_32gf_lc_swag-e1583746.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_32gf_lc_swag/imagenet1k_v1/regnet_y_32gf_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_y_32gf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    

@MODELS.register(name="regnet_y_128gf", arch="regnet")
class RegNet_Y_128GF(RegNet):
    """RegNet-Y-128GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_y_128gf"
    zoo : dict = {
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_y_128gf_swag-c8ce3e52.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_128gf_swag/imagenet1k_v1/regnet_y_128gf_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_lc_v1" : {
            "url"        : "https://download.pytorch.org/models/regnet_y_128gf_lc_swag-cbe8ce12.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_y_128gf_lc_swag/imagenet1k_v1/regnet_y_128gf_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_y_128gf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
 
@MODELS.register(name="regnet_x_400mf", arch="regnet")
class RegNet_X_400MF(RegNet):
    """RegNet-X-400MF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_x_400mf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_400mf/imagenet1k_v1/regnet_x_400mf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_400mf-62229a5f.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_400mf/imagenet1k_v2/regnet_x_400mf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_x_400mf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
   

@MODELS.register(name="regnet_x_800mf", arch="regnet")
class RegNet_X_800MF(RegNet):
    """RegNet-X-800MF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_x_800mf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_800mf/imagenet1k_v1/regnet_x_800mf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_800mf-94a99ebd.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_800mf/imagenet1k_v2/regnet_x_800mf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_x_800mf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
  

@MODELS.register(name="regnet_x_1_6gf", arch="regnet")
class RegNet_X_1_6GF(RegNet):
    """RegNet-X-1.6GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_x_1_6gf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_1_6gf/imagenet1k_v1/regnet_x_1_6gf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_1_6gf-a12f2b72.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_1_6gf/imagenet1k_v2/regnet_x_1_6gf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_x_1_6gf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
      
      
@MODELS.register(name="regnet_x_3_2gf", arch="regnet")
class RegNet_X_3_2GF(RegNet):
    """RegNet-X-3.2GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_x_3_2gf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_3_2gf/imagenet1k_v1/regnet_x_3_2gf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_3_2gf-7071aa85.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_3_2gf/imagenet1k_v2/regnet_x_3_2gf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "regnet_x_3_2gf",
        in_channels: int = 3,
        num_classes: int = 1000,
        weights    : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            num_classes = num_classes,
            weights     = weights,
            *args, **kwargs
        )
        
        if isinstance(self.weights, dict):
            in_channels = self.weights.get("in_channels", in_channels)
            num_classes = self.weights.get("num_classes", num_classes)
        self.in_channels  = in_channels or self.in_channels
        self.out_channels = num_classes or self.out_channels
        
        self.model = regnet_x_3_2gf(num_classes=self.out_channels)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="regnet_x_8gf", arch="regnet")
class RegNet_X_8GF(RegNet):
    """RegNet-X-8GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_x_8gf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_8gf/imagenet1k_v1/regnet_x_8gf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_8gf-2b70d774.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_8gf/imagenet1k_v2/regnet_x_8gf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_x_8gf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="regnet_x_16gf", arch="regnet")
class RegNet_X_16GF(RegNet):
    """RegNet-X-16GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_x_16gf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_16gf/imagenet1k_v1/regnet_x_16gf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_16gf-ba3796d7.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_16gf/imagenet1k_v2/regnet_x_16gf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_x_16gf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="regnet_x_32gf", arch="regnet")
class RegNetX_32GF(RegNet):
    """RegNet-X-32GF model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "regnet_x32gf"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x32gf/imagenet1k_v1/regnet_x_32gf_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/regnet_x_32gf-6eb8fdc6.pth",
            "path"       : ZOO_DIR / "vision/classify/regnet/regnet_x_32gf/imagenet1k_v2/regnet_x_32gf_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = regnet_x_32gf(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
