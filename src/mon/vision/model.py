#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class for all vision models."""

__all__ = [
    "VisionModel",
]

from abc import ABC
from copy import deepcopy

import torch

from mon import core, nn
from mon.nn import _size_2_t


# ----- Base Model -----
class VisionModel(nn.Model, ABC):
    """Base class for vision models with image/video input."""
    
    # ----- Initialize -----
    def compute_efficiency_score(self, image_size: _size_2_t = 512, channels: int = 3) -> tuple[float, float]:
        """Compute model efficiency score (FLOPs, params).

        Args:
            image_size: Input size as ``int`` or [H, W]. Default is ``512``.
            channels: Number of input channels as ``int``. Default is ``3``.

        Returns:
            Tuple of (FLOPs, parameter count) as ``float`` values.
        """
        from fvcore.nn import parameter_count
        from mon import vision
        
        h, w      = vision.image_size(image_size)
        datapoint = {"image": torch.rand(1, channels, h, w).to(self.device)}
        flops, params = core.thop.custom_profile(deepcopy(self), inputs=datapoint, verbose=False)
        params        = self.params if hasattr(self, "params") and params == 0 else params
        params        = parameter_count(self) if hasattr(self, "params")  else params
        params        = sum(params.values())  if isinstance(params, dict) else params
        return flops, params
        
    # ----- Predict -----
    def infer(
        self,
        datapoint : dict,
        image_size: _size_2_t = 512,
        resize    : bool = False,
        *args, **kwargs
    ) -> dict:
        """Infers model output with optional processing.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
            image_size: Input size as ``int`` or [H, W]. Default is ``512``.
            resize: Resize input to ``image_size`` if ``True``. Default is ``False``.
    
        Returns:
            ``dict`` of model predictions with inference time.
    
        Notes:
            Override for custom pre/post-processing; defaults to ``self.forward()``.
        """
        from mon.vision import types, transforms
        
        # Input
        image  = datapoint["image"]
        h0, w0 = types.image_size(image)
        for k, v in datapoint.items():
            if types.is_image(v):
                size         = image_size if resize else 32 * ((max(h0, w0) + 31) // 32)
                datapoint[k] = transforms.resize(v, size)
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Infer
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint, *args, **kwargs)
        timer.tock()
    
        # Post-processing
        for k, v in outputs.items():
            if types.is_image(v):
                h1, w1 = types.image_size(v)
                if h1 != h0 or w1 != w0:
                    outputs[k] = transforms.resize(v, (h0, w0))
        
        # Return
        return outputs | {
            "time": timer.avg_time
        }
