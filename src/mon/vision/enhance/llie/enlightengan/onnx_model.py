#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Union

import numpy as np
from onnxruntime import InferenceSession

from mon import core
from mon.constants import ZOO_DIR

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


def get_relative_path(root, *args):
    return os.path.join(os.path.dirname(root), *args)


class EnlightenOnnxModel:

    def __init__(
        self,
        model  : Union[bytes, str, None] = None,
        weights: core.Path = ZOO_DIR / "vision/enhance/llie/enlightengan/enlightengan/custom/enlightengan.onnx",
    ):
        self.graph = InferenceSession(
            model or weights or str(ZOO_DIR / "vision/enhance/llie/enlightengan/enlightengan.onnx"),
            providers=["AzureExecutionProvider", "CPUExecutionProvider"]
        )

    def __repr__(self):
        return f"<EnlightenGAN OnnxModel {id(self)}>"

    def _pad(self, img):
        h, w, _    = img.shape
        block_size = 16
        min_height = (h // block_size + 1) * block_size
        min_width  = (w // block_size + 1) * block_size
        img        = np.pad(img, ((0, min_height - h), (0, min_width - w), (0, 0)), mode="constant", constant_values=0)
        return img, (h, w)

    def _preprocess(self, img):
        if len(img.shape) != 3:
            raise ValueError(f"Incorrect shape: expected 3, got {len(img.shape)}")
        return np.expand_dims(np.transpose(img, (2, 0, 1)).astype(np.float32) / 255., 0)

    def predict(self, img):
        padded, (h, w) = self._pad(img)
        image_numpy,   = self.graph.run(["output"], {"input": self._preprocess(padded)})
        image_numpy    = (np.transpose(image_numpy[0], (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy    = np.clip(image_numpy, 0, 255)
        return image_numpy.astype("uint8")[:h, :w, :]
