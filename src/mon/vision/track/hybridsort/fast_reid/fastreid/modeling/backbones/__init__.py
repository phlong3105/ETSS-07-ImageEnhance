# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import BACKBONE_REGISTRY, build_backbone
from .mobilenet import build_mobilenetv2_backbone
from .mobilenetv3 import build_mobilenetv3_backbone
from .osnet import build_osnet_backbone
from .regnet import build_effnet_backbone, build_regnet_backbone
from .repvgg import build_repvgg_backbone
from .resnest import build_resnest_backbone
from .resnet import build_resnet_backbone
from .resnext import build_resnext_backbone
from .shufflenet import build_shufflenetv2_backbone
from .vision_transformer import build_vit_backbone
