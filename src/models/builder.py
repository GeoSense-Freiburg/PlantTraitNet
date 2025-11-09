
# This script is taken from GroupViT repository, Written by Jiarui Xu
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
# -------------------------------------------------------------------------

from mmcv.utils import Registry
from omegaconf import OmegaConf

MODELS = Registry('model')

def build_model(config):
    model = MODELS.build(OmegaConf.to_container(config, resolve=True))
    return model