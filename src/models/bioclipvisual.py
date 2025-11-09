import torch
import torch.nn as nn
from .builder import MODELS
from utils import get_logger
from bioclip import TreeOfLifeClassifier
import PIL
from typing import Optional

@MODELS.register_module()
class BioCLIP(nn.Module):
    def __init__(self):
        super(BioCLIP, self).__init__()
        self.logger = get_logger()
        classifier = TreeOfLifeClassifier()
        bioclipmodel = classifier.model
        self.model = bioclipmodel.visual
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.output_feat_size = 512

    @torch.no_grad()
    def forward(self, image):
        img_feat = self.model(image)
        return img_feat
