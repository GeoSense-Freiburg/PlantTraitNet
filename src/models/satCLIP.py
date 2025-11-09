import sys
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .builder import MODELS
from utils import get_logger

#To use SatCLIP code, install satclip package from "https://github.com/microsoft/satclip" and set the path accordingly
sys.path.append("../satclip/satclip")
from load import get_satclip


@MODELS.register_module()
class SatCLIP(nn.Module):
    def __init__(self, output_feat_size: Optional[int] = 256):
        super(SatCLIP, self).__init__()
        self.logger = get_logger()
        self.output_feat_size = output_feat_size

        # Only loads location encoder by default
        ckpt_path = hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt")
        self.model = get_satclip(ckpt_path, device='cpu')

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def forward_feature(self, loc):
        """
        Args:
            loc: [B, 2] tensor of locations (Longitude and Latitude)
                Reference: https://huggingface.co/microsoft/SatCLIP-ViT16-L40
        """
        with torch.no_grad():
            loc_emb = self.model(loc.double())
        return loc_emb

    def forward(self, loc):
        loc = loc.cuda()
        self.model.cuda()
        return self.forward_feature(loc)
