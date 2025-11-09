from typing import Optional

import torch
import torch.nn as nn

from .builder import MODELS
from geoclip import LocationEncoder


@MODELS.register_module()
class GeoCLIP(nn.Module):
    def __init__(self, output_feat_size: Optional[int] = 256):
        super(GeoCLIP, self).__init__()
        # self.logger = get_logger()
        self.output_feat_size = output_feat_size
        self.embedding_size = 512  # Embedding size

        self.model = LocationEncoder()
        for param in self.model.parameters():
            param.requires_grad = False

        self.Linear = None
        if self.output_feat_size != self.embedding_size:
            self.Linear = nn.Linear(self.embedding_size, self.output_feat_size)

    def forward_feature(self, loc):
        """
        Args:
            loc: [B, 2] tensor of locations 
                (Latitude and Longitude)
                Reference: https://github.com/VicenteVivan/geo-clip?tab=readme-ov-file
        """
        with torch.no_grad():
            # GeoCLIP expects [Latitude, Longitude], unlike SatCLIP which expects [Longitude, Latitude]
            loc = loc[:, [1, 0]]
            loc_emb = self.model(loc)

        if self.Linear is not None:
            loc_emb = self.Linear(loc_emb)

        return loc_emb

    def forward(self, loc):
        return self.forward_feature(loc)