from typing import Optional
import torch.nn as nn
from .builder import MODELS
from utils import get_logger
from depth_anything_v2.dinov2 import DINOv2


# This module uses encoder from Depth-Anything v2 which utilizes intermediate layer features from DINOv2
@MODELS.register_module()
class DepthMapEncoder(nn.Module):
    def __init__(self, depth: Optional[dict] = {"include_depth_encoding": False}, output_feat_size=384):
        super(DepthMapEncoder, self).__init__()
        self.output_feat_size = output_feat_size
        self.logger = get_logger()

        if depth["include_depth_encoding"]:
            self.depth_enc_str = depth["depth_encoder"]

            # Initialize pretrained DINOv2 depth encoder
            self.depth_encoder = DINOv2(model_name=self.depth_enc_str)
            for param in self.depth_encoder.parameters():
                param.requires_grad = False

            self.depth_layer_idx = depth["dino_layer_idx"][self.depth_enc_str]
            self.depth_pool_dim = depth["pool_dim"]
            self.depth_enc_dim = depth["out_dim"][self.depth_enc_str]
            self.num_depth_features = self.depth_enc_dim * self.depth_pool_dim

            # Linear projection if feature dimensions differ from target output size
            if self.num_depth_features != self.output_feat_size:
                self.compress_depth_enc = nn.Sequential(
                    nn.AdaptiveAvgPool1d(self.depth_pool_dim),
                    nn.Flatten(),
                    nn.Linear(self.num_depth_features, self.output_feat_size),
                )
            else:
                self.compress_depth_enc = nn.Identity()

    def forward(self, x):
        # Extract intermediate layer features from DINOv2 encoder
        dpt_enc_all = self.depth_encoder.get_intermediate_layers(
            x,
            self.depth_layer_idx,
            return_class_token=False,
        )

        # Permute dimensions: [B, N, C] → [B, C, N]
        dpt_enc_last = dpt_enc_all[-1].permute(0, 2, 1)

        # Compress or pass depth encoding as-is
        depth_encoding = self.compress_depth_enc(dpt_enc_last.float())

        return depth_encoding
