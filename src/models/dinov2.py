from typing import Optional

import torch
import torch.nn as nn

from .builder import MODELS
from utils import get_logger


@MODELS.register_module()
class Dinov2(nn.Module):
    def __init__(
        self,
        output_feat_size: Optional[int] = None,
        adaptive_pool_size: Optional[int] = 32
    ):
        super(Dinov2, self).__init__()
        self.logger = get_logger()

        # Load pretrained DINOv2 model and freeze its parameters
        self.dinomodel = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        for param in self.dinomodel.parameters():
            param.requires_grad = False

        # Define output dimensions
        self.output_feat_size = output_feat_size
        embedding_size = 768
        self.num_depth_features = adaptive_pool_size * embedding_size

        # Projection layer to compress DINOv2 patch features
        self.compress_dino_feat = nn.Sequential(
            nn.AdaptiveAvgPool1d(adaptive_pool_size),
            nn.Flatten(),
            nn.Linear(self.num_depth_features, self.output_feat_size),
        )

    def forward_feature(self, image):
        """Forward pass for training mode."""
        # Extract patch features from DINOv2
        patch_feat = self.dinomodel.forward_features(image)['x_norm_patchtokens']
        self.logger.debug(f"patch_feat: {patch_feat.shape}")

        # Rearrange dimensions: [B, N, C] → [B, C, N]
        patch_feat_permut = patch_feat.permute(0, 2, 1)
        self.logger.debug(f"patch_feat_permut: {patch_feat_permut.shape}")

        # Compress and project features to output dimension
        average_patch_feat = self.compress_dino_feat(patch_feat_permut)
        self.logger.debug(f"average_patch_feat: {average_patch_feat.shape}")

        return average_patch_feat

    @torch.no_grad()
    def forward_test(self, image):
        """Forward pass for inference mode."""
        patch_feat = self.dinomodel.forward_features(image)['x_norm_patchtokens']
        self.logger.debug(f"patch_feat: {patch_feat.shape}")

        patch_feat_permut = patch_feat.permute(0, 2, 1)
        self.logger.debug(f"patch_feat_permut: {patch_feat_permut.shape}")

        average_patch_feat = self.compress_dino_feat(patch_feat_permut)
        self.logger.debug(f"average_patch_feat: {average_patch_feat.shape}")

        return average_patch_feat

    def forward(self, image, train=True):
        """Unified forward interface."""
        if not train:
            return self.forward_test(image)
        return self.forward_feature(image)
