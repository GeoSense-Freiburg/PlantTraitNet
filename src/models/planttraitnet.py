import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from typing import Optional, cast, OrderedDict

from .builder import MODELS
from utils import get_logger, instantiate_object
from loss_util import WeightedL1NLLLoss, MixedNLLLoss, WeightedGaussianNLLLoss


def dist_collect(x):
    """Collect all tensors from all GPUs.

    Args:
        x: shape (mini_batch, ...)
    Returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [
        torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous()
        for _ in range(dist.get_world_size())
    ]
    return torch.cat(out_list, dim=0).contiguous()


def _named_sequential(*modules) -> nn.Sequential:
    return nn.Sequential(OrderedDict(modules))


class ResNet(nn.Module):
    """ResNet-style MLP model with multiple output heads (mu and logvar per trait)."""

    def __init__(
        self,
        *,
        d_in: int,
        n_traits: int,
        n_blocks: int,
        d_block: int,
        d_hidden: Optional[int] = None,
        d_hidden_multiplier: Optional[float],
        dropout1: float,
        dropout2: float,
    ) -> None:
        super().__init__()

        if n_blocks <= 0:
            raise ValueError(f"n_blocks must be positive, got {n_blocks=}")
        if d_hidden is None:
            if d_hidden_multiplier is None:
                raise ValueError("If d_hidden is None, then d_hidden_multiplier must be provided.")
            d_hidden = int(d_block * cast(float, d_hidden_multiplier))
        else:
            if d_hidden_multiplier is not None:
                raise ValueError("If d_hidden is provided, d_hidden_multiplier must be None.")

        self.n_traits = n_traits
        self.input_projection = nn.Linear(d_in, d_block)

        self.blocks = nn.ModuleList([
            _named_sequential(
                ("normalization", nn.BatchNorm1d(d_block)),
                ("linear1", nn.Linear(d_block, d_hidden)),
                ("activation", nn.LeakyReLU()),
                ("dropout1", nn.Dropout(dropout1)),
                ("linear2", nn.Linear(d_hidden, d_block)),
                ("dropout2", nn.Dropout(dropout2)),
            )
            for _ in range(n_blocks)
        ])

        # Separate mu and logvar heads per trait
        self.mu_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_block, 1),
                nn.Softplus()
            ) for _ in range(n_traits)
        ])

        self.logvar_linears = nn.ModuleList([
            nn.Linear(d_block, 1) for _ in range(n_traits)
        ])

        # Weight initialization
        for layer in self.blocks.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.zeros_(layer.bias)

        for mu_layer in self.mu_heads:
            if isinstance(mu_layer, nn.Linear):
                nn.init.kaiming_uniform_(mu_layer.weight, mode="fan_in")
                nn.init.zeros_(mu_layer.bias)

        for logvar_layer in self.logvar_linears:
            if isinstance(logvar_layer, nn.Linear):
                nn.init.normal_(logvar_layer.weight, 0, 0.01)
                nn.init.zeros_(logvar_layer.bias)

    def forward(self, x: torch.Tensor):
        """Forward pass returning mu and logvar per trait."""
        x = self.input_projection(x)
        for block in self.blocks:
            x = x + block(x)

        batch_size = x.size(0)
        mu_tensor = torch.zeros(batch_size, self.n_traits, device=x.device)
        logvar_tensor = torch.zeros(batch_size, self.n_traits, device=x.device)

        for i, (mu_layer, logvar_layer) in enumerate(zip(self.mu_heads, self.logvar_linears)):
            mu_tensor[:, i] = mu_layer(x).squeeze(-1)
            logvar_tensor[:, i] = logvar_layer(x).squeeze(-1)

        return mu_tensor, logvar_tensor


@MODELS.register_module()
class PlantTraitNet(nn.Module):
    """Multimodal multitrait model combining image, geo, and depth encoders."""

    def __init__(
        self,
        image_encoder,
        geo_encoder,
        depthmap_encoder,
        output_traits,
        fusion_dim=None,
        n_blocks=8,
        d_block=256,
        d_hidden_multiplier=2.0,
        dropout1=0.2,
        dropout2=0.2,
        fusion="ConcatFusionModule",
        loss="WeightedL1NLLLoss",
        layernorm=True,
        modality="multimodal",
    ):
        super(PlantTraitNet, self).__init__()

        self.logger = get_logger()
        #self.modality = modality
        self.output_traits = output_traits

        # Initialize all encoders
        self.image_encoder = MODELS.build(image_encoder)
        self.depth_encoder = MODELS.build(depthmap_encoder)
        self.geo_encoder = MODELS.build(geo_encoder)
        self.encoders = [self.image_encoder, self.depth_encoder, self.geo_encoder]

        # Determine fusion input size
        self.input_size = sum(encoder.output_feat_size for encoder in self.encoders)

        self.fusion = instantiate_object(fusion["type"])

        # Loss selection
        if loss == "MixedNLLLoss":
            self.criterion = MixedNLLLoss()
        elif loss == "WeightedGaussianNLLLoss":
            self.criterion = WeightedGaussianNLLLoss()
        else:
            self.criterion = WeightedL1NLLLoss()

        self.logger.info(f"Using loss function: {self.criterion.__class__.__name__}")

        # Input projection layer
        if fusion_dim is not None:
            self.input_proj = nn.Linear(self.input_size, fusion_dim)
        else:
            self.input_proj = nn.Identity()
            fusion_dim = self.input_size

        self.logger.info(f"Using fusion dimension: {fusion_dim}")
        self.logger.info(f"Using input projection: {self.input_proj.__class__.__name__}")

        # Optional LayerNorm
        self.layernorm = nn.LayerNorm(self.input_size, eps=1e-6, elementwise_affine=True) if layernorm else None

        # Initialize multitask ResNet backbone
        self.multimodal = ResNet(
            d_in=fusion_dim,
            n_traits=output_traits,
            n_blocks=n_blocks,
            d_block=fusion_dim,
            d_hidden=None,
            d_hidden_multiplier=d_hidden_multiplier,
            dropout1=dropout1,
            dropout2=dropout2,
        )

    def loss(self, mu, log_var, trait_mean):
        return self.criterion(mu, log_var, trait_mean)

    def _compute_embeddings(self, image, loc):
        embeddings = {
            "image": self.image_encoder(image),
            "depthmap": self.depth_encoder(image),
            "geo": self.geo_encoder(loc),
        }
        return self.fusion(embeddings)

    def get_embeddings(self, image, loc):
        return self.multimodal(self._compute_embeddings(image, loc))

    def get_image_embeddings(self, image):
        return {"image": self.image_encoder(image)}

    def forward_train(self, image, geo, label):
        embeddings = self._compute_embeddings(image, geo)
        if self.layernorm is not None:
            embeddings = self.layernorm(embeddings.float())

        embeddings = self.input_proj(embeddings)
        mu, log_var = self.multimodal(embeddings)

        losses = self.loss(mu, log_var, label)
        loss = {"nll": losses.mean()}

        for i in range(losses.shape[0]):
            loss[f"nll_{i}"] = losses[i]

        return loss, log_var

    def forward_test(self, image, geo):
        embeddings = self._compute_embeddings(image, geo)
        if self.layernorm is not None:
            embeddings = self.layernorm(embeddings.float())

        embeddings = self.input_proj(embeddings)
        mu, log_var = self.multimodal(embeddings)
        
        return mu, log_var

    def forward(self, image, geo=None, label=None, train=True):
        self.logger.debug(f"Image shape: {image.shape}")
        image, label, geo = image.cuda(), label.cuda(), geo.cuda()

        if train:
            return self.forward_train(image, geo, label)
        else:
            return self.forward_test(image, geo)
