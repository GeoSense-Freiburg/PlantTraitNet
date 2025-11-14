import torch
import torch.nn as nn
from .builder import MODELS
from utils import get_logger
from typing import Optional
import torch
from rshf.climplicit import Climplicit

@MODELS.register_module()
class ClimplicitModel(nn.Module):
    def __init__(self, output_feat_size: Optional[int] = 256):
        super(ClimplicitModel, self).__init__()
        self.logger = get_logger()
        self.embedding_size = 1024 #embedding size
        self.output_feat_size = output_feat_size
        self.model = Climplicit.from_pretrained("Jobedo/climplicit", config={"return_chelsa": False})

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.Linear = None
        if self.output_feat_size != self.embedding_size:
            self.Linear = nn.Linear(1024, self.output_feat_size)
        
    def forward_feature(self, loc):
        """
        Args:
            loc: [B, 2] tensor of locations
        """
        with torch.no_grad():
            loc_emb = self.model(loc)
        if self.Linear is not None:
            loc_emb = self.Linear(loc_emb)
        return loc_emb
    
    def forward(self, loc):
            loc.cuda()
            self.model.cuda()
            return self.forward_feature(loc)
