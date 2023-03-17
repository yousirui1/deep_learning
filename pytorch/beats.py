import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torchaudio.compliance.kaldi as ta_kaldi
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
#logger.setLevel(level = logging.INFO)
#handler = logging.FileHandler("log.txt")
#logger.addHandler(handler)

class BEATsConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = -1 # path size of patch embedding
        self.embed_dim: int = 512       # patch embedding dimension


class BEATs(nn.Module):
    def __init__(self, cfg: BEATsConfig,) -> None:
        super().__init__()
        logger.info(f"BEATs  Config: {cfg.__dict__}")

        self.cfg = cfg
        self.embed = cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )
        
        self.input_patch_size = cfg.input_patch_size

    def forward_padding_mask(self,
                        features: torch.Tensor,
                        padding_mask: torch.Tensor,) -> torch.Tensor:
        




conf = BEATsConfig()
beats = BEATs(conf)

