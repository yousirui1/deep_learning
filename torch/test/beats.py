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

        extra = padding_mask.size(1) % features.size(1)
    
    def preprocess(self,
            source: torch.Tensor,
            fbank_mean: float = 15.41663,
            fbank_std: float = 6.55582,
            )->torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins = 128, sample_frequency = 16000, frame_length = 25,
                    frame_shift = 10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim = 0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_features(self,
            source: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            fbank_mean: float = 15.41663,
            fbank_std: float = 6.55582,
            ):
        fbank = self.preprocess(source, fbank_mean = fbank_mean, fbank_std=fbank_std)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)

        x, layer_results = self.encoder(
            x,
            padding_mask = padding_mask,
        )

        if self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)

            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim = 1)
                logits = logits / (~padding_mask).sum(dim = 1).unsqueeze(-1).expand_as(logits)
            else: 
                logits = logits.mean(dim = 1)

            lprobs = torch.sigmoid(logits)

            return lprobs, padding_mask
        else:
            return x, padding_mask





conf = BEATsConfig()
beats = BEATs(conf)


# input -> log -> mask -> dropout -> encoder -> dropout -> dense -> mask -> sigmoid

# encoder 





