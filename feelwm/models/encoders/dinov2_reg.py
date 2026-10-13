"""Frozen DINOv2-with-registers patch encoder used to build every DINOv2 feature cache.

``facebook/dinov2-with-registers-base`` (ViT-B/14) with two changes:
  * the final LayerNorm keeps its normalisation but loses its learned affine
    (weight/bias removed), and
  * the 1 CLS + 4 register tokens are dropped, leaving the 256 patch tokens.

At 224x224 input this yields (B, 256, 768). The feature space matches the encoder of the
RAE-NWM latent decoder, so cached features can be decoded back to images by that model.
"""
from __future__ import annotations

import torch
from torch import nn

N_PREFIX_TOKENS = 5  # 1 CLS + 4 register tokens


class Dinov2RegEncoder(nn.Module):
    def __init__(self, model_id: str = "facebook/dinov2-with-registers-base",
                 normalize: bool = True) -> None:
        super().__init__()
        from transformers import Dinov2WithRegistersModel  # lazy: only the cachers need it

        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(model_id, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(model_id, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x, output_hidden_states=True)
        return out.last_hidden_state[:, N_PREFIX_TOKENS:]
