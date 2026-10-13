from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

PROPRIO_FEATURE_DIM = 5  # [roughness, slip_ratio, angular_instability, ke_rate, slope_power]


@dataclass
class ProprioFeatureHeadConfig:
    d_in: int = 384       # D_vis for full model, D_prop for baseline
    pool_patches: bool = True  # True: [B, N, d_in] → pool → [B, d_in]; False: input is already [B, d_in]
    out_dim: int = PROPRIO_FEATURE_DIM  # number of active proprio channels (default all 5)


class ProprioFeatureHead(nn.Module):
    """
    Predicts 5-dim proprio feature from a hidden state or proprio encoding.

    Full model  (pool_patches=True):  input [B, N, d_in] → output [B, 5]
    Baseline 1a (pool_patches=False): input [B, d_in]    → output [B, 5]

    Output column order: [roughness, slip_ratio, angular_instability, ke_rate, slope_power]
    No output activation — targets are normalized real values.
    """

    def __init__(self, cfg: ProprioFeatureHeadConfig = ProprioFeatureHeadConfig()) -> None:
        super().__init__()
        self.pool_patches = cfg.pool_patches
        d = cfg.d_in
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, cfg.out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.pool_patches:
            x = x.mean(dim=-2)  # [B, N, d_in] → [B, d_in]
        return self.mlp(x)      # [B, out_dim]
