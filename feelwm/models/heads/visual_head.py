from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor


@dataclass
class VisualHeadConfig:
    d_vis: int = 768
    hidden_dim: int = 768


class VisualHead(nn.Module):
    """Predicts next-step cached DINO feature tokens from hidden patch states."""

    def __init__(self, cfg: VisualHeadConfig = VisualHeadConfig()) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_vis, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.d_vis),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)
