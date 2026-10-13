from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor


@dataclass
class ActionEmbeddingConfig:
    d_action: int = 3
    d_vis: int = 768
    hidden_dim: int = 768


class ActionEmbedding(nn.Module):
    """Embeds body-frame pose-delta action into the visual feature dimension."""

    def __init__(self, cfg: ActionEmbeddingConfig = ActionEmbeddingConfig()) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_action, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.d_vis),
        )

    def forward(self, action: Tensor) -> Tensor:
        return self.net(action)
