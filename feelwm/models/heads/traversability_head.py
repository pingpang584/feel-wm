from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TraversabilityHeadConfig:
    d_in: int = 5
    hidden_dim: int = 32


class TraversabilityHead(nn.Module):
    """Predicts unsafe logit from the predicted semantic proprio feature."""

    def __init__(self, cfg: TraversabilityHeadConfig = TraversabilityHeadConfig()) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_in, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, f_prop: Tensor) -> Tensor:
        return self.net(f_prop)

    def predict_proba(self, f_prop: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(f_prop))
