from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor


@dataclass
class NextProprioHeadConfig:
    d_vis: int = 768   # dynamics hidden dim (pooled over patches)
    d_prop: int = 128  # proprio-encoder embedding dim (the FiLM conditioning space)


class NextProprioHead(nn.Module):
    """Predicts the next-step proprio conditioning embedding for aligned rollout.

    Input:  dynamics hidden h_t [B, N, d_vis] (pooled over patches).
    Output: a *residual* Δ ∈ R^{d_prop} added to the frozen t=0 encoder seed:
            ẑ_prop_{t+1} = seed + Δ(h_t).

    The final layer is zero-initialised so Δ ≈ 0 at the start of training; the
    aligned rollout therefore begins numerically identical to `rollout_proprio_mode=
    "freeze"` (the strong baseline) and only departs from it as the alignment loss
    teaches Δ to track the true evolving proprio.
    """

    def __init__(self, cfg: NextProprioHeadConfig = NextProprioHeadConfig()) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_vis, cfg.d_vis),
            nn.GELU(),
            nn.Linear(cfg.d_vis, cfg.d_prop),
        )
        # zero-init the residual so aligned rollout == freeze at init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: Tensor) -> Tensor:
        return self.mlp(h.mean(dim=-2))  # [B, N, d_vis] → pool → [B, d_prop]
