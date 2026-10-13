from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor


@dataclass
class FiLMConditioningConfig:
    d_vis: int = 768
    d_prop: int = 128
    hidden_dim: int = 768


class FiLMConditioning(nn.Module):
    """Generates per-channel FiLM parameters from action and optional proprio."""

    def __init__(self, cfg: FiLMConditioningConfig = FiLMConditioningConfig()) -> None:
        super().__init__()
        self.proprio_proj = nn.Linear(cfg.d_prop, cfg.d_vis)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_vis, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 2 * cfg.d_vis),
        )

    def forward(self, action_emb: Tensor, proprio: Tensor | None = None) -> tuple[Tensor, Tensor]:
        cond = action_emb
        if proprio is not None:
            cond = cond + self.proprio_proj(proprio)
        gamma_delta, beta = self.net(cond).chunk(2, dim=-1)
        gamma = 1.0 + gamma_delta
        return gamma, beta


@dataclass
class AdaLNConditioningConfig:
    d_vis: int = 768
    d_prop: int = 128


class AdaLNConditioning(nn.Module):
    """Produces a single conditioning vector for AdaLN-Zero block modulation.

    Same fusion as FiLM (additive action + projected proprio), but instead of
    emitting (gamma, beta) directly it returns a `d_vis` embedding consumed by the
    per-block bottleneck modulation MLPs inside `AdaLNDynamicsTransformer`. SiLU
    front matches the DiT conditioning-embedding convention.
    """

    def __init__(self, cfg: AdaLNConditioningConfig = AdaLNConditioningConfig()) -> None:
        super().__init__()
        self.proprio_proj = nn.Linear(cfg.d_prop, cfg.d_vis)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(cfg.d_vis, cfg.d_vis))

    def forward(self, action_emb: Tensor, proprio: Tensor | None = None) -> Tensor:
        cond = action_emb
        if proprio is not None:
            cond = cond + self.proprio_proj(proprio)
        return self.net(cond)
