from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ScheduledSamplingConfig:
    enabled: bool = True
    start_epoch: int = 5
    end_epoch: int = 15
    p_start: float = 1.0
    p_end: float = 0.5
    # How the teacher-force mask is drawn at probability p:
    #   "step"   – per-(sample, step) Bernoulli(p) (legacy default). Free-running chains
    #              are geometric with mean 1/p steps (~2 at p=0.5): the model almost never
    #              trains on a full-horizon self-feedback chain, while deployment
    #              (planning) is ALWAYS a full-H free chain from a real seed.
    #   "sample" – one Bernoulli(p) coin per SAMPLE, broadcast over the horizon: a (1−p)
    #              fraction of samples train fully free-running (deployment structure,
    #              real compounding error), the rest fully teacher-forced. Same expected
    #              teacher fraction p, so a p_end floor of 0.5 still bounds the teacher
    #              fraction in expectation.
    mask_granularity: str = "step"

    def __post_init__(self) -> None:
        if self.mask_granularity not in {"step", "sample"}:
            raise ValueError(f"unknown mask_granularity={self.mask_granularity!r}")


def teacher_probability(epoch: int, cfg: ScheduledSamplingConfig) -> float:
    if not cfg.enabled:
        return cfg.p_start
    if epoch <= cfg.start_epoch:
        return cfg.p_start
    if epoch >= cfg.end_epoch:
        return cfg.p_end
    span = max(cfg.end_epoch - cfg.start_epoch, 1)
    t = (epoch - cfg.start_epoch) / span
    p = cfg.p_start + (cfg.p_end - cfg.p_start) * t
    return max(cfg.p_end, min(cfg.p_start, p))


def sample_teacher_force_mask(
    batch_size: int,
    horizon: int,
    p_teacher: float,
    device: torch.device,
    granularity: str = "step",
) -> Tensor:
    if granularity not in {"step", "sample"}:
        raise ValueError(f"unknown granularity={granularity!r}")
    if p_teacher >= 1.0:
        return torch.ones(batch_size, horizon, dtype=torch.bool, device=device)
    if p_teacher <= 0.0:
        return torch.zeros(batch_size, horizon, dtype=torch.bool, device=device)
    if granularity == "sample":
        coin = torch.rand(batch_size, 1, device=device) < p_teacher
        return coin.expand(batch_size, horizon).contiguous()
    return torch.rand(batch_size, horizon, device=device) < p_teacher
