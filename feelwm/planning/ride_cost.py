"""ride_cost — predicted proprio features → scalar ride risk.

A portable, physically-grounded planning-risk signal. It is produced from the world
model's **predicted** proprio-feature rollout (the proprio head exists in every
trav-mode), so the cost is proprio-grounded regardless of how the binary
traversability head is wired (visual+proprio fusion happens at the score level).
It enters the planning score as ``λ_ride · ride_cost``; it is an ablation term
(the shipped planning config sets ``lambda_ride: 0`` for wheeled models).

Definition:
  - Channels ``[roughness, angular_instability, slope_power]``. ``slip_ratio`` is
    excluded (structurally val R²≈0 on TartanDrive; the rollout-proprio ablation
    found headroom only in the explicit 4-dim feature, not in the embedding).
  - Computed in **normalized z-space**: norm_stats put channels on a comparable
    dimensionless scale, so a plain weighted sum across channels is defensible.
    Physical-unit values are an interpretability appendix (``ride_cost_physical``).
  - ``ride_cost = mean_h ( Σ_c w_c · f̂_norm[h, c] )``, ``w`` uniform by default,
    averaged **only within the proprio horizon mask** (cum_dt ≤ 1.0 s). No horizon
    discount by default (``discount`` is a secondary config knob). Not clamped —
    a negative cost just means "smoother than the training average"; ranking is
    relative.

Safety-framed aggregations: the default ``mean`` treats the term as average exposure —
a comfort semantics. Under the safety framing (proprio-grounded risk, TAO-style) the
relevant statistic is the TAIL of the horizon, so ``agg`` adds ``max`` (worst step),
``cvar`` (mean of the worst ``cvar_alpha`` share of masked steps), and ``exceed`` (mean
superlinear penalty of the per-step value above ``exceed_z0`` — zero for candidates that
never cross the threshold). ``mean`` stays the default and is bit-identical to the original
definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from feelwm.data.normalize import NormStats, denormalize_feature

# Channels that enter the ride cost (slip_ratio deliberately excluded; see module docstring).
DEFAULT_RIDE_CHANNELS = ["roughness", "angular_instability", "slope_power"]


RIDE_AGGS = ("mean", "max", "cvar", "exceed")


@dataclass(frozen=True)
class RideCostConfig:
    """Knobs for :func:`ride_cost`. Defaults are the reference definition above."""

    ride_channels: list[str] = field(default_factory=lambda: list(DEFAULT_RIDE_CHANNELS))
    weights: list[float] | None = None  # per-ride-channel; None → uniform 1.0
    discount: float = 1.0  # γ horizon discount; 1.0 = none (spec default)
    # Horizon aggregation of the per-step channel sum (safety recomposition axis).
    # "mean" = default. "max"/"cvar" rank on the worst step(s); "exceed" is the
    # masked mean of relu(per_step − exceed_z0)^exceed_power, so a candidate that
    # never crosses z0 costs exactly 0 (an absolute threshold the pool-relative
    # z-standardization in score.py cannot express on its own).
    agg: str = "mean"
    cvar_alpha: float = 0.3      # tail share for agg="cvar" (worst ⌈α·K⌉ masked steps)
    exceed_z0: float = 1.0       # threshold on the per-step SUMMED channel value (z-units)
    exceed_power: float = 2.0    # superlinear exponent for agg="exceed"


def _resolve_columns(channel_names: list[str], ride_channels: list[str]) -> list[int]:
    """Map ride-channel names to their column indices in the model's active output."""
    cols = []
    for name in ride_channels:
        if name not in channel_names:
            raise ValueError(
                f"ride channel {name!r} is not an active model channel {channel_names}"
            )
        cols.append(channel_names.index(name))
    return cols


def ride_cost(
    f_prop_preds: Tensor,
    channel_names: list[str],
    horizon_mask: Tensor | None = None,
    cfg: RideCostConfig | None = None,
) -> Tensor:
    """Aggregate a normalized proprio-feature rollout into a scalar ride cost per trajectory.

    Args:
        f_prop_preds: normalized proprio predictions ``[B, H, C]`` (the ``f_prop_preds``
            output of ``TerrainWorldModel.forward_rollout``). ``C`` columns correspond,
            in order, to ``channel_names``.
        channel_names: the model's active proprio-channel names, in column order
            (e.g. ``["roughness", "slip_ratio", "angular_instability", "slope_power"]``).
        horizon_mask: optional ``[B, H]`` (or broadcastable ``[H]``) boolean/0-1 mask of
            steps to include — the dataset's ``proprio_horizon_mask`` (cum_dt ≤ 1 s).
            ``None`` averages over every step.
        cfg: :class:`RideCostConfig`; ``None`` uses the reference defaults.

    Returns:
        ``[B]`` ride cost in normalized z-units (not clamped).
    """
    cfg = cfg or RideCostConfig()
    if cfg.agg not in RIDE_AGGS:
        raise ValueError(f"unknown agg {cfg.agg!r}, expected one of {RIDE_AGGS}")
    per_step = ride_step_values(f_prop_preds, channel_names, cfg)  # [B, H]
    B, H = per_step.shape

    if cfg.agg in ("mean", "exceed"):
        if cfg.agg == "exceed":
            per_step = (per_step - cfg.exceed_z0).clamp_min(0.0) ** cfg.exceed_power
        # Per-step weight = horizon-membership × discount γ^h, then a weighted mean.
        if horizon_mask is None:
            step_w = per_step.new_ones(B, H)
        else:
            step_w = horizon_mask.to(per_step.dtype)
            if step_w.ndim == 1:
                step_w = step_w.expand(B, H)
        if cfg.discount != 1.0:
            gamma = cfg.discount ** torch.arange(H, device=per_step.device, dtype=per_step.dtype)
            step_w = step_w * gamma
        denom = step_w.sum(dim=-1).clamp_min(torch.finfo(per_step.dtype).eps)
        return (per_step * step_w).sum(dim=-1) / denom

    # Tail aggregations rank on the discounted per-step values within the mask.
    if cfg.discount != 1.0:
        gamma = cfg.discount ** torch.arange(H, device=per_step.device, dtype=per_step.dtype)
        per_step = per_step * gamma
    if horizon_mask is None:
        valid = per_step.new_ones(B, H, dtype=torch.bool)
    else:
        valid = horizon_mask.to(torch.bool)
        if valid.ndim == 1:
            valid = valid.expand(B, H)
        # An all-masked row falls back to the full horizon (mirror traversability_gate_stat).
        degenerate = ~valid.any(dim=-1, keepdim=True)
        valid = valid | degenerate

    if cfg.agg == "max":
        return per_step.masked_fill(~valid, float("-inf")).max(dim=-1).values

    # cvar: mean of the worst ⌈α·K⌉ masked steps (K = valid count per row).
    n_valid = valid.sum(dim=-1)                                     # [B]
    k = (cfg.cvar_alpha * n_valid).ceil().clamp_min(1).long()       # [B]
    sorted_vals = per_step.masked_fill(~valid, float("-inf")).sort(dim=-1, descending=True).values
    in_tail = torch.arange(H, device=per_step.device).expand(B, H) < k[:, None]
    # k ≤ n_valid, so the tail never reaches the -inf padding.
    return torch.where(in_tail, sorted_vals, sorted_vals.new_zeros(())).sum(dim=-1) / k.to(per_step.dtype)


def ride_step_values(
    f_prop_preds: Tensor,
    channel_names: list[str],
    cfg: RideCostConfig | None = None,
) -> Tensor:
    """Per-step weighted channel sum ``Σ_c w_c·f̂[h, c]`` — no horizon aggregation.

    The shared front half of :func:`ride_cost`; also used directly as the per-step
    statistic behind the proprio fusion gate (score.py) and per-replan logging.
    Returns ``[B, H]``.
    """
    if f_prop_preds.ndim != 3:
        raise ValueError(f"f_prop_preds must be [B, H, C], got {tuple(f_prop_preds.shape)}")
    cfg = cfg or RideCostConfig()
    cols = _resolve_columns(channel_names, cfg.ride_channels)
    sel = f_prop_preds[..., cols]  # [B, H, n_ride]
    if cfg.weights is None:
        w = sel.new_ones(len(cols))
    else:
        if len(cfg.weights) != len(cols):
            raise ValueError(
                f"weights has {len(cfg.weights)} entries, expected {len(cols)} (ride_channels)"
            )
        w = sel.new_tensor(cfg.weights)
    return (sel * w).sum(dim=-1)


def ride_cost_physical(
    f_prop_preds: Tensor,
    channel_names: list[str],
    stats: NormStats,
    horizon_mask: Tensor | None = None,
    cfg: RideCostConfig | None = None,
) -> dict[str, Tensor]:
    """Interpretability appendix: per-channel ride cost in physical units.

    Denormalizes each ride channel back to physical units (``denormalize_feature``)
    and reports its horizon-masked mean separately — for reporting only; the planner
    ranks on the dimensionless z-space :func:`ride_cost`.

    Returns a ``{channel_name: [B]}`` dict of physical-unit means.
    """
    from feelwm.data.normalize import resolve_channel_indices

    cfg = cfg or RideCostConfig()
    cols = _resolve_columns(channel_names, cfg.ride_channels)
    canonical = resolve_channel_indices([channel_names[c] for c in cols])

    sel = f_prop_preds[..., cols]  # [B, H, n_ride], normalized
    phys = torch.from_numpy(
        denormalize_feature(sel.detach().cpu().numpy(), stats, indices=canonical)
    ).to(sel)

    B, H = phys.shape[:2]
    if horizon_mask is None:
        step_w = phys.new_ones(B, H)
    else:
        step_w = horizon_mask.to(phys.dtype)
        if step_w.ndim == 1:
            step_w = step_w.expand(B, H)
    denom = step_w.sum(dim=-1).clamp_min(torch.finfo(phys.dtype).eps)  # [B]

    out: dict[str, Tensor] = {}
    for j, name in enumerate(cfg.ride_channels):
        out[name] = (phys[..., j] * step_w).sum(dim=-1) / denom
    return out
