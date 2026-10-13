from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from feelwm.data.normalize import FEATURE_CHANNELS, resolve_channel_indices
from feelwm.training.curriculum import CurriculumStage

# Re-exported from the data layer (single source of truth) for backward compatibility.
CHANNEL_NAMES = FEATURE_CHANNELS

__all__ = ["CHANNEL_NAMES", "LossConfig", "compute_total_loss", "resolve_channel_indices"]


@dataclass
class LossConfig:
    lambda_prop: float = 1.0
    lambda_trav: float = 0.5
    lambda_cos: float = 0.1
    # aligned rollout mode: weight on the next-proprio-embedding alignment loss
    # (ẑ_prop vs sg(encode(true future window))). 0 disables it (other rollout modes).
    lambda_prop_align: float = 0.0
    proprio_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 0.5, 0.5])
    # Active proprio output channels (subset of CHANNEL_NAMES, canonical order).
    # Default = all five. proprio_weights length must match this list.
    channel_names: list[str] = field(default_factory=lambda: list(CHANNEL_NAMES))


def compute_total_loss(
    preds: dict[str, Tensor | None],
    targets: dict[str, Tensor],
    cfg: LossConfig,
    stage: CurriculumStage,
    pos_weight: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    valid_mask = targets["valid_target_mask"].bool()
    # Per-horizon cutoff: proprio/trav are reliable only within
    # ~1s, so under multi-stride rollout the dataset emits a step mask; visual is always
    # supervised. Absent (legacy fixed-0.1s) → no cutoff.
    horizon_mask = targets.get("proprio_horizon_mask")
    if horizon_mask is not None:
        prop_valid = valid_mask & horizon_mask.bool().to(valid_mask.device)
    else:
        prop_valid = valid_mask
    z_pred = _get_pred(preds, "z_vis_preds", "z_vis_pred")
    z_target = targets["z_vis_target"]

    visual_mse = _masked_mean((z_pred - z_target) ** 2, valid_mask, extra_dims=2)
    cos = F.cosine_similarity(z_pred, z_target, dim=-1)
    visual_cos = _masked_mean(1.0 - cos, valid_mask, extra_dims=1)
    visual_loss = visual_mse + cfg.lambda_cos * visual_cos

    losses: dict[str, Tensor] = {
        "loss_total": visual_loss,
        "loss_visual": visual_loss.detach(),
        "loss_visual_mse": visual_mse.detach(),
        "loss_visual_cos": visual_cos.detach(),
    }
    total = visual_loss

    if stage.value >= CurriculumStage.VISUAL_PROPRIO.value and preds.get("f_prop_preds") is not None:
        f_pred = _get_pred(preds, "f_prop_preds", "f_prop_pred")
        f_target = targets["f_prop_target"]
        weights = torch.tensor(cfg.proprio_weights, device=f_pred.device, dtype=f_pred.dtype)
        per_channel = _masked_channel_mse(f_pred, f_target, prop_valid)
        proprio_loss = (weights * per_channel).sum()
        total = total + cfg.lambda_prop * proprio_loss
        losses["loss_proprio"] = proprio_loss.detach()
        for i, name in enumerate(cfg.channel_names):
            losses[f"loss_prop_{name}"] = per_channel[i].detach()

        # Aligned-rollout alignment: make ẑ_prop track encode(true future window). Masked by
        # valid_target_mask only (NOT the proprio_horizon cutoff): the conditioning
        # must stay correct over the FULL horizon, which is exactly the freeze→oracle
        # headroom at >1s the aligned scheme chases.
        align_pred = preds.get("z_prop_align_preds")
        align_target = preds.get("z_prop_align_target")
        if cfg.lambda_prop_align > 0 and align_pred is not None and align_target is not None:
            align_loss = _masked_mean((align_pred - align_target) ** 2, valid_mask, extra_dims=1)
            total = total + cfg.lambda_prop_align * align_loss
            losses["loss_prop_align"] = align_loss.detach()

    if stage is CurriculumStage.FULL and preds.get("c_pred_logits") is not None:
        logits = _get_pred(preds, "c_pred_logits", "c_pred_logit").squeeze(-1)
        target = targets["binary_label_target"].to(dtype=logits.dtype)
        trav_mask = prop_valid  # cost shares the proprio near-horizon cutoff
        if "binary_valid_target_mask" in targets:
            trav_mask = trav_mask & targets["binary_valid_target_mask"].bool()
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
            reduction="none",
        )
        trav_loss = _masked_mean(bce, trav_mask, extra_dims=0)
        total = total + cfg.lambda_trav * trav_loss
        losses["loss_traversability"] = trav_loss.detach()

    losses["loss_total"] = total
    return total, losses


def _get_pred(preds: dict[str, Tensor | None], plural_key: str, singular_key: str) -> Tensor:
    val = preds.get(plural_key)
    if val is None:
        val = preds.get(singular_key)
        if val is not None:
            val = val.unsqueeze(1)
    if val is None:
        raise KeyError(f"preds must contain {plural_key} or {singular_key}")
    return val


def _masked_mean(values: Tensor, valid_mask: Tensor, extra_dims: int) -> Tensor:
    mask = valid_mask.to(device=values.device, dtype=values.dtype)
    for _ in range(extra_dims):
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.expand_as(values).sum().clamp(min=1.0)


def _masked_channel_mse(pred: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    sq = (pred - target) ** 2
    mask = valid_mask.to(device=sq.device, dtype=sq.dtype).unsqueeze(-1)
    return (sq * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp(min=1.0)
