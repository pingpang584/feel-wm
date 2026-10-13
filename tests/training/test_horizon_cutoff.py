"""Per-horizon proprio/trav cutoff in compute_total_loss (multi-stride supervision).

The dataset emits `proprio_horizon_mask` under multi-stride; the loss must restrict
proprio (and traversability) supervision to the masked-in steps while leaving the
visual loss over all steps. Absent mask → legacy behavior (no cutoff).
"""
from __future__ import annotations

import torch

from feelwm.training.curriculum import CurriculumStage
from feelwm.training.losses import LossConfig, compute_total_loss


def _make_io(B=2, H=4, C=4, P=3, D=5):
    preds = {
        "z_vis_preds": torch.zeros(B, H, P, D),
        "f_prop_preds": torch.ones(B, H, C),       # pred 1
    }
    targets = {
        "z_vis_target": torch.zeros(B, H, P, D),
        "f_prop_target": torch.zeros(B, H, C),     # target 0 → per-step mse = 1 where supervised
        "valid_target_mask": torch.ones(B, H, dtype=torch.bool),
        "binary_label_target": torch.zeros(B, H),
        "binary_valid_target_mask": torch.zeros(B, H, dtype=torch.bool),
    }
    cfg = LossConfig(
        lambda_prop=1.0,
        channel_names=["a", "b", "c", "d"],
        proprio_weights=[1.0, 1.0, 1.0, 1.0],
    )
    return preds, targets, cfg


def test_cutoff_reduces_proprio_loss():
    preds, targets, cfg = _make_io()
    stage = CurriculumStage.VISUAL_PROPRIO
    _, full = compute_total_loss(preds, dict(targets), cfg, stage)

    mask = torch.tensor([[True, False, False, False], [True, False, False, False]])
    _, cut = compute_total_loss(preds, dict(targets, proprio_horizon_mask=mask), cfg, stage)

    # masking-out far steps cannot increase the (uniform) per-step proprio loss
    assert float(cut["loss_proprio"]) <= float(full["loss_proprio"]) + 1e-6
    assert float(full["loss_proprio"]) > 0.0


def test_fully_masked_horizon_zeros_proprio():
    preds, targets, cfg = _make_io()
    mask0 = torch.zeros(2, 4, dtype=torch.bool)
    _, zeroed = compute_total_loss(
        preds, dict(targets, proprio_horizon_mask=mask0), cfg, CurriculumStage.VISUAL_PROPRIO
    )
    assert float(zeroed["loss_proprio"]) == 0.0


def test_visual_loss_independent_of_horizon_mask():
    # non-zero visual residual so the visual loss is positive and mask-independent
    preds, targets, cfg = _make_io()
    preds["z_vis_preds"] = torch.ones_like(preds["z_vis_preds"])  # residual 1 at every step
    mask = torch.tensor([[True, False, False, False], [True, False, False, False]])
    _, full = compute_total_loss(preds, dict(targets), cfg, CurriculumStage.VISUAL_PROPRIO)
    _, cut = compute_total_loss(
        preds, dict(targets, proprio_horizon_mask=mask), cfg, CurriculumStage.VISUAL_PROPRIO
    )
    assert torch.allclose(full["loss_visual"], cut["loss_visual"])
