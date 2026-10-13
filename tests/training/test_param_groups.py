"""Two-group LR selector for fine-tuning with a freshly initialised token bottleneck.

A single lr tuned for 1.3M trainable params is wrong when 58.2M are trainable, 14.4M of
them freshly initialised. The selector splits them: the fresh bottleneck (its gate
included) trains at `lr_fresh`, everything pretrained at `lr`. These tests pin the split
itself — the warmup/cosine scaling is per group and already multiplicative
(`_set_warmup_lr`).
"""
from __future__ import annotations

import pytest
import torch

from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel
from feelwm.training.trainer import build_param_groups

LR, LR_FRESH = 6e-5, 6e-4
PREFIXES = ["dynamics.bottleneck."]


def _model(**overrides) -> TerrainWorldModel:
    kwargs = dict(
        context_len=3, rollout_horizon=3, n_patch=16, d_vis=32, d_prop=8, d_action=4,
        d_raw=11, n_layers=2, n_heads=2, ffn_mult=2, proprio_feature_dim=3,
        use_traversability_head=False,
        token_bottleneck={"type": "perceiver", "k": 4, "residual_exit": True},
    )
    kwargs.update(overrides)
    return TerrainWorldModel(ModelConfig(**kwargs))


def test_every_bottleneck_tensor_including_the_gate_gets_lr_fresh():
    model = _model()
    for name in ("film", "action_embedding", "visual_head"):     # run-2 freeze list
        for p in getattr(model, name).parameters():
            p.requires_grad_(False)
    groups = build_param_groups(model, lr=LR, lr_fresh=LR_FRESH, fresh_prefixes=PREFIXES)

    by_lr = {g["lr"]: {id(p) for p in g["params"]} for g in groups}
    assert set(by_lr) == {LR, LR_FRESH}
    expected_fresh = {
        id(p) for n, p in model.named_parameters() if n.startswith("dynamics.bottleneck.")
    }
    assert by_lr[LR_FRESH] == expected_fresh
    assert id(model.dynamics.bottleneck.exit_gate) in by_lr[LR_FRESH]

    # every OTHER trainable tensor is in the pretrained group, and nothing is in both
    expected_base = {
        id(p) for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("dynamics.bottleneck.")
    }
    assert by_lr[LR] == expected_base
    assert not (by_lr[LR] & by_lr[LR_FRESH])
    assert id(next(model.visual_head.parameters())) not in by_lr[LR] | by_lr[LR_FRESH]


def test_pretrained_group_is_first_so_current_lr_still_means_the_main_lr():
    groups = build_param_groups(_model(), lr=LR, lr_fresh=LR_FRESH, fresh_prefixes=PREFIXES)
    assert [g["name"] for g in groups] == ["pretrained", "fresh"]
    assert groups[0]["lr"] == LR


def test_lr_fresh_none_is_one_group_at_lr():
    groups = build_param_groups(_model(), lr=LR, lr_fresh=None, fresh_prefixes=PREFIXES)
    assert len(groups) == 1 and groups[0]["lr"] == LR
    assert len(groups[0]["params"]) == sum(1 for p in _model().parameters() if p.requires_grad)


def test_a_prefix_that_matches_nothing_is_an_error_not_a_silent_single_group():
    with pytest.raises(ValueError):
        build_param_groups(_model(token_bottleneck=None), lr=LR, lr_fresh=LR_FRESH,
                           fresh_prefixes=PREFIXES)


def test_the_optimizer_actually_steps_the_two_groups_at_different_rates():
    model = _model()
    groups = build_param_groups(model, lr=LR, lr_fresh=LR_FRESH, fresh_prefixes=PREFIXES)
    opt = torch.optim.AdamW(groups, lr=LR, weight_decay=0.0)
    gate = model.dynamics.bottleneck.exit_gate
    core = model.dynamics.out_norm.weight
    before = (gate.detach().clone(), core.detach().clone())
    for p in (gate, core):
        p.grad = torch.ones_like(p)
    opt.step()
    # AdamW's first step moves each param by ~lr, so the ratio of the two steps is the
    # ratio of the two lrs.
    d_gate = (gate.detach() - before[0]).abs().mean().item()
    d_core = (core.detach() - before[1]).abs().mean().item()
    assert d_gate / d_core == pytest.approx(LR_FRESH / LR, rel=1e-3)
