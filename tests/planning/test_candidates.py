"""Unit tests for candidate-pool assembly (planner candidate library + CEM samples).

Run: python -m pytest tests/planning/test_candidates.py -v
"""
from __future__ import annotations

import pytest
import torch

from feelwm.planning.candidates import (
    POSE_DELTA_DIMS,
    inject_expert,
    perturb_actions,
    stack_samples,
)


def _fake_sample(H: int, da: int = 4, C: int = 4, P: int = 4, D: int = 8, fill: float = 0.0):
    return {
        "action_sequence": torch.full((H, da), fill),
        "f_prop_target": torch.full((H, C), fill),
        "valid_target_mask": torch.ones(H, dtype=torch.bool),
        "proprio_horizon_mask": torch.ones(H, dtype=torch.bool),
        "z_vis_target": torch.full((H, P, D), fill),
    }


def test_stack_truncates_and_filters_short():
    samples = [_fake_sample(8, fill=1.0), _fake_sample(4, fill=2.0), _fake_sample(10, fill=3.0)]
    pool = stack_samples(samples, horizon=6, source_index=[10, 11, 12])
    assert len(pool) == 2  # the H=4 sample is dropped
    assert pool.actions.shape == (2, 6, 4)
    assert pool.f_prop_gt.shape == (2, 6, 4)
    assert pool.z_vis_goal.shape == (2, 4, 8)
    assert pool.source_index == [10, 12]


def test_stack_goal_is_final_frame():
    s = _fake_sample(8)
    s["z_vis_target"][5] = 7.0  # horizon=6 → goal is index 5
    pool = stack_samples([s], horizon=6)
    assert torch.allclose(pool.z_vis_goal[0], torch.full((4, 8), 7.0))


def test_stack_raises_when_none_qualify():
    with pytest.raises(ValueError):
        stack_samples([_fake_sample(3)], horizon=8)


def test_inject_expert_index_and_shapes():
    pool = stack_samples([_fake_sample(8), _fake_sample(8)], horizon=6)
    expert = torch.full((6, 4), 9.0)
    out = inject_expert(pool, expert, at=0)
    assert out.expert_index == 0
    assert len(out) == 3
    assert torch.allclose(out.actions[0], expert)
    # expert metadata is NaN-padded so it is excluded from GT-based stats
    assert torch.isnan(out.f_prop_gt[0]).all()
    assert not out.valid[0].any()


def test_inject_expert_bad_shape():
    pool = stack_samples([_fake_sample(8)], horizon=6)
    with pytest.raises(ValueError):
        inject_expert(pool, torch.zeros(5, 4))


def test_perturb_leaves_dt_untouched():
    base = torch.zeros(6, 4)
    base[:, 3] = 0.1  # Δt column
    pert = perturb_actions(base, n=16, sigma=1.0)
    assert pert.shape == (16, 6, 4)
    assert torch.allclose(pert[..., 3], torch.full((16, 6), 0.1))  # Δt preserved
    assert pert[..., list(POSE_DELTA_DIMS)].abs().sum() > 0  # pose dims moved


def test_perturb_reproducible_with_generator():
    base = torch.zeros(4, 4)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    assert torch.allclose(
        perturb_actions(base, 8, 0.5, generator=g1),
        perturb_actions(base, 8, 0.5, generator=g2),
    )
