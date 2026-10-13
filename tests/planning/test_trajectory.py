"""Unit tests for SE(2) trajectory integration + ADE/FDE (NWM-protocol metrics).

Run: python -m pytest tests/planning/test_trajectory.py -v
"""
from __future__ import annotations

import math

import numpy as np
import torch

from feelwm.data.normalize import NormStats
from feelwm.planning.trajectory import (
    actions_to_deltas,
    ade_fde,
    ate_rte,
    candidate_trajectories,
    integrate_se2,
)


def _identity_stats() -> NormStats:
    return NormStats(
        feature_mean=np.zeros(5, np.float32), feature_std=np.ones(5, np.float32),
        window_mean=np.zeros(14, np.float32), window_std=np.ones(14, np.float32),
        action_mean=np.zeros(3, np.float32), action_std=np.ones(3, np.float32),
    )


def test_integrate_straight_line():
    d = torch.zeros(1, 4, 3)
    d[..., 0] = 1.0  # 1 m forward per step, no yaw
    pos = integrate_se2(d)
    assert torch.allclose(pos[0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(pos[0, :, 1], torch.zeros(4), atol=1e-6)


def test_integrate_quarter_turns():
    # forward 1 m then turn 90° each step → unit square: (1,0) (1,1) (0,1) (0,0)
    d = torch.zeros(1, 4, 3)
    d[..., 0] = 1.0
    d[..., 2] = math.pi / 2
    pos = integrate_se2(d)
    expected = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    assert torch.allclose(pos[0], expected, atol=1e-5)


def test_yaw_applies_before_translation_of_next_step():
    # step 1: pure rotation 90°; step 2: forward 1 → should land at (0, 1)
    d = torch.zeros(1, 2, 3)
    d[0, 0, 2] = math.pi / 2
    d[0, 1, 0] = 1.0
    pos = integrate_se2(d)
    assert torch.allclose(pos[0, 1], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_actions_to_deltas_scales_by_stride():
    stats = _identity_stats()
    a = torch.zeros(1, 3, 4)
    a[..., 0] = 2.0   # rate: 2 m per 0.1s anchor
    a[..., 3] = 0.5   # Δt 0.5s → stride 5
    d = actions_to_deltas(a, stats)
    assert torch.allclose(d[..., 0], torch.full((1, 3), 10.0))  # 2 × 5
    # 3-dim legacy actions: no scaling
    d3 = actions_to_deltas(a[..., :3], stats)
    assert torch.allclose(d3[..., 0], torch.full((1, 3), 2.0))


def test_ade_fde_known_values():
    ref = torch.zeros(1, 3, 2)
    traj = torch.zeros(1, 3, 2)
    traj[0, :, 0] = torch.tensor([1.0, 2.0, 3.0])  # distances 1, 2, 3
    ade, fde = ade_fde(traj, ref)
    assert torch.allclose(ade, torch.tensor([2.0]))
    assert torch.allclose(fde, torch.tensor([3.0]))


def test_ate_is_the_rms_of_the_same_errors_ade_averages():
    """ATE and ADE differ only in the aggregator, so ATE >= ADE with equality iff the
    per-waypoint errors are all equal. Both properties are checked, because "ATE is just
    RMSE-ADE" is the whole reason our rows can be read against the NWM paper's columns."""
    ref = torch.zeros(1, 3, 2)
    traj = torch.zeros(1, 3, 2)
    traj[0, :, 0] = torch.tensor([1.0, 2.0, 3.0])
    ade, _ = ade_fde(traj, ref)
    ate, _ = ate_rte(traj, ref)
    assert torch.allclose(ade, torch.tensor([2.0]))
    assert torch.allclose(ate, torch.tensor([math.sqrt((1 + 4 + 9) / 3)]))
    assert (ate > ade).all()

    flat = torch.zeros(1, 3, 2)
    flat[0, :, 0] = 2.0                       # every waypoint off by the same 2 m
    ade_f, _ = ade_fde(flat, ref)
    ate_f, _ = ate_rte(flat, ref)
    assert torch.allclose(ate_f, ade_f)


def test_rte_is_per_step_and_does_not_accumulate():
    """RTE reads step displacement, so a constant offset (no per-step error) gives 0 while
    ATE grows — that separation is what makes the pair diagnostic."""
    ref = torch.zeros(1, 4, 2)
    ref[0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])      # expert: 1 m per step

    offset = ref.clone()
    offset[0, :, 1] = 5.0                                   # rigidly shifted sideways
    ate_o, rte_o = ate_rte(offset, ref)
    assert torch.allclose(ate_o, torch.tensor([5.0]))
    assert torch.allclose(rte_o, torch.zeros(1), atol=1e-6)

    fast = torch.zeros(1, 4, 2)
    fast[0, :, 0] = torch.tensor([1.5, 3.0, 4.5, 6.0])      # 1.5 m per step: +0.5 each
    ate_f, rte_f = ate_rte(fast, ref)
    assert torch.allclose(rte_f, torch.tensor([0.5]))       # constant per-step error
    # errors 0.5/1.0/1.5/2.0 accumulate linearly -> ATE is their RMS
    assert torch.allclose(ate_f, torch.tensor([math.sqrt((0.25 + 1 + 2.25 + 4) / 4)]))


def test_pure_speed_offset_pins_the_fde_over_ade_ratio():
    """A reference constant: a straight plan whose only error is a speed offset
    has err_t proportional to t, so FDE/ADE = H / mean(1..H) — 1.778 at the H=8 the
    planning queries use — regardless of how big the offset is. A row's distance from
    this number reads as "how speed-driven is this error"."""
    for H in (4, 8, 16):
        for offset in (0.1, 2.0):
            ref = torch.zeros(1, H, 2)
            ref[0, :, 0] = torch.arange(1, H + 1, dtype=torch.float32)
            traj = ref.clone()
            traj[0, :, 0] += offset * torch.arange(1, H + 1, dtype=torch.float32)
            ade, fde = ade_fde(traj, ref)
            expected = H / np.arange(1, H + 1).mean()
            assert math.isclose(float(fde / ade), expected, rel_tol=1e-5)
    assert math.isclose(8 / np.arange(1, 9).mean(), 1.7777, rel_tol=1e-3)


def test_candidate_trajectories_batch_shape():
    stats = _identity_stats()
    a = torch.zeros(5, 8, 4)
    a[..., 3] = 0.1
    pos = candidate_trajectories(a, stats)
    assert pos.shape == (5, 8, 2)
    # identical candidates → identical trajectories → zero ADE between them
    ade, fde = ade_fde(pos, pos[0:1])
    assert torch.allclose(ade, torch.zeros(5), atol=1e-6)
