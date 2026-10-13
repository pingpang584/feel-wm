"""Unit tests for the free-CEM init arm (W1) and the TD1 in-domain eval layer (W2).

Everything synthetic — no data, checkpoints, or GPU.

The two things worth pinning down here are both places where an "obvious" implementation is
silently wrong:

- **`--init-mean-scale` is a PHYSICAL scale.** The dataset action is a z-scored per-0.1 s
  rate, so the train prior sits at ~0 in normalized space; multiplying the normalized numbers
  moves the init almost not at all. The test asserts the arm reaches the spacing convention.
- **the frame→anchor join.** Our own export stores `source_anchors`, so the join is exact;
  NWM's export has no such key and must fall back to path-length matching. Getting this
  backwards would silently mis-pair every TD1 query by a few frames.

Run: python -m pytest tests/planning/test_cem_init_and_td1_layer.py -v
"""
from __future__ import annotations

import importlib

import numpy as np
import torch

from feelwm.data.normalize import NormStats

CEM = importlib.import_module("tools.eval.openloop_cem")

SPACING_M = 0.72
STRIDE = 4


def _stats(mean=(0.45, 0.007, 0.0002), std=(0.277, 0.089, 0.0165)) -> NormStats:
    """Action stats shaped like `data/norm_stats/v2.json`; other fields unused here."""
    z = np.zeros(5, dtype=np.float32)
    return NormStats(feature_mean=z, feature_std=z + 1, window_mean=z, window_std=z + 1,
                     action_mean=np.array(mean, dtype=np.float32),
                     action_std=np.array(std, dtype=np.float32))


# ------------------------------------------------------------------ W1: init mean scale


def test_forward_u_matches_hand_computation():
    st = _stats()
    mean = torch.tensor([-0.0158, -0.007, -0.004])
    expect = (-0.0158 * 0.277 + 0.45) * STRIDE / SPACING_M
    assert abs(CEM.forward_u_per_step(mean, st, STRIDE) - expect) < 1e-5


def test_scale_is_physical_not_normalized():
    """The convention arm must LAND at ~1 spacing unit/step; normalized scaling would not."""
    st = _stats()
    mean = torch.tensor([-0.0158, -0.007, -0.004])
    base = CEM.forward_u_per_step(mean, st, STRIDE)
    scaled = CEM.scale_init_mean(mean, st, 1.0 / base)
    assert abs(CEM.forward_u_per_step(scaled, st, STRIDE) - 1.0) < 1e-4

    naive = mean.clone()
    naive[:2] *= 1.0 / base                      # what scaling the normalized numbers gives
    assert abs(CEM.forward_u_per_step(naive, st, STRIDE) - base) < 0.05


def test_scale_is_exactly_proportional_in_physical_space():
    st = _stats()
    mean = torch.tensor([0.8, 0.2, -0.05])
    for f in (0.3922, 0.5, 2.0):
        assert abs(CEM.forward_u_per_step(CEM.scale_init_mean(mean, st, f), st, STRIDE)
                   - f * CEM.forward_u_per_step(mean, st, STRIDE)) < 1e-4


def test_scale_one_is_bit_exact_and_dyaw_never_moves():
    """Default must reproduce existing runs bit-for-bit; the arm touches (dx, dy) only."""
    st = _stats()
    mean = torch.tensor([-0.0158, -0.007, -0.004])
    assert torch.equal(CEM.scale_init_mean(mean, st, 1.0), mean)
    assert float(CEM.scale_init_mean(mean, st, 0.3922)[2]) == float(mean[2])


def test_aggregate_records_the_arm_and_drops_nwm_columns_when_absent():
    rows = [{"qid": 0, "pair_discrepancy_m": 0.0, "ade_goalonly": 1.0, "fde_goalonly": 2.0,
             "ade_full": 1.0, "fde_full": 2.0, "ate_goalonly": 1.0, "rte_goalonly": 1.0,
             "ate_full": 1.0, "rte_full": 1.0}]
    prior = {"mean": np.zeros(3), "std": np.ones(3)}
    cem = {"n_candidates": 8, "iterations": 2, "elite_frac": 0.25, "epsilon": 0.0,
           "init_mean_scale": 0.3922, "init_forward_u_per_step": 0.981}
    out = CEM._aggregate(rows, "val", 0, prior, cem, STRIDE, 8, 1.0, "off")
    assert out["cem"]["init_mean_scale"] == 0.3922
    assert "physical space" in out["cem"]["init"]
    assert "nwm_ade" not in out and "paired_win_rate_vs_nwm" not in out


# ------------------------------------------------------------------ W2: frame→anchor join



# ------------------------------------------------------------------ W2: query stratification


def _eligible(counts: dict[str, int]) -> list[tuple[str, int]]:
    return [(s, t) for s, c in counts.items() for t in range(c)]


