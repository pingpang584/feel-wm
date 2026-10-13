"""Unit tests for the ride_cost aggregation (ride cost predicted from proprio).

Run: python -m pytest tests/planning/test_ride_cost.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from feelwm.data.normalize import NormStats
from feelwm.planning.ride_cost import (
    DEFAULT_RIDE_CHANNELS,
    RideCostConfig,
    ride_cost,
    ride_cost_physical,
)

# v4 multistride joint model: 4 active channels (ke_rate dropped).
CHANNELS = ["roughness", "slip_ratio", "angular_instability", "slope_power"]


def test_slip_excluded_by_default():
    """Changing only the slip column must not move the default ride cost."""
    f = torch.zeros(2, 4, 4)
    base = ride_cost(f, CHANNELS)
    f[..., 1] = 5.0  # slip_ratio column
    assert torch.allclose(ride_cost(f, CHANNELS), base)


def test_uniform_sum_over_masked_steps():
    f = torch.zeros(1, 4, 4)
    # roughness=1, instability=2, slope_power=3 at every step; slip irrelevant.
    f[..., 0], f[..., 2], f[..., 3] = 1.0, 2.0, 3.0
    # uniform weights → per-step sum = 1+2+3 = 6; mean over steps = 6.
    assert torch.allclose(ride_cost(f, CHANNELS), torch.tensor([6.0]))


def test_horizon_mask_restricts_average():
    f = torch.zeros(1, 4, 4)
    f[:, :2, 0] = 1.0   # first two steps roughness=1
    f[:, 2:, 0] = 9.0   # last two steps roughness=9 (beyond horizon)
    mask = torch.tensor([[True, True, False, False]])
    # only first two steps counted → mean roughness = 1.0
    assert torch.allclose(ride_cost(f, CHANNELS, horizon_mask=mask), torch.tensor([1.0]))
    # 1-D mask broadcasts over batch
    assert torch.allclose(
        ride_cost(f, CHANNELS, horizon_mask=mask[0]), torch.tensor([1.0])
    )


def test_weights_and_discount():
    f = torch.zeros(1, 2, 4)
    f[:, 0, 0], f[:, 1, 0] = 1.0, 0.0  # roughness 1 then 0; others zero
    cfg = RideCostConfig(weights=[2.0, 1.0, 1.0], discount=0.5)
    # per-step sum: step0 = 2*1 = 2, step1 = 0. weighted mean with γ^h:
    # (2*1 + 0*0.5) / (1 + 0.5) = 2/1.5
    assert torch.allclose(ride_cost(f, CHANNELS, cfg=cfg), torch.tensor([2.0 / 1.5]))


def test_not_clamped_negative():
    f = torch.zeros(1, 3, 4)
    f[..., 0] = -2.0  # smoother than average → negative cost allowed
    assert (ride_cost(f, CHANNELS) < 0).all()


def test_unknown_channel_raises():
    with pytest.raises(ValueError):
        ride_cost(torch.zeros(1, 2, 4), CHANNELS, cfg=RideCostConfig(ride_channels=["bogus"]))


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        ride_cost(torch.zeros(2, 4), CHANNELS)


def test_default_channels_match_spec():
    assert DEFAULT_RIDE_CHANNELS == ["roughness", "angular_instability", "slope_power"]


def test_physical_appendix_keys_and_shape():
    stats = NormStats(
        feature_mean=np.zeros(5, np.float32),
        feature_std=np.ones(5, np.float32),
        window_mean=np.zeros(14, np.float32),
        window_std=np.ones(14, np.float32),
        action_mean=np.zeros(3, np.float32),
        action_std=np.ones(3, np.float32),
    )
    f = torch.zeros(3, 4, 4)
    f[..., 0] = 1.5  # roughness; identity stats → physical == normalized
    out = ride_cost_physical(f, CHANNELS, stats)
    assert set(out) == set(DEFAULT_RIDE_CHANNELS)
    assert out["roughness"].shape == (3,)
    assert torch.allclose(out["roughness"], torch.full((3,), 1.5))


# ── safety aggregations (ride-cost recomposition) ────────────────────────────────────


def test_agg_mean_is_default_and_unchanged():
    """agg='mean' must be bit-identical to the pre-recomposition locked spec."""
    torch.manual_seed(0)
    f = torch.randn(4, 6, 4)
    mask = torch.tensor([True, True, True, False, False, False])
    a = ride_cost(f, CHANNELS, horizon_mask=mask)
    b = ride_cost(f, CHANNELS, horizon_mask=mask, cfg=RideCostConfig(agg="mean"))
    assert torch.equal(a, b)


def test_agg_max_worst_masked_step():
    f = torch.zeros(1, 4, 4)
    f[:, 1, 0] = 2.0   # in-mask peak (per-step sum 2)
    f[:, 3, 0] = 9.0   # beyond horizon — must not count
    mask = torch.tensor([[True, True, True, False]])
    out = ride_cost(f, CHANNELS, horizon_mask=mask, cfg=RideCostConfig(agg="max"))
    assert torch.allclose(out, torch.tensor([2.0]))


def test_agg_max_degenerate_mask_falls_back_to_full_horizon():
    f = torch.zeros(1, 3, 4)
    f[:, 2, 0] = 4.0
    mask = torch.zeros(1, 3, dtype=torch.bool)
    out = ride_cost(f, CHANNELS, horizon_mask=mask, cfg=RideCostConfig(agg="max"))
    assert torch.allclose(out, torch.tensor([4.0]))


def test_agg_cvar_mean_of_worst_tail():
    f = torch.zeros(1, 4, 4)
    f[:, :, 0] = torch.tensor([1.0, 4.0, 3.0, 2.0])
    # α=0.5 over K=4 masked steps → worst 2 = {4, 3} → 3.5
    cfg = RideCostConfig(agg="cvar", cvar_alpha=0.5)
    assert torch.allclose(ride_cost(f, CHANNELS, cfg=cfg), torch.tensor([3.5]))
    # α small → ⌈α·K⌉ clamps to 1 → plain max
    cfg1 = RideCostConfig(agg="cvar", cvar_alpha=0.01)
    assert torch.allclose(ride_cost(f, CHANNELS, cfg=cfg1), torch.tensor([4.0]))


def test_agg_cvar_respects_mask():
    f = torch.zeros(1, 4, 4)
    f[:, :, 0] = torch.tensor([1.0, 2.0, 0.0, 9.0])
    mask = torch.tensor([[True, True, True, False]])
    # K=3 valid, α=0.34 → k=⌈1.02⌉=2 → worst 2 in-mask = {2, 1} → 1.5
    cfg = RideCostConfig(agg="cvar", cvar_alpha=0.34)
    assert torch.allclose(ride_cost(f, CHANNELS, horizon_mask=mask, cfg=cfg),
                          torch.tensor([1.5]))


def test_agg_exceed_zero_below_threshold_superlinear_above():
    f = torch.zeros(1, 2, 4)
    f[:, :, 0] = torch.tensor([0.5, 0.9])   # per-step sums below z0=1
    cfg = RideCostConfig(agg="exceed", exceed_z0=1.0, exceed_power=2.0)
    assert torch.allclose(ride_cost(f, CHANNELS, cfg=cfg), torch.tensor([0.0]))
    f[:, 1, 0] = 3.0   # exceed by 2 → 2² = 4; mean over 2 steps = 2
    assert torch.allclose(ride_cost(f, CHANNELS, cfg=cfg), torch.tensor([2.0]))


def test_agg_unknown_raises():
    with pytest.raises(ValueError):
        ride_cost(torch.zeros(1, 2, 4), CHANNELS, cfg=RideCostConfig(agg="median"))


def test_slip_only_config_reads_slip_column():
    """The λ_slip term reuses ride_cost with channels=[slip_ratio]."""
    from feelwm.planning.ride_cost import ride_step_values

    f = torch.zeros(2, 3, 4)
    f[..., 1] = torch.tensor([0.1, 0.8, 0.3])   # slip column
    cfg = RideCostConfig(ride_channels=["slip_ratio"], agg="max")
    assert torch.allclose(ride_cost(f, CHANNELS, cfg=cfg), torch.full((2,), 0.8))
    # per-step values (the gate/logging stat) expose the same column untouched
    assert torch.allclose(ride_step_values(f, CHANNELS, cfg),
                          torch.tensor([0.1, 0.8, 0.3]).expand(2, 3))
