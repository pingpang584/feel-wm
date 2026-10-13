"""Legged-schema additive support: registry-aware normalization + d_raw=53 / 5-ch model.

These lock the strictly-additive legged path (ANYmal-D schema) WITHOUT touching the
wheeled defaults — see tests/* elsewhere for the unchanged wheeled behavior.
"""
import numpy as np
import torch

from feelwm.data.normalize import (
    FEATURE_CHANNELS, LEGGED_FEATURE_CHANNELS, NormStats,
    normalize_feature, denormalize_feature, normalize_window, resolve_channel_indices,
)
from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel


def _legged_stats(d_raw=53):
    return NormStats(
        feature_mean=np.zeros(5, np.float32), feature_std=np.ones(5, np.float32),
        window_mean=np.zeros(d_raw, np.float32), window_std=np.ones(d_raw, np.float32),
        action_mean=np.zeros(3, np.float32), action_std=np.ones(3, np.float32),
        feature_set="legged",
    )


def test_resolve_channel_indices_autodetects_registry():
    assert resolve_channel_indices(LEGGED_FEATURE_CHANNELS) == [0, 1, 2, 3, 4]
    # wheeled subset still resolves against the wheeled registry (back-compat)
    assert resolve_channel_indices(["roughness", "slip_ratio", "ke_rate"]) == [0, 1, 3]


def test_legged_feature_set_skips_kerate_pretransform():
    """Index 3 (legged=slip) must NOT get signed_log1p. With identity stats, normalize is a no-op."""
    stats = _legged_stats()
    arr = np.array([[0.5, 0.1, -0.2, -0.9, 1.3]], np.float32)  # slip=-0.9 at idx 3
    out = normalize_feature(arr, stats)
    np.testing.assert_allclose(out, arr, atol=1e-6)              # pure z-score (identity here)
    np.testing.assert_allclose(denormalize_feature(out, stats), arr, atol=1e-6)
    # round-trip with explicit indices (idx 3 present) also clean
    np.testing.assert_allclose(
        denormalize_feature(out, stats, indices=[0, 1, 2, 3, 4]), arr, atol=1e-6)


def test_legged_window_skips_wheeled_clips():
    stats = _legged_stats()
    win = np.random.randn(4, 50, 53).astype(np.float32) * 3.0   # values that wheeled would clip
    out = normalize_window(win, stats)
    np.testing.assert_allclose(out, win, atol=1e-6)             # identity stats -> no clip, no scale


def test_model_instantiates_and_runs_with_d_raw_53():
    cfg = ModelConfig(
        context_len=3, rollout_horizon=2, n_patch=8, d_vis=32, d_prop=16, d_action=3,
        d_raw=53, proprio_feature_dim=5, n_layers=1, n_heads=4, ffn_mult=2, dropout=0.0,
    )
    model = TerrainWorldModel(cfg)
    batch = {
        "z_vis_context": torch.randn(2, 4, 8, 32),
        "proprio_window_context": torch.randn(2, 4, 50, 53),
        "action_context": torch.randn(2, 4, 3),
    }
    out = model(batch)              # accepts the 53-dim proprio window end-to-end
    assert out["f_prop_pred"].shape == (2, 5)
