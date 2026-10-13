"""Normalization utilities for TartanDrive v2 DataLoader.

Pre-transforms (clamp/scale) are applied first, then per-channel z-score.
All operations work on numpy arrays; the caller converts to torch.

Channel order:
  proprio_feature (5): roughness, slip_ratio, angular_instability, ke_rate, slope_power
  proprio_window (14): ax, ay, az, gx, gy, gz, vx, vy, wz, pitch_rad, roll_rad, throttle, brake, steer_cmd
  action (3): dx, dy, dyaw
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Canonical proprio-feature channel order (single source of truth, WHEELED/TartanDrive).
# The ke_rate signed_log1p pre-transform below is keyed to canonical index 3.
FEATURE_CHANNELS = ["roughness", "slip_ratio", "angular_instability", "ke_rate", "slope_power"]

# Legged (ANYmal-D, TartanGround+GrandTour) channel order.
# Separate registry; the wheeled FEATURE_CHANNELS is unchanged. Legged uses NO signed_log1p
# (index 3 here is `slip`, not ke_rate) and NO wheeled window pretransform: see feature_set below.
LEGGED_FEATURE_CHANNELS = [
    "roughness", "angular_instability", "slope_power", "slip", "cost_of_transport"
]
# Names that uniquely identify each registry (used to auto-detect in resolve_channel_indices).
_LEGGED_ONLY = set(LEGGED_FEATURE_CHANNELS) - set(FEATURE_CHANNELS)  # {slip, cost_of_transport}


def resolve_channel_indices(names: list[str], registry: list[str] | None = None) -> list[int]:
    """Map active proprio-channel names to their canonical registry indices.

    registry=None auto-detects: legged if any legged-exclusive name (slip / cost_of_transport)
    is present, else wheeled (default — back-compatible for all existing call sites)."""
    if registry is None:
        registry = LEGGED_FEATURE_CHANNELS if any(n in _LEGGED_ONLY for n in names) else FEATURE_CHANNELS
    idx = []
    for n in names:
        if n not in registry:
            raise ValueError(f"unknown proprio channel {n!r}; valid: {registry}")
        idx.append(registry.index(n))
    return idx


@dataclass(frozen=True)
class NormStats:
    feature_mean: np.ndarray   # (5,)
    feature_std: np.ndarray    # (5,)
    window_mean: np.ndarray    # wheeled (14,) / legged (53,)
    window_std: np.ndarray     # wheeled (14,) / legged (53,)
    action_mean: np.ndarray    # (3,)
    action_std: np.ndarray     # (3,)
    # "wheeled" (default) applies the TartanDrive pretransforms (signed_log1p ke_rate,
    # brake/steer/vx window clips). "legged" skips them (pure per-channel z-score).
    # Absent in legacy wheeled JSON -> defaults to "wheeled" -> bit-identical behavior.
    feature_set: str = "wheeled"

    @classmethod
    def from_json(cls, path: str | Path) -> "NormStats":
        d = json.loads(Path(path).read_text())
        return cls(
            feature_mean=np.array(d["proprio_feature"]["mean"], dtype=np.float32),
            feature_std=np.array(d["proprio_feature"]["std"], dtype=np.float32),
            window_mean=np.array(d["proprio_window"]["mean"], dtype=np.float32),
            window_std=np.array(d["proprio_window"]["std"], dtype=np.float32),
            action_mean=np.array(d["action"]["mean"], dtype=np.float32),
            action_std=np.array(d["action"]["std"], dtype=np.float32),
            feature_set=d.get("feature_set", "wheeled"),
        )


# --- pre-transforms (applied before z-score) ---

def _feature_pretransform(arr: np.ndarray) -> np.ndarray:
    """signed_log1p on ke_rate (ch3). Returns a copy."""
    arr = arr.copy()
    arr[..., 3] = np.sign(arr[..., 3]) * np.log1p(np.abs(arr[..., 3]))
    return arr


def _feature_inv_pretransform(arr: np.ndarray) -> np.ndarray:
    """Inverse of signed_log1p on ch3. Returns a copy."""
    arr = arr.copy()
    x = arr[..., 3]
    arr[..., 3] = np.sign(x) * (np.expm1(np.abs(x)))
    return arr


def _window_pretransform(arr: np.ndarray) -> np.ndarray:
    """Clamp/scale window channels. Returns a copy."""
    arr = arr.copy()
    arr[..., 6]  = np.clip(arr[..., 6],  -10.0,  30.0)   # vx
    arr[..., 7]  = np.clip(arr[..., 7],   -5.0,   5.0)   # vy
    arr[..., 12] = np.clip(arr[..., 12],   0.0, 500.0)   # brake
    arr[..., 13] = arr[..., 13] / 415.0                   # steer_cmd → [-1, 1]
    return arr


def _window_inv_pretransform(arr: np.ndarray) -> np.ndarray:
    """Approximate inverse of window pre-transforms. Returns a copy.

    Clamp inversions are identity (clamping is non-invertible); only
    steer_cmd scaling is inverted. Use only for approximate round-trip checks.
    """
    arr = arr.copy()
    arr[..., 13] = arr[..., 13] * 415.0
    return arr


# --- normalize / denormalize ---

def normalize_feature(arr: np.ndarray, stats: NormStats) -> np.ndarray:
    """(N, 5) or (..., 5) float32 → normalized. Legged skips the ke_rate pretransform."""
    if stats.feature_set == "wheeled":
        arr = _feature_pretransform(arr)
    return (arr - stats.feature_mean) / stats.feature_std


def denormalize_feature(
    arr: np.ndarray, stats: NormStats, indices: list[int] | None = None
) -> np.ndarray:
    """Inverse z-score, then (wheeled only) inverse pre-transform on ke_rate.

    indices: canonical channel indices that the columns of ``arr`` correspond to.
    None (default) means all 5 channels in canonical order. When a subset is given,
    the z-score uses the matching stats columns and the ke_rate (canonical index 3)
    inverse-pretransform is applied only if channel 3 is present, at its column.
    The ke_rate pretransform exists only in the wheeled registry; legged is pure z-score.
    """
    if indices is None:
        arr = arr * stats.feature_std + stats.feature_mean
        return _feature_inv_pretransform(arr) if stats.feature_set == "wheeled" else arr
    idx = list(indices)
    arr = arr * stats.feature_std[idx] + stats.feature_mean[idx]
    if stats.feature_set == "wheeled" and 3 in idx:  # ke_rate present in the subset
        arr = arr.copy()
        pos = idx.index(3)
        x = arr[..., pos]
        arr[..., pos] = np.sign(x) * np.expm1(np.abs(x))
    return arr


def normalize_window(arr: np.ndarray, stats: NormStats) -> np.ndarray:
    """(..., T_win, d_raw) float32 → normalized. Legged (d_raw=53) skips wheeled column clips."""
    if stats.feature_set == "wheeled":
        arr = _window_pretransform(arr)
    return (arr - stats.window_mean) / stats.window_std


def denormalize_window(arr: np.ndarray, stats: NormStats) -> np.ndarray:
    """Inverse z-score, then (wheeled only) approximate inverse pre-transforms."""
    arr = arr * stats.window_std + stats.window_mean
    return _window_inv_pretransform(arr) if stats.feature_set == "wheeled" else arr


def normalize_action(arr: np.ndarray, stats: NormStats) -> np.ndarray:
    """(..., 3) float32 → normalized."""
    return (arr - stats.action_mean) / stats.action_std


def denormalize_action(arr: np.ndarray, stats: NormStats) -> np.ndarray:
    return arr * stats.action_std + stats.action_mean
