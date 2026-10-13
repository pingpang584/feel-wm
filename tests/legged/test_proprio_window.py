"""Unit tests for the shared 53-column proprio-window builder (tools/legged/).

numpy/scipy only — no bag, no torch, no Isaac. The point of these is that ONE definition
serves both the offline extractor and an online window builder, so the risk being tested
is a silent change to that definition: the reference below is a literal transcription of
the pre-refactor extractor code, kept independent on purpose.

Run: python3 -m pytest tests/legged/test_proprio_window.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "tools" / "legged"
sys.path.insert(0, str(SCRIPTS))

import proprio_window as pw          # noqa: E402
from urdf_fk import CANONICAL_JOINTS  # noqa: E402


# --- reference: the extractor's window assembly BEFORE it was factored out ------------
def _ref_hold_last(src_ts, src_val, grid):
    if len(src_ts) == 0:
        return np.full((len(grid),) + src_val.shape[1:], np.nan, np.float32)
    idx = np.clip(np.searchsorted(src_ts, grid, side="right") - 1, 0, len(src_ts) - 1)
    return src_val[idx]


def _ref_causal(src_ts, src_val, t):
    grid = t - 0.01 * np.arange(49, -1, -1)
    return _ref_hold_last(src_ts, src_val, grid)


def _ref_window(sig, t):
    imu_all = np.concatenate([sig["acc"], sig["gyro"]], axis=1)
    tw3 = np.concatenate([sig["lin"][:, [0, 1]], sig["ang"][:, 2:3]], axis=1)
    jstack = np.concatenate([sig["jpos"], sig["jvel"], sig["jtau"]], axis=1)
    w_cmd = (_ref_causal(sig["cm_ts"], sig["cm"], t) if len(sig["cm_ts"])
             else np.full((50, 2), np.nan, np.float32))
    return np.concatenate([
        _ref_causal(sig["imu_ts"], imu_all, t),
        _ref_causal(sig["od_ts"], tw3, t),
        _ref_causal(sig["od_ts"], sig["pitch"][:, None], t),
        _ref_causal(sig["od_ts"], sig["roll"][:, None], t),
        w_cmd,
        _ref_causal(sig["jt_ts"], jstack, t),
        _ref_causal(sig["jt_ts"], sig["fz"], t),
    ], axis=1)


@pytest.fixture(scope="module")
def signals():
    """Synthetic episode at the collection contract's rates (IMU 200, joints/feet 100,
    odom/cmd 50 Hz), each channel a distinct ramp so a column swap cannot pass."""
    rng = np.random.default_rng(0)
    dur = 6.0
    imu_ts = np.arange(0, dur, 1 / 200.0)
    od_ts = np.arange(0, dur, 1 / 50.0)
    jt_ts = np.arange(0, dur, 1 / 100.0)
    cm_ts = np.arange(0, dur, 1 / 50.0)
    n_i, n_o, n_j, n_c = len(imu_ts), len(od_ts), len(jt_ts), len(cm_ts)
    return {
        "imu_ts": imu_ts,
        "acc": np.stack([imu_ts + k for k in (100.0, 200.0, 300.0)], axis=1),
        "gyro": np.stack([imu_ts + k for k in (400.0, 500.0, 600.0)], axis=1),
        "od_ts": od_ts,
        "lin": np.stack([od_ts + k for k in (700.0, 800.0, 900.0)], axis=1),
        "ang": np.stack([od_ts + k for k in (1000.0, 1100.0, 1200.0)], axis=1),
        "pitch": od_ts + 1300.0, "roll": od_ts + 1400.0,
        "cm_ts": cm_ts, "cm": np.stack([cm_ts + 1500.0, cm_ts + 1600.0], axis=1),
        "jt_ts": jt_ts,
        "jpos": jt_ts[:, None] + np.arange(12)[None] * 10 + 2000.0,
        "jvel": jt_ts[:, None] + np.arange(12)[None] * 10 + 3000.0,
        "jtau": jt_ts[:, None] + np.arange(12)[None] * 10 + 4000.0,
        "fz": jt_ts[:, None] + np.arange(4)[None] * 10 + 5000.0,
        "_noise": rng.standard_normal((n_i, n_o, n_j, n_c)[0]),
    }


def _sources(sig):
    return pw.window_sources(
        imu_ts=sig["imu_ts"], acc=sig["acc"], gyro=sig["gyro"],
        od_ts=sig["od_ts"], lin=sig["lin"], ang=sig["ang"],
        pitch=sig["pitch"], roll=sig["roll"],
        cm_ts=sig["cm_ts"], cmd=sig["cm"],
        jt_ts=sig["jt_ts"], jpos=sig["jpos"], jvel=sig["jvel"], jtau=sig["jtau"],
        fz=sig["fz"])


# --- layout ---------------------------------------------------------------------------

def test_column_groups_sum_to_d_raw():
    assert sum(w for _, w in pw.COLUMN_GROUPS) == pw.D_RAW == 53


def test_window_shape_and_dtype(signals):
    w = pw.build_window(_sources(signals), 3.0)
    assert w.shape == (pw.T_WIN_N, pw.D_RAW) == (50, 53)
    assert w.dtype == np.float32


# --- the refactor did not change the definition ----------------------------------------

@pytest.mark.parametrize("t", [1.0, 2.345, 3.0, 5.5])
def test_matches_pre_refactor_reference(signals, t):
    got = pw.build_window(_sources(signals), t)
    ref = _ref_window(signals, t).astype(np.float32)
    assert np.array_equal(got, ref)


def test_build_windows_matches_per_anchor(signals):
    src = _sources(signals)
    grid = np.arange(1.0, 5.0, 0.1)
    stacked = pw.build_windows(src, grid)
    assert stacked.shape == (len(grid), 50, 53)
    for i, t in enumerate(grid):
        assert np.array_equal(stacked[i], pw.build_window(src, float(t)))


# --- causality / hold-last semantics ----------------------------------------------------

def test_window_is_causal(signals):
    """No grid point may read a sample stamped after it — the whole reason resampling is
    hold-last and not interpolation."""
    t = 3.0
    # acc_x carries the raw IMU stamp here (the fixture's +100 offset would cost ~1e-5
    # of float32 resolution at that magnitude, which is the scale being measured)
    sig = dict(signals, acc=np.column_stack([signals["imu_ts"], signals["acc"][:, 1:]]))
    w = pw.build_window(_sources(sig), t)
    stamps = w[:, 0].astype(np.float64)
    grid = t - 0.01 * np.arange(49, -1, -1)
    assert np.all(stamps <= grid + 1e-6)
    assert np.all(np.diff(stamps) >= -1e-6)          # monotone, no reordering
    assert stamps[-1] == pytest.approx(t, abs=1 / 200.0)   # last point is fresh


def test_missing_stream_is_nan_not_zero(signals):
    """An absent topic must be NaN — zeros would look like a stationary robot."""
    sig = dict(signals)
    sig["cm_ts"] = np.zeros(0)
    sig["cm"] = np.zeros((0, 2))
    w = pw.build_window(_sources(sig), 3.0)
    assert np.all(np.isnan(w[:, 11:13]))
    assert np.isfinite(w[:, :11]).all() and np.isfinite(w[:, 13:]).all()


# --- joint permutation -------------------------------------------------------------------

def test_joint_permutation_maps_tree_to_canonical():
    tree = ["LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
            "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
            "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"]     # the recorder's real order
    perm = pw.joint_permutation(tree)
    assert [tree[i] for i in perm] == CANONICAL_JOINTS


def test_joint_permutation_rejects_incomplete_names():
    with pytest.raises(ValueError, match="missing canonical joints"):
        pw.joint_permutation(["LF_HAA", "LF_HFE"])


def test_quat_pitch_roll_column_order():
    """Column 9 is pitch and column 10 is roll — the schema's order, not the intuitive one."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_euler("xyz", [[0.3, -0.2, 1.1]]).as_quat()   # roll, pitch, yaw
    pitch, roll = pw.quat_pitch_roll(q)
    assert pitch[0] == pytest.approx(-0.2)
    assert roll[0] == pytest.approx(0.3)
