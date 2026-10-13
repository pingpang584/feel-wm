"""Unit tests for the Isaac-legged Stage-B extractor (tools/legged/).

numpy/scipy only — no bag, no torch — so they run in the gazebo container's system
python as well as the nwm env. The end-to-end contract is exercised on a real recorded
bag by the extractor itself, not here.

Run: python3 -m pytest tests/legged/test_extract_isaac_anymal.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "tools" / "legged"
sys.path.insert(0, str(SCRIPTS))

import extract_isaac_anymal as ex          # noqa: E402
from urdf_fk import CANONICAL_JOINTS, FootFK   # noqa: E402


# --- FK geometry -------------------------------------------------------------

@pytest.fixture(scope="module")
def fk():
    return FootFK()


def test_fk_zero_pose_symmetry(fk):
    p = fk.foot_positions(np.zeros((1, 12)))[0]        # LF, RF, LH, RH
    assert np.allclose(p[0, 2], p[:, 2], atol=1e-6)    # equal height
    assert np.allclose(p[0], p[1] * [1, -1, 1], atol=1e-4)   # LF == mirrored RF
    assert np.allclose(p[0], p[2] * [-1, 1, 1], atol=1e-4)   # LF == mirrored LH


def test_fk_stand_pose_matches_measured_base_height(fk):
    # IsaacLab anymal_c default stand: HAA 0, HFE +-0.4, KFE -+0.8. The campaign
    # collector measured base height 0.54 m — the FK feet must sit ~that far below.
    q = np.array([[0, 0.4, -0.8, 0, 0.4, -0.8, 0, -0.4, 0.8, 0, -0.4, 0.8]])
    z = fk.foot_positions(q)[0, :, 2]
    assert np.allclose(z, -0.54, atol=0.02)


def test_fk_offset_point_preserves_distance(fk):
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.6, 0.6, (5, 12))
    p_ball = fk.foot_positions(q)
    p_com = fk.foot_positions(q, offsets=fk.foot_coms)
    d = np.linalg.norm(p_com - p_ball, axis=2)         # (5,4)
    expect = np.linalg.norm(fk.foot_coms, axis=1)      # (4,)
    assert np.allclose(d, expect[None, :], atol=1e-9)


def test_fk_com_offsets_parsed(fk):
    # all four FOOT links carry a nonzero inertial origin in the vendored URDF
    assert fk.foot_coms.shape == (4, 3)
    assert (np.linalg.norm(fk.foot_coms, axis=1) > 0.1).all()


def test_canonical_joint_order():
    assert CANONICAL_JOINTS[:3] == ["LF_HAA", "LF_HFE", "LF_KFE"]
    assert len(CANONICAL_JOINTS) == 12
    # the Isaac tree order (campaign meta joint_order) must be a permutation of it
    tree = ["LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA", "LF_HFE", "LH_HFE", "RF_HFE",
            "RH_HFE", "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE"]
    perm = [tree.index(j) for j in CANONICAL_JOINTS]
    assert sorted(perm) == list(range(12))
    assert [tree[i] for i in perm] == CANONICAL_JOINTS


# --- stride detection + notch ------------------------------------------------

def _synthetic_fz(f0, fs=100.0, dur=30.0, seed=1):
    t = np.arange(0, dur, 1 / fs)
    rng = np.random.default_rng(seed)
    fz = np.zeros((len(t), 4))
    for k in range(4):
        phase = k * np.pi / 2
        fz[:, k] = 250 + 200 * np.clip(np.sin(2 * np.pi * f0 * t + phase), 0, None) \
            + rng.normal(0, 10, len(t))
    return fz


def test_detect_stride_synthetic():
    fz = _synthetic_fz(1.8)
    f0, method, snr = ex.detect_stride(fz, 100.0, 30.0, v_nom=0.7)
    assert method == "fz_psd"
    assert abs(f0 - 1.8) < 0.06
    assert snr > ex.STRIDE_MIN_SNR


def test_detect_stride_fallback_short():
    fz = _synthetic_fz(1.8, dur=3.0)
    f0, method, _ = ex.detect_stride(fz, 100.0, 3.0, v_nom=1.0)
    assert method == "fallback_speed"
    assert abs(f0 - 1.94) < 1e-6                        # speed-fallback line at v=1.0


def test_detect_stride_fallback_line_endpoints():
    for v, want in [(0.4, 1.61), (1.0, 1.94), (0.7, 1.775)]:
        f0, _, _ = ex.detect_stride(np.zeros((10, 4)), 100.0, 1.0, v_nom=v)
        assert abs(f0 - want) < 1e-9


def test_gait_notch_removes_harmonic_keeps_low_freq():
    fs, f0 = 200.0, 1.8
    t = np.arange(0, 30, 1 / fs)
    slow = np.sin(2 * np.pi * 0.3 * t)                  # "terrain" content
    gait = np.sin(2 * np.pi * 2 * f0 * t)               # trot bounce at 2x stride
    out = ex.gait_notch_multi(slow + gait, fs, f0)
    # gait component attenuated hard, slow component intact
    resid_gait = np.abs(np.fft.rfft(out - slow))[int(2 * f0 * 30)]
    assert np.std(out - slow) < 0.25 * np.std(gait)
    assert np.corrcoef(out, slow)[0, 1] > 0.97
    assert resid_gait < 0.2 * len(t) / 2


# --- trav label --------------------------------------------------------------

def _flat_odom(dur=10.0, fs=50.0):
    ts = np.arange(0, dur, 1 / fs)
    return ts, np.zeros_like(ts), np.zeros_like(ts)


def test_trav_label_rollover():
    od_ts, pitch, roll = _flat_odom()
    roll = roll.copy()
    roll[od_ts > 8.0] = np.deg2rad(70)                  # fall at t=8
    v = np.full_like(od_ts, 0.5)
    cm_ts = od_ts
    cm = np.column_stack([np.full_like(od_ts, 0.5), np.zeros_like(od_ts)])
    tilt = np.deg2rad(60)
    assert ex._trav_label(7.5, od_ts, pitch, roll, v, cm_ts, cm, tilt) == 1
    assert ex._trav_label(5.0, od_ts, pitch, roll, v, cm_ts, cm, tilt) == 0


def test_trav_label_stuck_rule():
    od_ts, pitch, roll = _flat_odom()
    v = np.full_like(od_ts, 0.01)                       # stalled
    cm_ts = od_ts
    cm = np.column_stack([np.full_like(od_ts, 0.6), np.zeros_like(od_ts)])
    tilt = np.deg2rad(60)
    assert ex._trav_label(4.0, od_ts, pitch, roll, v, cm_ts, cm, tilt) == 1
    v_ok = np.full_like(od_ts, 0.5)                     # tracking fine
    assert ex._trav_label(4.0, od_ts, pitch, roll, v_ok, cm_ts, cm, tilt) == 0


# --- stance-gated window means ----------------------------------------------

def test_stance_mean_gating():
    speed = np.array([1.0, 2.0, 3.0, 4.0])
    stance = np.array([[True], [False], [True], [False]])[:, 0]
    stance = np.column_stack([stance])                  # (4,1) -> use flat variant
    assert ex._stance_mean(speed, stance[:, 0], 0, 4) == pytest.approx(2.0)
    assert np.isnan(ex._stance_mean(speed, np.zeros(4, bool), 0, 4))
