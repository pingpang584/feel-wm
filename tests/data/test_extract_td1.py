"""TD1 extraction — frame conventions and TD2-canonical output shape.

The conventions tested here are the load-bearing ones: TD1's body axes, world-frame
twist and arbitrarily mounted IMU all differ from TD2, and every one of those
differences is silent at training time (wrong encoder columns still converge) but
corrupts the TD1→TD2 comparison.

Pure-algebra tests always run. Tests needing a real bag are skipped unless the
probe bags are present locally (not shipped with the repo).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from feelwm.config import data_path

from tools.tartandrive.extract import (
    ANCHOR_TOPIC, Controls, FourWheel, ImageStream, Imu, PoseTwist, Scalar, Span,
)
from tools.tartandrive.extract_td1 import (
    BODY_REMAP, attitude_deg_from_R, extract_topics_td1,
)
from tools.tartandrive.td1_imu_mount import MOUNT_ROTATION, quat_to_matrix

PROBE_ROOT = data_path("tartandrive1_probe")
PROBE_BAGS = {
    # bag path -> whether it also carries /span/status
    "20210826_heightmaps_1/20210826_heightmaps_1/20210826_51.bag": False,
    "20210903_heightmaps_1/20210903_heightmaps_1/20210903_26.bag": True,
    "20210910_heightmaps_1/20210910_heightmaps_1/20210910_6.bag": True,
}


def _available():
    return [(PROBE_ROOT / p, has_span) for p, has_span in PROBE_BAGS.items()
            if (PROBE_ROOT / p).exists()]


needs_probe = pytest.mark.skipif(not _available(), reason="TD1 probe bags absent")


# --- pure algebra -----------------------------------------------------------

def test_body_remap_is_a_proper_rotation():
    assert np.allclose(BODY_REMAP @ BODY_REMAP.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(BODY_REMAP), 1.0)


def test_body_remap_sends_rep103_axes_to_td1_axes():
    """REP-103 forward/left/up must land on TD1's +y / -x / +z."""
    assert np.allclose(BODY_REMAP @ [1, 0, 0], [0, 1, 0])     # forward  -> TD1 +y
    assert np.allclose(BODY_REMAP @ [0, 1, 0], [-1, 0, 0])    # left     -> TD1 -x
    assert np.allclose(BODY_REMAP @ [0, 0, 1], [0, 0, 1])     # up       -> TD1 +z


def test_mount_rotation_is_a_proper_rotation():
    assert np.allclose(MOUNT_ROTATION @ MOUNT_ROTATION.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(MOUNT_ROTATION), 1.0, atol=1e-6)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@pytest.mark.parametrize("deg", [7.0, -4.0, 0.0])
def test_attitude_formula_reads_pure_pitch(deg):
    """In the TD1 convention forward is +y, so pitch is rotation about the right
    axis (+x). One angle at a time: composing two would make the expected values
    depend on an arbitrary Euler order rather than on the formula under test."""
    R = _rot_x(np.deg2rad(deg))[None]
    pitch, roll, _ = attitude_deg_from_R(R)
    assert pitch[0] == pytest.approx(deg, abs=1e-4)
    assert roll[0] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("deg", [7.0, -4.0, 0.0])
def test_attitude_formula_reads_pure_roll(deg):
    """Roll is rotation about the forward axis (+y)."""
    R = _rot_y(np.deg2rad(deg))[None]
    pitch, roll, _ = attitude_deg_from_R(R)
    assert roll[0] == pytest.approx(deg, abs=1e-4)
    assert pitch[0] == pytest.approx(0.0, abs=1e-4)


# --- against real bags ------------------------------------------------------

@needs_probe
def test_output_matches_the_td2_canonical_contract():
    bag, _ = _available()[0]
    data = extract_topics_td1([bag])
    expected = {
        "/integrated_to_init": PoseTwist,
        "/novatel/imu/data": Imu,
        "/span/status": Span,
        "/controls": Controls,
        "/wheel_rpm": FourWheel,
        "/shock_pos": FourWheel,
        "/ros_talon/steering_angle": Scalar,
        "/traversability_cost": Scalar,
        ANCHOR_TOPIC: ImageStream,
    }
    assert set(data) == set(expected)
    for key, cls in expected.items():
        assert isinstance(data[key], cls), key
    # TD1 has no traversability teacher.
    assert len(data["/traversability_cost"].ts_ns) == 0


@needs_probe
def test_timestamps_are_ascending_after_the_imu_stamp_sort():
    """5.6% of MultiSense stamps arrive out of order; every windowed feature
    assumes ascending time, so an unsorted stream silently mis-windows."""
    bag, _ = _available()[0]
    data = extract_topics_td1([bag])
    for key in ("/integrated_to_init", "/novatel/imu/data", "/span/status",
                "/controls", "/wheel_rpm", "/shock_pos"):
        ts = data[key].ts_ns
        assert np.all(np.diff(ts) >= 0), key


@needs_probe
def test_anchor_stride_is_10hz_not_the_native_20hz():
    bag, _ = _available()[0]
    data = extract_topics_td1([bag])
    dt_ms = np.diff(data[ANCHOR_TOPIC].ts_ns) / 1e6
    assert np.median(dt_ms) == pytest.approx(100.0, abs=5.0)
    assert len(data[ANCHOR_TOPIC].data_bytes) == len(data[ANCHOR_TOPIC].ts_ns)


@needs_probe
def test_imu_is_gravity_aligned_after_the_mount_rotation():
    """Raw TD1 IMU has gravity on -x; `roughness` reads axis 2, so a missing mount
    rotation would compute vertical vibration from a horizontal axis."""
    for bag, _ in _available():
        imu = extract_topics_td1([bag])["/novatel/imu/data"]
        mean_a = imu.accel.mean(axis=0)
        assert mean_a[2] == pytest.approx(9.8, abs=0.4), bag.name
        assert abs(mean_a[0]) < 1.5 and abs(mean_a[1]) < 1.5, bag.name


@needs_probe
def test_twist_is_rotated_into_the_body_frame():
    """TD1 publishes twist in the world frame. In the body frame forward motion
    must dominate: |v_x| >> |v_y| once the vehicle is moving."""
    for bag, _ in _available():
        pose = extract_topics_td1([bag])["/integrated_to_init"]
        v = pose.twist_linear
        moving = np.hypot(v[:, 0], v[:, 1]) > 2.0
        assert moving.sum() > 100, bag.name
        assert np.mean(np.abs(v[moving, 0])) > 5 * np.mean(np.abs(v[moving, 1])), bag.name


@needs_probe
def test_poses_are_rebased_onto_the_run_start():
    """TD2's /integrated_to_init starts at the identity; TD1's map frame does not."""
    bag, _ = _available()[0]
    pose = extract_topics_td1([bag])["/integrated_to_init"]
    assert np.allclose(pose.pos[0], 0.0, atol=1e-6)
    assert np.allclose(quat_to_matrix(pose.quat[:1])[0], np.eye(3), atol=1e-6)


@needs_probe
def test_synthesised_span_matches_the_real_span_topic():
    """The odometry quaternion reproduces /span/status exactly under the recovered
    convention recovered for TD1's odometry frame."""
    from rosbags.highlevel import AnyReader

    bags = [b for b, has_span in _available() if has_span]
    if not bags:
        pytest.skip("no probe bag carries /span/status")
    bag = bags[0]

    ts, pitch, roll = [], [], []
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == "/span/status"]
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            ts.append(m.header.stamp.sec * 10**9 + m.header.stamp.nanosec)
            pitch.append(m.pitch_ins_deg)
            roll.append(m.roll_ins_deg)
    ts = np.asarray(ts, dtype=np.int64)

    span = extract_topics_td1([bag])["/span/status"]
    idx = np.clip(np.searchsorted(ts, span.ts_ns), 0, ts.size - 1)
    ok = np.abs(ts[idx] - span.ts_ns) < 1_000_000        # within 1 ms
    assert ok.sum() > 100
    assert np.max(np.abs(span.pitch_deg[ok] - np.asarray(pitch)[idx][ok])) < 1e-3
    assert np.max(np.abs(span.roll_deg[ok] - np.asarray(roll)[idx][ok])) < 1e-3
