"""Per-topic message extraction from TartanDrive 1.0 bags.

Companion to `extract.py`, which handles TD2. **This module deliberately returns
the exact same dict — same keys, same dataclasses, same conventions — so that
`features.py`, `cache.py`, `io_utils.py` and `build.py` run over TD1 unchanged.**
The label formulas are then literally shared with TD2 rather than reimplemented,
which is the point of the symmetric-domain experiment.

Everything TD1-specific is therefore a *translation* performed here:

| TD2 canonical key | TD1 source | translation |
|---|---|---|
| `/integrated_to_init` | `/odometry/filtered_odom` @ 50 Hz | axis remap to REP-103, twist rotated world→body, poses rebased to the run start |
| `/novatel/imu/data` | `/multisense/imu/imu_data` @ ~200-400 Hz | stamp sort, mount rotation into the body frame, REP-103 remap |
| `/span/status` | synthesised from the odometry quaternion | exact (matches `/span/status` where a bag carries both) |
| `/controls`, `/wheel_rpm`, `/shock_pos`, `/ros_talon/steering_angle` | same topics | direct |
| `/traversability_cost` | **absent in TD1** | empty (no traversability head is trained on TD1) |
| anchor images | `/multisense/left/image_rect_color` @ 20 Hz | every 2nd frame → the 10 Hz TD2 anchor stride |

Three conventions are load-bearing and were all established empirically by
probing TD1 bags, not assumed:

1. **TD1 body axes are forward `+y`, right `+x`, up `+z`** — not REP-103.
   Every pose, twist and IMU vector is remapped here so downstream code sees
   REP-103 (`+x` forward, `+y` left, `+z` up), which is what TD2 supplies.
2. **The odometry twist is in the world frame**, unlike TD2's body-frame twist
   so it is rotated by `Rᵀ` before being written.
3. **The IMU sits at an arbitrary mount rotation** (`td1_imu_mount.py`); gravity
   lies on `-x` in its raw frame, so the raw `linear_acceleration.z` is a
   horizontal axis and would silently corrupt `roughness`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as Rot

from .extract import (
    ANCHOR_TOPIC,
    Controls,
    FourWheel,
    ImageStream,
    Imu,
    PoseTwist,
    Scalar,
    Span,
)
from .td1_imu_mount import MOUNT_ROTATION, quat_to_matrix

TD1_ANCHOR_TOPIC = "/multisense/left/image_rect_color"
TD1_ODOM_TOPIC = "/odometry/filtered_odom"
TD1_IMU_TOPIC = "/multisense/imu/imu_data"

TD1_TOPICS = {
    TD1_ANCHOR_TOPIC,
    TD1_ODOM_TOPIC,
    TD1_IMU_TOPIC,
    "/controls",
    "/wheel_rpm",
    "/shock_pos",
    "/ros_talon/steering_angle",
}

ANCHOR_SUBSAMPLE = 2        # TD1 cameras run at 20 Hz; the anchor stride is 10 Hz

# TD1 body (fwd +y, right +x, up +z) → REP-103 (fwd +x, left +y, up +z).
# Columns are the REP-103 axes expressed in TD1 body coordinates, so
# `R_rep103 = R_td1 @ BODY_REMAP` and `v_rep103 = BODY_REMAPᵀ @ v_td1`.
BODY_REMAP = np.array([
    [0.0, -1.0, 0.0],       # REP-103 +x (forward) is TD1 +y
    [1.0, 0.0, 0.0],        # REP-103 +y (left)    is TD1 -x
    [0.0, 0.0, 1.0],        # REP-103 +z (up)      is TD1 +z
], dtype=np.float64)


def _header_ts_ns(msg) -> int | None:
    try:
        stamp = msg.header.stamp
    except AttributeError:
        return None
    ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return ns if ns > 0 else None


def _sorted(ts: list, *arrays):
    ts = np.asarray(ts, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    return (ts[order],) + tuple(np.asarray(a)[order] for a in arrays)


def attitude_deg_from_R(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SPAN-equivalent pitch / roll / azimuth from the TD1 odometry rotation.

    Verified against `/span/status` to 0.0000 deg on every probe bag that carries
    both.
    """
    pitch = np.degrees(np.arcsin(np.clip(R[:, 2, 1], -1.0, 1.0)))
    roll = -np.degrees(np.arctan2(R[:, 2, 0], R[:, 2, 2]))
    # Azimuth is carried for completeness only; nothing in the label path uses it.
    azimuth = np.degrees(np.arctan2(R[:, 0, 1], R[:, 1, 1])) % 360.0
    return pitch.astype(np.float32), roll.astype(np.float32), azimuth.astype(np.float32)


def extract_topics_td1(bag_paths: Iterable[Path]) -> dict:
    """Stream TD1 bag(s) once; return the TD2-canonical extraction dict."""
    bag_paths = [Path(p) for p in bag_paths]

    odom_ts, odom_pos, odom_quat, odom_vlin, odom_vang = [], [], [], [], []
    imu_ts, imu_acc, imu_gyr = [], [], []
    ctl_ts, ctl_thr, ctl_brk = [], [], []
    wheel = {"ts": [], "fl": [], "fr": [], "rl": [], "rr": []}
    shock = {"ts": [], "fl": [], "fr": [], "rl": [], "rr": []}
    steer_ts, steer_val = [], []
    img_ts, img_bytes = [], []
    img_hwe = [None, None, None]

    with AnyReader(bag_paths) as reader:
        conns = [c for c in reader.connections if c.topic in TD1_TOPICS]
        for conn, bag_ts, data in reader.messages(connections=conns):
            msg = reader.deserialize(data, conn.msgtype)
            ts = _header_ts_ns(msg)
            ts = ts if ts is not None else int(bag_ts)
            topic = conn.topic

            if topic == TD1_ODOM_TOPIC:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                vl, va = msg.twist.twist.linear, msg.twist.twist.angular
                odom_ts.append(ts)
                odom_pos.append([p.x, p.y, p.z])
                odom_quat.append([q.x, q.y, q.z, q.w])
                odom_vlin.append([vl.x, vl.y, vl.z])
                odom_vang.append([va.x, va.y, va.z])
            elif topic == TD1_IMU_TOPIC:
                a, g = msg.linear_acceleration, msg.angular_velocity
                imu_ts.append(ts)
                imu_acc.append([a.x, a.y, a.z])
                imu_gyr.append([g.x, g.y, g.z])
            elif topic == "/controls":
                ctl_ts.append(ts)
                ctl_thr.append(msg.throttle)
                ctl_brk.append(msg.brake)
            elif topic in ("/wheel_rpm", "/shock_pos"):
                buf = wheel if topic == "/wheel_rpm" else shock
                buf["ts"].append(ts)
                buf["fl"].append(msg.front_left)
                buf["fr"].append(msg.front_right)
                buf["rl"].append(msg.rear_left)
                buf["rr"].append(msg.rear_right)
            elif topic == "/ros_talon/steering_angle":
                steer_ts.append(ts)
                steer_val.append(msg.data)
            elif topic == TD1_ANCHOR_TOPIC:
                img_ts.append(ts)
                if img_hwe[0] is None:
                    img_hwe = [int(msg.height), int(msg.width), msg.encoding]
                img_bytes.append(np.asarray(msg.data, dtype=np.uint8))

    if not odom_ts:
        raise RuntimeError(f"{bag_paths[0].name}: no {TD1_ODOM_TOPIC}")
    if not img_ts:
        raise RuntimeError(f"{bag_paths[0].name}: no {TD1_ANCHOR_TOPIC}")

    out = {}

    # --- pose / twist -------------------------------------------------------
    ts, pos, quat, vlin, vang = _sorted(odom_ts, odom_pos, odom_quat,
                                        odom_vlin, odom_vang)
    R_td1 = quat_to_matrix(quat)                       # body(TD1) → world

    # Attitude must be read in the world-referenced frame, before the poses are
    # rebased onto the run start, because pitch/roll are angles against gravity.
    pitch_deg, roll_deg, azimuth_deg = attitude_deg_from_R(R_td1)

    R_rep = R_td1 @ BODY_REMAP                         # body(REP-103) → world
    # Twist is published in the world frame → rotate into body.
    v_body = np.einsum("nji,nj->ni", R_rep, vlin)
    w_body = np.einsum("nji,nj->ni", R_rep, vang)

    # Rebase onto the first pose so the frame matches TD2's per-run
    # `/integrated_to_init` (which starts at the identity by construction).
    R0t = R_rep[0].T
    pos_rebased = (pos - pos[0]) @ R0t.T
    R_rebased = np.einsum("ij,njk->nik", R0t, R_rep)

    out["/integrated_to_init"] = PoseTwist(
        ts_ns=ts,
        pos=pos_rebased.astype(np.float64),
        quat=Rot.from_matrix(R_rebased).as_quat().astype(np.float64),
        twist_linear=v_body.astype(np.float32),
        twist_angular=w_body.astype(np.float32),
    )

    # --- IMU ----------------------------------------------------------------
    # Sorting is mandatory, not cosmetic: 5.6 % of MultiSense header stamps arrive
    # out of order, and every windowed feature assumes ascending
    # timestamps.
    imu_to_rep = MOUNT_ROTATION @ BODY_REMAP
    i_ts, i_acc, i_gyr = _sorted(imu_ts, imu_acc, imu_gyr)
    out["/novatel/imu/data"] = Imu(
        ts_ns=i_ts,
        accel=(i_acc @ imu_to_rep).astype(np.float32),
        gyro=(i_gyr @ imu_to_rep).astype(np.float32),
    )

    # --- synthesised SPAN ---------------------------------------------------
    out["/span/status"] = Span(
        ts_ns=ts,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        azimuth_deg=azimuth_deg,
    )

    # --- Racepak ------------------------------------------------------------
    if ctl_ts:
        c_ts, thr, brk = _sorted(ctl_ts, ctl_thr, ctl_brk)
        out["/controls"] = Controls(ts_ns=c_ts,
                                    throttle=thr.astype(np.float32),
                                    brake=brk.astype(np.float32))
    else:
        out["/controls"] = Controls(ts_ns=np.zeros(0, np.int64),
                                    throttle=np.zeros(0, np.float32),
                                    brake=np.zeros(0, np.float32))

    for key, buf in (("/wheel_rpm", wheel), ("/shock_pos", shock)):
        if buf["ts"]:
            w_ts, fl, fr, rl, rr = _sorted(buf["ts"], buf["fl"], buf["fr"],
                                           buf["rl"], buf["rr"])
            out[key] = FourWheel(ts_ns=w_ts, fl=fl.astype(np.float32),
                                 fr=fr.astype(np.float32), rl=rl.astype(np.float32),
                                 rr=rr.astype(np.float32))
        else:
            z = np.zeros(0, np.float32)
            out[key] = FourWheel(ts_ns=np.zeros(0, np.int64), fl=z, fr=z, rl=z, rr=z)

    if steer_ts:
        s_ts, s_val = _sorted(steer_ts, steer_val)
        out["/ros_talon/steering_angle"] = Scalar(ts_ns=s_ts,
                                                  data=s_val.astype(np.float32))
    else:
        out["/ros_talon/steering_angle"] = Scalar(ts_ns=np.zeros(0, np.int64),
                                                  data=np.zeros(0, np.float32))

    # TD1 predates the TD2-era traversability pipeline, so there is no
    # teacher signal. Downstream this becomes an all-NaN column and the model is
    # configured with use_traversability_head=false.
    out["/traversability_cost"] = Scalar(ts_ns=np.zeros(0, np.int64),
                                         data=np.zeros(0, np.float32))

    # --- anchor images, 20 Hz → 10 Hz --------------------------------------
    a_ts = np.asarray(img_ts, dtype=np.int64)
    order = np.argsort(a_ts, kind="stable")[::ANCHOR_SUBSAMPLE]
    out[ANCHOR_TOPIC] = ImageStream(
        ts_ns=a_ts[order],
        height=img_hwe[0] or 0,
        width=img_hwe[1] or 0,
        encoding=img_hwe[2] or "",
        data_bytes=[img_bytes[i] for i in order],
    )
    return out
