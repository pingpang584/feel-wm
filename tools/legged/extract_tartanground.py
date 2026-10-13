"""TartanGround (sim ANYmal-D) legged label extractor.

TartanGround splits proprio across two sources on TWO clocks:
  - npy files (imu/*.npy): IMU acc/gyro + odometry, 0-based clock @100Hz, anchored by cam_time.
  - ROS1 bag (rosbags/*.bag): joint_states / contact_force / twist / pose / command twist, sim clock.
The two are aligned per-trajectory by base-SPEED cross-correlation (frame-invariant; yaw-rate fails
due to NED<->ENU sign flips). bag_time = npy_time + offset; everything is unified onto the npy clock.

Custom msgs (any_msgs/ExtendedJointState, ...) are registered from the bag-embedded definitions.
Output schema identical to the GrandTour extractor. Stage-1 slip = cmd-tracking-error.

Usage:
  python tools/legged/extract_tartanground.py <traj_dir>   # e.g. .../Gascola/Data_anymal/P2000
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from _common import (
    EPS, T_WIN_N, FOOT_ORDER, CHANNELS, gait_notch, quat_to_pitch_roll,
    causal_window, centered_slice, pose_mat, action_from_poses,
    cost_of_transport, print_summary,
)


def register_bag_types(reader, ts):
    """Register all custom msg types from bag-embedded concatenated definitions."""
    allt = {}
    for c in reader.connections:
        parts = re.split(r"^=+$", c.msgdef.data, flags=re.M)
        blocks = [(c.msgtype.replace("/msg/", "/"), parts[0])]
        for p in parts[1:]:
            mm = re.search(r"MSG:\s*(\S+)", p)
            if mm:
                blocks.append((mm.group(1), p[mm.end():]))
        for typ, body in blocks:
            try:
                allt.update(get_types_from_msg(body, typ))
            except Exception:
                pass
    ts.register(allt)


def read_topic(reader, ts, topic, extract):
    """Collect (stamps, values) for one topic. `extract(msg)` -> value row."""
    c = [x for x in reader.connections if x.topic == topic][0]
    stamps, vals = [], []
    for conn, t, raw in reader.messages(connections=[c]):
        m = ts.deserialize_ros1(raw, c.msgtype)
        s = m.header.stamp
        stamps.append(s.sec + s.nanosec * 1e-9)
        vals.append(extract(m))
    return np.array(stamps), np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir", type=Path)
    ap.add_argument("--out", default="labels_legged.npz")
    args = ap.parse_args()
    d = args.traj_dir

    # --- npy sources (0-based clock) ---
    imu_ts = np.load(d / "imu/imu_time.npy")
    acc = np.load(d / "imu/acc.npy")
    gyro = np.load(d / "imu/gyro.npy")
    vel_global = np.load(d / "imu/vel_global.npy")
    anchor_ts = np.load(d / "imu/cam_time.npy").astype(np.float64)
    imu_fs = 1.0 / np.median(np.diff(imu_ts))

    # --- bag sources (sim clock) ---
    bag = next((d / "rosbags").glob("*.bag"))
    ts = get_typestore(Stores.ROS1_NOETIC)
    with Reader(str(bag)) as r:
        register_bag_types(r, ts)
        jt_ts, jt = read_topic(r, ts, "/state_estimator/joint_states",
                               lambda m: np.concatenate([m.position, m.velocity, m.effort]))
        tw_ts, tw = read_topic(r, ts, "/state_estimator/twist",
                               lambda m: [m.twist.twist.linear.x, m.twist.twist.linear.y,
                                          m.twist.twist.angular.z])
        po_ts, po = read_topic(r, ts, "/state_estimator/pose_in_odom",
                               lambda m: [m.pose.pose.position.x, m.pose.pose.position.y,
                                          m.pose.pose.position.z, m.pose.pose.orientation.x,
                                          m.pose.pose.orientation.y, m.pose.pose.orientation.z,
                                          m.pose.pose.orientation.w])
        cm_ts, cm = read_topic(r, ts, "/path_planning_and_following/twist",
                               lambda m: [m.twist.linear.x, m.twist.angular.z])
        fz_cols = []
        for f in FOOT_ORDER:
            t_, v_ = read_topic(r, ts, f"/state_estimator/contact_force_{f.lower()}_foot",
                                lambda m: m.wrench.force.z)
            fz_cols.append((t_, v_))

    # --- clock alignment: bag_time = npy_time + offset (speed xcorr) ---
    bag_speed = np.linalg.norm(tw[:, :2], axis=1)
    npy_speed = np.linalg.norm(vel_global[:, :2], axis=1)
    fs = 50.0
    grid = np.arange(0, imu_ts[-1], 1 / fs)
    npy_g = np.interp(grid, imu_ts, npy_speed)
    best = (None, -1.0)
    for off in np.arange(tw_ts[0] - 5, tw_ts[0] + 25, 0.05):
        bg = np.interp(grid, tw_ts - off, bag_speed, left=np.nan, right=np.nan)
        mask = np.isfinite(bg)
        if mask.sum() < fs * 60:
            continue
        cc = np.corrcoef(npy_g[mask], bg[mask])[0, 1]
        if cc > best[1]:
            best = (off, cc)
    offset, corr = best
    print(f"clock align: bag_time = npy_time + {offset:.2f}  (speed corr={corr:+.3f})")
    if corr < 0.9:
        print("  WARNING: weak alignment (<0.90) — inspect this trajectory")

    # bring all bag stamps onto the npy clock
    jt_ts, tw_ts, po_ts, cm_ts = jt_ts - offset, tw_ts - offset, po_ts - offset, cm_ts - offset
    fz_cols = [(t_ - offset, v_) for t_, v_ in fz_cols]
    bag_speed_ts, bag_speed_v = tw_ts, bag_speed
    po_pitch, po_roll = quat_to_pitch_roll(po[:, 3:7])
    se_cot = cost_of_transport(jt[:, 24:36], jt[:, 12:24], np.interp(jt_ts, bag_speed_ts, bag_speed_v))

    # gait-notched IMU (label only)
    az_gf = gait_notch(acc[:, 2], imu_fs)
    wr_gf = gait_notch(gyro[:, 0], imu_fs)
    wp_gf = gait_notch(gyro[:, 1], imu_fs)

    # footFz on a common bag grid (each foot has its own stamps) -> interp to joint clock
    fz_grid = np.stack([np.interp(jt_ts, t_, v_) for t_, v_ in fz_cols], axis=1)  # (M,4)

    N = len(anchor_ts)
    proprio_window = np.full((N, T_WIN_N, 53), np.nan, np.float32)
    proprio_feature = np.full((N, 5), np.nan, np.float32)
    pose = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    valid = np.ones(N, np.uint8)

    for i, t in enumerate(anchor_ts):
        w_a = causal_window(imu_ts, acc, t)                 # (50,3)
        w_g = causal_window(imu_ts, gyro, t)                # (50,3)
        w_tw = causal_window(tw_ts, tw, t)                  # (50,3) vx,vy,wz
        w_pitch = causal_window(po_ts, po_pitch[:, None], t)
        w_roll = causal_window(po_ts, po_roll[:, None], t)
        w_cmd = causal_window(cm_ts, cm, t)                 # (50,2) v,w
        w_jpos = causal_window(jt_ts, jt[:, 0:12], t)
        w_jvel = causal_window(jt_ts, jt[:, 12:24], t)
        w_jtau = causal_window(jt_ts, jt[:, 24:36], t)
        w_fz = causal_window(jt_ts, fz_grid, t)
        proprio_window[i] = np.concatenate(
            [w_a, w_g, w_tw, w_pitch, w_roll, w_cmd[:, 0:1], w_cmd[:, 1:2],
             w_jpos, w_jvel, w_jtau, w_fz], axis=1)

        lo_i, hi_i = centered_slice(imu_ts, t)
        lo_s, hi_s = centered_slice(jt_ts, t)
        lo_p, hi_p = centered_slice(po_ts, t)
        lo_c, hi_c = centered_slice(cm_ts, t)
        if hi_i - lo_i < 2 or hi_s - lo_s < 2 or hi_p - lo_p < 1:
            valid[i] = 0
            continue
        az = az_gf[lo_i:hi_i]
        roughness = float(np.std(az - az.mean()))
        ang_inst = float(np.sqrt(wr_gf[lo_i:hi_i].var() + wp_gf[lo_i:hi_i].var()))
        vw = np.interp(np.linspace(t - 0.25, t + 0.25, 10), bag_speed_ts, bag_speed_v)
        pw = np.interp(np.linspace(t - 0.25, t + 0.25, 10), po_ts, po_pitch)
        slope_power = float(np.mean(vw * np.sin(pw)))
        cot = float(np.mean(se_cot[lo_s:hi_s]))
        if hi_c - lo_c >= 1:
            cv = cm[lo_c:hi_c, 0].mean()
            slip = float((cv - vw.mean()) / max(cv, EPS))
        else:
            slip = np.nan
        proprio_feature[i] = [roughness, ang_inst, slope_power, slip, cot]

        pi = np.clip(np.searchsorted(po_ts, t), 0, len(po_ts) - 1)
        pose[i] = pose_mat(po[pi, 0:3], po[pi, 3:7])
        if not np.isfinite(proprio_feature[i]).all():
            valid[i] = 0

    action = action_from_poses(pose, valid)
    out = d / args.out
    np.savez_compressed(
        out, anchor_idx=np.arange(N, dtype=np.int32), timestamp=anchor_ts,
        proprio_window=proprio_window, proprio_feature=proprio_feature,
        pose=pose, action=action, valid_mask=valid)
    print_summary(out, anchor_ts, proprio_feature, valid, npy_speed)


if __name__ == "__main__":
    main()
