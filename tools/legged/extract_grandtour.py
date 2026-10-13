"""GrandTour (real ANYmal-D) legged label extractor.

Produces per-mission labels.npz with the legged schema:
  proprio_window  (N, 50, 53)  causal [t-0.5s, t] @100Hz encoder input
  proprio_feature (N, 5)        centered [t-0.25s, t+0.25s] label
                                [roughness, angular_instability, slope_power, slip, cost_of_transport]
  pose (N,4,4), action (N,3), valid_mask (N,)

Stage-1 slip = cmd-tracking-error (no URDF). Stage-2 FK foot_slip is
only computed by the Isaac extractor (extract_isaac_anymal.py).
Anchors on hdr_front 10Hz timestamps. Timestamps are float64 UNIX seconds.

Usage:
  python tools/legged/extract_grandtour.py <mission_dir> [--out labels_legged.npz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

from _common import (
    EPS, T_WIN_N, FOOT_ORDER, gait_notch, quat_to_pitch_roll,
    causal_window, centered_slice, pose_mat, action_from_poses,
    cost_of_transport, print_summary,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mission_dir", type=Path)
    ap.add_argument("--out", default="labels_legged.npz")
    args = ap.parse_args()
    data = args.mission_dir / "data"

    def Z(topic):
        return zarr.open(str(data / topic), mode="r")

    # --- load topics ---
    cam = Z("hdr_front")
    anchor_ts = cam["timestamp"][:]                       # (N,) 10Hz anchor

    se = Z("anymal_state_state_estimator")
    se_ts = se["timestamp"][:]
    jpos, jvel, jtau = se["joint_positions"][:], se["joint_velocities"][:], se["joint_efforts"][:]
    pose_pos, pose_orien = se["pose_pos"][:], se["pose_orien"][:]
    twist_lin, twist_ang = se["twist_lin"][:], se["twist_ang"][:]
    footFz = np.stack([se[f"{f}_FOOT_wrench_force"][:, 2] for f in FOOT_ORDER], axis=1)  # (M,4)
    se_pitch, se_roll = quat_to_pitch_roll(pose_orien)
    se_speed = np.linalg.norm(twist_lin[:, :2], axis=1)
    se_cot = cost_of_transport(jtau, jvel, se_speed)

    imu = Z("anymal_imu")
    imu_ts = imu["timestamp"][:]
    lin_acc, ang_vel = imu["lin_acc"][:], imu["ang_vel"][:]
    # gait-notched copies used ONLY for the roughness / angular_instability labels
    # (raw lin_acc/ang_vel still feed the encoder window unchanged).
    imu_fs = 1.0 / np.median(np.diff(imu_ts))
    az_gf = gait_notch(lin_acc[:, 2], imu_fs)
    wr_gf = gait_notch(ang_vel[:, 0], imu_fs)
    wp_gf = gait_notch(ang_vel[:, 1], imu_fs)

    cmd = Z("anymal_command_twist")
    cmd_ts = cmd["timestamp"][:]
    cmd_v, cmd_w = cmd["linear"][:, 0], cmd["angular"][:, 2]

    N = len(anchor_ts)
    proprio_window = np.full((N, T_WIN_N, 53), np.nan, np.float32)
    proprio_feature = np.full((N, 5), np.nan, np.float32)
    pose = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    valid = np.ones(N, np.uint8)

    for i, t in enumerate(anchor_ts):
        # ---- encoder INPUT window (causal, 53-dim) ----
        w_imu_a = causal_window(imu_ts, lin_acc, t)          # (50,3)
        w_imu_g = causal_window(imu_ts, ang_vel, t)          # (50,3)
        w_vx = causal_window(se_ts, twist_lin, t)            # (50,3)
        w_wz = causal_window(se_ts, twist_ang, t)[:, 2:3]    # (50,1)
        w_pitch = causal_window(se_ts, se_pitch[:, None], t) # (50,1)
        w_roll = causal_window(se_ts, se_roll[:, None], t)
        w_cmdv = causal_window(cmd_ts, cmd_v[:, None], t)
        w_cmdw = causal_window(cmd_ts, cmd_w[:, None], t)
        w_jpos = causal_window(se_ts, jpos, t)
        w_jvel = causal_window(se_ts, jvel, t)
        w_jtau = causal_window(se_ts, jtau, t)
        w_fz = causal_window(se_ts, footFz, t)
        proprio_window[i] = np.concatenate(
            [w_imu_a, w_imu_g, w_vx[:, :2], w_wz, w_pitch, w_roll,
             w_cmdv, w_cmdw, w_jpos, w_jvel, w_jtau, w_fz], axis=1)

        # ---- decoder LABEL (centered window, 5-dim) ----
        lo_i, hi_i = centered_slice(imu_ts, t)
        lo_s, hi_s = centered_slice(se_ts, t)
        lo_c, hi_c = centered_slice(cmd_ts, t)
        if hi_i - lo_i < 2 or hi_s - lo_s < 2:
            valid[i] = 0
            continue
        az = az_gf[lo_i:hi_i]
        roughness = float(np.std(az - az.mean()))
        wr, wp = wr_gf[lo_i:hi_i], wp_gf[lo_i:hi_i]
        ang_inst = float(np.sqrt(wr.var() + wp.var()))
        v_w = se_speed[lo_s:hi_s]
        slope_power = float(np.mean(v_w * np.sin(se_pitch[lo_s:hi_s])))
        cot = float(np.mean(se_cot[lo_s:hi_s]))
        if hi_c - lo_c >= 1:
            cv = cmd_v[lo_c:hi_c]; vg = se_speed[lo_s:hi_s]
            slip = float(np.mean((cv.mean() - vg.mean()) / max(cv.mean(), EPS)))
        else:
            slip = np.nan
        proprio_feature[i] = [roughness, ang_inst, slope_power, slip, cot]

        # ---- pose ----
        si = np.clip(np.searchsorted(se_ts, t), 0, len(se_ts) - 1)
        pose[i] = pose_mat(pose_pos[si], pose_orien[si])
        if not np.isfinite(proprio_feature[i]).all():
            valid[i] = 0

    action = action_from_poses(pose, valid)
    out = args.mission_dir / args.out
    np.savez_compressed(
        out, anchor_idx=np.arange(N, dtype=np.int32), timestamp=anchor_ts,
        proprio_window=proprio_window, proprio_feature=proprio_feature,
        pose=pose, action=action, valid_mask=valid)
    print_summary(out, anchor_ts, proprio_feature, valid, se_speed)


if __name__ == "__main__":
    main()
