"""Shared legged label helpers for GrandTour + TartanGround extractors.

Both datasets share ONE schema:
  proprio_window  (N, 50, 53)  causal [t-0.5s, t] @100Hz encoder input
  proprio_feature (N, 5)        [roughness, angular_instability, slope_power, slip, cost_of_transport]
The two extractors differ only in I/O (zarr vs rosbag+npy); the formulas below are shared.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt, iirnotch
from scipy.spatial.transform import Rotation

# --- constants ---
ANYMAL_D_MASS_KG = 52.0   # nominal; CoT z-score absorbs scale
G = 9.81
EPS = 0.1
T_WIN_N = 50              # causal encoder window samples
ENC_DT = 0.01            # 100 Hz
LABEL_HALF = 0.25        # centered label half-window (s)
FOOT_ORDER = ["LF", "RF", "LH", "RH"]   # canonical array order
ACT_XY_MAX, ACT_YAW_MAX = 2.05, 0.17    # UniWM action sanity bounds
CHANNELS = ["roughness", "angular_instability", "slope_power", "slip", "cost_of_transport"]

# Gait-harmonic notch: ANYmal trot base-bounce at 3.1 Hz
# (= 2x the 1.49 Hz stride) + 6.3 Hz harmonic. Removes ~26% of raw roughness.
GAIT_NOTCH_HZ = (3.1, 6.3)
GAIT_NOTCH_Q = 4.0


def gait_notch(sig: np.ndarray, fs: float) -> np.ndarray:
    """filtfilt notch at the gait fundamental + harmonic over a full continuous signal."""
    out = sig.astype(np.float64)
    for f0 in GAIT_NOTCH_HZ:
        if f0 < fs / 2:
            b, a = iirnotch(f0, GAIT_NOTCH_Q, fs=fs)
            out = filtfilt(b, a, out)
    return out


def quat_to_pitch_roll(q_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,4) [x,y,z,w] -> (pitch, roll) in rad, gravity-referenced."""
    e = Rotation.from_quat(q_xyzw).as_euler("xyz")   # roll, pitch, yaw
    return e[:, 1], e[:, 0]


def causal_window(src_ts, src_val, t):
    """Hold-last resample onto 50-sample causal grid [t-0.49, t] @10ms. src_val (M,) or (M,k)."""
    grid = t - ENC_DT * np.arange(T_WIN_N - 1, -1, -1)
    idx = np.searchsorted(src_ts, grid, side="right") - 1
    idx = np.clip(idx, 0, len(src_ts) - 1)
    return src_val[idx]


def centered_slice(src_ts, t):
    lo = np.searchsorted(src_ts, t - LABEL_HALF, side="left")
    hi = np.searchsorted(src_ts, t + LABEL_HALF, side="right")
    return int(lo), int(hi)


def pose_mat(pos3: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = pos3
    return T


def action_from_poses(pose: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(N,4,4) body-to-world -> (N,3) (dx,dy,dyaw) body-frame delta to next anchor."""
    N = len(pose)
    action = np.zeros((N, 3), np.float32)
    for i in range(N - 1):
        dT = np.linalg.inv(pose[i]) @ pose[i + 1]
        dyaw = Rotation.from_matrix(dT[:3, :3]).as_euler("xyz")[2]
        action[i] = [dT[0, 3], dT[1, 3], dyaw]
        if abs(dT[0, 3]) > ACT_XY_MAX or abs(dT[1, 3]) > ACT_XY_MAX or abs(dyaw) > ACT_YAW_MAX:
            valid[i] = 0
    return action


def cost_of_transport(jtau: np.ndarray, jvel: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Per-sample CoT = sum_j|tau_j*qd_j| / (m g max(v,eps)). jtau/jvel (M,12), speed (M,)."""
    mech = np.abs(jtau * jvel).sum(axis=1)
    return mech / (ANYMAL_D_MASS_KG * G * np.maximum(speed, EPS))


def print_summary(name, anchor_ts, proprio_feature, valid, speed):
    N = len(anchor_ts)
    print(f"wrote {name}  N={N}  valid={int(valid.sum())} ({valid.mean()*100:.1f}%)")
    print(f"speed: mean={speed.mean():.2f} max={speed.max():.2f} m/s")
    vf = proprio_feature[valid == 1]
    for j, c in enumerate(CHANNELS):
        col = vf[:, j]
        col = col[np.isfinite(col)]
        print(f"  {c:18s} mean={col.mean():+.3f} std={col.std():.3f} "
              f"p5={np.percentile(col,5):+.3f} p95={np.percentile(col,95):+.3f}")
