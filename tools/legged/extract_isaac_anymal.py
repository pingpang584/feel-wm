"""Isaac-legged (ANYmal-C) label extractor: Isaac Sim episode bags -> legged labels.

Reads one bag per episode (layout `<episodes-root>/<Env>/<episode>.bag` + a `.meta.json`
sidecar, as written by the Isaac Sim collector) and writes one sequence directory per
episode:

    <out>/<Env>__<stem>__ep00/
        labels_legged.npz     # legged schema (below) — name matches the loader
                              #   convention (feelwm/data/legged_dataset.py)
        image/NNNNNN.jpg      # per-anchor camera frame (for DINO caching)
        meta.json             # episode meta subset + stride/notch + FK-vs-GT stats

Schema (shared legged schema; deltas for Isaac):
  - anchor          : 10 Hz grid over the odom span (15 Hz raw camera, nearest <=80 ms)
  - proprio_window  : (N,50,53) causal [t-0.5s,t] @100 Hz —
                      [acc3, gyro3, twist vx,vy,wz, pitch, roll, cmd_v, cmd_w,
                       jpos12, jvel12, jtau12, footFz4]; joints CANONICAL per-leg
                      order (bag /joint_states is Isaac tree order -> permuted here),
                      feet LF,RF,LH,RH
  - proprio_feature : (N,5) centered 0.5 s
                      [roughness, angular_instability, slope_power, slip, cost_of_transport]
                      * roughness / angular_instability are gait-notch-filtered with a
                        PER-EPISODE stride: detected from the foot-Fz fundamental
                        (Welch peak in the stride band), notches at k*f0, k=2..5, Q=4
                        (the measured stride, 1.61-1.94 Hz, is speed-dependent —
                        never the fixed 3.1/6.3 Hz). Fallback for short/weak
                        episodes: a speed->stride line, logged per episode.
                      * slip = STAGE-2 FK foot_slip: URDF FK foot
                        positions differentiated over time (dp/dt == J*qd), composed
                        with base twist, world-rotated, stance-gated by Fz.
  - side-by-side GT : foot_slip_gt from the PhysX GT foot-link world velocities —
                      same stance gate/windowing, so the FK-estimator noise floor
                      and the label's go/no-go check can be computed without
                      re-extraction. The GT topic reports the foot-link CoM velocity
                      (pilot-measured), so `fk_com_foot_speed` (FK at the CoM point)
                      is stored as the apples-to-apples FK side; `fk_foot_speed`
                      stays at the ball center (= the label's point). slip_track
                      (stage-1 cmd-tracking-error) is also kept for reference.
  - fz_peak / fz_rate_peak : (N,4) LOG-ONLY Fz impact-loading materials (conditional
                      label candidate — correlation-vs-roughness is left to analysis).
  - trav_gt         : forward-looking [t, t+1 s]: rollover (|roll| or |pitch| >
                      meta rollover_deg, legged 60°) or label-rule stuck (cmd_v>0.2,
                      v_ground<0.05, sustained >=1 s). The collector's progress-stuck
                      (30 s < 0.5 m) is a KNOWN divergent rule — episodes it ended may
                      have zero label positives; recorded as warnings, not decided here.
  - action          : (N,3) body-frame (dx,dy,dyaw) to next anchor (sim-generous yaw
                      clamp).

wall_timeout episodes are SKIPPED (marker meta only; bags renamed *.bag.wall_timeout).
Accounting is META-BASED (the .meta.json sidecars are authoritative) — the collector's
collection_log.jsonl can miss rows and must not be used to enumerate episodes.

Isaac legged is its own domain: sequences live under
${DATA_ROOT}/isaac/sim_anymal_isaac/sequences, with their own norm stats / feat cache.

Usage (full run):
  python3 tools/legged/extract_isaac_anymal.py --workers 8
Smoke:
  python3 tools/legged/extract_isaac_anymal.py --envs Deck --limit 1
Single bag:
  python3 tools/legged/extract_isaac_anymal.py --bag <path.bag>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.signal import filtfilt, iirnotch, welch
from scipy.spatial.transform import Rotation

from feelwm.config import data_path
from proprio_window import (
    ENC_DT,
    T_WIN_N,
    build_window,
    joint_permutation,
    quat_pitch_roll,
    window_sources,
)
from urdf_fk import FootFK

# --- constants ---------------------------------------------------------------
ANCHOR_HZ = 10.0
# ENC_DT (100 Hz window stride) and T_WIN_N (50 causal samples) come from
# proprio_window — the single causal-window definition shared by extraction.
LABEL_HALF = 0.25             # centered label half-window (s)
T_TRAV = 1.0                  # forward look-ahead for trav_gt (s)
CAM_TOL_S = 0.08
ACT_XY_MAX, ACT_YAW_MAX = 2.05, 0.6   # sim-generous yaw clamp
STUCK_CMD, STUCK_VGND, STUCK_SUSTAIN = 0.2, 0.05, 1.0
EPS = 0.1

# ANYmal-C in-sim mass: pilot-measured standing sumFz ~= 490 N -> 49.9 kg. Used for
# CoT + the stance gate; z-score absorbs scale (same principle as _common's ANYmal-D mass).
MASS_KG = 49.9
G = 9.81
TAU_STANCE_N = 0.15 * MASS_KG * G      # stance gate = 15% body weight (~73 N)

# Per-episode stride detection (foot-Fz fundamental). Fallback = pilot speed->stride
# line through the measured (v=0.4, 1.61 Hz) and (v=1.0, 1.94 Hz) endpoints.
STRIDE_BAND = (1.1, 2.4)               # Hz, generous around the measured 1.61-1.94
STRIDE_MIN_DUR_S = 8.0
STRIDE_MIN_SNR = 4.0                   # peak / median PSD in band
NOTCH_KS = (2, 3, 4, 5)
NOTCH_Q = 4.0

FOOT_TOPICS = ["lf", "rf", "lh", "rh"]                    # == FOOT_LINKS order
CHANNELS = ["roughness", "angular_instability", "slope_power", "slip", "cost_of_transport"]

_FK = None                             # lazy per-process FootFK


def _fk():
    global _FK
    if _FK is None:
        _FK = FootFK()
    return _FK


# --- bag IO ------------------------------------------------------------------

def _stamp(m):
    s = m.header.stamp
    return s.sec + s.nanosec * 1e-9


def read_topic(reader, ts, topic, extract, use_bag_time=False):
    c = [x for x in reader.connections if x.topic == topic]
    if not c:
        return np.zeros(0), np.zeros((0,))
    stamps, vals = [], []
    for conn, t, raw in reader.messages(connections=c[:1]):
        m = ts.deserialize_ros1(raw, c[0].msgtype)
        stamps.append(t * 1e-9 if use_bag_time else _stamp(m))
        vals.append(extract(m))
    return np.asarray(stamps, np.float64), np.asarray(vals)


def read_bag(bag: Path):
    """Isaac episode bag -> the signal dict."""
    ts = get_typestore(Stores.ROS1_NOETIC)
    with Reader(str(bag)) as r:
        imu_ts, imu = read_topic(r, ts, "/imu/data", lambda m: [
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z,
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        od_ts, od = read_topic(r, ts, "/ground_truth/odom", lambda m: [
            m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z,
            m.pose.pose.orientation.x, m.pose.pose.orientation.y,
            m.pose.pose.orientation.z, m.pose.pose.orientation.w,
            m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z,
            m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z])
        jt_c = [x for x in r.connections if x.topic == "/joint_states"][0]
        jt_ts, jt, joint_names = [], [], None
        for conn, t, raw in r.messages(connections=[jt_c]):
            m = ts.deserialize_ros1(raw, jt_c.msgtype)
            if joint_names is None:
                joint_names = list(m.name)
            jt_ts.append(_stamp(m))
            jt.append(np.concatenate([m.position, m.velocity, m.effort]))
        jt_ts, jt = np.asarray(jt_ts, np.float64), np.asarray(jt)
        cm_ts, cm = read_topic(r, ts, "/cmd_vel",
                               lambda m: [m.linear.x, m.angular.z], use_bag_time=True)
        fz_cols, gtv_cols = [], []
        for f in FOOT_TOPICS:
            t_, v_ = read_topic(r, ts, f"/contact_force_{f}_foot", lambda m: m.wrench.force.z)
            fz_cols.append((t_, v_))
            t_, v_ = read_topic(r, ts, f"/gt/foot_vel_{f}",
                                lambda m: [m.twist.linear.x, m.twist.linear.y, m.twist.linear.z])
            gtv_cols.append((t_, v_))
        cam_ts, cam_raw = _read_camera(r, ts)

    if joint_names is None or len(od_ts) < 20 or len(imu_ts) < 100:
        raise ValueError("bag missing core topics")
    perm = joint_permutation(joint_names)                     # tree -> canonical
    jpos = jt[:, 0:12][:, perm]
    jvel = jt[:, 12:24][:, perm]
    jtau = jt[:, 24:36][:, perm]
    fz = np.stack([np.interp(jt_ts, t_, v_) for t_, v_ in fz_cols], axis=1)      # (M,4)
    gtv = np.stack([np.stack([np.interp(jt_ts, t_, v_[:, k]) for k in range(3)], axis=1)
                    for t_, v_ in gtv_cols], axis=1)                             # (M,4,3)
    return dict(imu_ts=imu_ts, acc=imu[:, 0:3], gyro=imu[:, 3:6],
                od_ts=od_ts, pos=od[:, 0:3], quat=od[:, 3:7],
                lin=od[:, 7:10], ang=od[:, 10:13],
                jt_ts=jt_ts, jpos=jpos, jvel=jvel, jtau=jtau, fz=fz, gtv=gtv,
                cm_ts=cm_ts, cm=cm, cam_ts=cam_ts, cam_raw=cam_raw)


def _read_camera(reader, ts):
    """Raw rgb8 frames -> JPEG bytes (q90), stamp-deduped."""
    c = [x for x in reader.connections if x.topic.endswith("image_raw")]
    if not c:
        return np.zeros(0), []
    stamps, raws, seen = [], [], set()
    for conn, t, raw in reader.messages(connections=c[:1]):
        m = ts.deserialize_ros1(raw, c[0].msgtype)
        s = _stamp(m)
        key = round(s, 4)
        if key in seen:
            continue
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        if m.encoding.startswith("rgb"):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            continue
        seen.add(key)
        stamps.append(s)
        raws.append(buf.tobytes())
    return np.asarray(stamps, np.float64), raws


# --- per-episode stride + notch ---------------------------------------------

def detect_stride(fz, fs, dur_s, v_nom):
    """Foot-Fz fundamental via Welch PSD averaged over the 4 feet.

    Returns (f0_hz, method, snr). method: "fz_psd" or "fallback_speed"."""
    fallback = 1.61 + (1.94 - 1.61) / 0.6 * (float(v_nom) - 0.4)
    if dur_s < STRIDE_MIN_DUR_S or len(fz) < int(fs * STRIDE_MIN_DUR_S):
        return fallback, "fallback_speed", 0.0
    nper = min(len(fz), 1024)
    psd = None
    for k in range(4):
        f, p = welch(fz[:, k] - fz[:, k].mean(), fs=fs, nperseg=nper)
        psd = p if psd is None else psd + p
    band = (f >= STRIDE_BAND[0]) & (f <= STRIDE_BAND[1])
    wide = (f >= 0.8) & (f <= 3.0)
    if not band.any():
        return fallback, "fallback_speed", 0.0
    i = np.argmax(psd[band])
    f0 = float(f[band][i])
    snr = float(psd[band][i] / max(np.median(psd[wide]), 1e-12))
    if snr >= STRIDE_MIN_SNR:
        return f0, "fz_psd", snr
    # weak peak (irregular gait — stuck/timeout episodes): trust it only if it agrees
    # with the speed->stride prior; otherwise use the prior itself.
    if abs(f0 - fallback) <= 0.25:
        return f0, "fz_psd_lowsnr", snr
    return fallback, "fallback_speed", snr


def gait_notch_multi(sig, fs, f0):
    """filtfilt notches at k*f0 (k=2..5, Q=4) over the full continuous signal."""
    out = np.asarray(sig, np.float64)
    for k in NOTCH_KS:
        fk_hz = k * f0
        if fk_hz < fs / 2 * 0.95:
            b, a = iirnotch(fk_hz, NOTCH_Q, fs=fs)
            out = filtfilt(b, a, out)
    return out


# --- stage-2 foot slip (FK) + GT reference -----------------------------------

def foot_world_speeds(d):
    """Per-100Hz-sample world-frame planar foot speeds: FK at the BALL center (the
    slip label — contact-point proxy, correct zero), FK at the foot-link CoM (the
    apples-to-apples point for the GT comparison: the PhysX GT topic reports link
    CoM velocity — pilot-measured, see urdf_fk.FootFK), GT, and the stance mask.
    FK: v_w = R_wb (v_b + w x p + dp/dt); dp/dt == J*qd by the chain rule."""
    jt_ts = d["jt_ts"]
    fk = _fk()
    v_b = np.stack([np.interp(jt_ts, d["od_ts"], d["lin"][:, k]) for k in range(3)], axis=1)
    w_b = np.stack([np.interp(jt_ts, d["od_ts"], d["ang"][:, k]) for k in range(3)], axis=1)
    q = np.stack([np.interp(jt_ts, d["od_ts"], d["quat"][:, k]) for k in range(4)], axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = Rotation.from_quat(q).as_matrix()                      # (M,3,3) base->world
    T = fk.foot_transforms(d["jpos"])                          # (M,4,4,4)
    out = []
    for offsets in (None, fk.foot_coms):
        p = T[:, :, :3, 3]
        if offsets is not None:
            p = p + np.einsum("mfij,fj->mfi", T[:, :, :3, :3], offsets)
        dp = np.gradient(p, jt_ts, axis=0)
        v_foot_b = v_b[:, None, :] + np.cross(w_b[:, None, :], p) + dp
        v_fk = np.einsum("mij,mfj->mfi", R, v_foot_b)
        out.append(np.linalg.norm(v_fk[:, :, :2], axis=2))     # (M,4)
    fk_ball_xy, fk_com_xy = out
    gt_xy = np.linalg.norm(d["gtv"][:, :, :2], axis=2)
    # The stance gate is 15% of body weight, so it follows the corpus's own in-sim mass
    # (d["mass_kg"] when the reader provides one) rather than the Isaac constant.
    stance = d["fz"] > 0.15 * d.get("mass_kg", MASS_KG) * G
    return fk_ball_xy, fk_com_xy, gt_xy, stance


def _stance_mean(speed_xy, stance, lo, hi):
    """Mean over stance samples in [lo:hi); NaN if no stance contact."""
    s = stance[lo:hi]
    if not s.any():
        return np.nan
    return float(speed_xy[lo:hi][s].mean())


# --- geometry / labels -------------------------------------------------------

def pose_mat(pos3, quat_xyzw):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = pos3
    return T


def centered_slice(src_ts, t):
    lo = np.searchsorted(src_ts, t - LABEL_HALF, side="left")
    hi = np.searchsorted(src_ts, t + LABEL_HALF, side="right")
    return int(lo), int(hi)


def _trav_label(t, od_ts, pitch, roll, v_ground, cm_ts, cm, tilt_rad):
    lo = np.searchsorted(od_ts, t, side="left")
    hi = np.searchsorted(od_ts, t + T_TRAV, side="right")
    if hi - lo < 1:
        return 0
    if np.any(np.abs(pitch[lo:hi]) > tilt_rad) or np.any(np.abs(roll[lo:hi]) > tilt_rad):
        return 1
    if len(cm_ts):
        cmd_v = np.abs(np.interp(od_ts[lo:hi], cm_ts, cm[:, 0]))
        stuck = (cmd_v > STUCK_CMD) & (v_ground[lo:hi] < STUCK_VGND)
        if stuck.any():
            tt = od_ts[lo:hi][stuck]
            if tt.max() - tt.min() >= STUCK_SUSTAIN and stuck.sum() >= 3:
                return 1
    return 0


def _action_from_pose(pose, valid):
    N = len(pose)
    action = np.zeros((N, 3), np.float32)
    for i in range(N - 1):
        dT = np.linalg.inv(pose[i]) @ pose[i + 1]
        dyaw = Rotation.from_matrix(dT[:3, :3]).as_euler("xyz")[2]
        action[i] = [dT[0, 3], dT[1, 3], dyaw]
        if abs(dT[0, 3]) > ACT_XY_MAX or abs(dT[1, 3]) > ACT_XY_MAX or abs(dyaw) > ACT_YAW_MAX:
            valid[i] = 0
    return action


# --- main per-episode extraction ---------------------------------------------

def extract_episode(bag: Path, meta: dict, seq_dir: Path, save_images=True):
    d = read_bag(bag)
    imu_ts, od_ts, jt_ts = d["imu_ts"], d["od_ts"], d["jt_ts"]
    imu_fs = 1.0 / np.median(np.diff(imu_ts))
    jt_fs = 1.0 / np.median(np.diff(jt_ts))
    pitch, roll = quat_pitch_roll(d["quat"])
    v_ground = np.hypot(d["lin"][:, 0], d["lin"][:, 1])
    dur = float(od_ts[-1] - od_ts[0])

    # stride -> per-episode notch (labels only; encoder window sees raw signals)
    f0, method, snr = detect_stride(d["fz"], jt_fs, dur, meta.get("nominal_speed", 0.7))
    az_n = gait_notch_multi(d["acc"][:, 2], imu_fs, f0)
    gx_n = gait_notch_multi(d["gyro"][:, 0], imu_fs, f0)
    gy_n = gait_notch_multi(d["gyro"][:, 1], imu_fs, f0)

    # stage-2 slip machinery on the joint clock
    fk_xy, fk_com_xy, gt_xy, stance = foot_world_speeds(d)
    mass = float(d.get("mass_kg", MASS_KG))
    # Which FK point is apples-to-apples with the GT depends on what the GT topic
    # reports: Isaac/PhysX reports the foot-link CoM (the default); a reader whose GT is
    # at the foot ball centre sets d["gt_point"] = "ball". Comparing the wrong pair
    # would put a fixed geometric offset into the "noise floor".
    fk_ref = fk_xy if d.get("gt_point") == "ball" else fk_com_xy
    mech = np.abs(d["jtau"] * d["jvel"]).sum(axis=1)
    v_at_jt = np.interp(jt_ts, od_ts, v_ground)
    cot_s = mech / (mass * G * np.maximum(v_at_jt, EPS))
    dfz = np.abs(np.gradient(d["fz"], jt_ts, axis=0))

    rollover_deg = float(meta.get("termination_rules", {}).get("rollover_deg", 60.0))
    tilt_rad = np.deg2rad(rollover_deg)

    a0 = od_ts[0] + T_WIN_N * ENC_DT
    a1 = od_ts[-1] - (LABEL_HALF + 1.0 / ANCHOR_HZ)
    if a1 <= a0 or len(d["cam_ts"]) == 0:
        return None
    grid = np.arange(a0, a1, 1.0 / ANCHOR_HZ)
    if len(grid) < 5:
        return None

    N = len(grid)
    proprio_window = np.full((N, T_WIN_N, 53), np.nan, np.float32)
    proprio_feature = np.full((N, 5), np.nan, np.float32)
    trav_gt = np.zeros(N, np.uint8)
    pose = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    valid = np.ones(N, np.uint8)
    cam_idx_for = np.full(N, -1, np.int64)
    foot_slip_gt = np.full(N, np.nan, np.float32)
    slip_track = np.full(N, np.nan, np.float32)
    fk_foot_speed = np.full((N, 4), np.nan, np.float32)
    fk_com_foot_speed = np.full((N, 4), np.nan, np.float32)
    gt_foot_speed = np.full((N, 4), np.nan, np.float32)
    fz_peak = np.full((N, 4), np.nan, np.float32)
    fz_rate_peak = np.full((N, 4), np.nan, np.float32)

    # The window's column assembly lives in proprio_window.py (one definition of the
    # 53-column layout) — see that module's docstring.
    win_src = window_sources(
        imu_ts=imu_ts, acc=d["acc"], gyro=d["gyro"],
        od_ts=od_ts, lin=d["lin"], ang=d["ang"], pitch=pitch, roll=roll,
        cm_ts=d["cm_ts"], cmd=d["cm"],
        jt_ts=jt_ts, jpos=d["jpos"], jvel=d["jvel"], jtau=d["jtau"], fz=d["fz"])

    for i, t in enumerate(grid):
        ci = int(np.argmin(np.abs(d["cam_ts"] - t)))
        if abs(d["cam_ts"][ci] - t) > CAM_TOL_S:
            valid[i] = 0
            continue
        cam_idx_for[i] = ci

        proprio_window[i] = build_window(win_src, t)

        li, hi = centered_slice(imu_ts, t)
        lp, hp = centered_slice(od_ts, t)
        lj, hj = centered_slice(jt_ts, t)
        lc, hc = centered_slice(d["cm_ts"], t)
        if hi - li < 3 or hp - lp < 3 or hj - lj < 3:
            valid[i] = 0
            continue
        az = az_n[li:hi]
        rough = float(np.std(az - az.mean()))
        ang = float(np.sqrt(gx_n[li:hi].var() + gy_n[li:hi].var()))
        vg_c = v_ground[lp:hp]
        sp = float(np.mean(vg_c * np.sin(pitch[lp:hp])))
        cot = float(np.mean(cot_s[lj:hj]))
        slip_fk = _stance_mean(fk_xy, stance, lj, hj)
        foot_slip_gt[i] = _stance_mean(gt_xy, stance, lj, hj)
        for k in range(4):
            sk = stance[lj:hj, k]
            if sk.any():
                fk_foot_speed[i, k] = fk_xy[lj:hj, k][sk].mean()
                fk_com_foot_speed[i, k] = fk_com_xy[lj:hj, k][sk].mean()
                gt_foot_speed[i, k] = gt_xy[lj:hj, k][sk].mean()
        fz_peak[i] = d["fz"][lj:hj].max(axis=0)
        fz_rate_peak[i] = dfz[lj:hj].max(axis=0)
        if hc - lc >= 1:
            cv = d["cm"][lc:hc, 0].mean()
            slip_track[i] = float((cv - vg_c.mean()) / max(cv, EPS))
        proprio_feature[i] = [rough, ang, sp, slip_fk, cot]

        pi = int(np.clip(np.searchsorted(od_ts, t), 0, len(od_ts) - 1))
        pose[i] = pose_mat(d["pos"][pi], d["quat"][pi])
        trav_gt[i] = _trav_label(t, od_ts, pitch, roll, v_ground,
                                 d["cm_ts"], d["cm"], tilt_rad)
        if not np.isfinite(proprio_feature[i]).all():
            valid[i] = 0

    action = _action_from_pose(pose, valid)
    if valid.sum() < 5:
        return None

    seq_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        img_dir = seq_dir / "image"
        img_dir.mkdir(exist_ok=True)
        for i in range(N):
            if cam_idx_for[i] >= 0:
                (img_dir / f"{i:06d}.jpg").write_bytes(d["cam_raw"][cam_idx_for[i]])
    np.savez_compressed(
        seq_dir / "labels_legged.npz",
        anchor_idx=np.arange(N, dtype=np.int32), timestamp=grid.astype(np.float64),
        proprio_window=proprio_window, proprio_feature=proprio_feature,
        trav_gt=trav_gt, pose=pose, action=action, valid_mask=valid,
        foot_slip_gt=foot_slip_gt, slip_track=slip_track,
        fk_foot_speed=fk_foot_speed, fk_com_foot_speed=fk_com_foot_speed,
        gt_foot_speed=gt_foot_speed,
        fz_peak=fz_peak, fz_rate_peak=fz_rate_peak)

    # FK-vs-GT episode stats, evaluated at the CoM point (same point as GT — pilot:
    # swing corr 0.991 / ratio 0.997 there): stance residual = the noise floor;
    # swing corr/ratio = geometry guard (a wrong URDF breaks the swing match first).
    # stance_fk_ball_med checks the LABEL's zero point (~0.02 m/s when not slipping).
    st, sw = stance.ravel(), ~stance.ravel()
    # fk_b / fk_c always mean what they say (ball centre / link CoM); fk_r is whichever
    # of the two is apples-to-apples with THIS corpus's GT point, and it is the only one
    # the residual and the swing guard are computed from.
    fk_c, gt_r, fk_b = fk_com_xy.ravel(), gt_xy.ravel(), fk_xy.ravel()
    fk_r = fk_ref.ravel()
    stats = {
        "stride_hz": round(f0, 3), "stride_method": method, "stride_snr": round(snr, 1),
        "gt_point": d.get("gt_point", "com"), "mass_kg": round(mass, 3),
        "n_stance": int(st.sum()), "n_swing": int(sw.sum()),
        "stance_fk_ball_med": round(float(np.median(fk_b[st])), 4) if st.any() else None,
        "stance_fk_com_med": round(float(np.median(fk_c[st])), 4) if st.any() else None,
        "stance_gt_med": round(float(np.median(gt_r[st])), 4) if st.any() else None,
        "stance_resid_med": round(float(np.median(np.abs(fk_r[st] - gt_r[st]))), 4) if st.any() else None,
        "swing_corr": round(float(np.corrcoef(fk_r[sw], gt_r[sw])[0, 1]), 3) if sw.sum() > 10 else None,
        "swing_ratio_med": round(float(np.median(fk_r[sw] / np.maximum(gt_r[sw], 1e-3))), 3) if sw.sum() > 10 else None,
    }
    seq_meta = {
        "env": meta.get("env"), "episode": bag.stem,
        "termination": meta.get("termination"),
        "nominal_speed": meta.get("nominal_speed"),
        "fail_site": meta.get("fail_site"),
        "rollover_deg": rollover_deg,
        "joint_order": "canonical LF/RF/LH/RH x HAA/HFE/KFE (permuted from bag tree order)",
        "fk_gt": stats,
    }
    (seq_dir / "meta.json").write_text(json.dumps(seq_meta, indent=1))
    return {"anchors": N, "valid": int(valid.sum()), "trav_pos": int(trav_gt[valid == 1].sum()),
            **stats}


# --- batch driver ------------------------------------------------------------

SKIP_TERMINATIONS = ("wall_timeout",)
MIN_FREE_GB_START, MIN_FREE_GB_RUN = 40, 20


def find_episodes(root: Path, envs=None):
    out = []
    for meta_path in sorted(root.glob("*/*.meta.json")):
        env = meta_path.parent.name
        if envs and env not in envs:
            continue
        meta = json.loads(meta_path.read_text())
        meta["env"] = env
        bag = meta_path.with_name(meta_path.name.replace(".meta.json", ".bag"))
        out.append((env, bag, meta))
    return out


def process_one(args):
    env, bag, meta, out_root, save_images = args
    if meta["termination"] in SKIP_TERMINATIONS:
        return {"env": env, "bag": bag.name, "status": "skipped_wall_timeout"}
    if not bag.exists():
        return {"env": env, "bag": bag.name, "status": "missing_bag"}
    seq_dir = out_root / f"{env}__{bag.stem}__ep00"
    try:
        rec = extract_episode(bag, meta, seq_dir, save_images=save_images)
    except Exception as e:  # noqa: BLE001 — record and continue the batch
        return {"env": env, "bag": bag.name, "status": "error", "error": repr(e)}
    if rec is None:
        return {"env": env, "bag": bag.name, "status": "too_short"}
    out = {"env": env, "bag": bag.name, "status": "ok",
           "termination": meta["termination"], **rec}
    if meta["termination"] in ("rollover", "stuck") and rec["trav_pos"] == 0:
        out["warn"] = f"termination={meta['termination']} but trav_gt all-zero"
    elif meta["termination"] == "goal" and rec["trav_pos"] > 0:
        out["warn"] = f"termination=goal but {rec['trav_pos']} trav_gt positives"
    return out


def free_gb(path: Path):
    return shutil.disk_usage(path).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-root", type=Path,
                    default=data_path("isaac", "sim_anymal_isaac", "episodes"))
    ap.add_argument("--out-root", type=Path,
                    default=data_path("isaac", "sim_anymal_isaac", "sequences", "all"))
    ap.add_argument("--bag", type=Path, default=None, help="single-bag mode (needs .meta.json sibling)")
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    if args.bag:
        # NB: not Path.with_suffix — episode stems contain dots ("canonical__v0.7")
        meta_path = args.bag.with_name(args.bag.name[: -len(".bag")] + ".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.setdefault("termination", "unknown")
        meta["env"] = args.bag.parent.name
        eps = [(meta["env"], args.bag, meta)]
    else:
        eps = find_episodes(args.episodes_root, set(args.envs) if args.envs else None)
    if args.limit:
        eps = eps[:args.limit]
    if not eps:
        raise SystemExit("no episodes found")
    args.out_root.mkdir(parents=True, exist_ok=True)

    # disk guard (the dataset disk must not overflow). ~24 MB measured on
    # the 28.7 s pilot (mean ep is shorter) -> 815 eps <~ 20 GB; abort early rather
    # than fill the disk.
    if free_gb(args.out_root) < MIN_FREE_GB_START:
        raise SystemExit(f"only {free_gb(args.out_root):.0f} GB free — below the "
                         f"{MIN_FREE_GB_START} GB start margin, refusing to run")

    jobs = [(env, bag, meta, args.out_root, not args.no_images) for env, bag, meta in eps]
    results = []

    def _emit(i, rec):
        results.append(rec)
        line = f"[{i + 1}/{len(jobs)}] {rec['env']}/{rec['bag']}: {rec['status']}"
        if rec.get("stride_method") == "fallback_speed":
            line += " (stride fallback)"
        if rec.get("warn"):
            line += f" WARN {rec['warn']}"
        print(line, flush=True)
        if (i + 1) % 50 == 0 and free_gb(args.out_root) < MIN_FREE_GB_RUN:
            raise SystemExit(f"free space fell below {MIN_FREE_GB_RUN} GB — aborting")

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(process_one, jobs)):
                _emit(i, rec)
    else:
        for i, j in enumerate(jobs):
            _emit(i, process_one(j))

    ok = [r for r in results if r["status"] == "ok"]
    warns = [r for r in results if r.get("warn")]
    fallback = [r for r in ok if r.get("stride_method") == "fallback_speed"]
    # too_short = degenerate sub-1.4s episodes (a handful of tiny cov/fail goal paths
    # can't fit the 0.5 s causal window + label margins) — expected loss, not an error
    too_short = [r for r in results if r["status"] == "too_short"]
    errors = [r for r in results if r["status"] in ("error", "missing_bag")]
    sw_corr = [r["swing_corr"] for r in ok if r.get("swing_corr") is not None]
    summary = {
        "episodes_seen": len(results), "extracted": len(ok),
        "skipped_wall_timeout": sum(1 for r in results if r["status"] == "skipped_wall_timeout"),
        "too_short": [f"{r['env']}/{r['bag']}" for r in too_short],
        "errors": len(errors),
        "anchors_valid": sum(r["valid"] for r in ok),
        "trav_positives": sum(r["trav_pos"] for r in ok),
        "stride_fallback_eps": len(fallback),
        "swing_corr_min": min(sw_corr) if sw_corr else None,
        "swing_corr_med": float(np.median(sw_corr)) if sw_corr else None,
        "stance_resid_med_overall": float(np.median(
            [r["stance_resid_med"] for r in ok if r.get("stance_resid_med") is not None])) if ok else None,
        "warnings": warns, "error_records": errors,
    }
    out = args.out_root.parent / "extract_summary.json"
    out.write_text(json.dumps({"summary": summary, "episodes": results},
                              indent=1, ensure_ascii=False))
    print(f"\nextracted {len(ok)}/{len(results)} "
          f"(skipped {summary['skipped_wall_timeout']} wall_timeout, "
          f"{len(too_short)} too_short, {len(errors)} errors), "
          f"valid anchors {summary['anchors_valid']}, trav+ {summary['trav_positives']}, "
          f"stride-fallback {len(fallback)}, swing_corr med {summary['swing_corr_med']}, "
          f"warnings {len(warns)} -> {out}")


if __name__ == "__main__":
    main()
