"""NEGS-UGV (Gazebo) sim Husky label extractor: cycle bags -> per-episode labels.

Reads one ROS1 cycle bag (multiple episodes), segments it into episodes by
ground-truth-pose teleports (each path-replay reset jumps the robot), and writes
one sequence directory per episode:

    <out>/<bagstem>__ep{K}/
        labels.npz            # anchor_idx, timestamp, proprio_window (N,50,13),
                              #   proprio_feature (N,5), trav_gt (N,), pose (N,4,4),
                              #   action (N,3), valid_mask (N,)
        image/NNNNNN.jpg      # decoded camera frame per anchor (for DINO caching)

Schema:
  - anchor      : camera frames downsampled to a 10 Hz grid (contract is >=10 Hz;
                  raw stream is 15 Hz compressed JPEG). 10 Hz matches every existing
                  model convention (action dt, 100 Hz proprio window, H=8 ~= 0.8 s).
  - proprio_window (13-dim, causal [t-0.5s, t] @100Hz, T=50):
        0-2 imu accel, 3-5 imu gyro, 6-8 body twist vx,vy,wz,
        9 pitch, 10 roll, 11 cmd_v, 12 cmd_w
  - proprio_feature (5-dim, centered [t-0.25s, t+0.25s]):
        [roughness, slip_ratio, angular_instability, ke_rate, slope_power]
        (all 5 kept for sim; ke_rate is noise-free here, unlike on real logs)
  - trav_gt (uint8): forward-looking — 1 iff a failure event occurs in [t, t+T_TRAV]:
        rollover (|roll| or |pitch| > 45 deg), collision (bumper contact with a
        NON-ground body — see GROUND_MODELS), or
        stuck (|cmd_v|>0.2 and v_ground<0.05 sustained >= 1.0 s).
  - action (N,3): body-frame (dx, dy, dyaw) delta to the next anchor.

Usage (one bag):
  python tools/sim/extract_sim_husky.py <bag> --out-root <dir>
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# --- constants ---
R_WHEEL_M = 0.1651          # Husky A200 nominal (meta.json)
SLIP_EPS = 0.1
ANCHOR_HZ = 10.0
ENC_DT = 0.01               # 100 Hz proprio window stride
T_WIN_N = 50                # causal window samples (0.5 s)
LABEL_HALF = 0.25           # centered label half-window (s)
T_TRAV = 1.0                # forward look-ahead for trav_gt (s)
TILT_LIMIT_RAD = np.deg2rad(45.0)
STUCK_CMD = 0.2             # m/s commanded
STUCK_VGND = 0.05           # m/s realized
STUCK_SUSTAIN = 1.0         # s
EPISODE_JUMP_M = 2.0        # odom position jump => reset boundary
# trav_gt defines collision as "chassis contact with NON-ground". The world's
# terrain/scenery models must therefore be excluded from /bumper_states, otherwise the
# chassis grazing uneven ground is labelled trav_gt=1. Measured share of contact frames
# that are ground-ONLY: forest 41.2%, hill 30.4%, lake 23.3%, park 0% (park is flat, which
# is why earlier hill+park-only corpora never surfaced it). Ground model names differ per world:
# forest=terrain, lake=terreno_lago, hill=hill, park=parque/camino.
GROUND_MODELS = ("terrain", "terreno", "lago", "hill", "parque", "camino",
                 "ground", "path", "linea")
CAM_TOL_S = 0.08            # anchor<->camera nearest-neighbor tolerance
ACT_XY_MAX, ACT_YAW_MAX = 2.05, 0.6   # action sanity clamp (sim, generous yaw)

CHANNELS = ["roughness", "slip_ratio", "angular_instability", "ke_rate", "slope_power"]
WHEEL_NAMES = ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"]


# --- bag IO -----------------------------------------------------------------

def register_bag_types(reader, ts):
    """Register custom msg types (gazebo_msgs/ContactsState) from bag-embedded defs."""
    allt = {}
    for c in reader.connections:
        msgdef = c.msgdef if isinstance(c.msgdef, str) else c.msgdef.data
        parts = re.split(r"^=+$", msgdef, flags=re.M)
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


def _stamp(m) -> float:
    s = m.header.stamp
    return s.sec + s.nanosec * 1e-9


def _obstacle_contact(m):
    """1.0 iff the chassis touches a NON-ground body.

    gazebo_msgs/ContactState names both bodies as "model::link::collision"; the husky
    side can be either one. Ground/scenery contacts are dropped — on rough terrain the
    chassis grazes the ground constantly and would otherwise read as a collision.
    """
    for s in m.states:
        other = s.collision2_name.split("::")[0]
        if other.startswith("husky"):
            other = s.collision1_name.split("::")[0]
        if not other.startswith(GROUND_MODELS):
            return 1.0
    return 0.0


def read_topic(reader, ts, topic, extract, use_bag_time=False):
    """Collect (stamps, values). Header stamp by default; bag-record time for
    unstamped msgs (e.g. geometry_msgs/Twist on /cmd_vel)."""
    c = [x for x in reader.connections if x.topic == topic]
    if not c:
        return np.zeros(0), np.zeros(0)
    c = c[0]
    stamps, vals = [], []
    for conn, t, raw in reader.messages(connections=[c]):
        m = ts.deserialize_ros1(raw, c.msgtype)
        stamps.append(t * 1e-9 if use_bag_time else _stamp(m))
        vals.append(extract(m))
    return np.asarray(stamps, dtype=np.float64), np.asarray(vals)


# --- resampling helpers (mirror tools/tartandrive/features.py) ----------

def hold_last(src_ts, src_val, grid):
    if len(src_ts) == 0:
        return np.full((len(grid),) + src_val.shape[1:], np.nan, np.float32)
    idx = np.clip(np.searchsorted(src_ts, grid, side="right") - 1, 0, len(src_ts) - 1)
    return src_val[idx]


def causal_window(src_ts, src_val, t):
    grid = t - ENC_DT * np.arange(T_WIN_N - 1, -1, -1)
    return hold_last(src_ts, src_val, grid)


def centered_slice(src_ts, t):
    lo = np.searchsorted(src_ts, t - LABEL_HALF, side="left")
    hi = np.searchsorted(src_ts, t + LABEL_HALF, side="right")
    return int(lo), int(hi)


# --- 5-dim proprio features (formulas verbatim from features.py) ------------

def feat_roughness(az):
    return float(np.std(az - az.mean(), ddof=0)) if len(az) >= 3 else np.nan


def feat_angular(gx, gy):
    return float(np.sqrt(np.var(gx) + np.var(gy))) if len(gx) >= 3 else np.nan


def feat_slip(wheel_omega, v_ground):
    """sim slip: v_wheel = mean(omega)*r vs ground speed, symmetric denom, no rpm filter."""
    if len(wheel_omega) < 3:
        return np.nan
    v_wheel = R_WHEEL_M * wheel_omega.mean(axis=1)
    denom = np.maximum.reduce([v_wheel, v_ground, np.full_like(v_wheel, SLIP_EPS)])
    return float(np.mean((v_wheel - v_ground) / denom))


def feat_ke_rate(v, ts):
    if len(v) < 7:
        return np.nan
    wl = min(len(v), 5)
    if wl % 2 == 0:
        wl -= 1
    v_smooth = savgol_filter(v, wl, min(2, wl - 1)) if wl >= 3 else v
    dv = np.gradient(v_smooth)
    dt = np.diff(ts).astype(np.float64)
    dt = np.concatenate([[dt[0]], dt]) if len(dt) else np.ones_like(v)
    return float(np.mean(v * (dv / np.maximum(dt, 1e-6))))


def feat_slope_power(v, pitch):
    return float(np.mean(v * np.sin(pitch))) if len(v) >= 3 else np.nan


# --- geometry ---------------------------------------------------------------

def quat_pitch_roll(q_xyzw):
    e = Rotation.from_quat(q_xyzw).as_euler("xyz")   # roll, pitch, yaw
    return e[:, 1], e[:, 0]


def pose_mat(pos3, quat_xyzw):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = pos3
    return T


def segment_episodes(odom_ts, odom_pos):
    """Split into [start_ts, end_ts) episodes at odom position teleports."""
    if len(odom_ts) == 0:
        return []
    jumps = np.linalg.norm(np.diff(odom_pos[:, :2], axis=0), axis=1)
    cuts = np.where(jumps > EPISODE_JUMP_M)[0] + 1
    bounds = [0, *cuts.tolist(), len(odom_ts)]
    eps = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 10:
            continue
        eps.append((float(odom_ts[a]), float(odom_ts[b - 1])))
    return eps


# --- main extraction --------------------------------------------------------

def load_prune_decision(path, exempt=()):
    """Optional redundancy prune (a per-loop keep-list JSON from the data collector). Returns
    {seed_tag: (sorted_t_starts, keep_flags)} so an episode can be assigned to its loop by
    (bag-stem seed_tag, largest loop t_start <= episode_t0) and kept/dropped. None = no prune.
    exempt: seed_tags to leave OUT of the table (i.e. never pruned) — used to keep the
    held-out val batch (n1990) identical to the unpruned val set, so pruned-vs-unpruned
    val metrics stay directly comparable."""
    if not path:
        return None
    dec = json.load(open(path))
    per = {}
    for r in sorted(dec["loops"], key=lambda r: (r["seed_tag"], r["t_start"])):
        if r["seed_tag"] in exempt:
            continue
        per.setdefault(r["seed_tag"], ([], []))
        per[r["seed_tag"]][0].append(r["t_start"])
        per[r["seed_tag"]][1].append(bool(r["keep"]))
    return per


def _episode_kept(decision, seed_tag, t0):
    if decision is None:
        return True
    entry = decision.get(seed_tag)
    if entry is None:
        return True                       # seed_tag not in decision -> keep (don't silently drop)
    return _kept_by_loops(entry, t0)


def _kept_by_loops(entry, t0):
    """(sorted loop t_starts, keep flags) + episode start -> keep?  Assigns the episode to
    the last loop that started at or before it (same rule as the hill-cycle prune)."""
    starts, keep = entry
    i = bisect.bisect_right(starts, t0) - 1
    return keep[i] if 0 <= i < len(keep) else True


# --- forest col/nm episode filter (raw/forest_combined_keep.json) -------------
# Separate from prune_decision: different schema (round/kind/loop, no seed_tag, no t_start)
# and a different purpose — it drops BAD forest episodes (rollover from a bad spawn-z,
# vegetation never touched, clean run < 1.35 s, near-miss that actually touched), not
# redundant-but-valid traverses. The keep file has no timestamps, so loop -> t_start comes
# from the collector's `forest_{col,nm}_rollouts.jsonl` sitting next to the bags; from there
# the assignment is the same bisect as the prune path.
#   round1 = processed/forest_v1_colnm_archive/   round2 = processed/
# NOTE the `bag_dir` field inside the keep file points at STALE paths — ignore it and use
# the round token, which is what actually identifies the collection round.
FOREST_COLNM_RE = re.compile(r"^forest_(col|nm)_\d+$")
FOREST_ARCHIVE_DIR = "forest_v1_colnm_archive"


def forest_keep_entry(keep_path, bag: Path):
    """(starts, keeps) for this bag's (round, kind) group, or None if the filter doesn't
    apply (not a forest col/nm bag, or no keep file given). Pilots (forest_colpilot_*,
    forest_bushcolpilot_*) are NOT in the keep file and are deliberately left unfiltered."""
    if not keep_path:
        return None
    m = FOREST_COLNM_RE.match(bag.stem)
    if not m:
        return None
    kind = m.group(1)
    rnd = "round1" if bag.parent.name == FOREST_ARCHIVE_DIR else "round2"

    rollouts = bag.parent / f"forest_{kind}_rollouts.jsonl"
    if not rollouts.exists():
        raise FileNotFoundError(
            f"--forest-keep needs {rollouts} to map loop -> t_start for {bag.name}")
    loop_t = {}
    for line in rollouts.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            loop_t[int(rec["loop"])] = float(rec["t_start_sim"])

    decisions = json.load(open(keep_path))
    rows = []
    for key, v in decisions.items():
        parts = key.split("/")
        if len(parts) != 3 or parts[0] != rnd or parts[1] != kind:
            continue
        loop = int(v["loop"])
        if loop not in loop_t:
            raise KeyError(f"{key}: loop {loop} missing from {rollouts.name}")
        rows.append((loop_t[loop], bool(v["keep"])))
    if not rows:
        raise ValueError(f"no {rnd}/{kind} entries in {keep_path} (schema changed?)")
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows]


def extract_bag(bag: Path, out_root: Path, save_images: bool = True, prune_decision=None,
                forest_keep=None, name_prefix: str = ""):
    ts = get_typestore(Stores.ROS1_NOETIC)
    with Reader(str(bag)) as r:
        register_bag_types(r, ts)
        imu_ts, imu = read_topic(r, ts, "/imu/data", lambda m: [
            m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z,
            m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        od_ts, od = read_topic(r, ts, "/ground_truth/odom", lambda m: [
            m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z,
            m.pose.pose.orientation.x, m.pose.pose.orientation.y,
            m.pose.pose.orientation.z, m.pose.pose.orientation.w,
            m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z])
        jt_ts, jt = read_topic(r, ts, "/joint_states", lambda m: _wheel_vel(m))
        cm_ts, cm = read_topic(r, ts, "/cmd_vel", lambda m: [m.linear.x, m.angular.z],
                               use_bag_time=True)
        bp_ts, bp = read_topic(r, ts, "/bumper_states", _obstacle_contact)
        cam_ts, cam_raw = _read_camera(r, ts)

    pos = od[:, 0:3]
    quat = od[:, 3:7]
    twist = od[:, 7:10]                      # vx, vy, wz (body frame)
    pitch, roll = quat_pitch_roll(quat)
    v_ground = np.hypot(twist[:, 0], twist[:, 1])
    az_imu, gx_imu, gy_imu = imu[:, 2], imu[:, 3], imu[:, 4]

    episodes = segment_episodes(od_ts, pos)
    stem = bag.stem
    # seed_tag = the n<seed> token; only used for the hill-cycle prune_decision lookup.
    # unified names: "hill_run_n1000_c0" (cycles) / "hill_jt_005" / "park_jt_00" / "hill_cov_rev0".
    seed_tag = next((p for p in stem.split("_") if re.fullmatch(r"n\d+", p)),
                    stem.split("_")[0])               # non-cycle bags -> no n-token -> not in decision (kept)
    forest_entry = forest_keep_entry(forest_keep, bag)
    if forest_entry is not None and episodes and episodes[0][0] < forest_entry[0][0] - 1.0:
        # tripwire: episode stamps and the collector's t_start_sim must share the sim clock
        print(f"  [warn] {bag.name}: first episode t0={episodes[0][0]:.1f} precedes the first "
              f"loop t_start={forest_entry[0][0]:.1f} — clock mismatch? filter may mis-assign")
    n_written = 0
    n_pruned = 0
    for k, (t0, t1) in enumerate(episodes):
        if not _episode_kept(prune_decision, seed_tag, t0):
            n_pruned += 1                             # redundant OK-traverse -> skip (keeps ep index k stable)
            continue
        if forest_entry is not None and not _kept_by_loops(forest_entry, t0):
            n_pruned += 1                             # bad forest episode (rollover / veg-miss / too short)
            continue
        seq_dir = out_root / f"{name_prefix}{stem}__ep{k:02d}"
        n = _extract_episode(
            seq_dir, t0, t1, cam_ts, cam_raw,
            imu_ts, az_imu, gx_imu, gy_imu, imu,
            od_ts, pos, quat, twist, pitch, roll, v_ground,
            jt_ts, jt, cm_ts, cm, bp_ts, bp, save_images)
        if n > 0:
            n_written += 1
    return n_written, len(episodes), n_pruned


def _wheel_vel(m):
    name_to_v = dict(zip(m.name, m.velocity))
    return [name_to_v.get(w, np.nan) for w in WHEEL_NAMES]


def _read_camera(reader, ts):
    """Camera frames as JPEG bytes. Gazebo bags carry image_raw/compressed; Isaac bags
    (sim_husky_isaac) record raw sensor_msgs/Image rgb8 on image_raw — encode those to
    JPEG here so every downstream consumer sees one format."""
    comp = [x for x in reader.connections if x.topic.endswith("image_raw/compressed")]
    rawc = [x for x in reader.connections if x.topic.endswith("image_raw")]
    if not comp and not rawc:
        return np.zeros(0), []
    c = (comp or rawc)[0]
    is_raw = not comp
    stamps, raws, seen = [], [], set()
    for conn, t, raw in reader.messages(connections=[c]):
        m = ts.deserialize_ros1(raw, c.msgtype)
        s = _stamp(m)
        key = round(s, 4)
        if key in seen:                       # dedup sim-pause frame bursts (ANALYSIS.md)
            continue
        seen.add(key)
        stamps.append(s)
        if is_raw:
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
            if m.encoding.startswith("rgb"):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                stamps.pop(); seen.discard(key)
                continue
            raws.append(buf.tobytes())
        else:
            raws.append(bytes(m.data))
    return np.asarray(stamps, dtype=np.float64), raws


def _extract_episode(seq_dir, t0, t1, cam_ts, cam_raw,
                     imu_ts, az_imu, gx_imu, gy_imu, imu,
                     od_ts, pos, quat, twist, pitch, roll, v_ground,
                     jt_ts, jt, cm_ts, cm, bp_ts, bp, save_images):
    # anchors: 10 Hz grid. Leave 0.5 s causal history at the head; at the tail
    # keep only LABEL_HALF + one step (centered label + next-anchor action) so the
    # pre-failure anchors are NOT trimmed — failures happen AT the episode end, and
    # _trav_label clips its forward look-ahead to available data near the tail.
    a0 = t0 + T_WIN_N * ENC_DT
    a1 = t1 - (LABEL_HALF + 1.0 / ANCHOR_HZ)
    if a1 <= a0:
        return 0
    grid = np.arange(a0, a1, 1.0 / ANCHOR_HZ)
    if len(grid) < 5 or len(cam_ts) == 0:
        return 0

    N = len(grid)
    proprio_window = np.full((N, T_WIN_N, 13), np.nan, np.float32)
    proprio_feature = np.full((N, 5), np.nan, np.float32)
    trav_gt = np.zeros(N, np.uint8)
    pose = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    valid = np.ones(N, np.uint8)
    cam_idx_for = np.full(N, -1, np.int64)

    for i, t in enumerate(grid):
        ci = int(np.argmin(np.abs(cam_ts - t)))
        if abs(cam_ts[ci] - t) > CAM_TOL_S:
            valid[i] = 0
            continue
        cam_idx_for[i] = ci

        # 13-dim causal window
        w_imu = causal_window(imu_ts, imu, t)                 # (50,6)
        w_tw = causal_window(od_ts, twist, t)                 # (50,3)
        w_pitch = causal_window(od_ts, pitch[:, None], t)
        w_roll = causal_window(od_ts, roll[:, None], t)
        w_cmd = causal_window(cm_ts, cm, t) if len(cm_ts) else np.full((T_WIN_N, 2), np.nan, np.float32)
        proprio_window[i] = np.concatenate(
            [w_imu, w_tw, w_pitch, w_roll, w_cmd], axis=1)

        # 5-dim centered features
        li, hi = centered_slice(imu_ts, t)
        lp, hp = centered_slice(od_ts, t)
        lj, hj = centered_slice(jt_ts, t)
        if hi - li < 3 or hp - lp < 3:
            valid[i] = 0
            continue
        rough = feat_roughness(az_imu[li:hi])
        ang = feat_angular(gx_imu[li:hi], gy_imu[li:hi])
        vg_c = v_ground[lp:hp]
        if hj - lj >= 3:
            vg_at_j = np.interp(jt_ts[lj:hj], od_ts, v_ground)
            slip = feat_slip(jt[lj:hj], vg_at_j)
        else:
            slip = np.nan
        ke = feat_ke_rate(vg_c, od_ts[lp:hp])
        sp = feat_slope_power(vg_c, pitch[lp:hp])
        proprio_feature[i] = [rough, slip, ang, ke, sp]

        # pose (nearest odom)
        pi = int(np.clip(np.searchsorted(od_ts, t), 0, len(od_ts) - 1))
        pose[i] = pose_mat(pos[pi], quat[pi])

        # forward-looking trav_gt over [t, t+T_TRAV]
        trav_gt[i] = _trav_label(t, od_ts, pitch, roll, v_ground, cm_ts, cm, bp_ts, bp)

        if not np.isfinite(proprio_feature[i]).all():
            valid[i] = 0

    action = _action_from_pose(pose, valid)

    keep = valid.sum()
    if keep < 5:
        return 0

    seq_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        img_dir = seq_dir / "image"
        img_dir.mkdir(exist_ok=True)
        for i in range(N):
            if cam_idx_for[i] < 0:
                continue
            arr = np.frombuffer(cam_raw[cam_idx_for[i]], np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), img)
    np.savez_compressed(
        seq_dir / "labels.npz",
        anchor_idx=np.arange(N, dtype=np.int32), timestamp=grid.astype(np.float64),
        proprio_window=proprio_window, proprio_feature=proprio_feature,
        trav_gt=trav_gt, pose=pose, action=action, valid_mask=valid)
    return int(keep)


def _trav_label(t, od_ts, pitch, roll, v_ground, cm_ts, cm, bp_ts, bp):
    lo = np.searchsorted(od_ts, t, side="left")
    hi = np.searchsorted(od_ts, t + T_TRAV, side="right")
    if hi - lo < 1:
        return 0
    # rollover
    if np.any(np.abs(pitch[lo:hi]) > TILT_LIMIT_RAD) or np.any(np.abs(roll[lo:hi]) > TILT_LIMIT_RAD):
        return 1
    # collision (bumper non-empty in window)
    if len(bp_ts):
        blo = np.searchsorted(bp_ts, t, side="left")
        bhi = np.searchsorted(bp_ts, t + T_TRAV, side="right")
        if bhi > blo and np.any(bp[blo:bhi] > 0):
            return 1
    # stuck: commanded > STUCK_CMD but v_ground < STUCK_VGND sustained >= STUCK_SUSTAIN
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--prune-decision", type=str, default=None,
                    help="optional per-loop keep/drop JSON (redundancy prune); drops redundant "
                         "OK full-traverses. Default None = keep all.")
    ap.add_argument("--prune-exempt", nargs="*", default=[],
                    help="seed_tags never pruned even if in the decision (e.g. n1990 = held-out "
                         "val batch, kept identical to the unpruned set for val comparability)")
    ap.add_argument("--forest-keep", type=str, default=None,
                    help="raw/forest_combined_keep.json — drops BAD forest col/nm episodes "
                         "(rollover / vegetation never touched / clean run <1.35 s / near-miss "
                         "that touched). No-op for every non-forest_{col,nm}_N bag, so it is "
                         "safe to pass on the whole sweep. Needs forest_{col,nm}_rollouts.jsonl "
                         "next to the bag (loop -> t_start_sim).")
    ap.add_argument("--name-prefix", default="",
                    help="prepended to sequence dir names (Isaac episodes reuse bag stems "
                         "across envs — pass '<Env>__' to disambiguate)")
    args = ap.parse_args()
    decision = load_prune_decision(args.prune_decision, exempt=set(args.prune_exempt))
    n_ok, n_ep, n_pruned = extract_bag(args.bag, args.out_root, save_images=not args.no_images,
                                       prune_decision=decision, forest_keep=args.forest_keep,
                                       name_prefix=args.name_prefix)
    print(f"{args.bag.name}: wrote {n_ok}/{n_ep} episodes"
          + (f" (pruned {n_pruned} redundant)" if n_pruned else "") + f" -> {args.out_root}")


if __name__ == "__main__":
    main()
