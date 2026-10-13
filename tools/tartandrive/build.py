"""Build per-anchor label artifacts for one TartanDrive 2.0 sequence.

Entry point for the label-generation pipeline (bag → per-anchor proprio labels,
encoder-input windows, images). Idempotent: skips a sequence whose existing
meta.json already matches the current pipeline_version (unless --force).

Modes:
    full           bag → extract(+cache) → images + features + labels
    extract-only   bag → extract(+cache) + images      (no feature compute)
    features-only  load cache → features + labels      (no bag I/O, no images)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from feelwm.config import data_path  # noqa: E402
from tools.tartandrive.extract import ANCHOR_TOPIC, extract_topics  # noqa: E402
from tools.tartandrive.extract_td1 import extract_topics_td1  # noqa: E402
from tools.tartandrive.td1_imu_mount import MOUNT_PROVENANCE  # noqa: E402
from tools.tartandrive.cache import load_extracted, save_extracted  # noqa: E402
from tools.tartandrive.features import (  # noqa: E402
    T_WIN_N,
    R_TIRE_M,
    encoder_input_window,
    proprio_feature_5d,
    susp_rms,
)
from tools.tartandrive.io_utils import (  # noqa: E402
    write_images,
    write_meta,
    write_npz,
    write_sample_csv,
)


PIPELINE_VERSION = "1.1"
# v1.1:
#   - slip_ratio denominator changed to max(v_wheel, v_ground, eps) so |slip|
#     is bounded to [-1, 1] regardless of which speed is near zero.
#     Restores meaningful values during braking/coasting and spinning-on-spot.
#   - build.py tolerates sequences missing /integrated_to_init pose (was
#     IndexError on meadows_2023-09-14-12-07-28).
#   - duration_s in meta.json computed as median(dt) × N to ignore single
#     header.stamp glitches (was billions on 2023-11-14-15-22-01_figure_8).
# v1.0 (initial)
CACHE_FILENAME = "extracted.npz"

ANCHOR_TOL_NS = int(15e6)
SPAN_TOL_NS = int(25e6)
ACTION_DXY_MAX = 2.05
ACTION_DYAW_MAX = 0.17
START_END_BUFFER_NS = int(0.5 * 1e9)
SLIP_DROP_RATIO = 0.30

REP_SEQ_NAME = "2023-11-14-14-26-22_gupta"

# TD1 support. The TD2 path below is untouched:
# `dataset="td1"` only swaps the extractor, which returns the same canonical dict,
# so features / labels / cache / writers are literally the same code.
TD1_REP_SEQ_NAME = "20210826_heightmaps_1_20210826_51"


def td1_sequence_name(bag_path: Path) -> str:
    """`<archive>_<bag stem>` — the prefix NWM's TD1 split uses for its trajectory
    names (`data_splits/tartan_drive/*/traj_names.txt`), minus their segment index.
    A TD1 sequence is ONE bag (~30-40 s), unlike a TD2 sequence which is a folder
    of fragments spanning a whole drive."""
    return f"{bag_path.parent.name}_{bag_path.stem}"


def pose_to_mat(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    m[:3, 3] = pos.astype(np.float32)
    return m


def action_from_poses(T_t: np.ndarray, T_tp1: np.ndarray) -> np.ndarray:
    dT = np.linalg.inv(T_t) @ T_tp1
    return np.array(
        [float(dT[0, 3]), float(dT[1, 3]), float(np.arctan2(dT[1, 0], dT[0, 0]))],
        dtype=np.float32,
    )


def compute_features_and_labels(data: dict) -> tuple[dict, dict]:
    """Given the extracted data dict, compute per-anchor arrays + summary stats.

    Returns (npz_arrays, summary) where summary has 'n_anchors',
    'valid_fraction', 'has_proprio'.
    """
    pose = data["/integrated_to_init"]
    imu = data["/novatel/imu/data"]
    span = data["/span/status"]
    controls = data.get("/controls")
    steer = data.get("/ros_talon/steering_angle")
    wheel = data.get("/wheel_rpm")
    shock = data.get("/shock_pos")
    trav = data.get("/traversability_cost")
    images = data[ANCHOR_TOPIC]

    has_proprio = (
        controls is not None and len(controls.ts_ns) > 0
        and wheel is not None and len(wheel.ts_ns) > 0
        and shock is not None and len(shock.ts_ns) > 0
    )
    has_pose = pose is not None and len(pose.ts_ns) > 0

    anchor_ts = images.ts_ns.astype(np.int64)
    N = int(len(anchor_ts))
    if N == 0:
        raise RuntimeError("No anchor images")

    pose_dt_ns = np.full(N, 10**18, dtype=np.int64)
    # Default identity pose for the no-pose case; valid_mask will zero these
    pose_mats = np.broadcast_to(np.eye(4, dtype=np.float32), (N, 4, 4)).copy()
    if has_pose:
        for i, t in enumerate(anchor_ts):
            idx = int(np.searchsorted(pose.ts_ns, t))
            left = max(idx - 1, 0)
            right = min(idx, len(pose.ts_ns) - 1)
            dl = abs(int(pose.ts_ns[left]) - int(t))
            dr = abs(int(pose.ts_ns[right]) - int(t))
            j = left if dl <= dr else right
            pose_dt_ns[i] = min(dl, dr)
            pose_mats[i] = pose_to_mat(pose.pos[j], pose.quat[j])
    # When pose is missing (e.g., meadows_2023-09-14-12-07-28 lacks
    # /integrated_to_init), every anchor will have
    # pose_dt_ns > tolerance below, so valid_mask gets zeroed out
    # naturally. Pose matrices stay as identity (zeros + 1 on diag);
    # downstream code must respect valid_mask before using `pose`.

    actions = np.zeros((N, 3), dtype=np.float32)
    for i in range(N - 1):
        actions[i] = action_from_poses(pose_mats[i], pose_mats[i + 1])

    span_dt_ns = np.full(N, 10**18, dtype=np.int64)
    pitch_deg = np.full(N, np.nan, dtype=np.float32)
    roll_deg = np.full(N, np.nan, dtype=np.float32)
    if len(span.ts_ns) > 0:
        for i, t in enumerate(anchor_ts):
            j = int(np.searchsorted(span.ts_ns, t))
            left = max(j - 1, 0)
            right = min(j, len(span.ts_ns) - 1)
            dl = abs(int(span.ts_ns[left]) - int(t))
            dr = abs(int(span.ts_ns[right]) - int(t))
            k = left if dl <= dr else right
            span_dt_ns[i] = min(dl, dr)
            pitch_deg[i] = span.pitch_deg[k]
            roll_deg[i] = span.roll_deg[k]

    trav_cost_teacher = np.full(N, np.nan, dtype=np.float32)
    if trav is not None and len(trav.ts_ns) > 0:
        for i, t in enumerate(anchor_ts):
            j = int(np.searchsorted(trav.ts_ns, t))
            left = max(j - 1, 0)
            right = min(j, len(trav.ts_ns) - 1)
            dl = abs(int(trav.ts_ns[left]) - int(t))
            dr = abs(int(trav.ts_ns[right]) - int(t))
            k = left if dl <= dr else right
            trav_cost_teacher[i] = trav.data[k]

    proprio_feature = np.full((N, 5), np.nan, dtype=np.float32)
    slip_drop_bad = np.zeros(N, dtype=bool)
    for i, t in enumerate(anchor_ts):
        arr, diag = proprio_feature_5d(pose, imu, span, wheel if has_proprio else None, int(t))
        proprio_feature[i] = arr
        ok = diag["slip_n_ok"]
        drop = diag["slip_n_drop"]
        total = ok + drop
        if total > 0 and (drop / total) > SLIP_DROP_RATIO:
            slip_drop_bad[i] = True

    susp_rms_arr = np.full(N, np.nan, dtype=np.float32)
    if has_proprio and shock is not None:
        for i, t in enumerate(anchor_ts):
            susp_rms_arr[i] = susp_rms(shock, int(t))

    proprio_window = np.full((N, T_WIN_N, 14), np.nan, dtype=np.float32)
    for i, t in enumerate(anchor_ts):
        proprio_window[i] = encoder_input_window(
            pose, imu, span,
            controls if has_proprio else None,
            steer if steer is not None else None,
            int(t),
        )

    anchor_dt_ms = np.full(N, np.nan, dtype=np.float32)
    if N >= 2:
        anchor_dt_ms[:-1] = np.diff(anchor_ts).astype(np.float64) / 1e6

    valid_mask = np.ones(N, dtype=np.uint8)
    t_first, t_last = anchor_ts[0], anchor_ts[-1]
    buffer_start = t_first + START_END_BUFFER_NS
    buffer_end = t_last - START_END_BUFFER_NS
    valid_mask[(anchor_ts < buffer_start) | (anchor_ts > buffer_end)] = 0
    bad_action = (
        (np.abs(actions[:, 0]) > ACTION_DXY_MAX)
        | (np.abs(actions[:, 1]) > ACTION_DXY_MAX)
        | (np.abs(actions[:, 2]) > ACTION_DYAW_MAX)
    )
    valid_mask[bad_action] = 0
    valid_mask[slip_drop_bad] = 0
    nan_any = np.any(~np.isfinite(proprio_feature), axis=1)
    valid_mask[nan_any] = 0
    valid_mask[pose_dt_ns > ANCHOR_TOL_NS] = 0
    valid_mask[span_dt_ns > SPAN_TOL_NS] = 0
    if not has_proprio:
        valid_mask[:] = 0

    arrays = {
        "anchor_idx": np.arange(N, dtype=np.int32),
        "timestamp_ns": anchor_ts,
        "anchor_dt_ms": anchor_dt_ms,
        "pose": pose_mats,
        "action": actions,
        "proprio_feature": proprio_feature,
        "proprio_window": proprio_window,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "trav_cost_teacher": trav_cost_teacher,
        "susp_rms": susp_rms_arr,
        "valid_mask": valid_mask,
    }
    summary = {
        "n_anchors": N,
        "valid_fraction": float(valid_mask.mean()),
        "has_proprio": bool(has_proprio),
        "image_shape": [int(images.height), int(images.width)],
        "image_encoding_src": images.encoding,
    }
    return arrays, summary


def build_sequence(seq_dir: Path, out_dir: Path, mode: str = "full",
                   force: bool = False, dataset: str = "td2") -> dict:
    t_start = time.time()
    is_td1 = dataset == "td1"
    # A TD1 "sequence" is a single bag file, a TD2 one is a directory of fragments.
    seq_name = td1_sequence_name(seq_dir) if is_td1 else seq_dir.name
    seq_out = out_dir / seq_name
    meta_path = seq_out / "meta.json"
    cache_path = seq_out / CACHE_FILENAME

    # Idempotency (full mode only — features-only always rewrites)
    if mode == "full" and meta_path.exists() and not force:
        existing = json.loads(meta_path.read_text())
        if existing.get("pipeline_version") == PIPELINE_VERSION:
            print(f"[{seq_name}] already at pipeline_version={PIPELINE_VERSION}; skip.",
                  flush=True)
            return existing

    # --- Extract (full / extract-only) or load from cache (features-only) ---
    if mode == "features-only":
        if not cache_path.exists():
            raise RuntimeError(
                f"[{seq_name}] features-only mode but {cache_path} missing — "
                f"run full/extract-only first.")
        print(f"[{seq_name}] loading cache...", flush=True)
        data = load_extracted(cache_path)
    else:
        bags = [seq_dir] if is_td1 else sorted(seq_dir.glob("*.bag"))
        if not bags:
            raise RuntimeError(f"No .bag files in {seq_dir}")
        print(f"[{seq_name}] extracting from {len(bags)} bag(s)...", flush=True)
        data = extract_topics_td1(bags) if is_td1 else extract_topics(bags)
        seq_out.mkdir(parents=True, exist_ok=True)
        save_extracted(cache_path, data)

    images = data[ANCHOR_TOPIC]

    # --- Write images (full / extract-only only) ---
    if mode in ("full", "extract-only"):
        img_dir = seq_out / "images" / "left_color"
        n_imgs = write_images(img_dir, images)
        assert n_imgs == len(images.ts_ns), (
            f"image count {n_imgs} != anchor count {len(images.ts_ns)}")

    # --- Stop here in extract-only mode ---
    if mode == "extract-only":
        elapsed = round(time.time() - t_start, 1)
        # Write a minimal meta marker so the pipeline can later resume
        partial_meta = {
            "sequence_name": seq_name,
            "stage": "extract-only",
            "n_anchors": int(len(images.ts_ns)),
            "image_shape": [int(images.height), int(images.width)],
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "build_seconds": elapsed,
        }
        write_meta(meta_path, partial_meta)
        print(f"[{seq_name}] extract-only done. N={len(images.ts_ns)}, "
              f"elapsed={elapsed}s", flush=True)
        return partial_meta

    # --- Compute features and write outputs ---
    bags = [seq_dir] if is_td1 else sorted(seq_dir.glob("*.bag"))
    arrays, summary = compute_features_and_labels(data)
    seq_out.mkdir(parents=True, exist_ok=True)
    write_npz(seq_out / "labels.npz", arrays)
    if seq_name in (REP_SEQ_NAME, TD1_REP_SEQ_NAME):
        write_sample_csv(seq_out / "labels_sample.csv", arrays, n=200)

    # Robust duration: median frame-stride × n_anchors. Robust to a single
    # bogus header.stamp (e.g. 2023-11-14-15-22-01_figure_8 had one anchor
    # near epoch which made naive last-first ≈ 1.7e9 s).
    ts_arr = arrays["timestamp_ns"]
    if len(ts_arr) >= 2:
        dt_med_s = float(np.median(np.diff(ts_arr))) / 1e9
        duration_s = dt_med_s * (len(ts_arr) - 1)
    else:
        duration_s = 0.0
    meta = {
        "sequence_name": seq_name,
        "source_bags": [b.name for b in bags],
        "duration_s": duration_s,
        "n_anchors": summary["n_anchors"],
        "valid_fraction": round(summary["valid_fraction"], 4),
        "pipeline_version": PIPELINE_VERSION,
        "dataset": dataset,
        "pose_source": "odometry/filtered_odom" if is_td1 else "integrated_to_init",
        "pose_frame": "sensor_init_per_run",
        "r_tire_m": R_TIRE_M,
        "has_proprio": summary["has_proprio"],
        "d_raw": 14,
        "t_win": T_WIN_N,
        "image_shape": summary["image_shape"],
        "image_encoding_src": summary["image_encoding_src"],
        "mode_last_run": mode,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_seconds": round(time.time() - t_start, 1),
    }
    if is_td1:
        # Recorded so a rebuild can tell which mount calibration produced a
        # sequence — the constant is provisional until the full corpus lands.
        meta["imu_source"] = "multisense/imu/imu_data"
        meta["imu_mount_provenance"] = MOUNT_PROVENANCE
        meta["attitude_source"] = "odometry_quaternion"
        meta["has_trav_teacher"] = False
        meta["anchor_subsample"] = 2
    write_meta(meta_path, meta)
    print(f"[{seq_name}] {mode} done. N={summary['n_anchors']}, "
          f"valid_fraction={summary['valid_fraction']:.3f}, "
          f"has_proprio={summary['has_proprio']}, "
          f"elapsed={meta['build_seconds']}s", flush=True)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-name", default=REP_SEQ_NAME)
    ap.add_argument("--raw-root", default=str(data_path("tartandrive")))
    ap.add_argument("--out-root",
                    default=str(data_path("tartandrive2_processed")))
    ap.add_argument("--mode", choices=["full", "extract-only", "features-only"],
                    default="full")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dataset", choices=["td2", "td1"], default="td2",
                    help="td1: --seq-name is <archive>/<bag>.bag under --raw-root")
    args = ap.parse_args()
    build_sequence(
        Path(args.raw_root) / args.seq_name,
        Path(args.out_root),
        mode=args.mode,
        force=args.force,
        dataset=args.dataset,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
