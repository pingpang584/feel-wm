"""Validate — and if needed correct — the TD1 IMU mount rotation on a BUILT corpus.

`td1_imu_mount.py` solves the rotation from raw bags. Once
`td1_fetch_build_loop.sh` has run, the bags are gone (the loop deletes them to bound
disk use), but every sequence keeps `extracted.npz`, which holds the
already-rotated IMU together with the exact INS pitch/roll. That is enough to check
the shipped constant against the whole corpus, and enough to repair it in place.

Two independent checks, matching the two degrees of freedom the solve has:

**Tilt (2 DOF).** Kabsch the measured gravity direction on quiet samples against
the world-up direction implied by the stored attitude. What matters is not the size
of the fitted correction but whether applying it *helps*: a fit always returns some
rotation, and if the residual barely moves, the residual is vehicle acceleration
and sensor noise rather than misalignment.

**Azimuth (1 DOF).** Gravity cannot see rotation about the vertical. Fit the
horizontal gyro to the INS attitude rates: in REP-103 the map must come out as
`[[1, 0], [0, -1]]` (roll rate about `+x`, pitch rate about `-y`), and any rotation
on top of that is azimuth error. Euler rates are not body rates, so yaw-rate
coupling is a candidate explanation for a non-zero result and is tested by adding
the yaw axis and by gating on `|gyro_z|`.

`--fix DEG` rotates the cached IMU about the vertical by DEG and rewrites the
cache; `build.py --mode features-only` then regenerates the labels. The two label
channels are azimuth-invariant, so only encoder columns 0, 1, 3 and 4 change.

Usage:
    python -m tools.tartandrive.td1_check_imu_mount
    python -m tools.tartandrive.td1_check_imu_mount --fix -13.98
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from feelwm.config import data_path

import numpy as np

DEFAULT_ROOT = data_path("tartandrive1_processed")
QUIET_TOL = 0.3
QUIET_GYRO_RADS = 0.10
GRAVITY = 9.81
GYRO_SMOOTH_N = 9
# In REP-103 (x fwd, y left, z up) with the stored TD1-convention angles, roll turns
# about +x and pitch about -y, so a correctly aligned gyro fits this exactly.
EXPECTED = np.array([[1.0, 0.0], [0.0, -1.0]])


def archive_of(name: str) -> str:
    a, _, b = name.partition("_heightmaps_")
    return f"{a}_heightmaps_{b.split('_')[0]}"


def sequences(root: Path, per_archive: int | None):
    by = collections.defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        by[archive_of(d.name)].append(d)
    out = []
    for v in by.values():
        out.extend(v if per_archive is None else v[:per_archive])
    return out, len(by)


def up_in_body(pitch_rad, roll_rad):
    """World +z in REP-103 body coords, from the TD1-convention stored angles.

    Third row of the TD1 body-to-world rotation is
    `(-sin r cos p, sin p, cos r cos p)`, and REP-103 is
    `(fwd, left, up) = (+y_td1, -x_td1, +z_td1)`.
    """
    return np.stack([np.sin(pitch_rad),
                     np.sin(roll_rad) * np.cos(pitch_rad),
                     np.cos(roll_rad) * np.cos(pitch_rad)], -1)


def load(seq: Path):
    z = np.load(seq / "extracted.npz", allow_pickle=False)
    it = z["imu_ts_ns"].astype(np.float64) / 1e9
    pt = z["span_ts_ns"].astype(np.float64) / 1e9
    if it.size < 50 or pt.size < 20:
        return None
    pitch = np.interp(it, pt, np.deg2rad(z["span_pitch_deg"]))
    roll = np.interp(it, pt, np.deg2rad(z["span_roll_deg"]))
    kern = np.ones(GYRO_SMOOTH_N) / GYRO_SMOOTH_N
    gyro_s = np.column_stack([np.convolve(z["imu_gyro"][:, i], kern, "same")
                              for i in range(3)])
    rates = np.column_stack([
        np.interp(it, pt, np.gradient(np.deg2rad(z["span_roll_deg"]), pt)),
        np.interp(it, pt, np.gradient(np.deg2rad(z["span_pitch_deg"]), pt)),
    ])
    return z["imu_accel"], z["imu_gyro"], gyro_s, rates, up_in_body(pitch, roll)


def azimuth_of(W):
    u, sv, vt = np.linalg.svd(W[:2])
    resid = (u @ vt) @ np.linalg.inv(EXPECTED)
    return np.degrees(np.arctan2(resid[1, 0], resid[0, 0])), sv


def check(root: Path, per_archive: int | None):
    seqs, n_arch = sequences(root, per_archive)
    A, B, G, T, per = [], [], [], [], []
    for s in seqs:
        rec = load(s)
        if rec is None:
            continue
        acc, gyro, gyro_s, rates, up = rec
        quiet = ((np.abs(np.linalg.norm(acc, axis=1) - GRAVITY) < QUIET_TOL)
                 & (np.linalg.norm(gyro, axis=1) < QUIET_GYRO_RADS))
        if quiet.sum() >= 50:
            a_hat = acc[quiet] / np.linalg.norm(acc[quiet], axis=1, keepdims=True)
            A.append(a_hat)
            B.append(up[quiet])
            per.append(np.median(np.degrees(np.arccos(
                np.clip((a_hat * up[quiet]).sum(1), -1, 1)))))
        if np.isfinite(gyro_s).all() and np.isfinite(rates).all():
            G.append(gyro_s)
            T.append(rates)
    A, B = np.concatenate(A), np.concatenate(B)
    G, T = np.concatenate(G), np.concatenate(T)
    print(f"{len(seqs)} sequences over {n_arch} archives")

    U, _, Vt = np.linalg.svd(B.T @ A)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    dR = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    r_id = np.median(np.degrees(np.arccos(np.clip((A * B).sum(1), -1, 1))))
    r_dR = np.median(np.degrees(np.arccos(np.clip((A * (B @ dR.T)).sum(1), -1, 1))))
    print(f"\n[tilt]  {len(A)} quiet samples")
    print(f"  fitted correction {ang:.2f} deg, but it moves the median gravity-direction")
    print(f"  error only {r_id:.2f} -> {r_dR:.2f} deg, so the residual is noise, not tilt")
    print(f"  per-sequence error: median {np.median(per):.2f}  p90 "
          f"{np.percentile(per, 90):.2f}  max {max(per):.2f} deg")

    W, *_ = np.linalg.lstsq(G[:, :2], T, rcond=None)
    az, sv = azimuth_of(W)
    W3, *_ = np.linalg.lstsq(G, T, rcond=None)
    az3, _ = azimuth_of(W3)
    print(f"\n[azimuth]  {len(G)} samples")
    print(f"  residual azimuth {az:+.2f} deg   (singular values {np.round(sv, 3)})")
    print(f"  with the yaw axis added: {az3:+.2f} deg  <- rules out yaw-rate coupling")
    for gate in (0.30, 0.15, 0.08, 0.04):
        m = np.abs(G[:, 2]) < gate
        if m.sum() < 10000:
            continue
        Wg, *_ = np.linalg.lstsq(G[m][:, :2], T[m], rcond=None)
        print(f"  |gyro_z| < {gate:<5}: {azimuth_of(Wg)[0]:+.2f} deg  (n={int(m.sum())})")
    if abs(az) < 2.0:
        print("\nVERDICT: azimuth OK")
    else:
        print(f"\nVERDICT: azimuth off by {az:+.2f} deg — correct with --fix {-az:.2f}")
    return az


def fix_one(seq: Path, deg: float, Rz_row: np.ndarray, provenance: str) -> str:
    """Rotate one sequence's cached IMU and regenerate its labels.

    Labels are recomputed through `build.compute_features_and_labels`, the same
    function the build used, so this cannot drift from the pipeline. Images are
    untouched — the rotation only affects the IMU.
    """
    from tools.tartandrive.build import PIPELINE_VERSION, compute_features_and_labels
    from tools.tartandrive.cache import load_extracted, save_extracted
    from tools.tartandrive.io_utils import write_meta, write_npz

    meta_path, cache = seq / "meta.json", seq / "extracted.npz"
    if not (meta_path.exists() and cache.exists()):
        return "skipped (no cache)"
    meta = json.loads(meta_path.read_text())
    if abs(meta.get("imu_azimuth_fix_deg", 0.0) - deg) < 1e-9:
        return "already corrected"        # idempotent

    raw = dict(np.load(cache, allow_pickle=False))
    raw["imu_accel"] = (raw["imu_accel"] @ Rz_row).astype(np.float32)
    raw["imu_gyro"] = (raw["imu_gyro"] @ Rz_row).astype(np.float32)
    np.savez_compressed(cache, **raw)

    data = load_extracted(cache)
    arrays, summary = compute_features_and_labels(data)
    write_npz(seq / "labels.npz", arrays)
    meta.update({
        "imu_azimuth_fix_deg": deg,
        "imu_mount_provenance": provenance,
        "valid_fraction": round(summary["valid_fraction"], 4),
        "pipeline_version": PIPELINE_VERSION,
    })
    write_meta(meta_path, meta)
    return "corrected"


def fix(root: Path, deg: float) -> int:
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    # Row-vector form, since the pipeline stores vectors as rows: v_new = v @ Rz_row.
    Rz_row = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    from tools.tartandrive.td1_imu_mount import MOUNT_PROVENANCE

    counts: collections.Counter = collections.Counter()
    seqs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    for i, seq in enumerate(seqs, 1):
        counts[fix_one(seq, deg, Rz_row, MOUNT_PROVENANCE)] += 1
        if i % 50 == 0:
            print(f"  {i}/{len(seqs)}", flush=True)
    print(f"rotated the cached IMU by {deg:+.3f} deg about the vertical and "
          f"regenerated labels: {dict(counts)}")
    return counts["corrected"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--per-archive", type=int, default=6,
                    help="sequences sampled per archive (0 = all)")
    ap.add_argument("--fix", type=float, default=None, metavar="DEG",
                    help="rotate the cached IMU about the vertical by DEG")
    args = ap.parse_args()

    if args.fix is not None:
        return 0 if fix(args.processed_root, args.fix) >= 0 else 1
    check(args.processed_root, args.per_archive or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
