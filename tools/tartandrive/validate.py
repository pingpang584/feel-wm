"""Smoke-test validator: load a built sequence's artifacts, print shape/dtype/
range stats, and assert spec compliance. Exits non-zero on failure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from feelwm.config import data_path

REPO_ROOT = Path(__file__).resolve().parents[2]


EXPECTED_KEYS = {
    "anchor_idx": (("N",), np.int32),
    "timestamp_ns": (("N",), np.int64),
    "anchor_dt_ms": (("N",), np.float32),
    "pose": (("N", 4, 4), np.float32),
    "action": (("N", 3), np.float32),
    "proprio_feature": (("N", 5), np.float32),
    "proprio_window": (("N", 50, 14), np.float32),
    "pitch_deg": (("N",), np.float32),
    "roll_deg": (("N",), np.float32),
    "trav_cost_teacher": (("N",), np.float32),
    "susp_rms": (("N",), np.float32),
    "valid_mask": (("N",), np.uint8),
}


def run(seq_out: Path) -> int:
    failures = []
    meta_path = seq_out / "meta.json"
    labels_path = seq_out / "labels.npz"
    img_dir = seq_out / "images" / "left_color"
    csv_path = seq_out / "labels_sample.csv"

    print(f"=== validating {seq_out.name} ===")
    if not meta_path.exists():
        return failures.append("meta.json missing") or 1
    meta = json.loads(meta_path.read_text())
    print(f"  meta.json keys: {sorted(meta.keys())}")
    print(f"  n_anchors={meta['n_anchors']}  valid_fraction={meta['valid_fraction']}"
          f"  has_proprio={meta['has_proprio']}  build_seconds={meta.get('build_seconds')}s")

    if not labels_path.exists():
        print("  labels.npz MISSING")
        return 1
    z = np.load(labels_path)
    N = meta["n_anchors"]
    print(f"  labels.npz size = {labels_path.stat().st_size/1e6:.1f} MB, keys={sorted(list(z.keys()))}")

    # Key + shape + dtype checks
    for k, (shape_spec, dtype) in EXPECTED_KEYS.items():
        if k not in z:
            failures.append(f"missing key {k}")
            continue
        arr = z[k]
        expected_shape = tuple(N if s == "N" else s for s in shape_spec)
        if arr.shape != expected_shape:
            failures.append(f"{k} shape {arr.shape} != {expected_shape}")
        if arr.dtype != dtype:
            failures.append(f"{k} dtype {arr.dtype} != {dtype}")

    # Range / sanity checks on the *valid* subset
    vm = z["valid_mask"].astype(bool)
    n_valid = int(vm.sum())
    print(f"  valid anchors: {n_valid}/{N} ({n_valid/N*100:.1f}%)")

    def describe(name, arr, extra=""):
        if arr.ndim > 1:
            arr = arr.reshape(arr.shape[0], -1)
            arr = arr[vm] if len(arr) == N else arr
        else:
            arr = arr[vm] if len(arr) == N else arr
        if len(arr) == 0:
            print(f"    {name:20s}  (empty valid subset)")
            return
        finite = np.isfinite(arr)
        if arr.ndim == 1:
            vals = arr[finite]
            if len(vals):
                print(f"    {name:20s}  min={vals.min():+.4f}  max={vals.max():+.4f}  "
                      f"mean={vals.mean():+.4f}  std={vals.std():.4f}  "
                      f"nonfinite={int((~finite).sum())}  {extra}")
            else:
                print(f"    {name:20s}  ALL NON-FINITE")
        else:
            vals = arr[finite]
            if len(vals):
                print(f"    {name:20s}  shape={arr.shape}  "
                      f"min={vals.min():+.4f}  max={vals.max():+.4f}  "
                      f"mean={vals.mean():+.4f}  std={vals.std():.4f}  "
                      f"nonfinite={int((~finite).sum())}  {extra}")

    print("  valid-subset stats:")
    describe("action_dx",  z["action"][:, 0])
    describe("action_dy",  z["action"][:, 1])
    describe("action_dyaw", z["action"][:, 2])
    describe("roughness",              z["proprio_feature"][:, 0])
    describe("slip_ratio",             z["proprio_feature"][:, 1])
    describe("angular_instability",    z["proprio_feature"][:, 2])
    describe("ke_rate",                z["proprio_feature"][:, 3])
    describe("slope_power",            z["proprio_feature"][:, 4])
    describe("pitch_deg",  z["pitch_deg"])
    describe("roll_deg",   z["roll_deg"])
    describe("trav_cost",  z["trav_cost_teacher"])
    describe("susp_rms",   z["susp_rms"])
    describe("anchor_dt_ms", z["anchor_dt_ms"])
    describe("proprio_window", z["proprio_window"].reshape(N, -1))

    # Physical sanity checks
    rough = z["proprio_feature"][vm, 0]
    rough_finite = rough[np.isfinite(rough)]
    if len(rough_finite) and (rough_finite < 0).any():
        failures.append("roughness has negative values")

    if meta.get("has_proprio", False):
        slip = z["proprio_feature"][vm, 1]
        slip_finite = slip[np.isfinite(slip)]
        if len(slip_finite):
            pct_in = float(((slip_finite >= -1) & (slip_finite <= 1)).mean())
            print(f"    slip_ratio in [-1,1]: {pct_in*100:.1f}% of valid samples")
            if pct_in < 0.7:
                failures.append(f"slip_ratio in [-1,1] fraction = {pct_in:.2f} (< 0.70)")

    pitch = z["pitch_deg"][vm]
    if len(pitch) and (np.abs(pitch) > 90).any():
        failures.append("pitch_deg out of ±90°")

    # Image sanity
    n_imgs = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    print(f"  images: {n_imgs} PNG files in {img_dir.relative_to(seq_out)}")
    if n_imgs != N:
        failures.append(f"image count {n_imgs} != N={N}")

    # Sample CSV
    if seq_out.name == "2023-11-14-14-26-22_gupta":
        if not csv_path.exists():
            failures.append("labels_sample.csv missing for rep sequence")
        else:
            lines = csv_path.read_text().splitlines()
            print(f"  labels_sample.csv: {len(lines)-1} rows, {len(lines[0].split(','))} cols")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL CHECKS PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-name", default="2023-11-14-14-26-22_gupta")
    ap.add_argument("--out-root",
                    default=str(data_path("tartandrive2_processed")))
    args = ap.parse_args()
    rc = run(Path(args.out_root) / args.seq_name)
    sys.exit(rc)


if __name__ == "__main__":
    main()
