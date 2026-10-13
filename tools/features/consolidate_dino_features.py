#!/usr/bin/env python3
"""Consolidate per-anchor DINOv2 .npy files into per-sequence features.npy.

Current layout (slow — one file open per anchor per sample):
    {feat_root}/{feat_version}/{seq}/{anchor_idx}.npy   shape (256, 768) float16

Output (fast — one mmap per sequence):
    {feat_root}/{feat_version}/{seq}/features.npy       shape (N, 256, 768) float16
    {feat_root}/{feat_version}/{seq}/features_valid.npy shape (N,)          bool

features.npy is indexed by sequence position (== anchor_idx for TartanDrive2).
Missing anchors (no .npy on disk) are stored as zeros; features_valid.npy marks them False.

After running this script, the dataset loader uses np.load(..., mmap_mode='r')
and slices features[j] instead of opening {anchor_idx}.npy per call.

Usage:
    python tools/features/consolidate_dino_features.py \\
        --processed-root ${DATA_ROOT}/tartandrive2_processed \\
        --feat-root ${DATA_ROOT}/tartandrive2_processed_features \\
        --feat-version dinov2reg_v1

Options:
    --workers N     parallel sequences (default: 4)
    --overwrite     re-generate features.npy even if it exists
    --dry-run       print what would be done without writing
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from feelwm.config import data_path

N_PATCH = 256
D_VIS = 768


def _consolidate_seq(
    seq: str,
    processed_root: Path,
    feat_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[str, int, int]:
    """Process one sequence. Returns (seq, n_anchors, n_cached)."""
    out_feat = feat_dir / "features.npy"
    out_valid = feat_dir / "features_valid.npy"

    if out_feat.exists() and not overwrite:
        n = int(np.load(out_valid).shape[0])
        cached = int(np.load(out_valid).sum())
        return seq, n, cached

    labels = np.load(processed_root / seq / "labels.npz", allow_pickle=False)
    anchor_idx = labels["anchor_idx"]  # (N,) int32, sequential == position
    N = len(anchor_idx)

    features = np.zeros((N, N_PATCH, D_VIS), dtype=np.float16)
    valid = np.zeros(N, dtype=bool)

    for j in range(N):
        npy_path = feat_dir / f"{anchor_idx[j]}.npy"
        if npy_path.exists():
            features[j] = np.load(npy_path)
            valid[j] = True

    n_cached = int(valid.sum())

    if not dry_run:
        np.save(out_feat, features)
        np.save(out_valid, valid)

    return seq, N, n_cached


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate DINOv2 per-anchor .npy into features.npy")
    parser.add_argument("--processed-root", default=str(data_path("tartandrive2_processed")))
    parser.add_argument("--feat-root", default=str(data_path("tartandrive2_processed_features")))
    parser.add_argument("--feat-version", default="dinov2_v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    feat_base = Path(args.feat_root) / args.feat_version

    seqs = sorted(d.name for d in feat_base.iterdir() if d.is_dir())
    print(f"Found {len(seqs)} sequence directories under {feat_base}")
    if args.dry_run:
        print("DRY RUN — no files will be written")

    t0 = time.time()
    done = 0
    total_anchors = 0
    total_cached = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _consolidate_seq,
                seq,
                processed_root,
                feat_base / seq,
                args.overwrite,
                args.dry_run,
            ): seq
            for seq in seqs
        }
        for fut in as_completed(futures):
            seq = futures[fut]
            try:
                seq_name, n, cached = fut.result()
                done += 1
                total_anchors += n
                total_cached += cached
                status = "SKIP" if not args.overwrite and (feat_base / seq / "features.npy").exists() and not args.dry_run else "OK"
                print(f"  [{done:3d}/{len(seqs)}] {status}  {seq_name}  ({cached}/{n} anchors cached)")
            except Exception as exc:
                print(f"  ERROR {seq}: {exc}")

    elapsed = time.time() - t0
    print(f"\nDone. {total_cached:,}/{total_anchors:,} anchors consolidated in {elapsed:.1f}s")
    if not args.dry_run:
        size_gb = total_cached * N_PATCH * D_VIS * 2 / 1e9
        print(f"Estimated output size: {size_gb:.1f} GB")


if __name__ == "__main__":
    main()
