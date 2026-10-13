#!/usr/bin/env python3
"""Stamp a DINO feature-cache version directory with a ``meta.json`` provenance record.

Feature caches are referenced from configs by a bare ``feat_version`` string
(``dinov2reg_v1``, ``dinov3_v1``, ...). Nothing on disk said WHICH encoder, input
resolution or patch grid produced them, so an encoder A/B had to be reconstructed from
the cacher script's git history. An encoder comparison (e.g. DINOv3 vs DINOv2 on the
wheeled corpora) needs the variant / input resolution / patch grid / feature dim to be
recorded, so this writes them next to the features.

DECLARED vs MEASURED. Provenance (model id, input resolution, resize operator, LN-affine
disposition) cannot be recovered from the arrays -- it is DECLARED by the caller and must
match the cacher that actually ran. Everything else (patch grid, feature dim, dtype,
sequence/frame counts, feature variance) is MEASURED from the cache itself, so a wrong
declaration cannot fake a matching shape.

``feature_variance`` is measured because it is not cosmetic: ``visual_mse`` in
feelwm/training/losses.py is a RAW-feature MSE while the proprio (z-scored) and trav (BCE)
terms are scale-free, so the cache's absolute scale silently sets the visual:proprio:trav
loss balance. Two caches are loss-scale comparable only if this number matches.

Both on-disk layouts are supported:
  flat     {root}/{seq}/features.npy                        (TartanDrive2)
  per-seq  {seq_root}/{seq}/features/{feat_version}/...     (sim / Isaac corpora)

Usage:
    # TD2 (flat)
    python tools/features/write_cache_meta.py \
        ${DATA_ROOT}/tartandrive2_processed_features/dinov3_v1 \
        --encoder facebook/dinov3-vitb16-pretrain-lvd1689m \
        --image-size 256 --ln-affine kept --resize "F.interpolate bicubic"

    # Gazebo Husky (per-seq)
    python tools/features/write_cache_meta.py \
        ${DATA_ROOT}/negs-ugv/sim_husky_v3/sequences/all \
        --feat-version sim_dinov3_v1 \
        --encoder facebook/dinov3-vitb16-pretrain-lvd1689m \
        --image-size 256 --ln-affine kept --resize "F.interpolate bicubic"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Frames drawn per sampled sequence when measuring the feature scale. Small on purpose:
# the arrays are mmapped and each frame is 256x768, so a few hundred already pin the
# variance to far more digits than the comparison needs.
_SCALE_SEQS = 5
_SCALE_FRAMES = 150


def find_feature_files(root: Path, feat_version: str | None) -> tuple[list[Path], str]:
    """-> ([features.npy, ...], layout). Flat is tried first, then per-sequence."""
    flat = sorted(root.glob("*/features.npy"))
    if flat:
        return flat, "flat"
    if feat_version:
        nested = sorted(root.glob(f"*/features/{feat_version}/features.npy"))
        if nested:
            return nested, "per-sequence"
    raise SystemExit(
        f"no consolidated features.npy under {root}"
        + ("" if feat_version else " (pass --feat-version for the per-sequence layout)")
    )


def measure(files: list[Path]) -> dict:
    """Shape/dtype/counts over every sequence; feature scale over a sampled subset."""
    shapes, dtypes, n_frames, n_valid = set(), set(), 0, 0
    for f in files:
        a = np.load(f, mmap_mode="r")
        shapes.add(tuple(a.shape[1:]))
        dtypes.add(str(a.dtype))
        n_frames += int(a.shape[0])
        v = f.parent / "features_valid.npy"
        if v.exists():
            n_valid += int(np.load(v).sum())

    if len(shapes) != 1:
        raise SystemExit(f"inconsistent per-frame shapes across sequences: {sorted(shapes)}")
    (n_patch, d_vis), = shapes

    rng = np.random.default_rng(0)
    sample = [files[i] for i in rng.choice(len(files), min(_SCALE_SEQS, len(files)), replace=False)]
    var, absmean = [], []
    for f in sample:
        a = np.load(f, mmap_mode="r")
        take = min(_SCALE_FRAMES, a.shape[0])
        idx = np.sort(rng.choice(a.shape[0], take, replace=False))
        x = np.asarray(a[idx], dtype=np.float32)
        var.append(float(x.var()))
        absmean.append(float(np.abs(x).mean()))

    return {
        "n_sequences": len(files),
        "n_frames": n_frames,
        "n_frames_valid": n_valid or None,   # None = no features_valid.npy (flat TD2 path)
        "n_patch": int(n_patch),
        "d_vis": int(d_vis),
        "dtype": sorted(dtypes)[0] if len(dtypes) == 1 else sorted(dtypes),
        "feature_variance": round(float(np.mean(var)), 6),
        "feature_abs_mean": round(float(np.mean(absmean)), 6),
        "scale_sampled_sequences": len(sample),
    }


def build_meta(root: Path, args, files: list[Path], layout: str) -> dict:
    m = measure(files)
    patch_side = int(round(m["n_patch"] ** 0.5))
    meta = {
        "feat_version": args.feat_version or root.name,
        "layout": layout,
        "root": str(root),
        # --- declared provenance (must match the cacher that ran) ---
        "encoder": args.encoder,
        "image_size": args.image_size,
        "resize": args.resize,
        "ln_affine": args.ln_affine,
        "cacher": args.cacher,
        # --- measured ---
        "patch_grid": [patch_side, patch_side] if patch_side * patch_side == m["n_patch"] else None,
        **m,
    }
    if args.image_size and args.patch_size:
        side = args.image_size // args.patch_size
        meta["patch_size"] = args.patch_size
        if side * side != m["n_patch"]:
            raise SystemExit(
                f"declared image_size {args.image_size} / patch_size {args.patch_size} implies "
                f"{side * side} tokens, but the cache holds {m['n_patch']}"
            )
    if args.note:
        meta["notes"] = args.note
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", type=Path,
                    help="flat: the {feat_root}/{feat_version} dir; per-seq: the seq_root")
    ap.add_argument("--feat-version", default=None,
                    help="required for the per-sequence layout; defaults to root.name")
    ap.add_argument("--encoder", required=True, help="HuggingFace / hub model id")
    ap.add_argument("--image-size", type=int, default=None, help="encoder input side in px")
    ap.add_argument("--patch-size", type=int, default=None,
                    help="if given with --image-size, cross-checks the token count")
    ap.add_argument("--resize", default=None, help="resize operator, e.g. 'F.interpolate bicubic'")
    ap.add_argument("--ln-affine", choices=["kept", "stripped"], default=None,
                    help="final-LayerNorm affine disposition (sets the feature scale)")
    ap.add_argument("--cacher", default=None, help="script that produced the cache")
    ap.add_argument("--note", action="append", default=None,
                    help="free-form note; repeatable (side-effect inventory)")
    ap.add_argument("--out", type=Path, default=None, help="default: <cache dir>/meta.json")
    args = ap.parse_args()

    files, layout = find_feature_files(args.root, args.feat_version)
    meta = build_meta(args.root, args, files, layout)

    if args.out is not None:
        out = args.out
    elif layout == "flat":
        out = args.root / "meta.json"
    else:
        # per-sequence: no single cache dir, so the record lives beside the corpus
        out = args.root / f"{args.feat_version}.meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
