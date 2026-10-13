#!/usr/bin/env python3
"""DINOv2-with-registers patch feature caching for TartanDrive 2.0.

Multi-GPU, per-anchor .npy, resumable; the output format is what
``consolidate_dino_features.py`` consumes. The encoder (``feelwm.models.encoders.dinov2_reg``)
reproduces RAE-NWM's ``Dinov2withNorm`` (``facebook/dinov2-with-registers-base``).

WHY: we adopt the register feature space as the project baseline so RAE-NWM's pretrained
ViT-XL decoder can reconstruct our predicted features for evaluation (latent → frame
decoding is eval-only). Decoder compatibility REQUIRES the
exact encoder it was trained with:
  * ``Dinov2WithRegistersModel`` (facebook/dinov2-with-registers-base)
  * final LayerNorm affine REMOVED (normalize=True)
  * drop the first 5 tokens (1 CLS + 4 registers), keep 256 patch tokens
  * preprocessing identical to RAE.encode: bicubic resize to 224 + ImageNet normalize
This is a DIFFERENT space from torch-hub ``dinov2_vitb14`` (no registers, LN affine kept),
so it goes to its own feature version dir (``dinov2reg_v1``), never mixed with a plain
DINOv2 cache.

Output per anchor:
    {out_root}/<seq>/<anchor_idx>.npy  —  float16, shape (256, 768)
We cache RAW Dinov2withNorm output (pre RAE latent-normalization). The WM trains on raw
features as before; RAE's latent-norm is applied only on the eval-decode path.

Usage (all 8 GPUs):
    python tools/features/cache_dino_features_reg.py --gpus 0,1,2,3,4,5,6,7 --batch-size 64
Smoke test (one sequence):
    python tools/features/cache_dino_features_reg.py --gpus 0 \\
        --seq 2023-10-26-14-42-35_turnpike_afternoon_fall --batch-size 64
Then consolidate:
    python tools/features/consolidate_dino_features.py --feat-version dinov2reg_v1
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from feelwm.config import data_path
from feelwm.models.encoders.dinov2_reg import Dinov2RegEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = data_path("tartandrive2_processed")
OUT_ROOT = data_path("tartandrive2_processed_features", "dinov2reg_v1")

# AutoImageProcessor(facebook/dinov2-with-registers-base) == standard ImageNet stats.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
IMAGE_SIZE = 224
# DINOv2-with-registers ViT-B/14: patch_size=14, input=224 → (224/14)²=256, D=768
EXPECTED_SHAPE = (256, 768)


# ---------------------------------------------------------------------------
# Model + preprocessing  (called inside each worker process)
# ---------------------------------------------------------------------------
def load_reg_encoder(device: str) -> torch.nn.Module:
    """Load the frozen DINOv2-with-registers encoder (LN-affine removed, prefix tokens dropped)."""
    enc = Dinov2RegEncoder("facebook/dinov2-with-registers-base", normalize=True)
    enc.eval().to(device)
    for prm in enc.parameters():
        prm.requires_grad_(False)
    return enc


def preprocess(img: Image.Image) -> torch.Tensor:
    """Replicate RAE.encode preprocessing: [0,1] → bicubic 224 → ImageNet normalize."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W] in [0,1]
    t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False)
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    return t[0]  # [3,224,224]


@torch.inference_mode()
def forward_patches(enc: torch.nn.Module, imgs: torch.Tensor) -> np.ndarray:
    """imgs: (B,3,224,224) on device → numpy (B, 256, 768) float16."""
    return enc(imgs).half().cpu().numpy()


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------
def process_sequence(
    seq_dir: Path,
    out_dir: Path,
    enc: torch.nn.Module,
    device: str,
    batch_size: int,
    log_prefix: str = "",
) -> tuple[int, int, int]:
    """Returns (n_new, n_total, n_errors)."""
    img_dir = seq_dir / "images" / "left_color"
    if not img_dir.exists():
        return 0, 0, 0

    all_paths = sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not all_paths:
        return 0, 0, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [p for p in all_paths if not (out_dir / f"{p.stem}.npy").exists()]
    if not pending:
        return 0, len(all_paths), 0

    errors = 0
    new_count = 0

    for start in tqdm(
        range(0, len(pending), batch_size),
        desc=f"{log_prefix}{seq_dir.name[:35]}",
        unit="batch",
        leave=False,
    ):
        batch_paths = pending[start : start + batch_size]
        imgs: list[torch.Tensor] = []
        valid_paths: list[Path] = []
        for p in batch_paths:
            try:
                imgs.append(preprocess(Image.open(p)))
                valid_paths.append(p)
            except Exception as e:
                print(f"{log_prefix}[WARN] load {p.name}: {e}", file=sys.stderr, flush=True)
                errors += 1
        if not imgs:
            continue

        tensor = torch.stack(imgs).to(device)
        feats = forward_patches(enc, tensor)  # (B, 256, 768)

        for path, feat in zip(valid_paths, feats):
            if feat.shape != EXPECTED_SHAPE:
                print(f"{log_prefix}[WARN] shape {feat.shape} in {path.name}",
                      file=sys.stderr, flush=True)
                errors += 1
                continue
            if not np.all(np.isfinite(feat)):
                print(f"{log_prefix}[WARN] NaN/Inf in {seq_dir.name}/{path.name}",
                      file=sys.stderr, flush=True)
                errors += 1
                continue
            np.save(out_dir / f"{path.stem}.npy", feat)
            new_count += 1

    saved = sorted(out_dir.glob("*.npy"))
    if saved:
        probe = np.load(saved[0])
        assert probe.shape == EXPECTED_SHAPE, f"sanity shape {probe.shape}"
        assert probe.dtype == np.float16, f"sanity dtype {probe.dtype}"
        assert np.all(np.isfinite(probe)), "sanity NaN/Inf"

    return new_count, len(all_paths), errors


# ---------------------------------------------------------------------------
# Worker: one process per GPU
# ---------------------------------------------------------------------------
def worker_fn(
    gpu_id: int,
    seq_dirs: list[Path],
    out_root: Path,
    batch_size: int,
) -> tuple[int, int, int]:
    device = f"cuda:{gpu_id}"
    prefix = f"[GPU{gpu_id}] "

    print(f"{prefix}loading Dinov2withNorm (with-registers-base)…", flush=True)
    enc = load_reg_encoder(device)
    print(f"{prefix}ready, {len(seq_dirs)} sequences", flush=True)

    total_new = total_total = total_errors = 0
    for seq_dir in seq_dirs:
        new, total, err = process_sequence(
            seq_dir=seq_dir,
            out_dir=out_root / seq_dir.name,
            enc=enc,
            device=device,
            batch_size=batch_size,
            log_prefix=prefix,
        )
        if total == 0:
            print(f"{prefix}[SKIP] {seq_dir.name} — no images", flush=True)
            continue
        suffix = f" ({err} errors)" if err else ""
        print(f"{prefix}[OK] {seq_dir.name}: +{new}/{total}{suffix}", flush=True)
        total_new += new
        total_total += total
        total_errors += err

    print(f"{prefix}done. new={total_new} total={total_total} errors={total_errors}",
          flush=True)
    return total_new, total_total, total_errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--gpus", default="0", help="comma-separated GPU IDs (default: 0)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq", default=None, metavar="SEQ_NAME",
                    help="process a single sequence (smoke test)")
    args = ap.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    n_gpus = len(gpu_ids)

    for gid in gpu_ids:
        if not torch.cuda.is_available():
            sys.exit("CUDA not available. Set LD_PRELOAD to the correct libcuda.so.")
        if gid >= torch.cuda.device_count():
            sys.exit(f"GPU {gid} not found ({torch.cuda.device_count()} devices).")
        print(f"GPU {gid}: {torch.cuda.get_device_name(gid)}", flush=True)

    if args.seq:
        seq_dirs = [args.data_root / args.seq]
    else:
        seq_dirs = sorted(d for d in args.data_root.iterdir() if d.is_dir())

    print(f"\n{len(seq_dirs)} sequences → {n_gpus} GPU(s)", flush=True)
    print(f"Encoder     : Dinov2withNorm (facebook/dinov2-with-registers-base)", flush=True)
    print(f"Output root : {args.out_root}", flush=True)
    print(f"Anchor shape: {EXPECTED_SHAPE}, dtype float16", flush=True)
    print(f"Batch size  : {args.batch_size} per GPU\n", flush=True)

    t0 = time.time()

    if n_gpus == 1:
        worker_fn(gpu_ids[0], seq_dirs, args.out_root, args.batch_size)
    else:
        chunks = [seq_dirs[i::n_gpus] for i in range(n_gpus)]
        for gid, chunk in zip(gpu_ids, chunks):
            print(f"  GPU {gid}: {len(chunk)} sequences", flush=True)

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_gpus) as pool:
            results = pool.starmap(
                worker_fn,
                [(gid, chunk, args.out_root, args.batch_size)
                 for gid, chunk in zip(gpu_ids, chunks)],
            )
        total_new = sum(r[0] for r in results)
        total_total = sum(r[1] for r in results)
        total_errors = sum(r[2] for r in results)
        print(f"\nAll GPUs done in {(time.time() - t0) / 60:.1f} min. "
              f"new={total_new} total={total_total} errors={total_errors}", flush=True)
        if total_errors:
            sys.exit(1)


if __name__ == "__main__":
    main()
