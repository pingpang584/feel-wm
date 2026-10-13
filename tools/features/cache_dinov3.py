#!/usr/bin/env python3
"""DINOv3 ViT-B/16 patch feature caching for TartanDrive 2.0.

Alternative encoder to ``cache_dino_features_reg.py`` (facebook/dinov2-with-registers-base,
the default feature space). Same multi-GPU / per-anchor .npy / resume machinery and the
same on-disk format, so ``consolidate_dino_features.py`` and the dataset loader work
unchanged.

THREE DESIGN CHOICES that make this an encoder-only A/B:

1. ``facebook/dinov3-vitb16-pretrain-lvd1689m`` — ViT-B, hidden 768, size-matched to
   DINOv2 ViT-B/14. ViT-L/16 (hidden 1024) would change d_vis and every model config
   at the same time, i.e. two variables at once; keep it a separate axis.
2. ``--image-size 256``, NOT 224. DINOv3 is patch_size=16, so 224 -> 14x14 = 196
   tokens, while the whole pipeline hard-codes 256 patches (dataset.py n_patch,
   terrain_wm.py n_patch, consolidate_dino_features.py N_PATCH). 256px -> 16x16 = 256
   tokens reproduces the exact (256, 768) float16 record, so only ``feat_version``
   changes in a config. DINOv3 uses RoPE position embeddings computed from
   pixel_values, so a non-default input resolution is natively supported.
3. Final LayerNorm affine is KEPT (DINOv3ViTModel.last_hidden_state is already
   post-norm). The reg script strips it only because RAE-NWM's pretrained ViT-XL
   decoder demands that exact space -- and that decoder does NOT exist for DINOv3.
   CONSEQUENCE: the RAE latent-reconstruction eval path is unavailable on this cache.

Token layout is [CLS] + [registers] + [patches] (DINOv3ViTEmbeddings.forward), so the
prefix length is read from the checkpoint config rather than hard-coded.

Weights are GATED on HuggingFace: accept the license at
https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m first, otherwise the
download fails with 403.

Output per anchor (identical to dinov2reg_v1):
    {out_root}/<seq>/<anchor_idx>.npy  --  float16, shape (256, 768)

Usage (smoke test, one sequence):
    python tools/features/cache_dinov3.py --gpus 0 \\
        --seq 2023-10-26-14-42-35_turnpike_afternoon_fall --batch-size 64
Usage (full corpus, one worker per listed GPU):
    python tools/features/cache_dinov3.py --gpus 0,1,2,3 --batch-size 64
Then consolidate:
    python tools/features/consolidate_dino_features.py \\
        --processed-root ${DATA_ROOT}/tartandrive2_processed \\
        --feat-root ${DATA_ROOT}/tartandrive2_processed_features \\
        --feat-version dinov3_v1
"""

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = data_path("tartandrive2_processed")
OUT_ROOT = data_path("tartandrive2_processed_features", "dinov3_v1")
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
IMAGE_SIZE = 256  # patch_size 16 -> 16x16 = 256 tokens (see docstring choice 2)

# DINOv3 preprocessing is ImageNet mean/std; read from the processor when reachable.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Model + preprocessing  (called inside each worker process)
# ---------------------------------------------------------------------------
def load_encoder(model_id: str, device: str) -> torch.nn.Module:
    """Load DINOv3ViTModel. Raises with a pointer to the license gate on 403."""
    from transformers import AutoModel

    try:
        enc = AutoModel.from_pretrained(model_id)
    except Exception as e:
        if "403" in str(e) or "gated" in str(e).lower():
            raise SystemExit(
                f"{model_id} is gated. Accept the license at "
                f"https://huggingface.co/{model_id} with the account whose token is in "
                f"~/.cache/huggingface/token, then retry.\nOriginal error: {e}"
            ) from e
        raise
    enc.eval().to(device)
    for prm in enc.parameters():
        prm.requires_grad_(False)
    return enc


def expected_shape(enc: torch.nn.Module, image_size: int) -> tuple[int, int]:
    """(n_patch, hidden) implied by the checkpoint config at this input resolution."""
    cfg = enc.config
    if image_size % cfg.patch_size:
        raise SystemExit(
            f"--image-size {image_size} is not divisible by patch_size {cfg.patch_size}"
        )
    side = image_size // cfg.patch_size
    return side * side, cfg.hidden_size


def n_prefix_tokens(enc: torch.nn.Module) -> int:
    """CLS + register tokens preceding the patch tokens."""
    return 1 + int(getattr(enc.config, "num_register_tokens", 0) or 0)


def preprocess(img: Image.Image, image_size: int, mean, std) -> torch.Tensor:
    """[0,1] -> bicubic resize to image_size (no crop) -> normalize. Matches the
    reg-cache path: the FULL frame is resized, never centre-cropped."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    t = F.interpolate(
        t, size=(image_size, image_size), mode="bicubic", align_corners=False
    )
    t = (t - mean) / std
    return t[0]  # [3,S,S]


@torch.inference_mode()
def forward_patches(enc: torch.nn.Module, imgs: torch.Tensor, n_prefix: int) -> np.ndarray:
    """imgs: (B,3,S,S) on device -> numpy (B, n_patch, hidden) float16."""
    out = enc(pixel_values=imgs).last_hidden_state  # post final LayerNorm
    return out[:, n_prefix:, :].half().cpu().numpy()


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------
def process_sequence(
    seq_dir: Path,
    out_dir: Path,
    enc: torch.nn.Module,
    device: str,
    batch_size: int,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    exp_shape: tuple[int, int],
    n_prefix: int,
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
                imgs.append(preprocess(Image.open(p), image_size, mean, std))
                valid_paths.append(p)
            except Exception as e:
                print(f"{log_prefix}[WARN] load {p.name}: {e}", file=sys.stderr, flush=True)
                errors += 1
        if not imgs:
            continue

        tensor = torch.stack(imgs).to(device)
        feats = forward_patches(enc, tensor, n_prefix)

        for path, feat in zip(valid_paths, feats):
            if feat.shape != exp_shape:
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
        assert probe.shape == exp_shape, f"sanity shape {probe.shape}"
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
    model_id: str,
    image_size: int,
    mean_std: tuple[tuple[float, ...], tuple[float, ...]],
    cpu_threads: int = 1,
) -> tuple[int, int, int]:
    device = f"cuda:{gpu_id}"
    prefix = f"[GPU{gpu_id}] "

    # Preprocessing is CPU-side and PER IMAGE (bicubic resize of a ~1024x544 frame).
    # torch defaults to 32 intra-op threads, and on an op this small the thread-pool
    # sync costs MORE than the work: measured 21.5 ms @32 threads vs 3.4 ms @1 thread,
    # 79 ms vs 40 ms for the whole decode+preprocess. We already parallelize across
    # processes, so one thread each is both faster and what makes N workers fit on N
    # cores instead of oversubscribing 32N threads onto the box.
    torch.set_num_threads(cpu_threads)

    print(f"{prefix}loading {model_id}...", flush=True)
    enc = load_encoder(model_id, device)
    exp_shape = expected_shape(enc, image_size)
    n_prefix = n_prefix_tokens(enc)
    mean = torch.tensor(mean_std[0]).view(1, 3, 1, 1)
    std = torch.tensor(mean_std[1]).view(1, 3, 1, 1)
    print(f"{prefix}ready, {len(seq_dirs)} sequences, shape {exp_shape}, "
          f"drop {n_prefix} prefix tokens", flush=True)

    total_new = total_total = total_errors = 0
    for seq_dir in seq_dirs:
        new, total, err = process_sequence(
            seq_dir=seq_dir,
            out_dir=out_root / seq_dir.name,
            enc=enc,
            device=device,
            batch_size=batch_size,
            image_size=image_size,
            mean=mean,
            std=std,
            exp_shape=exp_shape,
            n_prefix=n_prefix,
            log_prefix=prefix,
        )
        if total == 0:
            print(f"{prefix}[SKIP] {seq_dir.name} -- no images", flush=True)
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
def resolve_mean_std(model_id: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Prefer the checkpoint's own processor stats; fall back to ImageNet."""
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(model_id)
        mean = tuple(float(x) for x in proc.image_mean)
        std = tuple(float(x) for x in proc.image_std)
        return mean, std
    except Exception as e:
        print(f"[WARN] could not read image processor ({type(e).__name__}); "
              f"falling back to ImageNet mean/std", file=sys.stderr, flush=True)
        return _IMAGENET_MEAN, _IMAGENET_STD


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                    help="input side in px; 256 keeps the 256-token record (default)")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU IDs (default: 0)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq", default=None, metavar="SEQ_NAME",
                    help="process a single sequence (smoke test)")
    ap.add_argument("--cpu-threads", type=int, default=1,
                    help="torch intra-op threads PER WORKER (default 1; >1 is slower "
                         "here -- the per-image resize is too small to parallelize)")
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

    mean_std = resolve_mean_std(args.model_id)

    print(f"\n{len(seq_dirs)} sequences -> {n_gpus} GPU(s)", flush=True)
    print(f"Encoder     : {args.model_id}", flush=True)
    print(f"Input size  : {args.image_size}px (full-frame bicubic resize, no crop)", flush=True)
    print(f"Norm        : mean={mean_std[0]} std={mean_std[1]}", flush=True)
    print(f"Output root : {args.out_root}", flush=True)
    print(f"Batch size  : {args.batch_size} per GPU", flush=True)
    print(f"CPU threads : {args.cpu_threads} per worker\n", flush=True)

    t0 = time.time()

    if n_gpus == 1:
        worker_fn(gpu_ids[0], seq_dirs, args.out_root, args.batch_size,
                  args.model_id, args.image_size, mean_std, args.cpu_threads)
        print(f"\nDone in {(time.time() - t0) / 60:.1f} min.", flush=True)
    else:
        chunks = [seq_dirs[i::n_gpus] for i in range(n_gpus)]
        for gid, chunk in zip(gpu_ids, chunks):
            print(f"  GPU {gid}: {len(chunk)} sequences", flush=True)

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_gpus) as pool:
            results = pool.starmap(
                worker_fn,
                [(gid, chunk, args.out_root, args.batch_size, args.model_id,
                  args.image_size, mean_std, args.cpu_threads)
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
