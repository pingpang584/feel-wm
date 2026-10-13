#!/usr/bin/env python3
"""DINO patch-feature caching for the wheeled per-sequence corpora (Gazebo Husky / Isaac
Husky / other per-sequence robot logs) — every corpus laid out as {seq}/image/NNNNNN.jpg + labels.npz.

Encoders (``--encoder``):
  reg     DINOv2-with-registers, 224², patch 14 (tools/features/cache_dino_features_reg.py).
          Default — the feature space of sim_dinov2reg_v1 / isaac_dinov2reg_v1.
  dinov3  DINOv3 ViT-B/16 @256² (tools/features/cache_dinov3.py). Same (256, 768) float16
          record, so n_patch/d_vis and the whole model config are unchanged; only
          feat_version moves. Resize operator is F.interpolate(bicubic), matching the reg
          path, so the encoder is the ONLY variable in a reg-vs-v3 A/B on these corpora.
          NOTE this differs from tools/legged/cache_dino_features.py, which uses
          PIL.resize for its own DINOv3 caches — the two are not byte-comparable.
          ⚠ GATED repo: needs a HuggingFace token with DINOv3 access.
          ⚠ DINOv3 features are ~5.8× lower-variance than the affine-free-LN reg cache,
          which rescales the raw-feature visual_mse term against lambda_prop/lambda_trav.
          Re-balance the loss weights before training on a dinov3 cache.

Writes CONSOLIDATED output directly —
{seq}/features/{feat_version}/features.npy (N,256,768) float16 + features_valid.npy (N,) bool —
exactly what feelwm/data/sim_dataset.py reads (mmap). No separate consolidate step.

Image layout (per-anchor; index = anchor_idx in labels.npz):
  {seq_root}/<episode>/image/NNNNNN.jpg

Usage:
  python tools/sim/cache_dino_features.py \
      --seq-root ${DATA_ROOT}/negs-ugv/sim_husky_v3/sequences/all --gpus 0,1,2,3
Smoke:
  python tools/sim/cache_dino_features.py --seq-root <...> --gpus 0 --limit 16 --max-seqs 1
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from cache_dino_features_reg import (  # noqa: E402
    load_reg_encoder, preprocess, forward_patches, EXPECTED_SHAPE,
)
import cache_dinov3  # noqa: E402


def build_encoder(encoder: str, device: str):
    """-> (forward_fn(batch)->np.ndarray, preprocess_fn(PIL)->Tensor, expected_shape)."""
    if encoder == "reg":
        enc = load_reg_encoder(device)
        return (lambda b: forward_patches(enc, b)), preprocess, EXPECTED_SHAPE
    if encoder == "dinov3":
        enc = cache_dinov3.load_encoder(cache_dinov3.MODEL_ID, device)
        size = cache_dinov3.IMAGE_SIZE
        n_prefix = cache_dinov3.n_prefix_tokens(enc)
        mean_std = cache_dinov3.resolve_mean_std(cache_dinov3.MODEL_ID)
        mean = torch.tensor(mean_std[0]).view(1, 3, 1, 1)
        std = torch.tensor(mean_std[1]).view(1, 3, 1, 1)
        return (
            (lambda b: cache_dinov3.forward_patches(enc, b, n_prefix)),
            (lambda img: cache_dinov3.preprocess(img, size, mean, std)),
            cache_dinov3.expected_shape(enc, size),
        )
    raise ValueError(f"unknown encoder {encoder!r}")


def find_sequences(seq_root: Path) -> list[tuple[Path, Path]]:
    out = []
    for lbl in sorted(seq_root.glob("*/labels.npz")):
        img = lbl.parent / "image"
        if img.is_dir():
            out.append((lbl.parent, img))
    return out


def _index_of(path: Path) -> int:
    return int(path.stem.split("_")[0])


def process_sequence(seq_dir, img_dir, feat_version, fwd, prep, exp_shape,
                     device, batch_size, limit, prefix):
    out_dir = seq_dir / "features" / feat_version
    feat_file, valid_file = out_dir / "features.npy", out_dir / "features_valid.npy"
    n_anchor = len(np.load(seq_dir / "labels.npz")["valid_mask"])
    if feat_file.exists() and valid_file.exists() and not limit:
        v = np.load(valid_file)
        if len(v) == n_anchor and v.any():
            return 0, n_anchor, 0   # resume skip

    paths = sorted(img_dir.glob("*.jpg"), key=_index_of)
    if limit:
        paths = paths[:limit]

    feats = np.zeros((n_anchor, *exp_shape), np.float16)
    valid = np.zeros(n_anchor, bool)
    errors = 0
    for start in range(0, len(paths), batch_size):
        imgs, idxs = [], []
        for p in paths[start:start + batch_size]:
            i = _index_of(p)
            if i >= n_anchor:
                continue
            try:
                imgs.append(prep(Image.open(p))); idxs.append(i)
            except Exception as e:
                print(f"{prefix}[WARN] {p.name}: {e}", file=sys.stderr, flush=True); errors += 1
        if not imgs:
            continue
        out = fwd(torch.stack(imgs).to(device))
        for i, f in zip(idxs, out):
            if f.shape == exp_shape and np.all(np.isfinite(f)):
                feats[i] = f; valid[i] = True
            else:
                errors += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_file, feats)
    np.save(valid_file, valid)
    return int(valid.sum()), n_anchor, errors


def worker_fn(gpu_id, seqs, feat_version, batch_size, limit, encoder="reg", cpu_threads=1):
    device = f"cuda:{gpu_id}"; prefix = f"[GPU{gpu_id}] "
    # One intra-op thread per worker: the per-image CPU resize is far too small to
    # parallelize (21.5 ms @32 threads vs 3.4 ms @1), and N workers x 32 threads
    # oversubscribes the box. Parallelism comes from the process pool.
    torch.set_num_threads(cpu_threads)
    print(f"{prefix}loading {encoder} encoder…", flush=True)
    fwd, prep, exp_shape = build_encoder(encoder, device)
    tot_new = tot = tot_err = 0
    for seq_dir, img_dir in seqs:
        new, n, err = process_sequence(seq_dir, img_dir, feat_version, fwd, prep,
                                       exp_shape, device, batch_size, limit, prefix)
        print(f"{prefix}[OK] {seq_dir.name}: {new}/{n}" + (f" ({err} err)" if err else ""), flush=True)
        tot_new += new; tot += n; tot_err += err
    print(f"{prefix}done. cached={tot_new}/{tot} errors={tot_err}", flush=True)
    return tot_new, tot, tot_err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-root", required=True, type=Path)
    ap.add_argument("--feat-version", default=None,
                    help="default sim_dinov2reg_v1 for --encoder reg; REQUIRED for dinov3")
    ap.add_argument("--encoder", default="reg", choices=["reg", "dinov3"])
    ap.add_argument("--cpu-threads", type=int, default=1,
                    help="torch intra-op threads PER WORKER (default 1; >1 is slower)")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seqs", type=int, default=0)
    args = ap.parse_args()

    # Naming tripwire: writing one encoder's features into the other's version dir is
    # silent and only surfaces as a bad training run.
    if args.feat_version is None:
        if args.encoder != "reg":
            sys.exit("--feat-version is required with --encoder dinov3 "
                     "(e.g. sim_dinov3_v1 / isaac_dinov3_v1)")
        args.feat_version = "sim_dinov2reg_v1"
    tag = args.feat_version.lower()
    if args.encoder == "dinov3" and "dinov2" in tag:
        sys.exit(f"refusing to write DINOv3 features into {args.feat_version!r}")
    if args.encoder == "reg" and "dinov3" in tag:
        sys.exit(f"refusing to write register features into {args.feat_version!r}")

    seqs = find_sequences(args.seq_root)
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    if not seqs:
        sys.exit(f"no sequences with labels+images under {args.seq_root}")

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    print(f"{len(seqs)} sequences → {len(gpu_ids)} GPU(s), encoder={args.encoder}, "
          f"feat_version={args.feat_version}", flush=True)
    t0 = time.time()
    if len(gpu_ids) == 1:
        worker_fn(gpu_ids[0], seqs, args.feat_version, args.batch_size, args.limit,
                  args.encoder, args.cpu_threads)
    else:
        chunks = [seqs[i::len(gpu_ids)] for i in range(len(gpu_ids))]
        with mp.get_context("spawn").Pool(len(gpu_ids)) as pool:
            res = pool.starmap(worker_fn, [(g, c, args.feat_version, args.batch_size,
                                            args.limit, args.encoder, args.cpu_threads)
                                           for g, c in zip(gpu_ids, chunks)])
        print(f"\nall done in {(time.time()-t0)/60:.1f} min. "
              f"cached={sum(r[0] for r in res)}/{sum(r[1] for r in res)} errors={sum(r[2] for r in res)}",
              flush=True)


if __name__ == "__main__":
    main()
