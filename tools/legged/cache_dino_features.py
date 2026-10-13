#!/usr/bin/env python3
"""DINO patch-feature caching for the legged track (GrandTour + TartanGround + Isaac ANYmal).

Writes CONSOLIDATED output directly — ``{seq}/features/{feat_version}/features.npy``
(N,256,768) float16 + ``features_valid.npy`` (N,) bool — which is exactly what
``feelwm/data/legged_dataset.py`` reads (mmap). No separate consolidate step.

Encoders (selected per dataset, ``--encoder`` overrides):
  reg     DINOv2-with-registers (``tools/features/cache_dino_features_reg.py``; 224², patch 14,
          drop 1 CLS + 4 registers → 256×768). GT/TG default — same space as the wheeled cache.
  dinov3  DINOv3 ViT-B/16 (``facebook/dinov3-vitb16-pretrain-lvd1689m``; native 256² input,
          patch 16, drop 1 CLS + 4 registers → 256×768 — same cache geometry as reg, so the
          model config (n_patch=256, d_vis=768) is unchanged). Isaac ANYmal default (the Isaac
          corpus is cached with DINOv3 from the start). Preprocessing mirrors the project
          convention: bicubic SQUARE resize (to 256) + ImageNet normalize — same
          aspect-distortion class as the 224 pipeline.
          ⚠ GATED repo: needs a HuggingFace token with DINOv3 access (env HF_TOKEN, or
          `hf auth login`); request access at the model page once per account.

Image layouts (per-anchor files; index = anchor_idx in labels_legged.npz):
  GrandTour     <root>/<mission>/images/hdr_front/NNNNNN.jpeg
  TartanGround  <root>/<env>/Data_anymal/<traj>/image_lcam_front/NNNNNN_lcam_front.png
  Isaac ANYmal  <root>/<episode>/image/NNNNNN.jpg   (root = …/sim_anymal_isaac/sequences/all)

Usage:
  python tools/legged/cache_dino_features.py --dataset isaac --gpus 0,1,2,3,4,5,6,7 --batch-size 64
Smoke (few images, one seq):
  python tools/legged/cache_dino_features.py --dataset isaac --gpus 0 --limit 16 --max-seqs 1
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from feelwm.config import data_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from cache_dino_features_reg import (  # noqa: E402
    load_reg_encoder, preprocess, forward_patches, EXPECTED_SHAPE,
)

GT_ROOT = data_path("grand_tour")
TG_ROOT = data_path("tartanground")
ISAAC_ROOT = data_path("isaac", "sim_anymal_isaac", "sequences", "all")

DINOV3_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DEFAULT_ENCODER = {"gt": "reg", "tg": "reg", "isaac": "dinov3"}
DEFAULT_FEAT_VERSION = {
    "gt": "gt_dinov2reg_v1", "tg": "tg_dinov2reg_v1", "isaac": "anymal_isaac_dinov3_v1",
}


def find_sequences(dataset: str, root: Path) -> list[tuple[Path, Path]]:
    """Return (seq_dir, image_dir) pairs. seq_dir is where features/ is written (next to labels)."""
    out = []
    if dataset == "gt":
        for lbl in sorted(root.glob("*/labels_legged.npz")):
            img = lbl.parent / "images" / "hdr_front"
            if img.is_dir():
                out.append((lbl.parent, img))
    elif dataset == "tg":
        for lbl in sorted(root.glob("*/Data_anymal/*/labels_legged.npz")):
            img = lbl.parent / "image_lcam_front"
            if img.is_dir():
                out.append((lbl.parent, img))
    else:  # isaac
        for lbl in sorted(root.glob("*/labels_legged.npz")):
            img = lbl.parent / "image"
            if img.is_dir():
                out.append((lbl.parent, img))
    return out


def _index_of(path: Path) -> int:
    return int(path.stem.split("_")[0])   # "000123" or "000123_lcam_front" → 123


# ---------------------------------------------------------------------------
# DINOv3 encoder (ViT-B/16 @ 256² → 1 CLS + 4 registers + 256 patches, 768-dim)
# ---------------------------------------------------------------------------

def load_dinov3_encoder(device: str):
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(DINOV3_ID)
    except Exception as e:
        raise SystemExit(
            f"cannot load {DINOV3_ID}: {e}\n"
            "DINOv3 is a GATED repo — request access on its HF model page, then provide a "
            "token via `export HF_TOKEN=…` or `hf auth login`."
        )
    model.eval().to(device)
    n_special = 1 + int(getattr(model.config, "num_register_tokens", 0))  # CLS + registers
    return model, n_special


def preprocess_dinov3(img: Image.Image) -> torch.Tensor:
    """Bicubic SQUARE resize to 256 + ImageNet normalize (project square-resize convention,
    at DINOv3's native pretrain resolution; 256/16 = 16 → 16×16 = 256 patch tokens)."""
    img = img.convert("RGB").resize((256, 256), Image.BICUBIC)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0   # np.array: writable copy
    return (x - _IMAGENET_MEAN) / _IMAGENET_STD


@torch.no_grad()
def forward_patches_dinov3(enc, batch: torch.Tensor) -> np.ndarray:
    model, n_special = enc
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h = model(pixel_values=batch).last_hidden_state       # (B, n_special+256, 768)
    patches = h[:, n_special:, :].float()                      # (B, 256, 768)
    return patches.cpu().numpy().astype(np.float16)


# ---------------------------------------------------------------------------


def process_sequence(seq_dir, img_dir, feat_version, fwd, prep, device, batch_size, limit, prefix):
    out_dir = seq_dir / "features" / feat_version
    feat_file, valid_file = out_dir / "features.npy", out_dir / "features_valid.npy"

    n_anchor = len(np.load(seq_dir / "labels_legged.npz")["valid_mask"])
    if feat_file.exists() and valid_file.exists() and not limit:
        v = np.load(valid_file)
        if len(v) == n_anchor and v.all():
            return 0, n_anchor, 0   # complete — resume skip

    paths = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.jpg"))
    paths = sorted(paths, key=_index_of)
    if limit:
        paths = paths[:limit]

    feats = np.zeros((n_anchor, *EXPECTED_SHAPE), np.float16)
    valid = np.zeros(n_anchor, bool)
    errors = 0
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        imgs, idxs = [], []
        for p in batch:
            i = _index_of(p)
            if i >= n_anchor:
                continue
            try:
                imgs.append(prep(Image.open(p))); idxs.append(i)
            except Exception as e:
                print(f"{prefix}[WARN] {p.name}: {e}", file=sys.stderr, flush=True); errors += 1
        if not imgs:
            continue
        out = fwd(torch.stack(imgs).to(device))   # (B,256,768) fp16
        for i, f in zip(idxs, out):
            if f.shape == EXPECTED_SHAPE and np.all(np.isfinite(f)):
                feats[i] = f; valid[i] = True
            else:
                errors += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_file, feats)
    np.save(valid_file, valid)
    return int(valid.sum()), n_anchor, errors


def worker_fn(gpu_id, seqs, feat_version, encoder, batch_size, limit):
    device = f"cuda:{gpu_id}"; prefix = f"[GPU{gpu_id}] "
    print(f"{prefix}loading {encoder} encoder…", flush=True)
    if encoder == "reg":
        enc = load_reg_encoder(device)
        fwd, prep = (lambda b: forward_patches(enc, b)), preprocess
    else:
        enc = load_dinov3_encoder(device)
        fwd, prep = (lambda b: forward_patches_dinov3(enc, b)), preprocess_dinov3
    tot_new = tot = tot_err = 0
    for seq_dir, img_dir in seqs:
        new, n, err = process_sequence(seq_dir, img_dir, feat_version, fwd, prep, device,
                                       batch_size, limit, prefix)
        print(f"{prefix}[OK] {seq_dir.name}: {new}/{n}" + (f" ({err} err)" if err else ""), flush=True)
        tot_new += new; tot += n; tot_err += err
    print(f"{prefix}done. cached={tot_new}/{tot} errors={tot_err}", flush=True)
    return tot_new, tot, tot_err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["gt", "tg", "isaac"])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--feat-version", default=None,
                    help="default gt_dinov2reg_v1 / tg_dinov2reg_v1 / anymal_isaac_dinov3_v1")
    ap.add_argument("--encoder", default=None, choices=["reg", "dinov3"],
                    help="default: reg for gt/tg, dinov3 for isaac")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="cap images/seq (smoke test)")
    ap.add_argument("--max-seqs", type=int, default=0, help="cap #sequences (smoke test)")
    args = ap.parse_args()

    roots = {"gt": GT_ROOT, "tg": TG_ROOT, "isaac": ISAAC_ROOT}
    root = args.root or roots[args.dataset]
    feat_version = args.feat_version or DEFAULT_FEAT_VERSION[args.dataset]
    encoder = args.encoder or DEFAULT_ENCODER[args.dataset]
    seqs = find_sequences(args.dataset, root)
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    if not seqs:
        sys.exit(f"no sequences with labels+images under {root}")

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    print(f"{len(seqs)} sequences → {len(gpu_ids)} GPU(s), "
          f"feat_version={feat_version}, encoder={encoder}", flush=True)
    t0 = time.time()
    if len(gpu_ids) == 1:
        worker_fn(gpu_ids[0], seqs, feat_version, encoder, args.batch_size, args.limit)
    else:
        chunks = [seqs[i::len(gpu_ids)] for i in range(len(gpu_ids))]
        with mp.get_context("spawn").Pool(len(gpu_ids)) as pool:
            res = pool.starmap(worker_fn, [(g, c, feat_version, encoder, args.batch_size, args.limit)
                                           for g, c in zip(gpu_ids, chunks)])
        print(f"\nall done in {(time.time()-t0)/60:.1f} min. "
              f"cached={sum(r[0] for r in res)}/{sum(r[1] for r in res)} errors={sum(r[2] for r in res)}",
              flush=True)


if __name__ == "__main__":
    main()
