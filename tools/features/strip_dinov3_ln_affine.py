#!/usr/bin/env python3
"""Derive a loss-scale-PARITY DINOv3 cache by inverting the encoder's final LayerNorm affine.

WHY (the DINOv3-vs-DINOv2 encoder ablation on the wheeled corpora).
``visual_mse`` in feelwm/training/losses.py is a RAW-feature MSE:

    visual_loss = mean((z_pred - z_target)**2) + lambda_cos * (1 - cos)
    total       = visual_loss + lambda_prop * proprio_loss + lambda_trav * trav_loss

The proprio term is computed on z-scored labels and the trav term is a BCE, so BOTH are
invariant to the visual cache's absolute scale -- but ``visual_mse`` is proportional to it.
The incumbent ``dinov2reg_v1`` cache is RAE's ``Dinov2withNorm(normalize=True)``, i.e. the
final LayerNorm affine is REMOVED, leaving plain LN output at variance 1.0. The DINOv3
cacher keeps the affine (tools/features/cache_dinov3.py, design choice 3 -- it was kept
only because no RAE decoder exists to demand otherwise), which lands at variance ~0.17.

Measured on TartanDrive2, 5 sequences x 150 frames:

    cache           var(elem)   mean|x|   MSE(z_t, z_t+10)
    dinov2reg_v1       1.0000    0.7284             0.4331
    dinov3_v1          0.1707    0.2780             0.0353     <- 5.9x / 12.3x smaller

Training the DINOv3 arm on the raw cache would therefore shrink ``visual_mse`` ~12x
against unchanged lambda_prop/lambda_trav -- an implicit loss REWEIGHTING riding along
with the encoder swap, i.e. a second variable in a comparison whose whole point is that
the encoder is the only one. Adam does not absorb this: it is scale-invariant to a global
loss factor, not to a change in the RATIO between loss terms.

WHAT THIS DOES. The final LayerNorm affine is elementwise, y = w * x_hat + b, so it is
exactly invertible from the cached features -- x_hat = (y - b) / w -- with NO GPU
re-encoding. The result is plain LN output at variance 1.0: the same convention as
dinov2reg_v1, restoring loss-scale parity and leaving the encoder as the only difference.
It is a per-dimension inverse, not a global rescale, so it also removes the affine's
per-channel anisotropy, which is what dinov2reg_v1's space likewise lacks.

Round-trip precision: the cache is float16 and w spans 0.247..11.19, so the inverse is
computed in float32 and re-cast. --verify reports the achieved variance and the RMS of the
float16 downcast error NORMALISED BY THE UNIT-VARIANCE SIGNAL -- not a per-element relative
error, which is meaningless where a feature crosses zero. Measured on real-robot frames the
downcast adds 2.1e-4 of 1 sigma (0.021%), against the 1.5e-4 (0.015%) float16 quantisation
floor that dinov2reg_v1 itself already carries, so the inverse costs ~nothing.

Only the CONSOLIDATED features.npy is produced (plus a copy of features_valid.npy) --
that is what feelwm/data/dataset.py and feelwm/data/sim_dataset.py mmap; the per-anchor .npy
files of the source cache are not needed and not duplicated.

Layouts (auto-detected, same as tools/features/write_cache_meta.py):
  flat     {feat_root}/{feat_version}/{seq}/features.npy               (TartanDrive2)
  per-seq  {seq_root}/{seq}/features/{feat_version}/features.npy       (sim corpora)

Usage (a multi-hundred-GB streaming copy -- expect a long run):
    # TD2: dinov3_v1 -> dinov3ln_v1
    python tools/features/strip_dinov3_ln_affine.py \
        --src ${DATA_ROOT}/tartandrive2_processed_features/dinov3_v1 \
        --dst ${DATA_ROOT}/tartandrive2_processed_features/dinov3ln_v1 \
        --verify

    # per-sequence corpus (e.g. Gazebo Husky): sim_dinov3_v1 -> sim_dinov3ln_v1 (in-corpus)
    python tools/features/strip_dinov3_ln_affine.py \
        --seq-root ${DATA_ROOT}/negs-ugv/sim_husky_v3/sequences/all \
        --src-version sim_dinov3_v1 --dst-version sim_dinov3ln_v1 --verify

Resumable: a sequence whose destination features.npy already exists is skipped unless
--overwrite. The source cache is never modified.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# Frames per write. 256x768 float32 is 0.75 MB/frame, so 512 keeps the working set
# ~400 MB while still writing in large sequential runs.
CHUNK = 512


# ---------------------------------------------------------------------------
# LayerNorm parameters
# ---------------------------------------------------------------------------
def load_ln_affine(model_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(weight, bias) of the encoder's FINAL LayerNorm, as float32 (D,)."""
    import torch
    from transformers import AutoModel

    enc = AutoModel.from_pretrained(model_id)
    mods = dict(enc.named_modules())
    ln = next((mods[k] for k in ("norm", "layernorm") if isinstance(mods.get(k), torch.nn.LayerNorm)), None)
    if ln is None:
        raise SystemExit(
            f"no final LayerNorm named 'norm'/'layernorm' on {type(enc).__name__}; "
            "inspect named_modules() before trusting any inverse"
        )
    if not ln.elementwise_affine:
        raise SystemExit(
            f"{model_id}'s final LayerNorm has no affine -- its cache is already at "
            "variance 1.0 and needs no stripping"
        )
    return (ln.weight.detach().float().numpy().copy(),
            ln.bias.detach().float().numpy().copy())


# ---------------------------------------------------------------------------
# Per-sequence transform
# ---------------------------------------------------------------------------
def strip_sequence(src: Path, dst: Path, w: np.ndarray, b: np.ndarray,
                   overwrite: bool, verify: bool) -> tuple[int, float | None, float | None]:
    """src/dst are the two features.npy paths. -> (n_frames, variance, downcast_rms).

    ``downcast_rms`` is normalised by the stripped features' own sigma (which is 1.0 by
    construction), so it reads directly as a fraction of the signal.
    """
    if dst.exists() and not overwrite:
        return 0, None, None

    a = np.load(src, mmap_mode="r")
    if a.shape[-1] != w.shape[0]:
        raise SystemExit(f"{src}: d_vis {a.shape[-1]} != LayerNorm width {w.shape[0]}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".npy.partial")     # never leave a half-written cache in place
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=a.shape)

    var_acc, n_acc, err_acc = 0.0, 0, 0.0
    for s in range(0, a.shape[0], CHUNK):
        y = np.asarray(a[s:s + CHUNK], dtype=np.float32)
        x = (y - b) / w
        out[s:s + CHUNK] = x.astype(np.float16)
        if verify:
            var_acc += float(x.var()) * x.shape[0]
            n_acc += x.shape[0]
            # Error the trainer actually sees: what the float16 downcast lost, measured
            # against the float32 inverse. NOT a per-element relative error -- features
            # cross zero, so that statistic is dominated by near-zero denominators.
            err_acc += float(((out[s:s + CHUNK].astype(np.float32) - x) ** 2).mean()) * x.shape[0]
    out.flush()
    del out
    tmp.replace(dst)

    src_valid = src.parent / "features_valid.npy"
    if src_valid.exists():
        shutil.copy2(src_valid, dst.parent / "features_valid.npy")

    if not verify:
        return a.shape[0], None, None
    var = var_acc / n_acc
    # normalise by the achieved sigma so the number reads as a fraction of the signal
    return a.shape[0], var, float(np.sqrt(err_acc / n_acc) / max(np.sqrt(var), 1e-12))


def resolve_pairs(args) -> list[tuple[Path, Path, str]]:
    """-> [(src features.npy, dst features.npy, seq name)]."""
    if args.seq_root:
        # .../{seq}/features/{version}/features.npy -> parents[2] is {seq}
        seqs = sorted(
            p.parents[2] for p in args.seq_root.glob(f"*/features/{args.src_version}/features.npy")
        )
        return [
            (s / "features" / args.src_version / "features.npy",
             s / "features" / args.dst_version / "features.npy",
             s.name)
            for s in seqs
        ]
    srcs = sorted(args.src.glob("*/features.npy"))
    return [(p, args.dst / p.parent.name / "features.npy", p.parent.name) for p in srcs]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_argument_group("flat layout (TartanDrive2)")
    g.add_argument("--src", type=Path, help="{feat_root}/{src_version}")
    g.add_argument("--dst", type=Path, help="{feat_root}/{dst_version}")
    h = ap.add_argument_group("per-sequence layout (sim corpora)")
    h.add_argument("--seq-root", type=Path)
    h.add_argument("--src-version", default="sim_dinov3_v1")
    h.add_argument("--dst-version", default="sim_dinov3ln_v1")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="measure output variance and float16 round-trip error (slower)")
    ap.add_argument("--limit", type=int, default=0, help="first N sequences only (smoke test)")
    args = ap.parse_args()

    if args.seq_root is None and not (args.src and args.dst):
        ap.error("give either --seq-root (per-sequence) or both --src and --dst (flat)")

    # Naming tripwire, mirroring tools/sim/cache_dino_features.py: writing stripped
    # features into a version name that reads as the raw cache is silent until a training
    # run comes out wrong.
    dst_tag = (args.dst.name if args.dst else args.dst_version).lower()
    src_tag = (args.src.name if args.src else args.src_version).lower()
    if dst_tag == src_tag:
        sys.exit("destination version equals the source version -- refusing to overwrite in place")
    if "dinov2" in dst_tag:
        sys.exit(f"refusing to write DINOv3 features into {dst_tag!r}")

    pairs = resolve_pairs(args)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        sys.exit("no source features.npy found -- check --src / --seq-root and the version names")

    w, b = load_ln_affine(args.model_id)
    print(f"{args.model_id}: final LayerNorm affine w[{w.shape[0]}] "
          f"min {w.min():.4f} max {w.max():.4f}, |b| mean {np.abs(b).mean():.4f}", flush=True)
    print(f"{len(pairs)} sequences: {src_tag} -> {dst_tag}\n", flush=True)

    t0 = time.time()
    n_done = n_skip = n_frames = 0
    vars_, rels = [], []
    for src, dst, name in pairs:
        n, var, rel = strip_sequence(src, dst, w, b, args.overwrite, args.verify)
        if n == 0:
            n_skip += 1
            print(f"[SKIP] {name} (destination exists)", flush=True)
            continue
        n_done += 1
        n_frames += n
        msg = f"[OK] {name}: {n} frames"
        if var is not None:
            vars_.append(var); msg += f", var {var:.4f}"
        if rel is not None:
            rels.append(rel); msg += f", downcast {rel:.2e} sigma"
        print(msg, flush=True)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min. "
          f"transformed={n_done} skipped={n_skip} frames={n_frames}", flush=True)
    if vars_:
        print(f"output variance: mean {np.mean(vars_):.4f} "
              f"(target 1.0 = the dinov2reg_v1 convention)", flush=True)
    if rels:
        print(f"float16 downcast error: max {max(rels):.2e} of 1 sigma "
              f"(dinov2reg_v1's own quantisation floor is 1.5e-4)", flush=True)


if __name__ == "__main__":
    main()
