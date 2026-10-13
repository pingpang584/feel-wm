"""Compute per-dataset legged norm-stats from labels_legged.npz files.

Per-channel z-score stats over VALID anchors only, written with `feature_set="legged"` so
feelwm/data/normalize.py skips the wheeled pretransforms. Per-dataset (GrandTour / TartanGround)
and TRAIN-split only — never a shared sim+real norm.

Usage:
  python tools/legged/compute_norm_stats.py <labels_glob...> --out data/norm_stats/legged_gt.json
  # e.g. '${DATA_ROOT}/grand_tour/*/labels_legged.npz'
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="glob(s) for labels_legged.npz")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="substrings; drop matching paths (held-out eval seqs — TRAIN-only norm)")
    args = ap.parse_args()

    paths = sorted({p for g in args.globs for p in glob.glob(g)})
    paths = [p for p in paths if not any(x in p for x in args.exclude)]
    if not paths:
        raise SystemExit(f"no labels matched: {args.globs}")

    feats, wins, acts = [], [], []
    n_total = n_valid = 0
    for p in paths:
        z = np.load(p)
        v = z["valid_mask"].astype(bool)
        n_total += len(v)
        n_valid += int(v.sum())
        feats.append(z["proprio_feature"][v])                       # (n,5)
        wins.append(z["proprio_window"][v].reshape(-1, z["proprio_window"].shape[-1]))  # (n*Tw,53)
        acts.append(z["action"][v])                                  # (n,3)
    F = np.concatenate(feats); W = np.concatenate(wins); A = np.concatenate(acts)

    def stats(x):
        # ignore non-finite per column; std floored to avoid div-by-zero on constant channels
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(float).tolist(), std.astype(float).tolist()

    fm, fs = stats(F); wm, ws = stats(W); am, as_ = stats(A)
    out = {
        "feature_set": "legged",
        "n_sequences": len(paths),
        "n_total_anchors": n_total,
        "n_valid_anchors": n_valid,
        "channels": ["roughness", "angular_instability", "slope_power", "slip", "cost_of_transport"],
        "proprio_feature": {"mean": fm, "std": fs},
        "proprio_window": {"mean": wm, "std": ws},   # 53-dim
        "action": {"mean": am, "std": as_},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}  from {len(paths)} seq, {n_valid}/{n_total} valid anchors")
    print(f"  feature mean={np.round(fm,3)}")
    print(f"  window dim={len(wm)}  action mean={np.round(am,3)}")


if __name__ == "__main__":
    main()
