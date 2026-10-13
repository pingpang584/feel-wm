"""Per-dataset / per-split offline eval for the legged WM (TartanGround + GrandTour).

Disentangles the held-out roughness/slip failure: domain-gap/OOD vs intrinsic difficulty.
Loads ONE checkpoint and reports free-running rollout R² (same path as training val:
forward_rollout, no oracle) per channel for three groups:
  gt_train  — real, IN-SAMPLE (GrandTour train missions)
  gt_val    — real, HELD-OUT  (December missions; reproduces the reported val R²)
  tg        — sim,  IN-SAMPLE (TartanGround)
R² is computed in each group's own (per-dataset) normalized space; affine-invariant so this
matches the denormalized R². Subsample counts keep it tractable.

Usage:
  python tools/legged/eval_per_dataset.py --config configs/legged_tg_gt.yaml \
      --checkpoint ${OUTPUT_ROOT}/legged_tg_gt/feelwm/checkpoint_best.pt --device cuda \
      --n-gt-train 12 --n-tg 16 --max-windows 4000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from feelwm.config import add_variant_arg, load_config
from feelwm.data.legged_dataset import LeggedDataset, LeggedDatasetConfig, discover_legged_specs
from feelwm.data.normalize import NormStats, denormalize_feature
from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel

CHANNELS = ["roughness", "angular_instability", "slope_power", "slip", "cost_of_transport"]


def r2_per_channel(pred, target, mask):
    """pred/target (M,5), mask (M,) bool. Returns (5,) R² = 1 - SSres/SStot per channel."""
    p, t = pred[mask], target[mask]
    ss_res = ((p - t) ** 2).sum(0)
    ss_tot = ((t - t.mean(0)) ** 2).sum(0)
    return (1.0 - ss_res / np.clip(ss_tot, 1e-12, None))


@torch.no_grad()
def eval_group(model, specs, ld_cfg, device, max_windows, stats):
    ds = LeggedDataset(specs, ld_cfg)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
    preds, targets, masks = [], [], []
    seen = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model.forward_rollout(
            batch, batch["action_sequence"],
            proprio_window_target=batch.get("proprio_window_target"),
        )
        preds.append(out["f_prop_preds"].cpu().numpy())
        targets.append(batch["f_prop_target"].cpu().numpy())
        masks.append(batch["valid_target_mask"].cpu().numpy())
        seen += batch["f_prop_target"].shape[0]
        if max_windows and seen >= max_windows:
            break
    P = np.concatenate(preds).reshape(-1, 5)
    T = np.concatenate(targets).reshape(-1, 5)
    M = np.concatenate(masks).reshape(-1).astype(bool)
    # MAE in denormalized (physical) units; legged norm is pure z-score per dataset.
    mae = np.abs(denormalize_feature(T[M], stats) - denormalize_feature(P[M], stats)).mean(axis=0)
    return r2_per_channel(P, T, M), mae, len(ds), int(M.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/legged_tg_gt.yaml")
    ap.add_argument("--checkpoint", default="runs/legged_tg_gt/feelwm/checkpoint_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-gt-train", type=int, default=12)
    ap.add_argument("--n-tg", type=int, default=16)
    ap.add_argument("--max-windows", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    add_variant_arg(ap)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config((repo / args.config), args.variant)
    d, m = cfg["data"], cfg["model"]

    def rp(x):
        return str(x if Path(x).is_absolute() else repo / x)

    model = TerrainWorldModel(ModelConfig(**{k: v for k, v in m.items()
                                             if k in ModelConfig.__dataclass_fields__}))
    ck = torch.load(rp(args.checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval().to(args.device)
    print(f"loaded {args.checkpoint} (best_val_r2={ck.get('best_val_r2'):.4f}, "
          f"saved after epoch {ck.get('next_epoch', '?')-1 if ck.get('next_epoch') else '?'})")

    specs = discover_legged_specs(
        grandtour_root=d["grandtour_root"], tartanground_root=d["tartanground_root"],
        feat_version_gt=d["feat_version_gt"], feat_version_tg=d["feat_version_tg"])
    heldout = set(d["heldout_grandtour_missions"])
    gt_train = [s for s in specs if s.tag == "gt" and s.name not in heldout]
    gt_val = [s for s in specs if s.tag == "gt" and s.name in heldout]
    tg = [s for s in specs if s.tag == "tg"]

    rng = np.random.default_rng(args.seed)
    def sample(lst, n):
        if n and len(lst) > n:
            idx = rng.choice(len(lst), n, replace=False)
            return [lst[i] for i in sorted(idx)]
        return lst

    norm = {"gt": rp(d["norm_stats_path_gt"]), "tg": rp(d["norm_stats_path_tg"])}
    ld_cfg = LeggedDatasetConfig(
        context_len=m["context_len"], rollout_horizon=m["rollout_horizon"],
        n_patch=m["n_patch"], d_vis=m["d_vis"],
        require_vis_context=True, require_vis_target=True, norm_stats=norm)

    stats_by_tag = {tag: NormStats.from_json(path) for tag, path in norm.items()}
    groups = [
        ("gt_train (real, in-sample)", sample(gt_train, args.n_gt_train), "gt"),
        ("gt_val   (real, HELD-OUT)", gt_val, "gt"),
        ("tg       (sim,  in-sample)", sample(tg, args.n_tg), "tg"),
    ]
    print(f"\n{'group':28} {'N_win':>7} {'valid':>7} | " + " ".join(f"{c[:6]:>7}" for c in CHANNELS) + f" {'mean4':>7}")
    print("-" * 100)
    for name, specs_g, tag in groups:
        r2, mae, nwin, nvalid = eval_group(model, specs_g, ld_cfg, args.device,
                                           args.max_windows, stats_by_tag[tag])
        mean4 = np.mean([r2[0], r2[1], r2[2], r2[4]])   # exclude slip
        print(f"{name:28} {nwin:>7} {nvalid:>7} | "
              + " ".join(f"{v:>7.3f}" for v in r2) + f" {mean4:>7.3f}")
        print(f"{'  └ MAE (denorm units)':28} {'':>7} {'':>7} | "
              + " ".join(f"{v:>7.4f}" for v in mae))
    print("\nmean4 = mean(roughness, angular_instability, slope_power, cost_of_transport); slip excluded")


if __name__ == "__main__":
    main()
