"""Evaluate a trained full-model checkpoint on a split (default: OOD test).

Reuses the trainer's validation machinery (FullModelTrainer.run_validation) so the
metrics are computed identically to training-time val: pooled R² per active proprio
channel, traversability AUC/F1, and per-sequence breakdown.

Single-process / single-GPU only — do NOT launch with torchrun. Inference only.

Usage:
  python tools/eval_offline.py CONFIG CHECKPOINT [--variant NAME] [--split test] [--out PATH]

Example (OOD test split):
  python tools/eval_offline.py \
      configs/tartandrive.yaml --variant td2 \
      ${OUTPUT_ROOT}/tartandrive/td2/checkpoint_best.pt \
      --split test --out out/td2_test.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feelwm.config import add_variant_arg, load_config
from feelwm.data.dataset import DatasetConfig, TartanDriveDataset
from feelwm.data.normalize import NormStats
from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel
from feelwm.training.curriculum import CurriculumConfig, CurriculumStage
from feelwm.training.losses import LossConfig
from feelwm.training.scheduled_sampling import ScheduledSamplingConfig
from feelwm.training.trainer import (
    FullModelTrainer,
    TrainerConfig,
    _IndexedDataset,
    _dataclass_from_dict,
    _replace_dataclass,
    _resolve_paths,
    _validate_proprio_channels,
)


def _fmt_r2(r2: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:+.3f}" for k, v in r2.items())


def _stationary_seqs(data_cfg: DatasetConfig, split: str, speed_floor: float,
                     max_moving_frac: float) -> list[str]:
    """Names of (near-)fully-stationary sequences in `split`: those where fewer than
    `max_moving_frac` of valid anchors have the robot moving faster than `speed_floor`
    (m/s). On such sequences R²=1−MSE/Var is undefined (target variance → 0), so they are
    excluded from regression eval.

    Uses per-anchor single-step speed (|Δx,Δy| over one 0.1s anchor step) from raw
    labels.npz — a property of the trajectory itself, independent of the eval rollout
    stride/horizon. This separates 'parked the whole recording' (sam_loop, trash) from
    'slow but moving' (slag_heap) and from bimodal park+drive (turnpike_skydio)."""
    import numpy as np
    split_seqs = json.loads(Path(data_cfg.split_path).read_text())[split]
    dt = 0.1  # anchors are sampled at 10 Hz
    flagged: list[str] = []
    for seq in split_seqs:
        p = Path(data_cfg.processed_root) / seq / "labels.npz"
        if not p.exists():
            continue
        z = np.load(p)
        a = z["action"][:, :2].astype(np.float32)
        vm = (z["valid_mask"] if "valid_mask" in z.files else np.ones(len(a), np.uint8)).astype(bool)
        if not vm.any():
            continue
        speed = np.linalg.norm(a[vm], axis=1) / dt          # per-anchor m/s
        if float(np.mean(speed >= speed_floor)) < max_moving_frac:
            flagged.append(seq)
    return flagged


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--split", default="test", help="split key in the split json (default: test)")
    p.add_argument("--out", default=None, help="optional markdown report path")
    p.add_argument("--oracle-proprio", action="store_true",
                   help="teacher-force GT proprio (projected 4-dim) into the rollout conditioning: "
                        "ceiling for the feedback scheme (freeze-vs-aligned-feedback diagnostic)")
    p.add_argument("--encode-oracle", action="store_true",
                   help="feed encode(true future raw window) directly as the conditioning: the "
                        "clean ceiling for the aligned rollout scheme (embedding space)")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--keep-stationary", action="store_true",
                   help="do NOT drop fully-stationary sequences (e.g. sam_loop, trash). By "
                        "default they are excluded: their target variance → 0 makes R² undefined")
    p.add_argument("--stationary-speed", type=float, default=0.5,
                   help="an anchor is 'moving' if robot speed ≥ this (m/s); stride-independent")
    p.add_argument("--stationary-frac", type=float, default=0.05,
                   help="drop a sequence when fewer than this fraction of its anchors are moving")
    add_variant_arg(p)
    args = p.parse_args(argv)
    if args.oracle_proprio and args.encode_oracle:
        p.error("--oracle-proprio and --encode-oracle are mutually exclusive")

    cfg = load_config(Path(args.config), args.variant)
    cwd = Path.cwd()

    trainer_cfg: TrainerConfig = _dataclass_from_dict(TrainerConfig, _resolve_paths(cfg["training"], cwd))
    # Eval is inference-only — never open a wandb run (the config carries wandb_project for
    # training; building the trainer here would otherwise create a junk run).
    trainer_cfg.wandb_project = None
    if args.max_batches is not None:
        trainer_cfg.max_val_batches = args.max_batches
    data_cfg: DatasetConfig = _dataclass_from_dict(DatasetConfig, _resolve_paths(cfg["data"], cwd))
    model_cfg: ModelConfig = _dataclass_from_dict(ModelConfig, cfg["model"])
    loss_cfg: LossConfig = _dataclass_from_dict(LossConfig, cfg["loss"])
    _validate_proprio_channels(model_cfg, loss_cfg, data_cfg)

    excluded: list[str] = []
    if not args.keep_stationary:
        excluded = _stationary_seqs(data_cfg, args.split, args.stationary_speed, args.stationary_frac)
        if excluded:
            print(f"[eval] dropping {len(excluded)} fully-stationary sequence(s) from {args.split} "
                  f"(<{args.stationary_frac:.0%} anchors move ≥{args.stationary_speed}m/s, R² undefined): {excluded}")
    eval_data = TartanDriveDataset(
        _replace_dataclass(data_cfg, split=args.split, exclude_sequences=excluded)
    )
    stats = NormStats.from_json(data_cfg.norm_stats_path)
    pos_rate = eval_data.positive_rate()
    pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-12)

    # eval_data is passed as both train and val: run_validation never touches the
    # train dataset, and this avoids loading the 94k-anchor train split.
    trainer = FullModelTrainer(
        model=TerrainWorldModel(model_cfg),
        train_dataset=eval_data,
        val_dataset=eval_data,
        train_stats=stats,
        trainer_cfg=trainer_cfg,
        loss_cfg=loss_cfg,
        curriculum_cfg=_dataclass_from_dict(CurriculumConfig, cfg["curriculum"]),
        sampling_cfg=_dataclass_from_dict(ScheduledSamplingConfig, cfg["scheduled_sampling"]),
        pos_weight=pos_weight,
    )
    trainer.load_checkpoint(args.checkpoint)

    loader = DataLoader(
        _IndexedDataset(eval_data),
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        num_workers=trainer_cfg.num_workers,
        pin_memory=trainer.device.type == "cuda",
        drop_last=False,
    )

    # stage=FULL so all heads (visual/proprio/traversability) are scored.
    summary, per_seq = trainer.run_validation(
        loader, epoch=0, stage=CurriculumStage.FULL,
        oracle_proprio=args.oracle_proprio,
        encode_oracle_proprio=args.encode_oracle,
    )

    r2 = summary.get("r2", {})
    binsafe = summary.get("binary_safety", {})
    mode = model_cfg.traversability_input_mode
    print(f"\n=== eval: {args.config}")
    print(f"    checkpoint: {args.checkpoint}")
    print(f"    split={args.split}  N_anchors={len(eval_data):,}  trav_mode={mode}  pos_rate={pos_rate:.4f}")
    print(f"    pooled R²: {_fmt_r2(r2)}")
    if r2:
        print(f"    mean R²: {sum(r2.values()) / len(r2):+.4f}")
    mae = summary.get("mae", {})
    if mae:
        print("    pooled MAE (denorm, physical units): "
              + "  ".join(f"{k}={v:.4f}" for k, v in mae.items()))
    print(f"    traversability: AUC={binsafe.get('auc', float('nan')):.4f}  "
          f"F1={binsafe.get('f1', float('nan')):.4f}  N={binsafe.get('n', 0)}")

    # per-sequence (outlier sequences are reported separately, not hidden in the pool)
    chans = trainer.channel_names
    print("\n    per-sequence R²:")
    for seq, vals in sorted(per_seq.items()):
        row = trainer._r2_np(vals["pred"], vals["target"])
        print(f"      {seq:42s} N={len(vals['target']):>6} | "
              + "  ".join(f"{c}={v:+.3f}" for c, v in zip(chans, row)))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Eval — {Path(args.config).stem} on `{args.split}`",
            "",
            f"- checkpoint: `{args.checkpoint}`",
            f"- trav_mode: **{mode}**, N_anchors={len(eval_data):,}, pos_rate={pos_rate:.4f}",
            (f"- excluded (fully-stationary, R² undefined): {excluded}" if excluded
             else "- excluded: none"),
            f"- traversability: **AUC={binsafe.get('auc', float('nan')):.4f}**, "
            f"F1={binsafe.get('f1', float('nan')):.4f}, N={binsafe.get('n', 0)}",
            "",
            "## Pooled R² (primary metric)",
            "",
            "| " + " | ".join(chans) + " | mean |",
            "|" + "---:|" * (len(chans) + 1),
            "| " + " | ".join(f"{r2[c]:+.3f}" for c in chans)
            + f" | {sum(r2.values()) / max(len(r2), 1):+.3f} |",
            "",
        ]
        if mae:
            lines += [
                "## Pooled MAE (denormalized, physical units)",
                "",
                "| " + " | ".join(chans) + " |",
                "|" + "---:|" * len(chans),
                "| " + " | ".join(f"{mae[c]:.4f}" for c in chans) + " |",
                "",
            ]
        lines += [
            "## Per-sequence R²",
            "",
            "| Sequence | N | " + " | ".join(chans) + " |",
            "|---|---:|" + "---:|" * len(chans),
        ]
        for seq, vals in sorted(per_seq.items()):
            row = trainer._r2_np(vals["pred"], vals["target"])
            lines.append(f"| {seq} | {len(vals['target'])} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
        # also dump the raw summary json alongside
        out.write_text("\n".join(lines) + "\n")
        out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\n    wrote {out} (+ .json)")


if __name__ == "__main__":
    main(sys.argv[1:])
