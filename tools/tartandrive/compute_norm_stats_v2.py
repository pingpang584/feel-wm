"""Compute per-channel normalization statistics for norm_stats/v2.json.

Key differences from v1:
  - Uses splits/v2.json (48 train sequences; 3 brake-corrupt sequences removed).
  - ke_rate (proprio_feature ch 3): signed-log1p applied BEFORE computing mean/std.
  - steer_cmd (proprio_window ch 13): divided by 415 to [-1, 1] BEFORE mean/std.
  - brake (proprio_window ch 12): clamp(0, 500) BEFORE mean/std.
  - vx (proprio_window ch 6): clamp(-10, 30) BEFORE mean/std.
  - vy (proprio_window ch 7): clamp(-5, 5) BEFORE mean/std.
  - All other channels: identical to v1 treatment.

Transform choices come from a per-channel distribution diagnostic of the v1 stats
(heavy-tailed ke_rate, raw-unit steer_cmd, outlier spikes in brake / vx / vy).

At DataLoader time, channels are pre-processed (same transforms as above)
BEFORE z-score normalization is applied. The pre-processing and the v2
mean/std are always applied together as a pair.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from feelwm.config import data_path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = REPO_ROOT / "data" / "splits" / "v2.json"
OUT_PATH = REPO_ROOT / "data" / "norm_stats" / "v2.json"

FEATURE_NAMES = ["roughness", "slip_ratio", "angular_instability", "ke_rate", "slope_power"]
WINDOW_NAMES = [
    "imu_ax", "imu_ay", "imu_az",
    "imu_gx", "imu_gy", "imu_gz",
    "vx", "vy", "yaw_rate",
    "pitch_rad", "roll_rad",
    "throttle", "brake", "steer_cmd",
]
ACTION_NAMES = ["dx", "dy", "dyaw"]

# Pre-processing transforms applied before z-score.
# DataLoader must apply the same transforms in the same order.
PRETRANSFORMS = {
    "feature": {
        3: ("signed_log1p", {}),      # ke_rate
    },
    "window": {
        6:  ("clamp", {"lo": -10.0, "hi": 30.0}),   # vx
        7:  ("clamp", {"lo":  -5.0, "hi":  5.0}),   # vy
        12: ("clamp", {"lo":   0.0, "hi": 500.0}),  # brake
        13: ("scale", {"factor": 1.0 / 415.0}),     # steer_cmd → [-1, 1]
    },
}


def _apply_feature_pretransform(arr: np.ndarray) -> np.ndarray:
    """arr: (N, 5). Returns transformed copy."""
    out = arr.copy().astype(np.float64)
    for ch, (fn, kw) in PRETRANSFORMS["feature"].items():
        if fn == "signed_log1p":
            out[:, ch] = np.sign(out[:, ch]) * np.log1p(np.abs(out[:, ch]))
    return out


def _apply_window_pretransform(arr: np.ndarray) -> np.ndarray:
    """arr: (M, 14) where M = N_valid * T_win. Returns transformed copy."""
    out = arr.copy().astype(np.float64)
    for ch, (fn, kw) in PRETRANSFORMS["window"].items():
        if fn == "clamp":
            out[:, ch] = np.clip(out[:, ch], kw["lo"], kw["hi"])
        elif fn == "scale":
            out[:, ch] = out[:, ch] * kw["factor"]
    return out


class WelfordAccumulator:
    def __init__(self, n_channels: int):
        self.n = np.zeros(n_channels, dtype=np.float64)
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.M2 = np.zeros(n_channels, dtype=np.float64)

    def update(self, arr: np.ndarray) -> None:
        if arr.ndim == 1:
            arr = arr[:, None]
        arr = arr.astype(np.float64)
        for c in range(arr.shape[1]):
            col = arr[:, c]
            finite = col[np.isfinite(col)]
            if len(finite) == 0:
                continue
            b_n = len(finite)
            b_mean = float(np.mean(finite))
            b_M2 = float(np.var(finite, ddof=0)) * b_n
            combined_n = self.n[c] + b_n
            delta = b_mean - self.mean[c]
            self.mean[c] += delta * b_n / combined_n
            self.M2[c] += b_M2 + delta ** 2 * self.n[c] * b_n / combined_n
            self.n[c] = combined_n

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        std = np.where(self.n > 1, np.sqrt(self.M2 / (self.n - 1)), 1.0)
        std = np.where(std < 1e-6, 1.0, std)
        mean = np.where(self.n > 0, self.mean, 0.0)
        return mean, std


def compute(out_root: Path, split_path: Path) -> dict:
    split = json.loads(split_path.read_text())
    train_seqs = split["train"]

    acc_feat = WelfordAccumulator(5)
    acc_win = WelfordAccumulator(14)
    acc_act = WelfordAccumulator(3)
    acc_pitch = WelfordAccumulator(1)
    acc_roll = WelfordAccumulator(1)
    acc_trav = WelfordAccumulator(1)

    n_total = n_valid_total = 0
    missing = []

    for seq in sorted(train_seqs):
        npz_path = out_root / seq / "labels.npz"
        if not npz_path.exists():
            missing.append(seq)
            continue

        z = np.load(npz_path)
        vm = z["valid_mask"].astype(bool)
        n_total += int(len(vm))
        n_valid_total += int(vm.sum())

        if vm.sum() == 0:
            continue

        feat_raw = z["proprio_feature"][vm]
        win_raw = z["proprio_window"][vm].reshape(-1, 14)
        act = z["action"][vm]

        acc_feat.update(_apply_feature_pretransform(feat_raw))
        acc_win.update(_apply_window_pretransform(win_raw))
        acc_act.update(act)
        acc_pitch.update(z["pitch_deg"][vm, None])
        acc_roll.update(z["roll_deg"][vm, None])
        acc_trav.update(z["trav_cost_teacher"][vm, None])

    if missing:
        print(f"WARNING: {len(missing)} sequences missing: {missing}")

    feat_mean, feat_std = acc_feat.result()
    win_mean, win_std = acc_win.result()
    act_mean, act_std = acc_act.result()
    pitch_mean, pitch_std = acc_pitch.result()
    roll_mean, roll_std = acc_roll.result()
    trav_mean, trav_std = acc_trav.result()

    stats = {
        "version": "v2",
        "pipeline_version": split["pipeline_version"],
        "split_version": "v2",
        "created_at": __import__("datetime").date.today().isoformat(),
        "n_train_sequences": len(train_seqs),
        "n_total_anchors": n_total,
        "n_valid_anchors": n_valid_total,
        "pretransforms": {
            "proprio_feature": {
                "ke_rate (ch3)": "signed_log1p: x -> sign(x)*log1p(|x|)"
            },
            "proprio_window": {
                "vx (ch6)": "clamp(-10, 30)",
                "vy (ch7)": "clamp(-5, 5)",
                "brake (ch12)": "clamp(0, 500)",
                "steer_cmd (ch13)": "scale by 1/415 -> [-1, 1]",
            },
            "note": (
                "Apply these transforms BEFORE z-score at DataLoader time. "
                "mean/std in this file are computed on the pre-transformed values."
            ),
        },
        "proprio_feature": {
            "channels": FEATURE_NAMES,
            "mean": feat_mean.tolist(),
            "std": feat_std.tolist(),
        },
        "proprio_window": {
            "channels": WINDOW_NAMES,
            "mean": win_mean.tolist(),
            "std": win_std.tolist(),
        },
        "action": {
            "channels": ACTION_NAMES,
            "mean": act_mean.tolist(),
            "std": act_std.tolist(),
        },
        "pitch_deg": {"mean": float(pitch_mean[0]), "std": float(pitch_std[0])},
        "roll_deg":  {"mean": float(roll_mean[0]),  "std": float(roll_std[0])},
        "trav_cost_teacher": {"mean": float(trav_mean[0]), "std": float(trav_std[0])},
    }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root",
                    default=str(data_path("tartandrive2_processed")))
    ap.add_argument("--split", default=str(SPLIT_PATH))
    ap.add_argument("--output", default=str(OUT_PATH))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    split_path = Path(args.split)
    output_path = Path(args.output)

    print(f"Computing norm stats v2 from {out_root}")
    print(f"Split: {split_path}")

    stats = compute(out_root, split_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {output_path}")

    print(f"\n  n_train_sequences : {stats['n_train_sequences']}")
    print(f"  n_valid_anchors   : {stats['n_valid_anchors']}")

    print("\n  proprio_feature (mean | std) [ke_rate is post-signed_log1p]:")
    for name, m, s in zip(
        stats["proprio_feature"]["channels"],
        stats["proprio_feature"]["mean"],
        stats["proprio_feature"]["std"],
    ):
        flag = " ← signed_log1p" if name == "ke_rate" else ""
        print(f"    {name:25s}  {m:+.4f}  {s:.4f}{flag}")

    print("\n  proprio_window (mean | std) [transforms applied]:")
    transforms = {"brake": " ← clamp(0,500)", "steer_cmd": " ← /415",
                  "vx": " ← clamp(-10,30)", "vy": " ← clamp(-5,5)"}
    for name, m, s in zip(
        stats["proprio_window"]["channels"],
        stats["proprio_window"]["mean"],
        stats["proprio_window"]["std"],
    ):
        flag = transforms.get(name, "")
        print(f"    {name:25s}  {m:+.4f}  {s:.4f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
