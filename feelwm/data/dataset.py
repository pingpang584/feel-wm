"""TartanDrive v2 sliding-window dataset.

Each sample provides:
  context: k+1 frames [t-k, t]   (all must have valid_mask=1 and DINOv2 feature)
  target:  H frames   [t+1, t+H] (per-frame valid_target_mask signals usability)

Sequence boundaries are never crossed. Normalization is applied at load time
using norm_stats/v2.json (pre-transforms + z-score).

DINOv2 features are cached as {feat_root}/{feat_version}/{seq}/{anchor_idx}.npy,
float16, shape (N_patch, D_vis). Missing features are zeros; valid_target_mask
reflects their absence when require_vis_target=True.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import data_path
from .normalize import (
    NormStats,
    normalize_action,
    normalize_feature,
    normalize_window,
    resolve_channel_indices,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _pose_delta(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """Body-frame SE(2) pose-delta (Δx, Δy, Δyaw) from pose_a to pose_b.

    ΔT = T_a⁻¹ · T_b (the same convention as the label extractor), read off x/y translation
    and the yaw of the rotation. Computed in float64 for inversion stability.
    """
    rel = np.linalg.inv(pose_a.astype(np.float64)) @ pose_b.astype(np.float64)
    dyaw = np.arctan2(rel[1, 0], rel[0, 0])
    return np.array([rel[0, 3], rel[1, 3], dyaw], dtype=np.float32)


@dataclass
class DatasetConfig:
    split: str = "train"
    split_path: str = str(_REPO_ROOT / "data" / "splits" / "v2.json")
    processed_root: str = str(data_path("tartandrive2_processed"))
    binary_label_root: str = str(_REPO_ROOT / "data" / "binary_labels" / "v2")
    feat_root: str = str(data_path("tartandrive2_processed_features"))
    norm_stats_path: str = str(_REPO_ROOT / "data" / "norm_stats" / "v2.json")
    feat_version: str = "dinov2_v1"
    context_len: int = 3          # k; context window has k+1 frames
    rollout_horizon: int = 8      # H
    n_patch: int = 256            # DINOv2 ViT-B/14 at 224px: 16×16 patches
    d_vis: int = 768              # ViT-B feature dim
    require_vis_context: bool = True   # exclude anchors lacking DINOv2 context
    require_vis_target: bool = True    # valid_target_mask=0 when DINOv2 missing
    # Active proprio output channels by name (canonical order; subset of the 5).
    # None = all 5. f_prop_target and current_feature are sliced to these channels.
    active_proprio_channels: list[str] | None = None
    # --- Multi-stride / Δt-conditioning ---
    # include_dt=False reproduces the legacy fixed-0.1s, 3-dim-action behavior exactly
    # (uses the precomputed `action`, no Δt channel, no horizon mask, stride 1).
    # include_dt=True: action = [normalize(pose_delta/stride), Δt] (4-dim); each rollout
    # step spans `stride` anchors (Δt = stride·0.1s); emits `proprio_horizon_mask`.
    include_dt: bool = False
    train_strides: list[int] = field(default_factory=lambda: [1])  # anchor strides (train split)
    val_stride: int = 1                # single stride for val/test loops
    proprio_horizon_s: float = 1.0     # proprio/trav supervised only where cum. Δt ≤ this
    # Multi-stride epoch sizing (train split only; val/test always use val_stride):
    #   "all"    – index every (anchor, stride) pair: one epoch sees each anchor at every
    #              stride, so the epoch is ×len(train_strides) larger (legacy behavior).
    #   "random" – index each anchor once; __getitem__ samples one of that anchor's VALID
    #              strides uniformly per draw. Epoch = #anchors; stride coverage spreads
    #              across epochs. Rationale: the reference recipe trained on ~3M samples
    #              ≈ 20 random-stride epochs, so "all"-mode 20 epochs (~12M) is 4× that
    #              budget for ~4× the wall-clock. Draws use Python's `random` (per-worker
    #              seeded by the DataLoader), so stride picks vary across epochs and workers.
    stride_sampling: str = "all"
    # Sequences to drop from this split at load time (by exact name). Used to exclude
    # fully-stationary val/test recordings (e.g. sam_loop_…12-42-17, trash_…12-03-21)
    # from regression eval, where R²=1−MSE/Var is undefined as target variance → 0.
    # Reversible/non-destructive: the v2 split file and norm stats are untouched.
    exclude_sequences: list[str] = field(default_factory=list)
    # Encoder-input column subset of proprio_window (column order of the extracted window).
    # None = all 14. Applied AFTER normalization, so the full-schema norm stats (and the
    # wheeled per-column clips) treat the kept columns identically to a full-input run —
    # a portable-input ablation differs from its baseline ONLY in what the encoder sees.
    # Portable-11 (= sim/Husky window schema cols 0–10): [0,...,10] drops the
    # TartanDrive-only actuator columns (throttle, brake, steer). model.d_raw must match.
    window_columns: list[int] | None = None

    def __post_init__(self) -> None:
        if any(s < 1 for s in self.train_strides) or self.val_stride < 1:
            raise ValueError("strides must be >= 1")
        if not self.include_dt and any(s != 1 for s in self.train_strides):
            raise ValueError(
                "train_strides != [1] requires include_dt=True "
                "(the Δt channel is needed to disambiguate stride)"
            )
        if self.stride_sampling not in {"all", "random"}:
            raise ValueError(f"unknown stride_sampling={self.stride_sampling!r}")
        if self.window_columns is not None:
            if len(self.window_columns) == 0:
                raise ValueError("window_columns must be None or non-empty")
            if len(set(self.window_columns)) != len(self.window_columns):
                raise ValueError("window_columns has duplicate indices")
            if any(c < 0 for c in self.window_columns):
                raise ValueError("window_columns indices must be >= 0")


class _SeqRecord(NamedTuple):
    """Per-sequence data held in memory after normalization."""
    name: str
    valid_mask: np.ndarray      # (N,) uint8
    feature: np.ndarray         # (N, 5) float32, normalized
    window: np.ndarray          # (N, T_win, 14) float32, normalized
    action: np.ndarray          # (N, 3) float32, normalized
    binary_unsafe: np.ndarray   # (N,) uint8 from safety.npz
    binary_valid: np.ndarray    # (N,) uint8 safety.npz valid_mask
    feat_dir: Path
    cached_set: frozenset       # set of int anchor positions with .npy files
    features: np.ndarray | None # mmap'd (N, N_patch, d_vis) float16 when consolidated
    pose: np.ndarray | None     # (N, 4, 4) float32 body-to-world SE(3); for strided actions


class TartanDriveDataset(Dataset):
    """Sliding-window dataset over TartanDrive v2 sequences.

    Args:
        config: DatasetConfig controlling split, paths, context/horizon sizes.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.cfg = config
        self.stats = NormStats.from_json(config.norm_stats_path)
        if config.active_proprio_channels is None:
            self._chan_idx = list(range(5))
        else:
            self._chan_idx = resolve_channel_indices(config.active_proprio_channels)
        self._n_chan = len(self._chan_idx)
        self._records: list[_SeqRecord] = []
        # (seq_id, anchor_t, stride). In random stride-sampling mode (train split) the
        # third element is instead a tuple of that anchor's valid strides, and
        # __getitem__ draws one uniformly per access.
        self._index: list[tuple[int, int, int | tuple[int, ...]]] = []
        self._build()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        split_data = json.loads(Path(self.cfg.split_path).read_text())
        seqs = split_data[self.cfg.split]
        excluded = set(self.cfg.exclude_sequences)
        if excluded:
            kept = [s for s in seqs if s not in excluded]
            dropped = [s for s in seqs if s in excluded]
            if dropped:
                print(f"[dataset] {self.cfg.split}: excluding {len(dropped)} sequence(s): {dropped}")
            seqs = kept
        k = self.cfg.context_len
        strides = self._strides()

        for seq in seqs:
            rec = self._load_seq(seq)
            sid = len(self._records)
            self._records.append(rec)
            N = len(rec.valid_mask)
            if self.cfg.include_dt and rec.pose is None:
                raise ValueError(
                    f"include_dt=True requires 'pose' in labels.npz (missing for {seq})"
                )

            if self._random_stride():
                # Anchor-level index: one entry per anchor carrying its valid strides.
                for t in range(min(strides) * k, N):
                    valid = tuple(
                        s for s in strides if t >= k * s and self._context_ok(rec, t, s)
                    )
                    if valid:
                        self._index.append((sid, t, valid))
            else:
                for s in strides:
                    for t in range(k * s, N):
                        if self._context_ok(rec, t, s):
                            self._index.append((sid, t, s))

    def _context_ok(self, rec: _SeqRecord, t: int, s: int) -> bool:
        """All k+1 context positions [t-k*s, t] at stride s are valid (and cached)."""
        ctx = range(t - self.cfg.context_len * s, t + 1, s)
        if any(rec.valid_mask[j] != 1 for j in ctx):
            return False
        if self.cfg.require_vis_context and any(j not in rec.cached_set for j in ctx):
            return False
        return True

    def _random_stride(self) -> bool:
        return self.cfg.stride_sampling == "random" and self.cfg.split == "train"

    def _strides(self) -> list[int]:
        """Anchor strides for the current split (train multi-stride; val/test single)."""
        if self.cfg.split == "train":
            return list(self.cfg.train_strides)
        return [self.cfg.val_stride]

    def _load_seq(self, seq: str) -> _SeqRecord:
        proc = Path(self.cfg.processed_root)
        blr = Path(self.cfg.binary_label_root)
        feat_root = Path(self.cfg.feat_root)

        z = np.load(proc / seq / "labels.npz", allow_pickle=False)
        feature = normalize_feature(z["proprio_feature"].astype(np.float32), self.stats)
        window = normalize_window(z["proprio_window"].astype(np.float32), self.stats)
        if self.cfg.window_columns is not None:
            if max(self.cfg.window_columns) >= window.shape[-1]:
                raise ValueError(
                    f"window_columns max index {max(self.cfg.window_columns)} out of range "
                    f"for proprio_window with d_raw={window.shape[-1]}"
                )
            window = np.ascontiguousarray(window[:, :, self.cfg.window_columns])
        action = normalize_action(z["action"].astype(np.float32), self.stats)
        valid_mask = z["valid_mask"].astype(np.uint8)
        pose = z["pose"].astype(np.float32) if "pose" in z.files else None
        N = len(valid_mask)

        bl_path = blr / seq / "safety.npz"
        if bl_path.exists():
            bl = np.load(bl_path, allow_pickle=False)
            binary_unsafe = bl["unsafe"].astype(np.uint8)
            binary_valid = bl["valid_mask"].astype(np.uint8)
        else:
            binary_unsafe = np.zeros(N, dtype=np.uint8)
            binary_valid = np.zeros(N, dtype=np.uint8)

        feat_dir = feat_root / self.cfg.feat_version / seq
        feat_file = feat_dir / "features.npy"
        valid_file = feat_dir / "features_valid.npy"
        if feat_file.exists() and valid_file.exists():
            features = np.load(feat_file, mmap_mode="r")
            valid = np.load(valid_file)
            cached_set = frozenset(np.where(valid)[0].tolist())
        elif feat_dir.is_dir():
            features = None
            cached_set = frozenset(
                int(f.stem) for f in feat_dir.iterdir() if f.suffix == ".npy" and f.stem.isdigit()
            )
        else:
            features = None
            cached_set = frozenset()

        return _SeqRecord(
            name=seq,
            valid_mask=valid_mask,
            feature=feature,
            window=window,
            action=action,
            binary_unsafe=binary_unsafe,
            binary_valid=binary_valid,
            feat_dir=feat_dir,
            cached_set=cached_set,
            features=features,
            pose=pose,
        )

    # ------------------------------------------------------------------
    # Getitem
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        sid, t, s = self._index[idx]
        if isinstance(s, tuple):  # random stride-sampling mode: draw one valid stride
            s = random.choice(s)
        rec = self._records[sid]
        k = self.cfg.context_len
        H = self.cfg.rollout_horizon
        N = len(rec.valid_mask)

        # --- context [t-k*s, t] at stride s ---
        ctx_pos = list(range(t - k * s, t + 1, s))

        z_vis_ctx = np.stack([self._load_vis(rec, j) for j in ctx_pos])  # (k+1, P, D)
        window_ctx = rec.window[ctx_pos].copy()                          # (k+1, T_win, 14)
        action_ctx, action_seq = self._build_actions(rec, t, s, k, H, N)

        # --- target [t+s, t+H*s] at stride s ---
        f_prop_rows: list[np.ndarray] = []
        z_vis_rows: list[np.ndarray] = []
        window_tgt_rows: list[np.ndarray] = []
        binary_rows: list[int] = []
        binary_valid_rows: list[bool] = []
        valid_rows: list[bool] = []
        t_win = rec.window.shape[1]

        for h in range(1, H + 1):
            tp = t + h * s
            in_bounds = tp < N
            label_valid = in_bounds and bool(rec.valid_mask[tp] == 1)
            has_vis = in_bounds and (tp in rec.cached_set)

            if self.cfg.require_vis_target:
                step_valid = label_valid and has_vis
            else:
                step_valid = label_valid

            f_prop_rows.append(
                rec.feature[tp][self._chan_idx]
                if label_valid
                else np.zeros(self._n_chan, np.float32)
            )
            # raw future proprio window: alignment target for next_proprio_head (aligned mode).
            # Invalid steps are zeros (masked by valid_target_mask in the loss).
            window_tgt_rows.append(
                rec.window[tp].copy()
                if label_valid
                else np.zeros((t_win, rec.window.shape[2]), np.float32)
            )
            z_vis_rows.append(self._load_vis(rec, tp) if (label_valid and has_vis) else self._zero_vis())

            if in_bounds and bool(rec.binary_valid[tp]):
                binary_rows.append(int(rec.binary_unsafe[tp]))
                binary_valid_rows.append(True)
            else:
                binary_rows.append(0)
                binary_valid_rows.append(False)

            valid_rows.append(step_valid)

        sample = {
            "z_vis_context": torch.from_numpy(z_vis_ctx),                             # (k+1, P, D)
            "proprio_window_context": torch.from_numpy(window_ctx),                   # (k+1, T_win, 14)
            "action_context": torch.from_numpy(action_ctx),                           # (k+1, d_action)
            "action_sequence": torch.from_numpy(action_seq),                          # (H, d_action)
            "current_feature": torch.from_numpy(rec.feature[t][self._chan_idx].copy()),  # (n_chan,)
            "f_prop_target": torch.from_numpy(np.stack(f_prop_rows)),                 # (H, n_chan)
            "proprio_window_target": torch.from_numpy(np.stack(window_tgt_rows)),     # (H, T_win, 14)
            "z_vis_target": torch.from_numpy(np.stack(z_vis_rows)),                   # (H, P, D)
            "binary_label_target": torch.tensor(binary_rows, dtype=torch.float32),    # (H,)
            "binary_valid_target_mask": torch.tensor(binary_valid_rows, dtype=torch.bool),  # (H,)
            "valid_target_mask": torch.tensor(valid_rows, dtype=torch.bool),          # (H,)
        }
        if self.cfg.include_dt:
            # proprio/trav supervised only where cumulative Δt = h·s·0.1s ≤ proprio_horizon_s
            cum_dt = np.arange(1, H + 1, dtype=np.float32) * (s * 0.1)
            sample["proprio_horizon_mask"] = torch.from_numpy(
                cum_dt <= self.cfg.proprio_horizon_s + 1e-9
            )                                                                         # (H,) bool
        return sample

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_actions(
        self, rec: _SeqRecord, t: int, s: int, k: int, H: int, N: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Context and rollout action arrays.

        Legacy (include_dt=False): precomputed normalized 0.1s pose-deltas, 3-dim.
        Δt path: per step a→a+s, action = [normalize(pose_delta(a, a+s) / s), Δt],
        where dividing by s makes the rate stride-invariant (reuses the 0.1s action
        norm-stats) and Δt = s·0.1s carries the integration time.
        """
        if not self.cfg.include_dt:
            action_ctx = rec.action[t - k : t + 1].copy()                 # (k+1, 3)
            action_seq = rec.action[t : min(t + H, N)].copy()             # (<=H, 3)
            if action_seq.shape[0] < H:
                pad = np.zeros((H - action_seq.shape[0], rec.action.shape[1]), dtype=np.float32)
                action_seq = np.concatenate([action_seq, pad], axis=0)
            return action_ctx, action_seq

        assert rec.pose is not None
        dt = float(s) * 0.1
        ctx_pos = list(range(t - k * s, t + 1, s))
        action_ctx = np.stack([self._action_row(rec, p, s, dt, N) for p in ctx_pos])      # (k+1, 4)
        action_seq = np.stack([self._action_row(rec, t + h * s, s, dt, N) for h in range(H)])  # (H, 4)
        return action_ctx, action_seq

    def _action_row(self, rec: _SeqRecord, a: int, s: int, dt: float, N: int) -> np.ndarray:
        b = a + s
        if a < 0 or b >= N:
            return np.zeros(4, dtype=np.float32)  # out of bounds → padded, masked by valid_target_mask
        rate = _pose_delta(rec.pose[a], rec.pose[b]) / s
        norm3 = normalize_action(rate[None, :], self.stats)[0]
        return np.concatenate([norm3, np.array([dt], dtype=np.float32)]).astype(np.float32)

    def _load_vis(self, rec: _SeqRecord, pos: int) -> np.ndarray:
        if pos in rec.cached_set:
            if rec.features is not None:
                return rec.features[pos].astype(np.float32)
            return np.load(rec.feat_dir / f"{pos}.npy").astype(np.float32)
        return self._zero_vis()

    def _zero_vis(self) -> np.ndarray:
        return np.zeros((self.cfg.n_patch, self.cfg.d_vis), dtype=np.float32)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def sequence_names(self) -> list[str]:
        return [r.name for r in self._records]

    def positive_rate(self) -> float:
        """Fraction of target steps in the index with binary_label=1 (train split).

        In random stride-sampling mode each anchor's valid strides are all counted —
        i.e. the expectation over the uniform stride draw, up to a negligible
        reweighting when anchors differ in how many strides are valid.
        """
        total = pos = 0
        H = self.cfg.rollout_horizon
        for sid, t, s in self._index:
            rec = self._records[sid]
            N = len(rec.valid_mask)
            for stride in (s if isinstance(s, tuple) else (s,)):
                for h in range(1, H + 1):
                    tp = t + h * stride
                    if tp < N and rec.binary_valid[tp]:
                        total += 1
                        pos += int(rec.binary_unsafe[tp])
        return pos / total if total else 0.0
