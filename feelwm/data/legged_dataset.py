"""Legged sliding-window dataset — ANYmal-D, TartanGround (sim) + GrandTour (real).

SEPARATE from the wheeled TartanDriveDataset (zero shared mutation). Reads `labels_legged.npz`
(proprio_window 53-dim, proprio_feature 5-dim, pose, action, valid_mask) from a manifest of
sequences, each tagged by dataset so PER-DATASET norm stats apply (never a shared sim+real
norm). Emits the SAME batch keys the trainer/model expect, so trainer.py is unchanged. The base
mode is single-stride (include_dt=False, 3-dim action); no binary-trav labels.

--- Binary traversability GT (Isaac-legged corpus) ---
TG/GT extractions carry no `trav_gt`, and for them the binary keys stay all-zero (lambda_trav=0
ignores them — bit-identical to the pre-isaac loader). The Isaac ANYmal extraction DOES write a
per-anchor `trav_gt` (rollover/stuck odom thresholds, same definition class as the wheeled Isaac
corpus — no collision channel), so when the npz has the field, `binary_label_target` /
`binary_valid_target_mask` carry it with the exact sim_dataset.py semantics (label at t+h·s,
valid where in-bounds and label-valid; proprio_horizon_mask scopes the trav loss in the trainer).

Visual features: {feat_dir}/features.npy (+ features_valid.npy) consolidated, or {feat_dir}/{i}.npy
per-anchor, or absent → zeros (require_vis=False lets the proprio path run before DINO is cached).

--- Multi-stride / Δt-conditioning (ported from feelwm/data/sim_dataset.py) ---
include_dt=False (default) reproduces the single-stride, 3-dim-action behavior exactly (uses
the precomputed `action`, no Δt channel, no horizon mask, stride 1).
include_dt=True: action = [normalize(pose_delta/stride), Δt] (4-dim); each rollout step spans
`stride` anchors, Δt = stride·`anchor_dt_s`. Both extractions anchor on their 10 Hz camera
(verified on the extracted labels): TartanGround 137 seqs / 177,035 steps
all at 0.1000 s, GrandTour 48 seqs / 200,039 steps at 0.1000 s except 2 dropped-frame gaps
(mission 2024-10-01-12-00-49, ≤0.6 s). Windows spanning such a gap would carry a Δt that does not
match their real elapsed time, so in Δt mode they are dropped at index time (`_grid_ok`).

--- Active output channels (config-selectable, as in the wheeled loaders) ---
`active_proprio_channels` selects which of the 5 legged label channels the model is asked
to predict (the multi-stride config drops `slip`). Labels stay 5-dim on disk; the slice
happens after normalization, against LEGGED_FEATURE_CHANNELS — never the wheeled registry.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import _pose_delta
from .normalize import (
    LEGGED_FEATURE_CHANNELS,
    NormStats,
    normalize_action,
    normalize_feature,
    normalize_window,
    resolve_channel_indices,
)


@dataclass
class LeggedSeqSpec:
    tag: str                 # "gt" | "tg" | "isaac" — selects norm stats + feat version
    name: str                # display name (mission / trajectory id)
    label_npz: Path          # labels_legged.npz
    feat_dir: Path           # directory with features.npy or {i}.npy (may not exist yet)


@dataclass
class LeggedDatasetConfig:
    context_len: int = 3
    rollout_horizon: int = 8
    n_patch: int = 256
    d_vis: int = 768
    require_vis_context: bool = True   # set False to run the proprio path before DINO is cached
    require_vis_target: bool = True
    norm_stats: dict[str, str] = field(default_factory=dict)   # {"gt": path, "tg": path}
    split: str = "train"               # "train" | "val" — selects the stride set
    # --- Multi-stride / Δt-conditioning (see module docstring) ---
    include_dt: bool = False
    train_strides: list[int] = field(default_factory=lambda: [1])  # anchor strides (train split)
    val_stride: int = 1                # single stride for the val loop
    proprio_horizon_s: float = 1.0     # proprio supervised only where cum. Δt ≤ this
    stride_sampling: str = "all"       # "all" (every anchor×stride) | "random" (one draw/anchor)
    anchor_dt_s: float = 0.1           # camera-anchor period; 10 Hz on BOTH extractions
    anchor_dt_tol_s: float = 1e-3      # grid-uniformity tolerance (dropped-frame detection)
    # Active proprio OUTPUT channels by name, resolved against the LEGGED registry
    # (normalize.LEGGED_FEATURE_CHANNELS = [roughness, angular_instability, slope_power,
    # slip, cost_of_transport]) — the config-selectable mechanism, ported from
    # dataset.py / sim_dataset.py. f_prop_target and current_feature are sliced to these
    # channels AFTER normalization, in the order given here. None = all 5 = bit-identical
    # to the original single-stride behavior.
    # The label registry itself is never touched; only what the model is asked to predict
    # (the multi-stride config drops `slip`: single-stride val R² was 0.000 at every epoch).
    active_proprio_channels: list[str] | None = None

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
        if self.active_proprio_channels is not None:
            if not self.active_proprio_channels:
                raise ValueError("active_proprio_channels must be None or non-empty")
            # Fail on an unknown/wheeled-only name here, at config time, rather than at
            # the first strided sample. Explicit registry (no auto-detect): a legged
            # subset can consist entirely of names the wheeled registry also has
            # (e.g. [roughness, angular_instability, slope_power]), which would silently
            # resolve to the WRONG canonical indices.
            resolve_channel_indices(self.active_proprio_channels, LEGGED_FEATURE_CHANNELS)


class _LeggedRecord(NamedTuple):
    spec: LeggedSeqSpec
    name: str                    # = spec.name; trainer per-sequence logging reads record.name
    valid_mask: np.ndarray       # (N,)
    feature: np.ndarray          # (N, 5) normalized
    window: np.ndarray           # (N, T_win, 53) normalized
    action: np.ndarray           # (N, 3) normalized
    pose: np.ndarray             # (N, 4, 4)
    cached_set: frozenset
    features: np.ndarray | None  # mmap (N, P, D) when consolidated
    gap_cum: np.ndarray | None   # (N,) cumulative off-grid anchor steps; None outside Δt mode
    trav_gt: np.ndarray | None   # (N,) binary trav GT (Isaac corpus); None = TG/GT (no field)


class LeggedDataset(Dataset):
    def __init__(self, specs: list[LeggedSeqSpec], config: LeggedDatasetConfig) -> None:
        self.cfg = config
        self._stats = {tag: NormStats.from_json(p) for tag, p in config.norm_stats.items()}
        for tag, st in self._stats.items():
            if st.feature_set != "legged":
                raise ValueError(f"norm stats {tag!r} must have feature_set='legged'")
        if config.active_proprio_channels is None:
            self._chan_idx = list(range(len(LEGGED_FEATURE_CHANNELS)))
        else:
            self._chan_idx = resolve_channel_indices(
                config.active_proprio_channels, LEGGED_FEATURE_CHANNELS
            )
        self._n_chan = len(self._chan_idx)
        self._records: list[_LeggedRecord] = []
        # (record_id, anchor_t, stride). In random stride-sampling mode (train split) the
        # third element is instead a tuple of that anchor's valid strides, and __getitem__
        # draws one uniformly per access (mirrors sim_dataset.py).
        self._index: list[tuple[int, int, int | tuple[int, ...]]] = []
        for spec in specs:
            self._load(spec)
        if not self._index:
            raise ValueError("LeggedDataset is empty (no valid context windows)")

    def _load(self, spec: LeggedSeqSpec) -> None:
        st = self._stats[spec.tag]
        z = np.load(spec.label_npz, allow_pickle=False)
        rec = _LeggedRecord(
            spec=spec,
            name=spec.name,
            valid_mask=z["valid_mask"].astype(np.uint8),
            feature=normalize_feature(z["proprio_feature"].astype(np.float32), st),
            window=normalize_window(z["proprio_window"].astype(np.float32), st),
            action=normalize_action(z["action"].astype(np.float32), st),
            pose=z["pose"].astype(np.float32),
            gap_cum=self._gap_cum(z["timestamp"] if "timestamp" in z else None),
            trav_gt=z["trav_gt"].astype(np.float32) if "trav_gt" in z.files else None,
            **self._load_feats(spec.feat_dir),
        )
        rid = len(self._records)
        self._records.append(rec)
        k = self.cfg.context_len
        N = len(rec.valid_mask)
        strides = self._strides()

        if self._random_stride():
            for t in range(min(strides) * k, N):
                valid = tuple(s for s in strides if t >= k * s and self._window_ok(rec, t, s))
                if valid:
                    self._index.append((rid, t, valid))
        else:
            for s in strides:
                for t in range(k * s, N):
                    if self._window_ok(rec, t, s):
                        self._index.append((rid, t, s))

    def _gap_cum(self, timestamp: np.ndarray | None) -> np.ndarray | None:
        """Cumulative count of off-grid anchor steps, for the Δt-mode grid check.

        None outside Δt mode (the legacy path never reads it) or when the extraction
        predates the `timestamp` field — then no window can be screened and the emitted
        Δt rests on the 10 Hz contract alone.
        """
        if not self.cfg.include_dt or timestamp is None:
            return None
        d = np.diff(np.asarray(timestamp, dtype=np.float64))
        off_grid = np.abs(d - self.cfg.anchor_dt_s) > self.cfg.anchor_dt_tol_s
        return np.concatenate([[0], np.cumsum(off_grid)]).astype(np.int64)

    def _window_ok(self, rec: _LeggedRecord, t: int, s: int) -> bool:
        return self._context_ok(rec, t, s) and self._grid_ok(rec, t, s)

    def _grid_ok(self, rec: _LeggedRecord, t: int, s: int) -> bool:
        """No dropped-frame gap anywhere in the anchors this sample spans.

        Δt is emitted as stride·anchor_dt_s, so a window containing a gap would claim an
        elapsed time it does not have (GrandTour: 2 such steps in 200k — see docstring).
        """
        if rec.gap_cum is None:
            return True
        lo = t - self.cfg.context_len * s
        hi = min(t + self.cfg.rollout_horizon * s, len(rec.gap_cum) - 1)
        return bool(rec.gap_cum[hi] == rec.gap_cum[lo])

    def _random_stride(self) -> bool:
        return self.cfg.stride_sampling == "random" and self.cfg.split == "train"

    def _strides(self) -> list[int]:
        """Anchor strides for the current split (train multi-stride; val single)."""
        if self.cfg.split == "train":
            return list(self.cfg.train_strides)
        return [self.cfg.val_stride]

    def _load_feats(self, feat_dir: Path) -> dict:
        feat_file, valid_file = feat_dir / "features.npy", feat_dir / "features_valid.npy"
        if feat_file.exists() and valid_file.exists():
            return {"features": np.load(feat_file, mmap_mode="r"),
                    "cached_set": frozenset(np.where(np.load(valid_file))[0].tolist())}
        if feat_dir.is_dir():
            return {"features": None,
                    "cached_set": frozenset(int(f.stem) for f in feat_dir.iterdir()
                                            if f.suffix == ".npy" and f.stem.isdigit())}
        return {"features": None, "cached_set": frozenset()}

    def _context_ok(self, rec: _LeggedRecord, t: int, s: int = 1) -> bool:
        """All k+1 context positions [t-k*s, t] at stride s are valid (and cached)."""
        ctx = range(t - self.cfg.context_len * s, t + 1, s)
        if any(rec.valid_mask[j] != 1 for j in ctx):
            return False
        if self.cfg.require_vis_context and any(j not in rec.cached_set for j in ctx):
            return False
        return True

    def __len__(self) -> int:
        return len(self._index)

    def _zero_vis(self) -> np.ndarray:
        return np.zeros((self.cfg.n_patch, self.cfg.d_vis), np.float32)

    def _load_vis(self, rec: _LeggedRecord, pos: int) -> np.ndarray:
        if pos in rec.cached_set:
            if rec.features is not None:
                return rec.features[pos].astype(np.float32)
            return np.load(rec.spec.feat_dir / f"{pos}.npy").astype(np.float32)
        return self._zero_vis()

    def __getitem__(self, idx: int) -> dict:
        rid, t, s = self._index[idx]
        if isinstance(s, tuple):  # random stride-sampling mode: draw one valid stride
            s = random.choice(s)
        rec = self._records[rid]
        k, H, N = self.cfg.context_len, self.cfg.rollout_horizon, len(rec.valid_mask)
        t_win, d_raw = rec.window.shape[1], rec.window.shape[2]

        ctx_pos = list(range(t - k * s, t + 1, s))
        z_vis_ctx = np.stack([self._load_vis(rec, j) for j in ctx_pos])      # (k+1,P,D)
        window_ctx = rec.window[ctx_pos].copy()
        action_ctx, action_seq = self._build_actions(rec, t, s, k, H, N)

        f_prop_rows, z_vis_rows, window_tgt_rows, valid_rows = [], [], [], []
        bin_rows, bin_valid_rows = [], []
        for h in range(1, H + 1):
            tp = t + h * s
            in_b = tp < N
            label_valid = in_b and bool(rec.valid_mask[tp] == 1)
            has_vis = in_b and (tp in rec.cached_set)
            step_valid = label_valid and (has_vis if self.cfg.require_vis_target else True)
            f_prop_rows.append(rec.feature[tp][self._chan_idx].copy() if label_valid
                               else np.zeros(self._n_chan, np.float32))
            window_tgt_rows.append(rec.window[tp].copy() if label_valid
                                   else np.zeros((t_win, d_raw), np.float32))
            z_vis_rows.append(self._load_vis(rec, tp) if (label_valid and has_vis) else self._zero_vis())
            valid_rows.append(step_valid)
            if rec.trav_gt is not None:  # Isaac corpus: real GT, sim_dataset.py semantics
                bin_rows.append(float(rec.trav_gt[tp]) if label_valid else 0.0)
                bin_valid_rows.append(label_valid)

        if rec.trav_gt is None:
            # TG/GT: no binary-trav GT; zeros keep trainer keys present (lambda_trav=0 ignores).
            bin_label = torch.zeros(H, dtype=torch.float32)
            bin_valid = torch.zeros(H, dtype=torch.bool)
        else:
            bin_label = torch.tensor(bin_rows, dtype=torch.float32)
            bin_valid = torch.tensor(bin_valid_rows, dtype=torch.bool)

        sample = {
            "z_vis_context": torch.from_numpy(z_vis_ctx),
            "proprio_window_context": torch.from_numpy(window_ctx),
            "action_context": torch.from_numpy(action_ctx),
            "action_sequence": torch.from_numpy(action_seq),
            "current_feature": torch.from_numpy(rec.feature[t][self._chan_idx].copy()),
            "f_prop_target": torch.from_numpy(np.stack(f_prop_rows)),
            "proprio_window_target": torch.from_numpy(np.stack(window_tgt_rows)),
            "z_vis_target": torch.from_numpy(np.stack(z_vis_rows)),
            "binary_label_target": bin_label,
            "binary_valid_target_mask": bin_valid,
            "valid_target_mask": torch.tensor(valid_rows, dtype=torch.bool),
        }
        if self.cfg.include_dt:
            # proprio supervised only where cumulative Δt = h·s·anchor_dt_s ≤ proprio_horizon_s
            cum_dt = np.arange(1, H + 1, dtype=np.float32) * (s * self.cfg.anchor_dt_s)
            sample["proprio_horizon_mask"] = torch.from_numpy(
                cum_dt <= self.cfg.proprio_horizon_s + 1e-9
            )
        return sample

    def _build_actions(
        self, rec: _LeggedRecord, t: int, s: int, k: int, H: int, N: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Context and rollout action arrays (mirrors sim_dataset.py._build_actions).

        Legacy (include_dt=False): precomputed normalized single-anchor pose-deltas, 3-dim.
        Δt path: per step a→a+s, action = [normalize(pose_delta(a, a+s) / s), Δt], where
        dividing by s makes the rate stride-invariant (so the single-anchor action norm-stats
        still apply) and Δt = s·anchor_dt_s carries the integration time.
        """
        if not self.cfg.include_dt:
            action_ctx = rec.action[t - k : t + 1].copy()
            action_seq = rec.action[t : min(t + H, N)].copy()
            if action_seq.shape[0] < H:
                action_seq = np.concatenate(
                    [action_seq, np.zeros((H - action_seq.shape[0], 3), np.float32)], axis=0)
            return action_ctx, action_seq

        dt = float(s) * self.cfg.anchor_dt_s
        ctx_pos = list(range(t - k * s, t + 1, s))
        action_ctx = np.stack([self._action_row(rec, p, s, dt, N) for p in ctx_pos])
        action_seq = np.stack([self._action_row(rec, t + h * s, s, dt, N) for h in range(H)])
        return action_ctx, action_seq

    def _action_row(self, rec: _LeggedRecord, a: int, s: int, dt: float, N: int) -> np.ndarray:
        b = a + s
        if a < 0 or b >= N:
            return np.zeros(4, dtype=np.float32)
        rate = _pose_delta(rec.pose[a], rec.pose[b]) / s
        norm3 = normalize_action(rate[None, :], self._stats[rec.spec.tag])[0]
        return np.concatenate([norm3, np.array([dt], dtype=np.float32)]).astype(np.float32)

    def sequence_names(self) -> list[str]:
        return [r.spec.name for r in self._records]


def discover_legged_specs(
    grandtour_root: str | Path | None = None,
    tartanground_root: str | Path | None = None,
    feat_version_gt: str = "gt_dinov2reg_v1",
    feat_version_tg: str = "tg_dinov2reg_v1",
    isaac_root: str | Path | None = None,
    feat_version_isaac: str = "anymal_isaac_dinov3_v1",
    feat_root: str | Path | None = None,
    exclude: set[str] | None = None,
) -> list[LeggedSeqSpec]:
    """Scan dataset trees for labels_legged.npz. feat_dir defaults next to labels under
    <seq>/features/<feat_version>/ unless feat_root is given (mirrors the cache layout).

    isaac_root points at the Isaac ANYmal extraction's flat sequence dir
    (…/sim_anymal_isaac/sequences/all — one dir per episode, e.g.
    Asphalt1__cov_02__v0.7__ep00), tag "isaac"."""
    exclude = exclude or set()
    specs: list[LeggedSeqSpec] = []

    def feat_dir_for(seq_dir: Path, name: str, fv: str) -> Path:
        base = Path(feat_root) / fv / name if feat_root else seq_dir / "features" / fv
        return base

    if grandtour_root:
        for lbl in sorted(Path(grandtour_root).glob("*/labels_legged.npz")):
            name = lbl.parent.name
            if name in exclude:
                continue
            specs.append(LeggedSeqSpec("gt", name, lbl, feat_dir_for(lbl.parent, name, feat_version_gt)))
    if tartanground_root:
        for lbl in sorted(Path(tartanground_root).glob("*/Data_anymal/*/labels_legged.npz")):
            name = f"{lbl.parents[2].name}_{lbl.parent.name}"   # e.g. Gascola_P2000
            if name in exclude:
                continue
            specs.append(LeggedSeqSpec("tg", name, lbl, feat_dir_for(lbl.parent, name, feat_version_tg)))
    if isaac_root:
        for lbl in sorted(Path(isaac_root).glob("*/labels_legged.npz")):
            name = lbl.parent.name
            if name in exclude:
                continue
            specs.append(LeggedSeqSpec("isaac", name, lbl,
                                       feat_dir_for(lbl.parent, name, feat_version_isaac)))
    return specs
