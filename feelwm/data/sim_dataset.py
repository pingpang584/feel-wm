"""Sim Husky sliding-window dataset — NEGS-UGV Gazebo.

SEPARATE from the wheeled TartanDriveDataset and the LeggedDataset (zero shared mutation).
Reads per-episode `labels.npz` (proprio_window 13-dim, proprio_feature 5-dim wheeled channels,
trav_gt, pose, action, valid_mask) produced by tools/sim/extract_sim_husky.py, with a SINGLE
sim norm (feature_set="sim" → pure z-score; normalize.py gates are all == "wheeled").

Unlike the legged loader, sim has TRUE binary traversability GT (rollover/collision/stuck), so
`binary_label_target` carries the real `trav_gt` and the trav head is trained (lambda_trav>0).

Visual features: {seq}/features/{feat_version}/features.npy (+ features_valid.npy) consolidated,
or absent → zeros (require_vis_*=False runs the proprio path before DINO is cached).

--- Multi-stride / Δt-conditioning (mirrors feelwm/data/dataset.py) ---
include_dt=False (default) reproduces the original fixed-0.1s, 3-dim-action behavior exactly
(uses the precomputed `action`, no Δt channel, no horizon mask, stride 1).
include_dt=True: action = [normalize(pose_delta/stride), Δt] (4-dim); each rollout step spans
`stride` anchors (Δt = stride·0.1s, since sim anchors are a fixed 10 Hz grid); emits
`proprio_horizon_mask` (cum. Δt ≤ proprio_horizon_s) for losses.py to consume.

--- Multi-corpus (combined Gazebo+Isaac training) ---
A spec list may mix corpora: SimSeqSpec carries its own label/feature paths plus a `domain`
tag, and the dataset never touches cfg.seq_root/feat_version when loading. ONE norm-stats
file normalizes the pooled corpus (joint stats), so the corpora must share the
window schema (13-dim), the anchor rate (10 Hz) and the visual feature space (dinov2reg
768×256). Corpus composition — which roots, which held-out substrings per root — is the
builder's job (feelwm/training/trainer.py: build_sim_datasets).
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
    NormStats, normalize_action, normalize_feature, normalize_window,
    resolve_channel_indices,
)


@dataclass
class SimSeqSpec:
    name: str                # episode dir name, e.g. hill_run_n1000_c0__ep02
    label_npz: Path
    feat_dir: Path
    # Corpus tag for MULTI-ROOT runs (Gazebo+Isaac combined training). Empty string = the
    # single-corpus case: record names, indices and per-sequence logs stay bit-identical to
    # the pre-multi-corpus loader. Non-empty prefixes the record name "<domain>/<name>", so
    # a pooled per-sequence log is unambiguous (the two corpora have independent naming
    # schemes) and groupable by domain. Split/exclude substring matching always uses the
    # bare `name`, never the prefixed one.
    domain: str = ""


@dataclass
class SimDatasetConfig:
    seq_root: str = ""
    feat_version: str = "sim_dinov2reg_v1"
    norm_stats_path: str = ""
    split: str = "train"                      # "train" | "val"
    heldout_substrings: list[str] = field(default_factory=list)   # name contains any → val
    # Sequences dropped from the TRAIN side ONLY (name contains any → contributes no train
    # window). Deliberately NOT symmetric with the TartanDrive knob of the same name
    # (feelwm/data/dataset.py:91), which drops a sequence from whichever split loads it: here
    # the point is to vary the TRAINING corpus while the val composition — held-out
    # sequences AND the tail-val slices of excluded sequences — stays bit-identical to the
    # empty-list case, so two arms differing only in train data are directly comparable
    # (e.g. the same recipe with one recording site removed from train). Match semantics
    # are substring, like heldout_substrings / tail_val_substrings.
    exclude_sequences: list[str] = field(default_factory=list)
    context_len: int = 3
    rollout_horizon: int = 8
    n_patch: int = 256
    d_vis: int = 768
    require_vis_context: bool = True
    require_vis_target: bool = True
    # --- Multi-stride / Δt-conditioning (see module docstring) ---
    include_dt: bool = False
    train_strides: list[int] = field(default_factory=lambda: [1])  # anchor strides (train split)
    val_stride: int = 1                # single stride for val loop
    proprio_horizon_s: float = 1.0     # proprio/trav supervised only where cum. Δt ≤ this
    stride_sampling: str = "all"       # "all" (every anchor×stride) | "random" (one draw/anchor)
    # Active proprio output channels by name (canonical FEATURE_CHANNELS order; subset of
    # the 5). None = all 5. f_prop_target and current_feature are sliced to these channels
    # AFTER normalization — mirrors dataset.py (the 4-ch configs drop ke_rate).
    active_proprio_channels: list[str] | None = None
    # Localization-jump screen (real-robot data; OFF by default → sim indices unchanged).
    # Real LiDAR-odom bags contain teleports where the pose estimate relocalizes: the
    # action vectors are POSE-DERIVED, so any window whose used anchors straddle a jump
    # trains on an action that never happened. The extractor already zeroes valid_mask on
    # the jump anchor itself, which only screens stride-1 windows — a strided window can
    # skip over the jump anchor and still span the jump. Set to the per-step displacement
    # (metres, on the 10 Hz anchor grid) above which a step is a teleport, e.g. 0.5 = 5 m/s.
    max_pose_step_m: float | None = None
    # Within-sequence TAIL holdout — a second val axis for sequences that have no repeat
    # or reverse traversal to hold out (some real-robot recordings). Sequences whose name
    # matches `tail_val_substrings` are split in time:
    # the head trains, the last `tail_val_fraction` of their ELIGIBLE anchors (valid, and
    # not jump-adjacent when the screen is on — i.e. counted after the screen) validates.
    # Windows straddling the cut are dropped from BOTH sides, so no train window's context
    # or rollout ever touches the val region. OFF by default (fraction 0.0).
    tail_val_substrings: list[str] = field(default_factory=list)
    tail_val_fraction: float = 0.0
    # Encoder-input window column subset, applied AFTER normalization with the full-schema
    # stats (mirrors dataset.py). Portable-11 = [0..10] of the 13-dim sim/real schema
    # (ax ay az gx gy gz vx vy wz pitch roll) — the shared cross-platform prefix, so a
    # portable-11 TartanDrive donor encoder can consume sim/real windows unchanged.
    window_columns: list[int] | None = None

    def __post_init__(self) -> None:
        if any(s < 1 for s in self.train_strides) or self.val_stride < 1:
            raise ValueError("strides must be >= 1")
        if self.window_columns is not None:
            if len(self.window_columns) == 0:
                raise ValueError("window_columns must be None or non-empty")
            if len(set(self.window_columns)) != len(self.window_columns):
                raise ValueError("window_columns has duplicate indices")
            if any(c < 0 for c in self.window_columns):
                raise ValueError("window_columns indices must be >= 0")
        if not self.include_dt and any(s != 1 for s in self.train_strides):
            raise ValueError(
                "train_strides != [1] requires include_dt=True "
                "(the Δt channel is needed to disambiguate stride)"
            )
        if self.stride_sampling not in {"all", "random"}:
            raise ValueError(f"unknown stride_sampling={self.stride_sampling!r}")
        if not 0.0 <= self.tail_val_fraction < 1.0:
            raise ValueError("tail_val_fraction must be in [0, 1)")
        if bool(self.tail_val_substrings) != (self.tail_val_fraction > 0.0):
            raise ValueError(
                "tail_val_substrings and tail_val_fraction must be set together "
                "(both empty/zero = tail holdout off)")


class _SimRecord(NamedTuple):
    spec: SimSeqSpec
    name: str
    valid_mask: np.ndarray       # (N,)
    feature: np.ndarray          # (N, 5) normalized
    window: np.ndarray           # (N, T_win, 13) normalized
    action: np.ndarray           # (N, 3) normalized
    pose: np.ndarray             # (N, 4, 4)
    trav_gt: np.ndarray          # (N,) uint8 — true binary traversability
    cached_set: frozenset
    features: np.ndarray | None
    jump_cum: np.ndarray | None  # (N,) cumulative pose teleports; None when screen is off
    tail_cut: int | None         # first anchor of the tail-val region; None = whole-seq record


def jump_cum_from_pose(pose: np.ndarray, max_pose_step_m: float | None) -> np.ndarray | None:
    """Cumulative count of localization teleports along a sequence (see max_pose_step_m)."""
    if max_pose_step_m is None:
        return None
    step = np.linalg.norm(np.diff(pose[:, :3, 3].astype(np.float64), axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(step > max_pose_step_m)]).astype(np.int64)


def tail_cut_index(
    valid_mask: np.ndarray, jump_cum: np.ndarray | None, fraction: float
) -> int:
    """First anchor of the tail-val region of one sequence.

    The cut is the `1 - fraction` quantile of the sequence's ELIGIBLE anchors — valid, and
    not adjacent to a teleport when the jump screen is on — so the tail holds ~`fraction`
    of the USABLE data even in a bag where the screen removes a third of it ("sliced after
    the jump screen"). It depends only on the sequence, never on the stride set, so the
    train and val datasets — separate objects with different strides — always agree on the
    boundary. Norm-stats computation calls this too, so the stats stay head-only
    (tools/sim/compute_norm_stats.py --tail-exclude).
    """
    eligible = valid_mask == 1
    if jump_cum is not None:
        step_jump = np.diff(jump_cum) > 0                        # step i -> i+1 teleports
        eligible[:-1] &= ~step_jump
        eligible[1:] &= ~step_jump
    idx = np.flatnonzero(eligible)
    if len(idx) == 0:
        return len(valid_mask)                                   # nothing usable → no tail
    head = min(int(round(len(idx) * (1.0 - fraction))), len(idx) - 1)
    return int(idx[head])


def discover_sim_specs(
    seq_root: str | Path, feat_version: str, domain: str = "",
    labels_name: str = "labels.npz",
) -> list[SimSeqSpec]:
    """All episode dirs with `labels_name` under seq_root (see SimSeqSpec.domain).

    A spec carries absolute label/feature paths, so a list may mix corpora with different
    seq_roots AND different feat_versions — the dataset reads only the spec paths. The
    caller is responsible for the invariants pooling requires: same d_vis/n_patch feature
    space, same d_raw window schema, same anchor rate (10 Hz), one shared norm-stats file.

    `labels_name` selects WHICH label file in each episode dir to read (default
    `labels.npz` — every config to date). It exists so a relabelled corpus can live in the
    SAME tree as the corpus it derives from instead of duplicating 19 GB of images and
    features: a relabelling pass can write e.g. `labels_collision_v1.npz` beside
    `labels.npz`, identical except that `trav_gt` also carries a recovered
    chassis-collision channel.
    Because the spec's `label_npz` is what the dataset AND the pos_weight computation read,
    setting this is the whole switch — nothing downstream needs to know.
    """
    specs = []
    for lbl in sorted(Path(seq_root).glob(f"*/{labels_name}")):
        specs.append(
            SimSeqSpec(lbl.parent.name, lbl, lbl.parent / "features" / feat_version, domain)
        )
    return specs


class SimHuskyDataset(Dataset):
    def __init__(self, specs: list[SimSeqSpec], config: SimDatasetConfig) -> None:
        self.cfg = config
        self.stats = NormStats.from_json(config.norm_stats_path)
        if config.active_proprio_channels is None:
            self._chan_idx = list(range(5))
        else:
            self._chan_idx = resolve_channel_indices(config.active_proprio_channels)
        self._n_chan = len(self._chan_idx)
        self._records: list[_SimRecord] = []
        # (record_id, anchor_t, stride). In random stride-sampling mode (train split) the
        # third element is instead a tuple of that anchor's valid strides, and
        # __getitem__ draws one uniformly per access (mirrors dataset.py).
        self._index: list[tuple[int, int, int | tuple[int, ...]]] = []
        for spec in specs:
            self._load(spec)
        if not self._index:
            raise ValueError(f"SimHuskyDataset ({config.split}) is empty (no valid context windows)")

    def _load(self, spec: SimSeqSpec) -> None:
        st = self.stats
        z = np.load(spec.label_npz, allow_pickle=False)
        window = normalize_window(z["proprio_window"].astype(np.float32), st)
        if self.cfg.window_columns is not None:
            if max(self.cfg.window_columns) >= window.shape[-1]:
                raise ValueError(
                    f"window_columns max index {max(self.cfg.window_columns)} out of range "
                    f"for proprio_window with d_raw={window.shape[-1]}"
                )
            window = np.ascontiguousarray(window[:, :, self.cfg.window_columns])
        jump_cum = self._jump_cum(z["pose"])
        valid_mask = z["valid_mask"].astype(np.uint8)
        tail_cut = self._tail_cut(spec.name, valid_mask, jump_cum)
        # a tail-val record shares its labels with the train record of the same sequence —
        # name it apart so per-sequence val logs are unambiguous about which slice they are
        tail_val = tail_cut is not None and self.cfg.split != "train"
        display = f"{spec.domain}/{spec.name}" if spec.domain else spec.name
        rec = _SimRecord(
            spec=spec, name=f"{display}__tail" if tail_val else display,
            valid_mask=valid_mask,
            feature=normalize_feature(z["proprio_feature"].astype(np.float32), st),
            window=window,
            action=normalize_action(z["action"].astype(np.float32), st),
            pose=z["pose"].astype(np.float32),
            trav_gt=z["trav_gt"].astype(np.uint8),
            jump_cum=jump_cum,
            tail_cut=tail_cut,
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

    def _tail_cut(
        self, name: str, valid_mask: np.ndarray, jump_cum: np.ndarray | None
    ) -> int | None:
        """First anchor of this sequence's tail-val region (None if it has none)."""
        if self.cfg.tail_val_fraction <= 0.0:
            return None
        if not any(sub in name for sub in self.cfg.tail_val_substrings):
            return None
        if any(sub in name for sub in self.cfg.heldout_substrings):
            return None      # already a whole-sequence val record; it has no head in train
        return tail_cut_index(valid_mask, jump_cum, self.cfg.tail_val_fraction)

    def _jump_cum(self, pose: np.ndarray) -> np.ndarray | None:
        return jump_cum_from_pose(pose, self.cfg.max_pose_step_m)

    def _load_feats(self, feat_dir: Path) -> dict:
        feat_file, valid_file = feat_dir / "features.npy", feat_dir / "features_valid.npy"
        if feat_file.exists() and valid_file.exists():
            return {"features": np.load(feat_file, mmap_mode="r"),
                    "cached_set": frozenset(np.where(np.load(valid_file))[0].tolist())}
        return {"features": None, "cached_set": frozenset()}

    def _window_ok(self, rec: _SimRecord, t: int, s: int) -> bool:
        return (self._context_ok(rec, t, s) and self._no_pose_jump(rec, t, s)
                and self._tail_side_ok(rec, t, s))

    def _tail_side_ok(self, rec: _SimRecord, t: int, s: int) -> bool:
        """Keep only windows lying wholly on this split's side of the tail cut.

        A window straddling the cut is dropped from BOTH splits: on the train side its
        rollout targets would reach into the val region, on the val side its context would
        come from trained-on anchors. Records without a tail cut are unaffected.
        """
        if rec.tail_cut is None:
            return True
        lo = t - self.cfg.context_len * s
        hi = min(t + self.cfg.rollout_horizon * s, len(rec.valid_mask) - 1)
        return hi < rec.tail_cut if self.cfg.split == "train" else lo >= rec.tail_cut

    def _no_pose_jump(self, rec: _SimRecord, t: int, s: int) -> bool:
        """No localization teleport anywhere in the anchors this sample spans.

        Checked over the whole used span [t-k*s, t+H*s], not just the context: every
        action in the sample (context AND rollout) is a pose delta, so a jump anywhere
        inside it produces an action the robot never executed.
        """
        if rec.jump_cum is None:
            return True
        lo = t - self.cfg.context_len * s
        hi = min(t + self.cfg.rollout_horizon * s, len(rec.jump_cum) - 1)
        return bool(rec.jump_cum[hi] == rec.jump_cum[lo])

    def _context_ok(self, rec: _SimRecord, t: int, s: int) -> bool:
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
        """Anchor strides for the current split (train multi-stride; val single)."""
        if self.cfg.split == "train":
            return list(self.cfg.train_strides)
        return [self.cfg.val_stride]

    def positive_rate(self) -> float:
        """Fraction of target steps in the index with trav_gt=1 (train split).

        In random stride-sampling mode each anchor's valid strides are all counted — i.e.
        the expectation over the uniform stride draw (mirrors dataset.py.positive_rate).
        """
        total = pos = 0
        H = self.cfg.rollout_horizon
        for rid, t, s in self._index:
            rec = self._records[rid]
            N = len(rec.valid_mask)
            for stride in (s if isinstance(s, tuple) else (s,)):
                for h in range(1, H + 1):
                    tp = t + h * stride
                    if tp < N and rec.valid_mask[tp] == 1:
                        total += 1
                        pos += int(rec.trav_gt[tp])
        return pos / total if total else 0.0

    def __len__(self) -> int:
        return len(self._index)

    def _zero_vis(self) -> np.ndarray:
        return np.zeros((self.cfg.n_patch, self.cfg.d_vis), np.float32)

    def _load_vis(self, rec: _SimRecord, pos: int) -> np.ndarray:
        if pos in rec.cached_set and rec.features is not None:
            return rec.features[pos].astype(np.float32)
        return self._zero_vis()

    def __getitem__(self, idx: int) -> dict:
        rid, t, s = self._index[idx]
        if isinstance(s, tuple):  # random stride-sampling mode: draw one valid stride
            s = random.choice(s)
        rec = self._records[rid]
        k, H, N = self.cfg.context_len, self.cfg.rollout_horizon, len(rec.valid_mask)
        t_win, d_raw = rec.window.shape[1], rec.window.shape[2]

        ctx_pos = list(range(t - k * s, t + 1, s))
        z_vis_ctx = np.stack([self._load_vis(rec, j) for j in ctx_pos])
        window_ctx = rec.window[ctx_pos].copy()
        action_ctx, action_seq = self._build_actions(rec, t, s, k, H, N)

        f_rows, vis_rows, win_rows, valid_rows = [], [], [], []
        bin_rows, bin_valid_rows = [], []
        for h in range(1, H + 1):
            tp = t + h * s
            in_b = tp < N
            label_valid = in_b and bool(rec.valid_mask[tp] == 1)
            has_vis = in_b and (tp in rec.cached_set)
            step_valid = label_valid and (has_vis if self.cfg.require_vis_target else True)
            f_rows.append(rec.feature[tp][self._chan_idx]
                          if label_valid else np.zeros(self._n_chan, np.float32))
            win_rows.append(rec.window[tp].copy() if label_valid
                            else np.zeros((t_win, d_raw), np.float32))
            vis_rows.append(self._load_vis(rec, tp) if (label_valid and has_vis) else self._zero_vis())
            valid_rows.append(step_valid)
            bin_rows.append(float(rec.trav_gt[tp]) if label_valid else 0.0)
            bin_valid_rows.append(label_valid)

        sample = {
            "z_vis_context": torch.from_numpy(z_vis_ctx),
            "proprio_window_context": torch.from_numpy(window_ctx),
            "action_context": torch.from_numpy(action_ctx),
            "action_sequence": torch.from_numpy(action_seq),
            "current_feature": torch.from_numpy(rec.feature[t][self._chan_idx].copy()),
            "f_prop_target": torch.from_numpy(np.stack(f_rows)),
            "proprio_window_target": torch.from_numpy(np.stack(win_rows)),
            "z_vis_target": torch.from_numpy(np.stack(vis_rows)),
            "binary_label_target": torch.tensor(bin_rows, dtype=torch.float32),
            "binary_valid_target_mask": torch.tensor(bin_valid_rows, dtype=torch.bool),
            "valid_target_mask": torch.tensor(valid_rows, dtype=torch.bool),
        }
        if self.cfg.include_dt:
            # proprio/trav supervised only where cumulative Δt = h·s·0.1s ≤ proprio_horizon_s
            cum_dt = np.arange(1, H + 1, dtype=np.float32) * (s * 0.1)
            sample["proprio_horizon_mask"] = torch.from_numpy(
                cum_dt <= self.cfg.proprio_horizon_s + 1e-9
            )
        return sample

    def _build_actions(
        self, rec: _SimRecord, t: int, s: int, k: int, H: int, N: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Context and rollout action arrays (mirrors dataset.py._build_actions).

        Legacy (include_dt=False): precomputed normalized 0.1s pose-deltas, 3-dim.
        Δt path: per step a→a+s, action = [normalize(pose_delta(a, a+s) / s), Δt], where
        dividing by s makes the rate stride-invariant (reuses the 0.1s action norm-stats)
        and Δt = s·0.1s carries the integration time (sim anchors are a fixed 10 Hz grid).
        """
        if not self.cfg.include_dt:
            action_ctx = rec.action[t - k : t + 1].copy()
            action_seq = rec.action[t : min(t + H, N)].copy()
            if action_seq.shape[0] < H:
                action_seq = np.concatenate(
                    [action_seq, np.zeros((H - action_seq.shape[0], 3), np.float32)], axis=0)
            return action_ctx, action_seq

        dt = float(s) * 0.1
        ctx_pos = list(range(t - k * s, t + 1, s))
        action_ctx = np.stack([self._action_row(rec, p, s, dt, N) for p in ctx_pos])
        action_seq = np.stack([self._action_row(rec, t + h * s, s, dt, N) for h in range(H)])
        return action_ctx, action_seq

    def _action_row(self, rec: _SimRecord, a: int, s: int, dt: float, N: int) -> np.ndarray:
        b = a + s
        if a < 0 or b >= N:
            return np.zeros(4, dtype=np.float32)
        rate = _pose_delta(rec.pose[a], rec.pose[b]) / s
        norm3 = normalize_action(rate[None, :], self.stats)[0]
        return np.concatenate([norm3, np.array([dt], dtype=np.float32)]).astype(np.float32)

    def sequence_names(self) -> list[str]:
        return [r.spec.name for r in self._records]

    def domain_counts(self) -> dict[str, int]:
        """Windows in the index per corpus tag — the mixing check for combined training.

        Counts INDEX ENTRIES, not anchors: in random stride-sampling mode one entry is one
        anchor (with its stride drawn per access), so this is the sampling weight each
        corpus actually gets under natural concatenation (no reweighting — the combined
        training's mixing rule).
        """
        counts: dict[str, int] = {}
        for rid, _t, _s in self._index:
            key = self._records[rid].spec.domain
            counts[key] = counts.get(key, 0) + 1
        return counts


def sim_corpora(cfg_data: dict) -> list[dict]:
    """Normalize a sim config `data` block to a list of corpora (multi-corpus support).

    Two accepted forms, and they are mutually exclusive:

    * FLAT (every sim/isaac/real config to date) — `seq_root` / `feat_version` /
      `heldout_substrings` at the top of the block. Yields ONE corpus with domain "", which
      the loader renders exactly as before: bare record names, unchanged indices and
      per-sequence logs.
    * `data.corpora: [{name, seq_root, feat_version, heldout_substrings}, ...]` — the
      multi-root form (Gazebo+Isaac pooling). Each entry keeps its OWN held-out substrings,
      because the two corpora have independent naming schemes and independently settled val
      splits; `name` becomes the record-name prefix and the per-domain val tag.

    Both forms also accept `drop_substrings` — sequences that belong to NEITHER split (see
    drop_specs). Three different knobs, three different scopes, easy to confuse:
      heldout_substrings  name matches -> the sequence VALIDATES (and never trains)
      drop_substrings     name matches -> the sequence is INVISIBLE (neither split)
      exclude_sequences   name matches -> no TRAIN windows, val composition untouched
                          (SimDatasetConfig field, run-level not corpus-level)

    Only the corpus composition varies here — normalization does NOT: `norm_stats_path` is a
    single file for the whole run (joint stats over the pooled corpus). Lives in the data
    layer, not the trainer, so tools/sim/compute_norm_stats.py can derive the SAME
    corpus/exclusion set
    from the training config without importing torch — stats and loader cannot drift.
    """
    flat = [k for k in ("seq_root", "feat_version", "heldout_substrings", "drop_substrings",
                        "labels_name") if k in cfg_data]
    if "corpora" not in cfg_data:
        return [{
            "name": "", "seq_root": cfg_data["seq_root"],
            "feat_version": cfg_data.get("feat_version", "sim_dinov2reg_v1"),
            "heldout_substrings": list(cfg_data.get("heldout_substrings", [])),
            "drop_substrings": list(cfg_data.get("drop_substrings", [])),
            "labels_name": cfg_data.get("labels_name", "labels.npz"),
        }]
    if flat:
        raise ValueError(
            f"data.corpora is set, so the flat corpus keys {flat} are ambiguous — move them "
            "into a corpora entry (per-corpus seq_root/feat_version/heldout_substrings)")
    corpora = []
    for i, c in enumerate(cfg_data["corpora"]):
        name = str(c.get("name", "")).strip()
        if not name:
            raise ValueError(f"data.corpora[{i}] needs a non-empty name (the domain tag)")
        if "/" in name or ":" in name:
            raise ValueError(
                f"data.corpora[{i}].name {name!r} must not contain '/' or ':' — they "
                "separate the record-name prefix and the per-domain val key")
        if "seq_root" not in c:
            raise ValueError(f"data.corpora[{i}] ({name}) needs a seq_root")
        corpora.append({
            "name": name, "seq_root": c["seq_root"],
            "feat_version": c.get("feat_version", "sim_dinov2reg_v1"),
            "heldout_substrings": list(c.get("heldout_substrings", [])),
            "drop_substrings": list(c.get("drop_substrings", [])),
            # per-corpus: in a Gazebo+Isaac run only the Isaac corpus has collision-relabelled
            # labels — Gazebo's trav_gt already includes a real bumper channel.
            "labels_name": c.get("labels_name", "labels.npz"),
        })
    names = [c["name"] for c in corpora]
    if len(set(names)) != len(names):
        raise ValueError(f"data.corpora names must be unique, got {names}")
    return corpora


def drop_specs(specs: list[SimSeqSpec], drop_substrings: list[str]) -> list[SimSeqSpec]:
    """Corpus-composition filter: matching sequences land in NEITHER split.

    Applied BEFORE split_sim_specs, so a dropped sequence is not a train sequence and not a
    val sequence — it is as if it were never collected. Neither existing knob expresses this:
    `heldout_substrings` would send it to val (growing the val set) and `exclude_sequences`
    only removes train windows.

    Why it exists: an incremental collection can add same-path variants of sequences the
    val split already
    holds out (`humji2__cov_02__speedvar` beside the held-out `humji2__cov_02__v0.7`). Those
    variants are near-duplicate visuals of val, so training on them leaks; putting them in
    val instead would break the arm's requirement that val stay BIT-IDENTICAL to the arm it
    is compared against. Dropping them satisfies both.

    Substring match on the bare `name`, like the other two knobs. Empty list = no-op.
    """
    if not drop_substrings:
        return list(specs)
    return [s for s in specs if not any(sub in s.name for sub in drop_substrings)]


def split_sim_specs(
    specs: list[SimSeqSpec], heldout_substrings: list[str]
) -> tuple[list[SimSeqSpec], list[SimSeqSpec]]:
    """(train, val): val = specs whose name contains any held-out substring."""
    def is_val(s: SimSeqSpec) -> bool:
        return any(sub in s.name for sub in heldout_substrings)
    return [s for s in specs if not is_val(s)], [s for s in specs if is_val(s)]


def drop_excluded_specs(
    specs: list[SimSeqSpec], exclude_sequences: list[str]
) -> list[SimSeqSpec]:
    """Train-side spec filter for SimDatasetConfig.exclude_sequences (substring match).

    Separate from split_sim_specs because the caller must apply it AFTER the tail-val pool
    is taken: excluding a sequence removes its TRAIN windows only, never its val records
    (see the field docstring). An empty list returns the input unchanged.
    """
    if not exclude_sequences:
        return list(specs)
    return [s for s in specs if not any(sub in s.name for sub in exclude_sequences)]
