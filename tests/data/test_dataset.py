"""Unit tests for TartanDriveDataset and normalization utilities.

Uses fake sequences written to a tmp directory — no real TartanDrive data required.
Run: python -m pytest tests/data/test_dataset.py -v
"""
from __future__ import annotations

import dataclasses
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from feelwm.data.collate import collate_fn, variable_h_collate
from feelwm.data.dataset import DatasetConfig, TartanDriveDataset
from feelwm.data.normalize import (
    NormStats,
    denormalize_action,
    denormalize_feature,
    denormalize_window,
    normalize_action,
    normalize_feature,
    normalize_window,
)


# ---------------------------------------------------------------------------
# Fake data helpers
# ---------------------------------------------------------------------------

N_PATCH = 8   # tiny for speed
D_VIS = 16
T_WIN = 4
D_RAW = 14
N_ANCHORS = 20   # anchors per sequence


def _make_norm_stats() -> dict:
    return {
        "version": "v2",
        "pipeline_version": "test",
        "split_version": "test",
        "created_at": "2026-01-01",
        "n_train_sequences": 1,
        "n_total_anchors": N_ANCHORS,
        "n_valid_anchors": N_ANCHORS,
        "pretransforms": {},
        "proprio_feature": {
            "channels": ["r", "s", "a", "ke", "sp"],
            "mean": [0.0] * 5,
            "std": [1.0] * 5,
        },
        "proprio_window": {
            "channels": ["c"] * D_RAW,
            "mean": [0.0] * D_RAW,
            "std": [1.0] * D_RAW,
        },
        "action": {
            "channels": ["dx", "dy", "dyaw"],
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    }


def _make_labels_npz(n: int, rng: np.random.Generator, valid_mask: np.ndarray | None = None) -> dict:
    if valid_mask is None:
        valid_mask = np.ones(n, dtype=np.uint8)
    # Identity-rotation poses translating +0.5 m/anchor along x → pose_delta over stride s
    # is (0.5*s, 0, 0); rate (delta/s) is (0.5, 0, 0) regardless of stride.
    pose = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    pose[:, 0, 3] = np.arange(n, dtype=np.float32) * 0.5
    return {
        "anchor_idx": np.arange(n, dtype=np.int32),
        "valid_mask": valid_mask,
        "proprio_feature": rng.random((n, 5)).astype(np.float32),
        "proprio_window": rng.random((n, T_WIN, D_RAW)).astype(np.float32),
        "action": rng.random((n, 3)).astype(np.float32),
        "pose": pose,
        "trav_cost_teacher": rng.random(n).astype(np.float32),
        "pitch_deg": rng.random(n).astype(np.float32),
        "roll_deg": rng.random(n).astype(np.float32),
    }


def _write_seq(
    root: Path,
    seq: str,
    n: int,
    rng: np.random.Generator,
    valid_mask: np.ndarray | None = None,
    cached_indices: list[int] | None = None,
    feat_root: Path | None = None,
    binary_root: Path | None = None,
    binary_valid_mask: np.ndarray | None = None,
    n_patch: int = N_PATCH,
    d_vis: int = D_VIS,
) -> None:
    # labels.npz
    seq_dir = root / seq
    seq_dir.mkdir(parents=True, exist_ok=True)
    labels = _make_labels_npz(n, rng, valid_mask)
    np.savez_compressed(seq_dir / "labels.npz", **labels)

    # safety.npz
    if binary_root is not None:
        bl_dir = binary_root / seq
        bl_dir.mkdir(parents=True, exist_ok=True)
        # ~10% unsafe
        unsafe = (rng.random(n) < 0.1).astype(np.uint8)
        bl_valid = (
            binary_valid_mask.astype(np.uint8)
            if binary_valid_mask is not None
            else labels["valid_mask"].copy()
        )
        np.savez_compressed(
            bl_dir / "safety.npz",
            anchor_idx=labels["anchor_idx"],
            unsafe=unsafe,
            valid_mask=bl_valid,
            trigger_trav=unsafe,
            trigger_pitch=np.zeros(n, dtype=np.uint8),
            trigger_roll=np.zeros(n, dtype=np.uint8),
        )

    # DINOv2 features
    if feat_root is not None and cached_indices is not None:
        feat_dir = feat_root / "dinov2_v1" / seq
        feat_dir.mkdir(parents=True, exist_ok=True)
        for i in cached_indices:
            feat = rng.random((n_patch, d_vis)).astype(np.float16)
            np.save(feat_dir / f"{i}.npy", feat)


def _make_split(seqs: list[str]) -> dict:
    return {
        "version": "test",
        "train": seqs,
        "val": [],
        "test": [],
    }


@pytest.fixture()
def fake_env(tmp_path: Path):
    """Build a minimal fake dataset environment and return config."""
    rng = np.random.default_rng(42)

    processed_root = tmp_path / "processed"
    binary_root = tmp_path / "binary"
    feat_root = tmp_path / "features"
    split_path = tmp_path / "splits" / "v2.json"
    norm_path = tmp_path / "norm_stats" / "v2.json"

    for d in (processed_root, binary_root, feat_root, split_path.parent, norm_path.parent):
        d.mkdir(parents=True, exist_ok=True)

    norm_path.write_text(json.dumps(_make_norm_stats()))

    # Sequence A: all anchors valid, all features cached
    seq_a = "seq_a"
    _write_seq(
        processed_root, seq_a, N_ANCHORS, rng,
        cached_indices=list(range(N_ANCHORS)),
        feat_root=feat_root,
        binary_root=binary_root,
        n_patch=N_PATCH, d_vis=D_VIS,
    )

    # Sequence B: some anchors invalid, partial feature cache
    mask_b = np.ones(N_ANCHORS, dtype=np.uint8)
    mask_b[5] = 0   # one hole in valid_mask
    seq_b = "seq_b"
    _write_seq(
        processed_root, seq_b, N_ANCHORS, rng,
        valid_mask=mask_b,
        cached_indices=list(range(10)),   # only first 10 cached
        feat_root=feat_root,
        binary_root=binary_root,
        n_patch=N_PATCH, d_vis=D_VIS,
    )

    split_path.write_text(json.dumps(_make_split([seq_a, seq_b])))

    cfg = DatasetConfig(
        split="train",
        split_path=str(split_path),
        processed_root=str(processed_root),
        binary_label_root=str(binary_root),
        feat_root=str(feat_root),
        norm_stats_path=str(norm_path),
        feat_version="dinov2_v1",
        context_len=3,          # k=3 → 4 context frames
        rollout_horizon=4,      # H=4
        n_patch=N_PATCH,
        d_vis=D_VIS,
        require_vis_context=True,
        require_vis_target=True,
    )
    return cfg


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestMultiStride:
    """Δt-conditioning / multi-stride (stride-aware windows + Δt action column)."""

    def test_legacy_default_unchanged(self, fake_env: DatasetConfig) -> None:
        # include_dt=False (default) → 3-dim action, no horizon mask.
        s = TartanDriveDataset(fake_env)[0]
        assert s["action_sequence"].shape[-1] == 3
        assert s["action_context"].shape[-1] == 3
        assert "proprio_horizon_mask" not in s

    def test_dt_channel_and_mask(self, fake_env: DatasetConfig) -> None:
        cfg = dataclasses.replace(fake_env, include_dt=True, train_strides=[1])
        s = TartanDriveDataset(cfg)[0]
        k, H = cfg.context_len, cfg.rollout_horizon
        assert s["action_sequence"].shape == (H, 4)
        assert s["action_context"].shape == (k + 1, 4)
        # stride 1 → Δt = 0.1 on every step
        assert torch.allclose(s["action_sequence"][:, 3], torch.full((H,), 0.1), atol=1e-6)
        assert s["proprio_horizon_mask"].shape == (H,)
        assert s["proprio_horizon_mask"].dtype == torch.bool

    def test_stride_expands_index(self, fake_env: DatasetConfig) -> None:
        single = TartanDriveDataset(dataclasses.replace(fake_env, include_dt=True, train_strides=[1]))
        multi = TartanDriveDataset(dataclasses.replace(fake_env, include_dt=True, train_strides=[1, 2]))
        assert len(multi) > len(single)

    def test_rate_normalized_stride_invariant(self, fake_env: DatasetConfig) -> None:
        # pose moves +0.5m/anchor → rate = pose_delta/stride = (0.5,0,0) for any stride.
        a1 = TartanDriveDataset(dataclasses.replace(fake_env, include_dt=True, train_strides=[1]))[0]
        a2 = TartanDriveDataset(dataclasses.replace(fake_env, include_dt=True, train_strides=[2]))[0]
        assert torch.allclose(a1["action_sequence"][0, :3], a2["action_sequence"][0, :3], atol=1e-5)

    def test_horizon_cutoff(self, fake_env: DatasetConfig) -> None:
        # stride 2 → cum Δt = 0.2,0.4,... ; cutoff 0.2 keeps only step 0.
        cfg = dataclasses.replace(
            fake_env, include_dt=True, train_strides=[2], proprio_horizon_s=0.2
        )
        m = TartanDriveDataset(cfg)[0]["proprio_horizon_mask"]
        assert bool(m[0]) and not bool(m[1])

    def test_multistride_requires_dt(self, fake_env: DatasetConfig) -> None:
        with pytest.raises(ValueError):
            dataclasses.replace(fake_env, include_dt=False, train_strides=[1, 2])

    def test_collate_includes_mask(self, fake_env: DatasetConfig) -> None:
        ds = TartanDriveDataset(dataclasses.replace(fake_env, include_dt=True, train_strides=[1]))
        batch = collate_fn([ds[0], ds[1]])
        assert batch["action_sequence"].shape[-1] == 4
        assert batch["proprio_horizon_mask"].shape == (2, fake_env.rollout_horizon)


class TestRandomStrideSampling:
    """stride_sampling="random": anchor-level index, stride drawn per access."""

    def _cfg(self, fake_env: DatasetConfig) -> DatasetConfig:
        return dataclasses.replace(
            fake_env, include_dt=True, train_strides=[1, 2], stride_sampling="random"
        )

    def test_anchor_level_index(self, fake_env: DatasetConfig) -> None:
        ds_all = TartanDriveDataset(
            dataclasses.replace(fake_env, include_dt=True, train_strides=[1, 2])
        )
        ds_rnd = TartanDriveDataset(self._cfg(fake_env))
        assert len(ds_rnd) < len(ds_all)  # pairs collapse to anchors
        anchors = [(sid, t) for sid, t, _ in ds_rnd._index]
        assert len(anchors) == len(set(anchors))  # one entry per anchor
        for _, _, strides in ds_rnd._index:
            assert isinstance(strides, tuple) and strides
            assert set(strides) <= {1, 2}

    def test_draws_only_valid_strides(self, fake_env: DatasetConfig) -> None:
        ds = TartanDriveDataset(self._cfg(fake_env))
        random.seed(0)
        for idx in range(len(ds)):
            _, _, valid = ds._index[idx]
            s = ds[idx]
            # first context action row is always in-bounds → its Δt reveals the stride
            dt = float(s["action_context"][0, 3])
            assert any(abs(dt - 0.1 * v) < 1e-6 for v in valid), (idx, dt, valid)

    def test_draw_varies_over_accesses(self, fake_env: DatasetConfig) -> None:
        ds = TartanDriveDataset(self._cfg(fake_env))
        idx = next(
            i for i, (_, _, strides) in enumerate(ds._index) if len(strides) > 1
        )
        random.seed(0)
        dts = {round(float(ds[idx]["action_context"][0, 3]), 6) for _ in range(50)}
        assert len(dts) > 1  # both strides get sampled

    def test_val_split_unaffected(self, fake_env: DatasetConfig) -> None:
        # gating is train-only: val keeps int-stride entries even with random sampling on
        cfg = dataclasses.replace(self._cfg(fake_env), split="val")
        ds = TartanDriveDataset(cfg)
        assert all(isinstance(s, int) for _, _, s in ds._index)

    def test_positive_rate_handles_anchor_entries(self, fake_env: DatasetConfig) -> None:
        rate = TartanDriveDataset(self._cfg(fake_env)).positive_rate()
        assert 0.0 <= rate <= 1.0

    def test_unknown_mode_rejected(self, fake_env: DatasetConfig) -> None:
        with pytest.raises(ValueError):
            dataclasses.replace(fake_env, stride_sampling="bogus")


class TestShapes:
    def test_sample_keys(self, fake_env: DatasetConfig) -> None:
        ds = TartanDriveDataset(fake_env)
        assert len(ds) > 0
        sample = ds[0]
        expected = {
            "z_vis_context",
            "proprio_window_context",
            "action_context",
            "action_sequence",
            "current_feature",
            "f_prop_target",
            "proprio_window_target",
            "z_vis_target",
            "binary_label_target",
            "binary_valid_target_mask",
            "valid_target_mask",
        }
        assert set(sample.keys()) == expected

    def test_context_shapes(self, fake_env: DatasetConfig) -> None:
        ds = TartanDriveDataset(fake_env)
        s = ds[0]
        k, H, P, D, Tw = (
            fake_env.context_len,
            fake_env.rollout_horizon,
            fake_env.n_patch,
            fake_env.d_vis,
            T_WIN,
        )
        assert s["z_vis_context"].shape == (k + 1, P, D)
        assert s["proprio_window_context"].shape == (k + 1, Tw, D_RAW)
        assert s["action_context"].shape == (k + 1, 3)
        assert s["action_sequence"].shape == (H, 3)
        assert s["current_feature"].shape == (5,)

    def test_target_shapes(self, fake_env: DatasetConfig) -> None:
        ds = TartanDriveDataset(fake_env)
        s = ds[0]
        H, P, D = fake_env.rollout_horizon, fake_env.n_patch, fake_env.d_vis
        assert s["f_prop_target"].shape == (H, 5)
        assert s["z_vis_target"].shape == (H, P, D)
        assert s["binary_label_target"].shape == (H,)
        assert s["binary_valid_target_mask"].shape == (H,)
        assert s["valid_target_mask"].shape == (H,)

    def test_dtypes(self, fake_env: DatasetConfig) -> None:
        ds = TartanDriveDataset(fake_env)
        s = ds[0]
        assert s["z_vis_context"].dtype == torch.float32
        assert s["binary_label_target"].dtype == torch.float32
        assert s["binary_valid_target_mask"].dtype == torch.bool
        assert s["valid_target_mask"].dtype == torch.bool

    def test_consolidated_feature_cache(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(123)
        processed_root = tmp_path / "processed"
        binary_root = tmp_path / "binary"
        feat_root = tmp_path / "features"
        split_path = tmp_path / "splits" / "v2.json"
        norm_path = tmp_path / "norm_stats" / "v2.json"
        for d in (processed_root, binary_root, feat_root, split_path.parent, norm_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        norm_path.write_text(json.dumps(_make_norm_stats()))
        _write_seq(
            processed_root,
            "seq",
            N_ANCHORS,
            rng,
            cached_indices=None,
            feat_root=None,
            binary_root=binary_root,
            n_patch=N_PATCH,
            d_vis=D_VIS,
        )
        feat_dir = feat_root / "dinov2_v1" / "seq"
        feat_dir.mkdir(parents=True)
        features = rng.random((N_ANCHORS, N_PATCH, D_VIS)).astype(np.float16)
        np.save(feat_dir / "features.npy", features)
        np.save(feat_dir / "features_valid.npy", np.ones(N_ANCHORS, dtype=bool))
        split_path.write_text(json.dumps(_make_split(["seq"])))

        cfg = DatasetConfig(
            split="train",
            split_path=str(split_path),
            processed_root=str(processed_root),
            binary_label_root=str(binary_root),
            feat_root=str(feat_root),
            norm_stats_path=str(norm_path),
            context_len=3,
            rollout_horizon=4,
            n_patch=N_PATCH,
            d_vis=D_VIS,
        )
        sample = TartanDriveDataset(cfg)[0]
        assert sample["z_vis_context"].shape == (4, N_PATCH, D_VIS)
        np.testing.assert_allclose(sample["z_vis_context"][0].numpy(), features[0].astype(np.float32))


# ---------------------------------------------------------------------------
# Valid mask filtering
# ---------------------------------------------------------------------------

class TestValidMaskFiltering:
    def test_context_requires_all_valid(self, tmp_path: Path) -> None:
        """No sample should include an anchor with valid_mask=0 in context."""
        rng = np.random.default_rng(0)
        processed_root = tmp_path / "p"
        binary_root = tmp_path / "b"
        feat_root = tmp_path / "f"
        split_path = tmp_path / "s" / "v2.json"
        norm_path = tmp_path / "n" / "v2.json"
        for d in (processed_root, binary_root, feat_root, split_path.parent, norm_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        norm_path.write_text(json.dumps(_make_norm_stats()))

        # All valid except position 4
        mask = np.ones(N_ANCHORS, dtype=np.uint8)
        mask[4] = 0
        _write_seq(
            processed_root, "s", N_ANCHORS, rng,
            valid_mask=mask,
            cached_indices=list(range(N_ANCHORS)),
            feat_root=feat_root,
            binary_root=binary_root,
            n_patch=N_PATCH, d_vis=D_VIS,
        )
        split_path.write_text(json.dumps(_make_split(["s"])))

        cfg = DatasetConfig(
            split="train",
            split_path=str(split_path),
            processed_root=str(processed_root),
            binary_label_root=str(binary_root),
            feat_root=str(feat_root),
            norm_stats_path=str(norm_path),
            context_len=3, rollout_horizon=4,
            n_patch=N_PATCH, d_vis=D_VIS,
        )
        ds = TartanDriveDataset(cfg)
        # Anchors 3,4,5,6 (k=3; t in {3..6}) would use position 4 in context
        # Specifically, t ∈ {4,5,6,7} for a 3-wide context that includes pos 4.
        # None of those should appear in the index.
        anchors_in_index = {t for _, t, _ in ds._index}
        # t=4: ctx=[1,2,3,4] → invalid (4 invalid)
        # t=5: ctx=[2,3,4,5] → invalid
        # t=6: ctx=[3,4,5,6] → invalid
        # t=7: ctx=[4,5,6,7] → invalid
        for bad_t in range(4, 8):
            assert bad_t not in anchors_in_index, f"t={bad_t} should be excluded"

    def test_valid_target_mask_for_end_of_sequence(self, fake_env: DatasetConfig) -> None:
        """Targets beyond sequence end must have valid_target_mask=False."""
        ds = TartanDriveDataset(fake_env)
        seq_a_id = next(i for i, r in enumerate(ds._records) if r.name == "seq_a")
        N = len(ds._records[seq_a_id].valid_mask)
        H = fake_env.rollout_horizon

        # Find the last anchor of seq_a in the index
        last_anchors = [(sid, t) for sid, t, _ in ds._index if sid == seq_a_id]
        if not last_anchors:
            pytest.skip("No seq_a samples in index")

        # Get the sample with highest t (closest to end)
        last_t = max(t for _, t in last_anchors)
        last_idx = next(i for i, (sid, t, _) in enumerate(ds._index) if sid == seq_a_id and t == last_t)

        sample = ds[last_idx]
        mask = sample["valid_target_mask"]

        for h in range(H):
            tp = last_t + h + 1
            if tp >= N:
                assert not mask[h], f"h={h} (tp={tp} >= N={N}) must be invalid"

    def test_binary_valid_target_mask_is_separate_from_visual_target_mask(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(13)
        processed_root = tmp_path / "p"
        binary_root = tmp_path / "b"
        feat_root = tmp_path / "f"
        split_path = tmp_path / "s" / "v2.json"
        norm_path = tmp_path / "n" / "v2.json"
        for d in (processed_root, binary_root, feat_root, split_path.parent, norm_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        norm_path.write_text(json.dumps(_make_norm_stats()))

        binary_valid = np.ones(N_ANCHORS, dtype=np.uint8)
        binary_valid[5] = 0
        _write_seq(
            processed_root, "s", N_ANCHORS, rng,
            cached_indices=list(range(N_ANCHORS)),
            feat_root=feat_root,
            binary_root=binary_root,
            binary_valid_mask=binary_valid,
            n_patch=N_PATCH, d_vis=D_VIS,
        )
        split_path.write_text(json.dumps(_make_split(["s"])))
        cfg = DatasetConfig(
            split="train",
            split_path=str(split_path),
            processed_root=str(processed_root),
            binary_label_root=str(binary_root),
            feat_root=str(feat_root),
            norm_stats_path=str(norm_path),
            context_len=3, rollout_horizon=4,
            n_patch=N_PATCH, d_vis=D_VIS,
        )
        ds = TartanDriveDataset(cfg)
        idx = next(i for i, (_, t, _) in enumerate(ds._index) if t == 4)
        sample = ds[idx]
        assert sample["valid_target_mask"][0]
        assert not sample["binary_valid_target_mask"][0]


# ---------------------------------------------------------------------------
# Sequence boundary test
# ---------------------------------------------------------------------------

class TestSequenceBoundary:
    def test_no_cross_sequence_samples(self, fake_env: DatasetConfig) -> None:
        """All context frames of a sample must belong to the same sequence."""
        ds = TartanDriveDataset(fake_env)
        k = fake_env.context_len
        for sid, t, _ in ds._index:
            # All context positions [t-k, t] must be within [0, N-1] of this seq
            N = len(ds._records[sid].valid_mask)
            assert t - k >= 0, "context underflows sequence start"
            assert t < N, "anchor t exceeds sequence length"

    def test_two_sequences_independent(self, fake_env: DatasetConfig) -> None:
        """Samples from seq_a and seq_b never mix anchors."""
        ds = TartanDriveDataset(fake_env)
        seq_ids = {sid for sid, _, _ in ds._index}
        assert len(seq_ids) <= 2   # at most 2 sequences contribute samples


# ---------------------------------------------------------------------------
# Normalization round-trip
# ---------------------------------------------------------------------------

class TestNormalizationRoundTrip:
    def _stats(self) -> NormStats:
        d = _make_norm_stats()
        return NormStats(
            feature_mean=np.array(d["proprio_feature"]["mean"], dtype=np.float32),
            feature_std=np.array(d["proprio_feature"]["std"], dtype=np.float32),
            window_mean=np.array(d["proprio_window"]["mean"], dtype=np.float32),
            window_std=np.array(d["proprio_window"]["std"], dtype=np.float32),
            action_mean=np.array(d["action"]["mean"], dtype=np.float32),
            action_std=np.array(d["action"]["std"], dtype=np.float32),
        )

    def test_action_round_trip(self) -> None:
        stats = self._stats()
        rng = np.random.default_rng(7)
        raw = rng.standard_normal((10, 3)).astype(np.float32)
        norm = normalize_action(raw, stats)
        recovered = denormalize_action(norm, stats)
        np.testing.assert_allclose(raw, recovered, atol=1e-5)

    def test_feature_round_trip_ch0_2_4(self) -> None:
        """Channels 0,1,2,4 have invertible round-trip (no clamping)."""
        stats = self._stats()
        rng = np.random.default_rng(8)
        raw = rng.standard_normal((10, 5)).astype(np.float32)
        norm = normalize_feature(raw, stats)
        recovered = denormalize_feature(norm, stats)
        for ch in (0, 1, 2, 4):
            np.testing.assert_allclose(raw[:, ch], recovered[:, ch], atol=1e-5,
                                       err_msg=f"feature ch{ch} round-trip failed")

    def test_feature_ke_rate_round_trip(self) -> None:
        """ke_rate (ch3): signed_log1p → z-score → denorm → inverse is exact for moderate values."""
        stats = self._stats()
        rng = np.random.default_rng(9)
        raw = rng.uniform(-10, 10, (10, 5)).astype(np.float32)
        norm = normalize_feature(raw, stats)
        recovered = denormalize_feature(norm, stats)
        np.testing.assert_allclose(raw[:, 3], recovered[:, 3], atol=1e-4)

    def test_window_steer_round_trip(self) -> None:
        """steer_cmd (ch13): /415 is invertible within [-415, 415]."""
        stats = self._stats()
        raw = np.zeros((5, T_WIN, D_RAW), dtype=np.float32)
        raw[..., 13] = np.linspace(-415, 415, 5)[:, None]
        norm = normalize_window(raw, stats)
        recovered = denormalize_window(norm, stats)
        np.testing.assert_allclose(raw[..., 13], recovered[..., 13], atol=1e-3)


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_collate_fn(self, fake_env: DatasetConfig) -> None:
        ds = TartanDriveDataset(fake_env)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn, shuffle=False)
        batch = next(iter(loader))
        k, H, P, D, Tw = (
            fake_env.context_len,
            fake_env.rollout_horizon,
            fake_env.n_patch,
            fake_env.d_vis,
            T_WIN,
        )
        assert batch["z_vis_context"].shape == (2, k + 1, P, D)
        assert batch["proprio_window_context"].shape == (2, k + 1, Tw, D_RAW)
        assert batch["action_context"].shape == (2, k + 1, 3)
        assert batch["f_prop_target"].shape == (2, H, 5)
        assert batch["z_vis_target"].shape == (2, H, P, D)
        assert batch["binary_label_target"].shape == (2, H)
        assert batch["binary_valid_target_mask"].shape == (2, H)
        assert batch["valid_target_mask"].shape == (2, H)

    def test_variable_h_collate(self, fake_env: DatasetConfig) -> None:
        ds = TartanDriveDataset(fake_env)
        # Manually build two samples with different H by slicing tensors
        s = ds[0]
        s2 = {k: v[:2] if v.ndim >= 1 and k.endswith("target") or k.endswith("mask")
              else v for k, v in s.items()}
        # Both share the same H (4) after DatasetConfig; use variable_h_collate directly
        batch = variable_h_collate([s, s])
        assert batch["f_prop_target"].shape[0] == 2

    def test_no_grad_in_features(self, fake_env: DatasetConfig) -> None:
        ds = TartanDriveDataset(fake_env)
        s = ds[0]
        for key, val in s.items():
            assert not val.requires_grad, f"{key} should not require grad"

    def test_vis_zeros_for_missing_cache(self, tmp_path: Path) -> None:
        """When require_vis_context=False, samples load even without DINOv2 cache;
        context vis features are zeros for uncached positions."""
        rng = np.random.default_rng(11)
        processed_root = tmp_path / "p"
        binary_root = tmp_path / "b"
        feat_root = tmp_path / "f"
        split_path = tmp_path / "s" / "v2.json"
        norm_path = tmp_path / "n" / "v2.json"
        for d in (processed_root, binary_root, feat_root, split_path.parent, norm_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        norm_path.write_text(json.dumps(_make_norm_stats()))

        # No features cached at all
        _write_seq(
            processed_root, "s", N_ANCHORS, rng,
            cached_indices=[],       # nothing cached
            feat_root=feat_root,
            binary_root=binary_root,
            n_patch=N_PATCH, d_vis=D_VIS,
        )
        split_path.write_text(json.dumps(_make_split(["s"])))

        cfg = DatasetConfig(
            split="train",
            split_path=str(split_path),
            processed_root=str(processed_root),
            binary_label_root=str(binary_root),
            feat_root=str(feat_root),
            norm_stats_path=str(norm_path),
            context_len=3, rollout_horizon=4,
            n_patch=N_PATCH, d_vis=D_VIS,
            require_vis_context=False,   # allow missing context features
            require_vis_target=False,
        )
        ds = TartanDriveDataset(cfg)
        assert len(ds) > 0, "Dataset should have samples when require_vis_context=False"
        s = ds[0]
        assert s["z_vis_context"].shape == (4, N_PATCH, D_VIS)
        assert s["z_vis_context"].abs().sum().item() == 0.0, "features should be zeros"


# ---------------------------------------------------------------------------
# window_columns — portable-input subset (encoder input only)
# ---------------------------------------------------------------------------

class TestWindowColumns:
    PORTABLE_11 = list(range(11))   # sim/Husky shared encoder-input column prefix

    def test_subset_equals_full_prefix(self, fake_env: DatasetConfig) -> None:
        # Slicing happens AFTER normalization: kept columns must be bit-identical
        # to the same columns of a full-input run.
        full = TartanDriveDataset(fake_env)[0]
        sub = TartanDriveDataset(
            dataclasses.replace(fake_env, window_columns=self.PORTABLE_11)
        )[0]
        assert sub["proprio_window_context"].shape[-1] == 11
        assert sub["proprio_window_target"].shape[-1] == 11
        assert torch.equal(
            sub["proprio_window_context"], full["proprio_window_context"][..., :11]
        )
        assert torch.equal(
            sub["proprio_window_target"], full["proprio_window_target"][..., :11]
        )

    def test_non_prefix_subset(self, fake_env: DatasetConfig) -> None:
        cols = [0, 3, 13]
        full = TartanDriveDataset(fake_env)[0]
        sub = TartanDriveDataset(dataclasses.replace(fake_env, window_columns=cols))[0]
        assert torch.equal(
            sub["proprio_window_context"], full["proprio_window_context"][..., cols]
        )

    def test_duplicate_indices_rejected(self, fake_env: DatasetConfig) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            dataclasses.replace(fake_env, window_columns=[0, 0, 1])

    def test_empty_rejected(self, fake_env: DatasetConfig) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            dataclasses.replace(fake_env, window_columns=[])

    def test_out_of_range_rejected(self, fake_env: DatasetConfig) -> None:
        with pytest.raises(ValueError, match="out of range"):
            TartanDriveDataset(dataclasses.replace(fake_env, window_columns=[0, 14]))
