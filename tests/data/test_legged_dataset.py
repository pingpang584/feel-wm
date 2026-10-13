"""Legged loader contract: trainer-compatible batch from labels_legged.npz, per-dataset norm,
runs the proprio path with zero-vis before DINO is cached. Uses synthesized npz (portable).

Second half: the multistride/Δt upgrade —
stride-aware validity, the Δt column, the proprio horizon mask, and the dropped-frame guard."""
import json

import numpy as np
import pytest
import torch
import yaml

from pathlib import Path

from feelwm.config import load_config
from feelwm.data.legged_dataset import (
    LeggedDataset, LeggedDatasetConfig, LeggedSeqSpec, discover_legged_specs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_labels(path, n=40, t_win=50, d_raw=53, timestamp=None, valid_mask=None, trav_gt=None):
    rng = np.random.default_rng(0)
    pose = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    pose[:, 0, 3] = np.arange(n) * 0.1   # moving forward so actions are non-trivial
    extra = {} if trav_gt is None else {"trav_gt": trav_gt}   # Isaac corpus writes this field
    np.savez(
        path,
        anchor_idx=np.arange(n, dtype=np.int32),
        timestamp=np.arange(n, dtype=np.float64) if timestamp is None else timestamp,
        proprio_window=rng.standard_normal((n, t_win, d_raw)).astype(np.float32),
        proprio_feature=rng.standard_normal((n, 5)).astype(np.float32),
        pose=pose,
        action=rng.standard_normal((n, 3)).astype(np.float32) * 0.1,
        valid_mask=np.ones(n, np.uint8) if valid_mask is None else valid_mask,
        **extra,
    )


def _write_stats(path, d_raw=53):
    path.write_text(json.dumps({
        "feature_set": "legged",
        "proprio_feature": {"mean": [0] * 5, "std": [1] * 5},
        "proprio_window": {"mean": [0] * d_raw, "std": [1] * d_raw},
        "action": {"mean": [0] * 3, "std": [1] * 3},
    }))


def test_legged_loader_batch_contract(tmp_path):
    lbl = tmp_path / "labels_legged.npz"
    _write_labels(lbl)
    stats = tmp_path / "legged_stats.json"
    _write_stats(stats)
    specs = [LeggedSeqSpec("gt", "m0", lbl, tmp_path / "nofeat")]
    cfg = LeggedDatasetConfig(context_len=3, rollout_horizon=8, n_patch=4, d_vis=16,
                              require_vis_context=False, require_vis_target=False,
                              norm_stats={"gt": str(stats)})
    ds = LeggedDataset(specs, cfg)
    assert len(ds) == 40 - 3            # anchors with full context
    # trainer per-sequence logging reads record.name (regression: AttributeError on val)
    assert ds._records[ds._index[0][0]].name == "m0"

    b = ds[0]
    assert b["proprio_window_context"].shape == (4, 50, 53)
    assert b["f_prop_target"].shape == (8, 5)
    assert b["z_vis_context"].shape == (4, 4, 16)
    assert b["action_sequence"].shape == (8, 3)
    # zero-vis placeholder before caching
    assert torch.count_nonzero(b["z_vis_context"]) == 0


def test_legged_loader_rejects_wheeled_stats(tmp_path):
    lbl = tmp_path / "labels_legged.npz"; _write_labels(lbl)
    bad = tmp_path / "wheeled.json"
    bad.write_text(json.dumps({   # no feature_set -> defaults wheeled
        "proprio_feature": {"mean": [0] * 5, "std": [1] * 5},
        "proprio_window": {"mean": [0] * 53, "std": [1] * 53},
        "action": {"mean": [0] * 3, "std": [1] * 3}}))
    specs = [LeggedSeqSpec("gt", "m0", lbl, tmp_path / "nofeat")]
    cfg = LeggedDatasetConfig(require_vis_context=False, norm_stats={"gt": str(bad)})
    try:
        LeggedDataset(specs, cfg)
        assert False, "should reject non-legged stats"
    except ValueError as e:
        assert "feature_set" in str(e)


def test_discover_specs(tmp_path):
    gt = tmp_path / "grand_tour"
    (gt / "2024-mission").mkdir(parents=True)
    _write_labels(gt / "2024-mission" / "labels_legged.npz")
    tg = tmp_path / "tartanground" / "Gascola" / "Data_anymal" / "P2000"
    tg.mkdir(parents=True)
    _write_labels(tg / "labels_legged.npz")
    specs = discover_legged_specs(grandtour_root=gt, tartanground_root=tmp_path / "tartanground")
    tags = sorted(s.tag for s in specs)
    names = sorted(s.name for s in specs)
    assert tags == ["gt", "tg"]
    assert names == ["2024-mission", "Gascola_P2000"]


def test_discover_specs_isaac(tmp_path):
    """Isaac ANYmal layout: flat <root>/<episode>/labels_legged.npz, tag 'isaac',
    feat dir defaults to the DINOv3 cache name."""
    root = tmp_path / "sim_anymal_isaac" / "sequences" / "all"
    for ep in ("Asphalt1__cov_02__v0.7__ep00", "Hill2__canonical__v1__ep00"):
        (root / ep).mkdir(parents=True)
        _write_labels(root / ep / "labels_legged.npz", trav_gt=np.zeros(40, np.uint8))
    specs = discover_legged_specs(isaac_root=root)
    assert [s.tag for s in specs] == ["isaac", "isaac"]
    assert sorted(s.name for s in specs) == [
        "Asphalt1__cov_02__v0.7__ep00", "Hill2__canonical__v1__ep00"]
    assert specs[0].feat_dir.name == "anymal_isaac_dinov3_v1"


# --- binary trav GT (Isaac ANYmal corpus) --------------------------------------------


def test_trav_gt_reaches_binary_keys_with_stride_indexing(tmp_path):
    """When the npz has trav_gt, binary_label/valid follow sim_dataset.py semantics:
    label at t+h·s, valid where in-bounds and label-valid."""
    n = 40
    trav = (np.arange(n) % 3 == 0).astype(np.uint8)   # known non-trivial pattern
    lbl = tmp_path / "labels_legged.npz"
    _write_labels(lbl, n=n, timestamp=np.arange(n, dtype=np.float64) * 0.1, trav_gt=trav)
    stats = tmp_path / "legged_stats.json"
    _write_stats(stats)
    specs = [LeggedSeqSpec("isaac", "ep0", lbl, tmp_path / "nofeat")]
    s = 2
    ds = LeggedDataset(specs, LeggedDatasetConfig(
        context_len=K, rollout_horizon=H, n_patch=4, d_vis=16,
        require_vis_context=False, require_vis_target=False,
        norm_stats={"isaac": str(stats)}, include_dt=True, split="val", val_stride=s))
    b = ds[0]
    _, t, _ = ds._index[0]
    exp_label = [float(trav[t + h * s]) if t + h * s < n else 0.0 for h in range(1, H + 1)]
    exp_valid = [t + h * s < n for h in range(1, H + 1)]
    assert b["binary_label_target"].tolist() == exp_label
    assert b["binary_valid_target_mask"].tolist() == exp_valid
    assert b["binary_label_target"].sum() > 0      # the pattern actually reached the sample


def test_no_trav_gt_keeps_binary_keys_zero(legged_10hz):
    """TG/GT regression: extractions without trav_gt keep the all-zero binary keys
    (lambda_trav=0 path, bit-identical to the pre-isaac loader)."""
    specs, stats = legged_10hz
    b = LeggedDataset(specs, _cfg(stats, val_stride=1))[0]
    assert torch.count_nonzero(b["binary_label_target"]) == 0
    assert torch.count_nonzero(b["binary_valid_target_mask"]) == 0


# --- multistride / Δt upgrade -----------------------------------------------------

K, H, N_ANCH = 3, 8, 60


@pytest.fixture
def legged_10hz(tmp_path):
    """One 10 Hz mission + legged stats — the real anchor grid of both extractions."""
    lbl = tmp_path / "labels_legged.npz"
    _write_labels(lbl, n=N_ANCH, timestamp=np.arange(N_ANCH, dtype=np.float64) * 0.1)
    stats = tmp_path / "legged_stats.json"
    _write_stats(stats)
    return [LeggedSeqSpec("gt", "m0", lbl, tmp_path / "nofeat")], str(stats)


def _cfg(stats, **kw):
    base = dict(context_len=K, rollout_horizon=H, n_patch=4, d_vis=16,
                require_vis_context=False, require_vis_target=False,
                norm_stats={"gt": stats}, include_dt=True, split="val")
    base.update(kw)
    return LeggedDatasetConfig(**base)


@pytest.mark.parametrize("stride", [1, 2, 5, 10])
def test_dt_column_matches_stride(legged_10hz, stride):
    specs, stats = legged_10hz
    ds = LeggedDataset(specs, _cfg(stats, val_stride=stride))
    b = ds[0]
    _, t, _ = ds._index[0]
    assert b["action_context"].shape == (K + 1, 4)
    assert b["action_sequence"].shape == (H, 4)
    dt = b["action_sequence"][:, 3]
    # step h spans anchors t+h·stride → t+(h+1)·stride; past the sequence end the row is
    # zero-padded (as in sim_dataset) and its target step is masked out.
    in_bounds = torch.tensor([t + (h + 1) * stride < N_ANCH for h in range(H)])
    assert in_bounds.any()
    assert torch.allclose(dt[in_bounds], torch.tensor(stride * 0.1), atol=1e-6)
    assert torch.all(dt[~in_bounds] == 0.0)


@pytest.mark.parametrize("stride, expected_on", [(1, 8), (2, 5), (5, 2), (10, 1)])
def test_proprio_horizon_mask_per_stride(legged_10hz, stride, expected_on):
    """proprio_horizon_s=1.0 → supervise steps with cum Δt = h·stride·0.1 s ≤ 1.0 s."""
    specs, stats = legged_10hz
    ds = LeggedDataset(specs, _cfg(stats, val_stride=stride))
    mask = ds[0]["proprio_horizon_mask"]
    assert torch.equal(mask, torch.tensor([True] * expected_on + [False] * (H - expected_on)))


def test_stride_context_respects_valid_mask(legged_10hz, tmp_path):
    """A window is indexed only if every USED (strided) context anchor is valid."""
    lbl = tmp_path / "invalid.npz"
    vm = np.ones(N_ANCH, np.uint8)
    vm[0] = 0
    _write_labels(lbl, n=N_ANCH, timestamp=np.arange(N_ANCH, dtype=np.float64) * 0.1,
                  valid_mask=vm)
    _, stats = legged_10hz
    specs = [LeggedSeqSpec("gt", "m0", lbl, tmp_path / "nofeat")]
    anchors = {t for _, t, _ in LeggedDataset(specs, _cfg(stats, val_stride=5))._index}
    assert 15 not in anchors     # context {0,5,10,15} touches the invalid anchor 0
    assert 20 in anchors         # context {5,10,15,20} does not


def test_dropped_frame_windows_are_dropped(legged_10hz, tmp_path):
    """Δt is emitted as stride·0.1 s, so windows spanning a camera gap must not be indexed
    (GrandTour 2024-10-01-12-00-49 has two such steps)."""
    ts = np.arange(N_ANCH, dtype=np.float64) * 0.1
    ts[21:] += 0.5                       # 20 -> 21 spans 0.6 s instead of 0.1 s
    lbl = tmp_path / "gap.npz"
    _write_labels(lbl, n=N_ANCH, timestamp=ts)
    _, stats = legged_10hz
    specs = [LeggedSeqSpec("gt", "m0", lbl, tmp_path / "nofeat")]

    anchors = {t for _, t, _ in LeggedDataset(specs, _cfg(stats, val_stride=1))._index}
    assert anchors == set(range(K, N_ANCH)) - set(range(13, 24))   # spans [t-3, t+8]
    # legacy single-stride path is untouched: no Δt claim, no screening
    legacy = LeggedDataset(specs, _cfg(stats, include_dt=False, val_stride=1))
    assert len(legacy) == N_ANCH - K


def test_random_stride_sampling_is_train_only(legged_10hz):
    specs, stats = legged_10hz
    train = LeggedDataset(specs, _cfg(stats, split="train", stride_sampling="random",
                                      train_strides=[1, 2, 5, 10]))
    assert all(isinstance(s, tuple) for _, _, s in train._index)   # one draw per access
    assert train[0]["action_sequence"].shape == (H, 4)
    val = LeggedDataset(specs, _cfg(stats, split="val", stride_sampling="random",
                                    train_strides=[1, 2, 5, 10], val_stride=1))
    assert all(s == 1 for _, _, s in val._index)                   # val is single-stride


def test_both_datasets_load_multistride(legged_10hz, tmp_path):
    """GT + TG in one dataset, each normed by its own stats (per-corpus normalization)."""
    specs, gt_stats = legged_10hz
    tg_lbl = tmp_path / "tg_labels.npz"
    _write_labels(tg_lbl, n=N_ANCH, timestamp=np.arange(N_ANCH, dtype=np.float64) * 0.1)
    tg_stats = tmp_path / "tg_stats.json"
    _write_stats(tg_stats)
    specs = specs + [LeggedSeqSpec("tg", "Gascola_P2000", tg_lbl, tmp_path / "nofeat")]
    ds = LeggedDataset(specs, _cfg(gt_stats, split="train", stride_sampling="random",
                                   train_strides=[1, 2, 5, 10],
                                   norm_stats={"gt": gt_stats, "tg": str(tg_stats)}))
    assert ds.sequence_names() == ["m0", "Gascola_P2000"]
    assert ds[len(ds) - 1]["action_sequence"].shape == (H, 4)


# --- active proprio channel selection ---------------------------------------------------


def test_active_channels_default_is_the_full_registry(legged_10hz):
    """None = all 5, bit-identical to the v1 single-stride behavior."""
    specs, stats = legged_10hz
    base = LeggedDataset(specs, _cfg(stats, val_stride=1))
    explicit = LeggedDataset(specs, _cfg(stats, val_stride=1, active_proprio_channels=None))
    assert base[0]["f_prop_target"].shape == (H, 5)
    assert base[0]["current_feature"].shape == (5,)
    torch.testing.assert_close(base[0]["f_prop_target"], explicit[0]["f_prop_target"])


def test_active_channels_select_by_name_in_config_order(legged_10hz):
    """Columns are sliced from the LEGGED registry
    [roughness, angular_instability, slope_power, slip, cost_of_transport],
    in the order the config lists them — not the registry's order."""
    specs, stats = legged_10hz
    full = LeggedDataset(specs, _cfg(stats, val_stride=1))[0]
    v2 = LeggedDataset(specs, _cfg(stats, val_stride=1, active_proprio_channels=[
        "roughness", "angular_instability", "slope_power", "cost_of_transport"]))[0]
    assert v2["f_prop_target"].shape == (H, 4)
    assert v2["current_feature"].shape == (4,)
    torch.testing.assert_close(v2["f_prop_target"], full["f_prop_target"][:, [0, 1, 2, 4]])
    torch.testing.assert_close(v2["current_feature"], full["current_feature"][[0, 1, 2, 4]])

    reordered = LeggedDataset(specs, _cfg(stats, val_stride=1, active_proprio_channels=[
        "cost_of_transport", "roughness"]))[0]
    torch.testing.assert_close(reordered["f_prop_target"], full["f_prop_target"][:, [4, 0]])


def test_active_channels_resolve_against_the_legged_registry(legged_10hz):
    """`slip_ratio`/`ke_rate` are WHEELED names — rejected; and an all-shared-name subset
    must resolve to legged indices (auto-detection would silently pick the wheeled ones)."""
    specs, stats = legged_10hz
    with pytest.raises(ValueError, match="unknown proprio channel"):
        LeggedDatasetConfig(norm_stats={"gt": stats}, active_proprio_channels=["slip_ratio"])
    with pytest.raises(ValueError, match="non-empty"):
        LeggedDatasetConfig(norm_stats={"gt": stats}, active_proprio_channels=[])

    full = LeggedDataset(specs, _cfg(stats, val_stride=1))[0]
    shared = LeggedDataset(specs, _cfg(stats, val_stride=1, active_proprio_channels=[
        "roughness", "angular_instability", "slope_power"]))[0]
    torch.testing.assert_close(shared["f_prop_target"], full["f_prop_target"][:, [0, 1, 2]])


def test_v2_config_channels_reach_the_sample(legged_10hz):
    """Parse-through: the shipped v2 config's data block → dataset config → sample width
    equals its model.proprio_feature_dim (a mismatch crashes the first training step)."""
    specs, stats = legged_10hz
    cfg = load_config(REPO_ROOT / "configs/legged_tg_gt.yaml")
    ds = LeggedDataset(specs, _cfg(
        stats, val_stride=1,
        active_proprio_channels=cfg["data"]["active_proprio_channels"],
        proprio_horizon_s=cfg["data"]["proprio_horizon_s"]))
    assert ds[0]["f_prop_target"].shape[-1] == cfg["model"]["proprio_feature_dim"] == 4


def test_builder_probe_rejects_a_width_mismatch():
    """The build-time guard: a loader that ignored the channel selection must fail HERE,
    not inside DDP teacher forcing on the first train batch."""
    from feelwm.models.terrain_wm import ModelConfig
    from feelwm.training.trainer import _assert_sample_proprio_dim

    m = ModelConfig(proprio_feature_dim=4)
    _assert_sample_proprio_dim(
        {"f_prop_target": torch.zeros(H, 4), "current_feature": torch.zeros(4)}, m, "legged train")
    with pytest.raises(ValueError, match="active_proprio_channels"):
        _assert_sample_proprio_dim({"f_prop_target": torch.zeros(H, 5)}, m, "legged train")
    with pytest.raises(ValueError, match="current_feature"):
        _assert_sample_proprio_dim(
            {"f_prop_target": torch.zeros(H, 4), "current_feature": torch.zeros(5)},
            m, "legged val")


def _write_feats(feat_dir: Path, n: int, n_patch: int, d_vis: int) -> None:
    feat_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_dir / "features.npy", np.zeros((n, n_patch, d_vis), np.float32))
    np.save(feat_dir / "features_valid.npy", np.ones(n, bool))


def _tiny_legged_yaml(tmp_path, *, active, feature_dim, n=40, d_raw=8, n_patch=4, d_vis=16):
    """Minimal but REAL config → build_legged_trainer_from_config path (synthetic data)."""
    gt = tmp_path / "grand_tour"
    for name in ("train-mission", "2024-12-heldout"):
        (gt / name).mkdir(parents=True)
        _write_labels(gt / name / "labels_legged.npz", n=n, d_raw=d_raw,
                      timestamp=np.arange(n, dtype=np.float64) * 0.1)
        _write_feats(gt / name / "features" / "gt_dinov2reg_v1", n, n_patch, d_vis)
    stats = tmp_path / "legged_stats.json"
    _write_stats(stats, d_raw=d_raw)
    cfg = {
        "model": {"context_len": K, "rollout_horizon": H, "n_patch": n_patch, "d_vis": d_vis,
                  "d_prop": 8, "d_action": 4, "d_raw": d_raw, "n_layers": 1, "n_heads": 2,
                  "ffn_mult": 2, "dropout": 0.0, "use_proprio_input": True,
                  "use_proprio_feature_head": True, "use_traversability_head": False,
                  "proprio_feature_dim": feature_dim},
        "data": {"norm_stats_path_gt": str(stats), "norm_stats_path_tg": str(stats),
                 "grandtour_root": str(gt), "tartanground_root": None,
                 "context_len": K, "rollout_horizon": H,
                 "active_proprio_channels": active,
                 "include_dt": True, "train_strides": [1, 2], "stride_sampling": "random",
                 "val_stride": 1, "proprio_horizon_s": 1.0,
                 "heldout_grandtour_missions": ["2024-12-heldout"]},
        "loss": {"lambda_prop": 0.5, "lambda_trav": 0.0, "lambda_cos": 0.1,
                 "channel_names": list(active), "proprio_weights": [1.0] * len(active)},
        "curriculum": {"enabled": False},
        "scheduled_sampling": {"enabled": True, "start_epoch": 5, "end_epoch": 15,
                               "p_start": 1.0, "p_end": 0.5},
        "training": {"epochs": 1, "batch_size": 2, "num_workers": 0, "device": "cpu",
                     "amp": False, "progress": False, "seed": 0,
                     "checkpoint_path": str(tmp_path / "ckpt.pt"),
                     "metrics_path": str(tmp_path / "metrics.jsonl"),
                     "per_sequence_dir": str(tmp_path / "per_sequence")},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_legged_builder_plumbs_active_channels(tmp_path):
    """End-to-end: config `data.active_proprio_channels` reaches the dataset (the plumbing
    that was once missing on the legged path)."""
    from feelwm.training.trainer import build_legged_trainer_from_config

    active = ["roughness", "angular_instability", "slope_power", "cost_of_transport"]
    path = _tiny_legged_yaml(tmp_path, active=active, feature_dim=4)
    trainer = build_legged_trainer_from_config(str(path))
    assert trainer.train_dataset[0]["f_prop_target"].shape == (H, 4)
    assert trainer.val_dataset[0]["current_feature"].shape == (4,)
    assert trainer.channel_indices == [0, 1, 2, 4]   # legged registry, not the wheeled one


def test_legged_builder_train_tags_filter(tmp_path):
    """`data.train_tags: [tg]` (TG-only zero-shot row): train specs
    restrict to the listed tags while val (held-out GT) is untouched — a corpus-only
    delta on the v2 recipe."""
    from feelwm.training.trainer import build_legged_trainer_from_config

    active = ["roughness", "angular_instability", "slope_power", "cost_of_transport"]
    path = _tiny_legged_yaml(tmp_path, active=active, feature_dim=4)
    cfg = yaml.safe_load(path.read_text())
    tg = tmp_path / "tartanground"
    seq = tg / "Gascola" / "Data_anymal" / "P2000"
    seq.mkdir(parents=True)
    _write_labels(seq / "labels_legged.npz", n=40, d_raw=8,
                  timestamp=np.arange(40, dtype=np.float64) * 0.1)
    _write_feats(seq / "features" / "tg_dinov2reg_v1", 40, 4, 16)
    cfg["data"]["tartanground_root"] = str(tg)
    cfg["data"]["train_tags"] = ["tg"]
    path.write_text(yaml.safe_dump(cfg))
    trainer = build_legged_trainer_from_config(str(path))
    assert {r.spec.tag for r in trainer.train_dataset._records} == {"tg"}
    assert {r.spec.tag for r in trainer.val_dataset._records} == {"gt"}
    assert [r.spec.name for r in trainer.val_dataset._records] == ["2024-12-heldout"]

    cfg["data"]["train_tags"] = ["isaac"]            # matches nothing on this tree
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="train_tags"):
        build_legged_trainer_from_config(str(path))


def test_legged_builder_rejects_config_channel_mismatch(tmp_path):
    """loss.channel_names must equal data.active_proprio_channels — the dataset emits
    exactly the active channels, in that order."""
    from feelwm.training.trainer import build_legged_trainer_from_config

    active = ["roughness", "angular_instability", "slope_power", "cost_of_transport"]
    path = _tiny_legged_yaml(tmp_path, active=active, feature_dim=4)
    cfg = yaml.safe_load(path.read_text())
    cfg["loss"]["channel_names"] = active[:3]        # data says 4 channels, loss says 3
    cfg["loss"]["proprio_weights"] = [1.0] * 3
    cfg["model"]["proprio_feature_dim"] = 3
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="active_proprio_channels"):
        build_legged_trainer_from_config(str(path))


def test_legged_builder_probe_catches_a_loader_that_ignores_the_selection(tmp_path, monkeypatch):
    """The channel-width crash class, reproduced: config fully self-consistent (4 channels
    everywhere) but the LOADER emits 5 → must fail at BUILD time, not on the first DDP
    teacher-forced batch."""
    import feelwm.data.legged_dataset as legged_mod
    from feelwm.training.trainer import build_legged_trainer_from_config

    original = legged_mod.LeggedDataset.__getitem__

    def ignores_selection(self, idx):
        sample = original(self, idx)
        sample["f_prop_target"] = torch.zeros(H, 5)   # pre-fix behavior
        return sample

    monkeypatch.setattr(legged_mod.LeggedDataset, "__getitem__", ignores_selection)
    path = _tiny_legged_yaml(
        tmp_path,
        active=["roughness", "angular_instability", "slope_power", "cost_of_transport"],
        feature_dim=4,
    )
    with pytest.raises(ValueError, match="f_prop_target with last dim 5"):
        build_legged_trainer_from_config(str(path))


def _tiny_isaac_yaml(tmp_path, *, n=40, d_raw=8, n_patch=4, d_vis=16):
    """Minimal but REAL isaac-mode config → build_legged_trainer_from_config (synthetic).
    Two path-groups: the cov_02 group is held out by substring, the canonical group trains.
    Train trav_gt has a known positive rate so the pos_weight computation is checkable."""
    root = tmp_path / "sequences" / "all"
    trav_train = np.zeros(n, np.uint8); trav_train[::4] = 1     # 25% positives
    for ep, trav in (("Deck__canonical__v0.7__ep00", trav_train),
                     ("Deck__cov_02__v0.7__ep00", np.zeros(n, np.uint8))):
        (root / ep).mkdir(parents=True)
        _write_labels(root / ep / "labels_legged.npz", n=n, d_raw=d_raw,
                      timestamp=np.arange(n, dtype=np.float64) * 0.1, trav_gt=trav)
        _write_feats(root / ep / "features" / "anymal_isaac_dinov3_v1", n, n_patch, d_vis)
    stats = tmp_path / "isaac_stats.json"
    _write_stats(stats, d_raw=d_raw)
    active = ["roughness", "angular_instability", "slope_power", "slip", "cost_of_transport"]
    cfg = {
        "model": {"context_len": K, "rollout_horizon": H, "n_patch": n_patch, "d_vis": d_vis,
                  "d_prop": 8, "d_action": 4, "d_raw": d_raw, "n_layers": 1, "n_heads": 2,
                  "ffn_mult": 2, "dropout": 0.0, "use_proprio_input": True,
                  "use_proprio_feature_head": True, "use_traversability_head": True,
                  "proprio_feature_dim": 5},
        "data": {"isaac_root": str(root), "norm_stats_path": str(stats),
                 "feat_version": "anymal_isaac_dinov3_v1",
                 "context_len": K, "rollout_horizon": H,
                 "active_proprio_channels": active,
                 "include_dt": True, "train_strides": [1, 2], "stride_sampling": "random",
                 "val_stride": 1, "proprio_horizon_s": 4.0,
                 "heldout_substrings": ["Deck__cov_02"]},
        "loss": {"lambda_prop": 0.5, "lambda_trav": 1.0, "lambda_cos": 0.1,
                 "channel_names": active, "proprio_weights": [1.0] * 5},
        "curriculum": {"enabled": False},
        # sample-granularity scheduled sampling goes through the SHARED
        # ScheduledSamplingConfig — prove the legged builder path parses it (the Isaac
        # config relies on it)
        "scheduled_sampling": {"enabled": True, "start_epoch": 5, "end_epoch": 15,
                               "p_start": 1.0, "p_end": 0.5, "mask_granularity": "sample"},
        "training": {"epochs": 1, "batch_size": 2, "num_workers": 0, "device": "cpu",
                     "amp": False, "progress": False, "seed": 0,
                     "checkpoint_path": str(tmp_path / "ckpt.pt"),
                     "metrics_path": str(tmp_path / "metrics.jsonl"),
                     "per_sequence_dir": str(tmp_path / "per_sequence")},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_legged_builder_isaac_mode(tmp_path):
    """Isaac single-corpus branch: substring val split, single norm stats, pos_weight from
    the train positive rate (25% → 3.0), binary labels reach the batch."""
    from feelwm.training.trainer import build_legged_trainer_from_config

    trainer = build_legged_trainer_from_config(str(_tiny_isaac_yaml(tmp_path)))
    assert trainer.train_dataset.sequence_names() == ["Deck__canonical__v0.7__ep00"]
    assert trainer.val_dataset.sequence_names() == ["Deck__cov_02__v0.7__ep00"]
    assert trainer.pos_weight is not None
    assert abs(float(trainer.pos_weight) - 3.0) < 1e-6   # (1-0.25)/0.25
    assert trainer.sampling_cfg.mask_granularity == "sample"   # sample-mask SS plumbs through
    b = trainer.train_dataset[0]
    assert b["binary_valid_target_mask"].any()
    assert b["f_prop_target"].shape[-1] == 5             # 5-ch (all legged channels)


def test_legged_builder_isaac_mode_requires_heldout(tmp_path):
    from feelwm.training.trainer import build_legged_trainer_from_config

    path = _tiny_isaac_yaml(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    del cfg["data"]["heldout_substrings"]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="heldout_substrings"):
        build_legged_trainer_from_config(str(path))



def _tiny_joint_yaml(tmp_path, *, n=40, d_raw=8, n_patch=4, d_vis=16):
    """Minimal REAL joint-mode config: gt (train+heldout) + tg + isaac (train+heldout),
    per-tag stats, isaac trav 25% positives → checkable pos_weight."""
    gt = tmp_path / "grand_tour"
    for name in ("gt-train", "2024-12-heldout"):
        (gt / name).mkdir(parents=True)
        _write_labels(gt / name / "labels_legged.npz", n=n, d_raw=d_raw,
                      timestamp=np.arange(n, dtype=np.float64) * 0.1)
        _write_feats(gt / name / "features" / "gt_dinov3_v1", n, n_patch, d_vis)
    tg = tmp_path / "tartanground" / "Gascola" / "Data_anymal" / "P2000"
    tg.mkdir(parents=True)
    _write_labels(tg / "labels_legged.npz", n=n, d_raw=d_raw,
                  timestamp=np.arange(n, dtype=np.float64) * 0.1)
    _write_feats(tg / "features" / "tg_dinov3_v1", n, n_patch, d_vis)
    isaac = tmp_path / "isaac" / "all"
    trav_train = np.zeros(n, np.uint8); trav_train[::4] = 1     # 25% positives
    for ep, trav in (("Deck__canonical__v0.7__ep00", trav_train),
                     ("Deck__cov_02__v0.7__ep00", np.zeros(n, np.uint8))):
        (isaac / ep).mkdir(parents=True)
        _write_labels(isaac / ep / "labels_legged.npz", n=n, d_raw=d_raw,
                      timestamp=np.arange(n, dtype=np.float64) * 0.1, trav_gt=trav)
        _write_feats(isaac / ep / "features" / "anymal_isaac_dinov3_v1", n, n_patch, d_vis)
    stats = tmp_path / "stats.json"
    _write_stats(stats, d_raw=d_raw)
    active = ["roughness", "angular_instability", "slope_power", "cost_of_transport"]
    cfg = {
        "model": {"context_len": K, "rollout_horizon": H, "n_patch": n_patch, "d_vis": d_vis,
                  "d_prop": 8, "d_action": 4, "d_raw": d_raw, "n_layers": 1, "n_heads": 2,
                  "ffn_mult": 2, "dropout": 0.0, "use_proprio_input": True,
                  "use_proprio_feature_head": True, "use_traversability_head": True,
                  "proprio_feature_dim": 4},
        "data": {"grandtour_root": str(gt), "tartanground_root": str(tmp_path / "tartanground"),
                 "isaac_root": str(isaac),
                 "norm_stats_path_gt": str(stats), "norm_stats_path_tg": str(stats),
                 "norm_stats_path_isaac": str(stats),
                 "context_len": K, "rollout_horizon": H,
                 "active_proprio_channels": active,
                 "include_dt": True, "train_strides": [1, 2], "stride_sampling": "random",
                 "val_stride": 1, "proprio_horizon_s": 4.0,
                 "heldout_grandtour_missions": ["2024-12-heldout"],
                 "heldout_substrings": ["Deck__cov_02"]},
        "loss": {"lambda_prop": 0.5, "lambda_trav": 1.0, "lambda_cos": 0.1,
                 "channel_names": active, "proprio_weights": [1.0] * 4},
        "curriculum": {"enabled": False},
        "scheduled_sampling": {"enabled": True, "start_epoch": 5, "end_epoch": 15,
                               "p_start": 1.0, "p_end": 0.5, "mask_granularity": "sample"},
        "training": {"epochs": 1, "batch_size": 2, "num_workers": 0, "device": "cpu",
                     "amp": False, "progress": False, "seed": 0,
                     "checkpoint_path": str(tmp_path / "ckpt.pt"),
                     "metrics_path": str(tmp_path / "metrics.jsonl"),
                     "per_sequence_dir": str(tmp_path / "per_sequence")},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_legged_builder_joint_mode(tmp_path):
    """3-corpus joint mode: per-tag norm, TAG-PURE vals (main = gt heldout, extra 'isaac'),
    pos_weight from the ISAAC train anchors only (25% → 3.0), all three tags in train."""
    from feelwm.training.trainer import build_legged_trainer_from_config

    trainer = build_legged_trainer_from_config(str(_tiny_joint_yaml(tmp_path)))
    train_tags = {s.tag for s in (r.spec for r in trainer.train_dataset._records)}
    assert train_tags == {"gt", "tg", "isaac"}
    assert trainer.val_dataset.sequence_names() == ["2024-12-heldout"]        # tag-pure gt
    assert set(trainer.extra_val_datasets) == {"isaac"}
    assert trainer.extra_val_datasets["isaac"].sequence_names() == ["Deck__cov_02__v0.7__ep00"]
    assert abs(float(trainer.pos_weight) - 3.0) < 1e-6   # isaac anchors only, (1-.25)/.25
    b = trainer.train_dataset[0]
    assert b["f_prop_target"].shape[-1] == 4             # 4-ch joint (slip slot excluded)


def test_legged_builder_refuses_a_zero_window_corpus(tmp_path):
    """The zero-window failure class: a corpus whose feature cache is
    missing silently drops ALL its train windows (require_vis_context) and the run
    becomes a partial-corpus model without anyone noticing. Build must FAIL loudly."""
    import shutil
    from feelwm.training.trainer import build_legged_trainer_from_config

    path = _tiny_joint_yaml(tmp_path)
    shutil.rmtree(tmp_path / "grand_tour" / "gt-train" / "features")   # train-gt cache gone
    with pytest.raises(ValueError, match="0 train windows"):
        build_legged_trainer_from_config(str(path))


def test_legged_builder_joint_mode_requires_both_heldouts(tmp_path):
    from feelwm.training.trainer import build_legged_trainer_from_config

    path = _tiny_joint_yaml(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    del cfg["data"]["heldout_substrings"]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="heldout"):
        build_legged_trainer_from_config(str(path))



def test_isaac_v1_noprop_config_is_v1_verbatim_plus_the_single_knob():
    """Legged proprio-input OFF arm: identical to the legged_isaac `feelwm` variant except
    use_proprio_input=false and its own output dirs — any other drift breaks the
    single-variable A/B."""
    v1 = load_config(REPO_ROOT / "configs/legged_isaac.yaml", "feelwm")
    off = load_config(REPO_ROOT / "configs/legged_isaac.yaml", "noprop")
    assert off["model"]["use_proprio_input"] is False
    assert off["model"]["use_proprio_feature_head"] and off["model"]["use_traversability_head"]
    v1m = dict(v1["model"]); offm = dict(off["model"])
    v1m.pop("use_proprio_input"); offm.pop("use_proprio_input")
    assert v1m == offm
    assert off["data"] == v1["data"]
    assert off["loss"] == v1["loss"]
    assert off["scheduled_sampling"] == v1["scheduled_sampling"]
    v1t = dict(v1["training"]); offt = dict(off["training"])
    for k in ("checkpoint_path", "metrics_path", "per_sequence_dir"):
        assert "legged_isaac/noprop/" in offt.pop(k)
        v1t.pop(k)
    assert v1t == offt


def test_v2_multistride_config_is_multistride_without_the_v4_knobs():
    cfg = load_config(REPO_ROOT / "configs/legged_tg_gt.yaml")
    assert cfg["model"]["d_action"] == 4
    d = cfg["data"]
    assert (d["include_dt"], d["train_strides"], d["stride_sampling"]) == (
        True, [1, 2, 5, 10], "random")
    assert (d["val_stride"], d["proprio_horizon_s"]) == (1, 1.0)
    # sample-granularity scheduled sampling is explicitly out of scope for this config
    assert "mask_granularity" not in cfg["scheduled_sampling"]
    # slip dropped from ACTIVE channels (a single-stride run had slip R^2 = 0.000 at all
    # 20 epochs); label registry stays 5-dim — this is a config-level selection only.
    active = d["active_proprio_channels"]
    assert active == ["roughness", "angular_instability", "slope_power", "cost_of_transport"]
    assert cfg["model"]["proprio_feature_dim"] == len(active) == 4
    assert cfg["loss"]["channel_names"] == active
    assert len(cfg["loss"]["proprio_weights"]) == 4
    # own output dirs — never inherit another run's per_sequence_dir
    assert "legged_tg_gt/feelwm/" in cfg["training"]["per_sequence_dir"]
