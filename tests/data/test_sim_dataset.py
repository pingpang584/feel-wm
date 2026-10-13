"""build_sim_datasets — the split/normalization construction shared by the sim trainer
(feelwm/training/trainer.py: build_sim_trainer_from_config) and offline eval
(tools/eval/horizon_sweep.py --sim). Synthetic episodes, no DINO cache."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from feelwm.data.sim_dataset import sim_corpora
from feelwm.training.scheduled_sampling import ScheduledSamplingConfig
from feelwm.training.trainer import _dataclass_from_dict, build_sim_datasets

REPO_ROOT = Path(__file__).resolve().parents[2]

N_ANCHOR, T_WIN, D_RAW = 40, 5, 13


def _write_episode(root, name: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    pose = np.tile(np.eye(4, dtype=np.float32), (N_ANCHOR, 1, 1))
    pose[:, 0, 3] = np.arange(N_ANCHOR, dtype=np.float32) * 0.1   # straight-line drive
    np.savez(
        d / "labels.npz",
        proprio_window=rng.standard_normal((N_ANCHOR, T_WIN, D_RAW)).astype(np.float32),
        proprio_feature=rng.standard_normal((N_ANCHOR, 5)).astype(np.float32),
        trav_gt=(rng.random(N_ANCHOR) < 0.2).astype(np.uint8),
        pose=pose,
        action=rng.standard_normal((N_ANCHOR, 3)).astype(np.float32),
        valid_mask=np.ones(N_ANCHOR, dtype=np.uint8),
    )


@pytest.fixture
def sim_cfg(tmp_path):
    root = tmp_path / "sequences"
    for name in ("hill_run_a__ep00", "hill_run_b__ep00", "lake_run_12__ep00"):
        _write_episode(root, name)
    norm = tmp_path / "norm.json"
    norm.write_text(json.dumps({
        "feature_set": "sim",
        "proprio_feature": {"mean": [0.0] * 5, "std": [1.0] * 5},
        "proprio_window": {"mean": [0.0] * D_RAW, "std": [1.0] * D_RAW},
        "action": {"mean": [0.0] * 3, "std": [1.0] * 3},
    }))
    return {
        "data": {
            "seq_root": str(root), "norm_stats_path": str(norm),
            "heldout_substrings": ["lake_run_12"],
            "include_dt": True, "train_strides": [1, 2], "val_stride": 1,
            "active_proprio_channels": ["roughness", "slip_ratio",
                                        "angular_instability", "slope_power"],
        },
        "model": {"context_len": 3, "rollout_horizon": 8, "n_patch": 4, "d_vis": 8},
    }


def test_split_is_by_heldout_substring(sim_cfg):
    ds, base = build_sim_datasets(sim_cfg, require_vis_context=False, require_vis_target=False)
    assert ds["val"].sequence_names() == ["lake_run_12__ep00"]
    assert sorted(ds["train"].sequence_names()) == ["hill_run_a__ep00", "hill_run_b__ep00"]
    assert base.split == "train"          # per-split copies are made downstream
    assert base.norm_stats_path == sim_cfg["data"]["norm_stats_path"]


def test_splits_arg_builds_only_what_is_asked(sim_cfg):
    ds, _ = build_sim_datasets(sim_cfg, splits=("val",),
                               require_vis_context=False, require_vis_target=False)
    assert set(ds) == {"val"}


def test_overrides_reach_the_dataset(sim_cfg):
    ds, base = build_sim_datasets(sim_cfg, splits=("val",), rollout_horizon=16, val_stride=5,
                                  require_vis_context=False, require_vis_target=False)
    assert (base.rollout_horizon, base.val_stride) == (16, 5)
    sample = ds["val"][0]
    assert sample["f_prop_target"].shape == (16, 4)
    assert sample["action_sequence"].shape == (16, 4)          # include_dt → 4-dim action
    assert sample["action_sequence"][0, 3] == pytest.approx(0.5)   # Δt = 5 x 0.1 s


def test_unknown_split_rejected(sim_cfg):
    with pytest.raises(ValueError, match="unknown sim split"):
        build_sim_datasets(sim_cfg, splits=("test",))


@pytest.mark.parametrize(
    "stride, expected_on",
    [
        (1, 8),    # cum Δt 0.1..0.8 s — all 8 steps supervised
        (5, 8),    # cum Δt 0.5..4.0 s — 4.0 s boundary INCLUSIVE (stride-5 full-H)
        (10, 4),   # cum Δt 1.0..8.0 s — h1-4 on (h4 = 4.0 s), h5-8 masked
    ],
)
def test_proprio_horizon_4s_mask_per_stride(sim_cfg, stride, expected_on):
    """v4 knob 1: data.proprio_horizon_s=4.0 → proprio_horizon_mask per stride
    (config dict → SimDatasetConfig → __getitem__, the sim build path)."""
    sim_cfg["data"]["proprio_horizon_s"] = 4.0
    ds, base = build_sim_datasets(
        sim_cfg, splits=("val",), val_stride=stride,
        require_vis_context=False, require_vis_target=False,
    )
    assert base.proprio_horizon_s == 4.0
    mask = ds["val"][0]["proprio_horizon_mask"]
    assert mask.shape == (8,)
    expected = torch.tensor([True] * expected_on + [False] * (8 - expected_on))
    assert torch.equal(mask, expected)



# --- localization-jump screen (real-robot data) -------------------------------------

def _jump_cfg(tmp_path, jump_at: int = 20, jump_m: float = 40.0):
    """One 60-anchor sequence whose pose teleports between anchors `jump_at`/`jump_at+1`."""
    root = tmp_path / "sequences"
    d = root / "real_run__ep00"
    d.mkdir(parents=True)
    n = 60
    rng = np.random.default_rng(0)
    pose = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    pose[:, 0, 3] = np.arange(n, dtype=np.float32) * 0.05      # 0.5 m/s crawl
    pose[jump_at + 1:, 0, 3] += jump_m                          # relocalization teleport
    np.savez(
        d / "labels.npz",
        proprio_window=rng.standard_normal((n, T_WIN, D_RAW)).astype(np.float32),
        proprio_feature=rng.standard_normal((n, 5)).astype(np.float32),
        trav_gt=np.zeros(n, np.uint8), pose=pose,
        action=rng.standard_normal((n, 3)).astype(np.float32),
        valid_mask=np.ones(n, np.uint8),          # extractor screening NOT simulated
    )
    norm = tmp_path / "norm.json"
    norm.write_text(json.dumps({
        "feature_set": "sim",
        "proprio_feature": {"mean": [0.0] * 5, "std": [1.0] * 5},
        "proprio_window": {"mean": [0.0] * D_RAW, "std": [1.0] * D_RAW},
        "action": {"mean": [0.0] * 3, "std": [1.0] * 3},
    }))
    return {
        "data": {"seq_root": str(root), "norm_stats_path": str(norm),
                 "heldout_substrings": ["real_run"], "include_dt": True,
                 "train_strides": [1], "val_stride": 1},
        "model": {"context_len": 3, "rollout_horizon": 8, "n_patch": 4, "d_vis": 8},
    }


def test_pose_jump_screen_drops_spanning_windows(tmp_path):
    cfg = _jump_cfg(tmp_path)
    kw = dict(splits=("val",), require_vis_context=False, require_vis_target=False)
    off, _ = build_sim_datasets(cfg, **kw)
    cfg["data"]["max_pose_step_m"] = 0.5          # 5 m/s on the 10 Hz grid
    on, base = build_sim_datasets(cfg, **kw)
    assert base.max_pose_step_m == 0.5

    anchors = lambda ds: {t for _, t, _ in ds["val"]._index}
    # a window spans [t-3, t+8]; the jump sits between anchors 20 and 21
    assert anchors(off) - anchors(on) == set(range(13, 24))
    assert anchors(on) == set(range(3, 60)) - set(range(13, 24))


def test_pose_jump_screen_is_off_by_default(tmp_path):
    """Sim/isaac configs must be untouched: no knob → no screening, identical index."""
    cfg = _jump_cfg(tmp_path)
    kw = dict(splits=("val",), require_vis_context=False, require_vis_target=False)
    default, base = build_sim_datasets(cfg, **kw)
    assert base.max_pose_step_m is None
    assert len(default["val"]) == 60 - 3


def test_pose_jump_screen_catches_what_valid_mask_misses(tmp_path):
    """The real reason for the screen: at stride > 1 a window can step OVER the jump
    anchor, so zeroing valid_mask there (what the extractor does) misses it."""
    cfg = _jump_cfg(tmp_path)
    cfg["data"]["train_strides"] = [5]
    z = np.load(f"{cfg['data']['seq_root']}/real_run__ep00/labels.npz")
    vm = z["valid_mask"].copy()
    vm[20] = 0                                    # extractor-style anchor invalidation
    np.savez(f"{cfg['data']['seq_root']}/real_run__ep00/labels.npz",
             **{k: (vm if k == "valid_mask" else z[k]) for k in z})
    kw = dict(splits=("val",), val_stride=5,
              require_vis_context=False, require_vis_target=False)
    off, _ = build_sim_datasets(cfg, **kw)
    # anchor 25 uses context {10,15,20,25} -> valid_mask catches it; 30 uses {15,20,25,30}
    # only through anchor 20 as well, but 35 (context {20,...}) aside, a window at 40
    # (context {25,30,35,40}) skips the invalid anchor while its rollout still spans it.
    assert 25 not in {t for _, t, _ in off["val"]._index}
    spans_jump_via_rollout = {t for _, t, _ in off["val"]._index if t <= 20 < t + 40}
    assert spans_jump_via_rollout, "expected strided windows that straddle the jump"
    cfg["data"]["max_pose_step_m"] = 0.5
    on, _ = build_sim_datasets(cfg, **kw)
    assert not {t for _, t, _ in on["val"]._index if t <= 20 < t + 40}


# --- within-sequence tail holdout (real-robot data) ----------------------------------

def _spans(ds, name_startswith):
    """(lo, hi) anchor span of every window of the matching record(s)."""
    out = []
    for rid, t, s in ds._index:
        rec = ds._records[rid]
        if not rec.name.startswith(name_startswith):
            continue
        for stride in (s if isinstance(s, tuple) else (s,)):
            lo = t - ds.cfg.context_len * stride
            hi = min(t + ds.cfg.rollout_horizon * stride, len(rec.valid_mask) - 1)
            out.append((lo, hi))
    return out


def _tail_cfg(sim_cfg, **data):
    sim_cfg["data"].setdefault("tail_val_substrings", ["hill_run_a"])
    sim_cfg["data"].setdefault("tail_val_fraction", 0.3)
    sim_cfg["data"].update(**data)
    return sim_cfg


def test_tail_holdout_puts_head_in_train_and_tail_in_val(sim_cfg):
    cfg = _tail_cfg(sim_cfg)
    ds, _ = build_sim_datasets(cfg, require_vis_context=False, require_vis_target=False)
    # the tail record joins the whole-sequence heldout record in val, under its own name
    assert sorted(r.name for r in ds["val"]._records) == [
        "hill_run_a__ep00__tail", "lake_run_12__ep00"]
    assert "hill_run_a__ep00" in [r.name for r in ds["train"]._records]

    train, val = _spans(ds["train"], "hill_run_a"), _spans(ds["val"], "hill_run_a")
    assert train and val
    # every train window ends before every val window begins: no shared anchor, so no
    # train rollout target is ever a val context frame
    assert max(hi for _, hi in train) < min(lo for lo, _ in val)


def test_tail_holdout_off_by_default(sim_cfg):
    ds, base = build_sim_datasets(sim_cfg, require_vis_context=False, require_vis_target=False)
    assert (base.tail_val_substrings, base.tail_val_fraction) == ([], 0.0)
    assert [r.name for r in ds["val"]._records] == ["lake_run_12__ep00"]


def test_tail_holdout_needs_both_knobs(sim_cfg):
    sim_cfg["data"]["tail_val_fraction"] = 0.3          # substrings missing
    with pytest.raises(ValueError, match="must be set together"):
        build_sim_datasets(sim_cfg, splits=("val",))


def test_tail_holdout_never_slices_a_heldout_sequence(sim_cfg):
    """A whole-sequence val seq has no head in train, so a matching substring must not
    turn it into a 30% slice (real configs: a short site substring can also match the
    name of a whole-sequence held-out run)."""
    cfg = _tail_cfg(sim_cfg, tail_val_substrings=["hill_run_a", "lake"])
    ds, _ = build_sim_datasets(cfg, require_vis_context=False, require_vis_target=False)
    names = sorted(r.name for r in ds["val"]._records)
    assert names == ["hill_run_a__ep00__tail", "lake_run_12__ep00"]   # lake NOT sliced
    lake = [r for r in ds["val"]._records if r.name == "lake_run_12__ep00"][0]
    assert lake.tail_cut is None


def test_tail_cut_is_taken_after_the_jump_screen(sim_cfg, tmp_path):
    """The fraction counts anchors that survive the screen, so a bag whose HEAD is
    jump-corrupted gives up a later slice than a naive fraction-of-N cut would."""
    lbl = f"{sim_cfg['data']['seq_root']}/hill_run_a__ep00/labels.npz"
    z = dict(np.load(lbl, allow_pickle=False))
    pose = z["pose"].copy()
    for i in range(2, 20, 2):                 # teleports through the first half only
        pose[i:, 0, 3] += 30.0
    z["pose"] = pose
    np.savez(lbl, **z)

    cfg = _tail_cfg(sim_cfg, max_pose_step_m=0.5)
    ds, _ = build_sim_datasets(cfg, require_vis_context=False, require_vis_target=False)
    cut_screened = [r for r in ds["val"]._records if r.name.endswith("__tail")][0].tail_cut
    cfg_naive = _tail_cfg(dict(sim_cfg, data=dict(sim_cfg["data"], max_pose_step_m=None)))
    ds_naive, _ = build_sim_datasets(cfg_naive, require_vis_context=False,
                                     require_vis_target=False)
    cut_naive = [r for r in ds_naive["val"]._records if r.name.endswith("__tail")][0].tail_cut
    assert cut_screened > cut_naive


# --- train-only sequence exclusion (real-robot new-site arm) -------------------------

_NO_VIS = dict(require_vis_context=False, require_vis_target=False)


def _index_by_name(ds) -> list[tuple[str, int, object]]:
    """The split's index keyed by record NAME (record ids shift when a spec is dropped)."""
    return [(ds._records[rid].name, t, s) for rid, t, s in ds._index]


def test_exclude_sequences_empty_is_bit_identical_to_not_setting_it(sim_cfg):
    """Guarantee for every existing sim/isaac/real config: the knob's default changes
    nothing. Both val axes are on so the tail path is covered too."""
    cfg = _tail_cfg(sim_cfg)
    before, base_before = build_sim_datasets(cfg, **_NO_VIS)
    cfg_empty = dict(cfg, data=dict(cfg["data"], exclude_sequences=[]))
    after, base_after = build_sim_datasets(cfg_empty, **_NO_VIS)
    assert base_before.exclude_sequences == [] == base_after.exclude_sequences
    for split in ("train", "val"):
        assert _index_by_name(before[split]) == _index_by_name(after[split])



def test_exclude_sequences_removes_exactly_that_sequences_train_windows(sim_cfg):
    cfg = _tail_cfg(sim_cfg)                       # hill_run_a also has a tail-val slice
    full, _ = build_sim_datasets(cfg, **_NO_VIS)
    cfg_excl = dict(cfg, data=dict(cfg["data"], exclude_sequences=["hill_run_b"]))
    excl, base = build_sim_datasets(cfg_excl, **_NO_VIS)

    assert base.exclude_sequences == ["hill_run_b"]
    assert [r.name for r in excl["train"]._records] == ["hill_run_a__ep00"]
    kept = [e for e in _index_by_name(full["train"]) if e[0] != "hill_run_b__ep00"]
    assert _index_by_name(excl["train"]) == kept          # the rest of train is untouched
    assert _index_by_name(excl["val"]) == _index_by_name(full["val"])   # both val axes


def test_excluding_a_tail_sequence_keeps_its_val_tail(sim_cfg):
    """Train-only semantics at their sharpest: the excluded sequence has a tail-val slice,
    and that slice must survive intact — val composition is fixed across arms."""
    cfg = _tail_cfg(sim_cfg)
    full, _ = build_sim_datasets(cfg, **_NO_VIS)
    cfg_excl = dict(cfg, data=dict(cfg["data"], exclude_sequences=["hill_run_a"]))
    excl, _ = build_sim_datasets(cfg_excl, **_NO_VIS)

    assert [r.name for r in excl["train"]._records] == ["hill_run_b__ep00"]
    assert sorted(r.name for r in excl["val"]._records) == [
        "hill_run_a__ep00__tail", "lake_run_12__ep00"]
    assert _index_by_name(excl["val"]) == _index_by_name(full["val"])


def test_exclude_sequences_matching_only_val_is_a_no_op(sim_cfg):
    cfg = _tail_cfg(sim_cfg)
    full, _ = build_sim_datasets(cfg, **_NO_VIS)
    cfg_excl = dict(cfg, data=dict(cfg["data"], exclude_sequences=["lake"]))
    excl, _ = build_sim_datasets(cfg_excl, **_NO_VIS)
    for split in ("train", "val"):
        assert _index_by_name(excl[split]) == _index_by_name(full[split])


def test_exclude_sequences_emptying_train_is_an_error(sim_cfg):
    cfg = dict(sim_cfg, data=dict(sim_cfg["data"], exclude_sequences=["hill_run"]))
    with pytest.raises(ValueError, match="left no train sequences"):
        build_sim_datasets(cfg, **_NO_VIS)



# --- multi-corpus pooling (combined Gazebo+Isaac training) ---------------------------
# One recipe, three corpus arms (Gazebo, Isaac, Gazebo+Isaac). The loader must pool two
# roots with DIFFERENT feature caches and DIFFERENT held-out lists under ONE norm-stats
# file, and must keep the two domains separable at readout.

def _write_features(root, name: str, feat_version: str, n_patch: int, d_vis: int) -> None:
    """Fully-cached visual features for one episode, under its corpus's feat_version dir."""
    d = root / name / "features" / feat_version
    d.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash(feat_version + name)) % (2**32))
    np.save(d / "features.npy",
            rng.standard_normal((N_ANCHOR, n_patch, d_vis)).astype(np.float32))
    np.save(d / "features_valid.npy", np.ones(N_ANCHOR, dtype=bool))


@pytest.fixture
def combined_cfg(tmp_path, sim_cfg):
    """Two corpora, each with its own root, feat_version and held-out substring."""
    g_root, i_root = tmp_path / "gazebo", tmp_path / "isaac"
    for name in ("hill_run_a__ep00", "hill_run_b__ep00", "lake_run_12__ep00"):
        _write_episode(g_root, name)
        _write_features(g_root, name, "sim_dinov2reg_v1", 4, 8)
    for name in ("Asphalt1__cov_01__v1", "Hill3__cov_01__v1", "Hill4__fail_01__v1"):
        _write_episode(i_root, name)
        _write_features(i_root, name, "isaac_dinov2reg_v1", 4, 8)
    data = {k: v for k, v in sim_cfg["data"].items()
            if k not in ("seq_root", "feat_version", "heldout_substrings")}
    data["corpora"] = [
        {"name": "gazebo", "seq_root": str(g_root), "feat_version": "sim_dinov2reg_v1",
         "heldout_substrings": ["lake_run_12"]},
        {"name": "isaac", "seq_root": str(i_root), "feat_version": "isaac_dinov2reg_v1",
         "heldout_substrings": ["Hill4__fail_01"]},
    ]
    return dict(sim_cfg, data=data)


def test_flat_config_is_one_unnamed_corpus(sim_cfg):
    """The form every existing sim/isaac/real config uses — one corpus, empty domain tag,
    so record names and per-sequence logs stay bit-identical to the pre-multi-corpus loader."""
    (c,) = sim_corpora(sim_cfg["data"])
    assert c["name"] == ""
    assert c["seq_root"] == sim_cfg["data"]["seq_root"]
    assert c["heldout_substrings"] == ["lake_run_12"]



def test_corpora_rejects_leftover_flat_keys(combined_cfg):
    data = dict(combined_cfg["data"], seq_root="/somewhere")
    with pytest.raises(ValueError, match="flat corpus keys"):
        sim_corpora(data)


@pytest.mark.parametrize("mutate, match", [
    (lambda cs: [dict(cs[0], name=""), cs[1]], "non-empty name"),
    (lambda cs: [cs[0], dict(cs[1], name="gazebo")], "unique"),
    (lambda cs: [cs[0], {k: v for k, v in cs[1].items() if k != "seq_root"}], "seq_root"),
    (lambda cs: [dict(cs[0], name="a/b"), cs[1]], "must not contain"),
])
def test_corpora_validation(combined_cfg, mutate, match):
    data = dict(combined_cfg["data"], corpora=mutate(combined_cfg["data"]["corpora"]))
    with pytest.raises(ValueError, match=match):
        sim_corpora(data)


def test_multi_corpus_pools_train_and_tags_every_record(combined_cfg):
    ds, base = build_sim_datasets(combined_cfg)
    train = ds["train"]
    assert sorted(r.name for r in train._records) == [
        "gazebo/hill_run_a__ep00", "gazebo/hill_run_b__ep00",
        "isaac/Asphalt1__cov_01__v1", "isaac/Hill3__cov_01__v1",
    ]
    # both domains actually reach the sampler, and the realized (unweighted) share is visible
    counts = train.domain_counts()
    assert set(counts) == {"gazebo", "isaac"} and min(counts.values()) > 0
    assert sum(counts.values()) == len(train)
    # the union of held-out lists is what the dataset-level guard sees
    assert base.heldout_substrings == ["lake_run_12", "Hill4__fail_01"]


def test_each_corpus_keeps_its_own_heldout_list(combined_cfg):
    """Per-corpus splits, not a shared substring list: the two corpora were split
    independently (settled at different times, different naming schemes)."""
    ds, _ = build_sim_datasets(combined_cfg)
    assert sorted(r.name for r in ds["val"]._records) == [
        "gazebo/lake_run_12__ep00", "isaac/Hill4__fail_01__v1"]


def test_per_domain_val_slices_partition_the_pooled_val(combined_cfg):
    """The A/B's endpoint needs each domain's val on its own — a pooled number cannot say
    whether pooling beat each single-domain corpus ON ITS OWN domain."""
    ds, _ = build_sim_datasets(combined_cfg)
    assert set(ds) == {"train", "val", "val:gazebo", "val:isaac"}
    assert [r.name for r in ds["val:gazebo"]._records] == ["gazebo/lake_run_12__ep00"]
    assert [r.name for r in ds["val:isaac"]._records] == ["isaac/Hill4__fail_01__v1"]
    # same windows as the pooled set, just partitioned (same config, same specs)
    assert len(ds["val:gazebo"]) + len(ds["val:isaac"]) == len(ds["val"])
    assert ds["val:gazebo"].cfg.split == "val"


def test_per_domain_val_is_skippable_and_absent_for_one_corpus(combined_cfg, sim_cfg):
    ds, _ = build_sim_datasets(combined_cfg, per_domain_val=False)
    assert set(ds) == {"train", "val"}
    single, _ = build_sim_datasets(sim_cfg, require_vis_context=False,
                                   require_vis_target=False)
    assert set(single) == {"train", "val"}


def test_multi_corpus_reads_each_corpus_own_feature_cache(combined_cfg):
    """Each corpus has its own DINO cache (sim_dinov2reg_v1 / isaac_dinov2reg_v1); the
    loader must resolve feat_version per spec, not once for the run."""
    ds, _ = build_sim_datasets(combined_cfg, splits=("train",))
    by_name = {r.name: r for r in ds["train"]._records}
    for name, version in (("gazebo/hill_run_a__ep00", "sim_dinov2reg_v1"),
                          ("isaac/Asphalt1__cov_01__v1", "isaac_dinov2reg_v1")):
        rec = by_name[name]
        assert rec.spec.feat_dir.name == version
        assert rec.features is not None and len(rec.cached_set) == N_ANCHOR
    # ...and the vis features really differ per corpus (not silently zero-filled)
    assert not np.allclose(by_name["gazebo/hill_run_a__ep00"].features[0],
                           by_name["isaac/Asphalt1__cov_01__v1"].features[0])


def test_a_corpus_with_no_heldout_match_is_an_error(combined_cfg):
    corpora = [combined_cfg["data"]["corpora"][0],
               dict(combined_cfg["data"]["corpora"][1], heldout_substrings=["nothing"])]
    cfg = dict(combined_cfg, data=dict(combined_cfg["data"], corpora=corpora))
    with pytest.raises(ValueError, match="no held-out sim episodes for corpus isaac"):
        build_sim_datasets(cfg)


def test_norm_stats_from_config_matches_the_loader_train_split(combined_cfg, tmp_path):
    """The joint stats must cover exactly the anchors training sees. compute_norm_stats
    --from-config re-derives the corpora and the per-corpus val exclusion through the SAME
    sim_corpora + substring rule, so the two cannot drift (the reason the mode exists:
    a single global --exclude list would have to be proven collision-free across two
    independent naming schemes)."""
    cns = importlib.import_module("tools.sim.compute_norm_stats")
    cfg_file = tmp_path / "combined.yaml"
    cfg_file.write_text(yaml.safe_dump(combined_cfg))
    paths, counts = cns.paths_from_config(cfg_file)

    ds, _ = build_sim_datasets(combined_cfg, splits=("train",))
    assert sorted(paths) == sorted(str(r.spec.label_npz) for r in ds["train"]._records)
    assert {k: v["n_train_seqs"] for k, v in counts.items()} == {"gazebo": 2, "isaac": 2}
    assert counts["isaac"]["feat_version"] == "isaac_dinov2reg_v1"


# --- the 3-way arms are one variable apart ------------------------------------------


# --- Isaac+excitation arm: the excitation increment is the ONLY variable vs Isaac-only --

_REGIMES = ("speedvar", "weave")



def test_drop_substrings_puts_a_sequence_in_neither_split(sim_cfg):
    """The knob the other two could not express: heldout_substrings would send it to val,
    exclude_sequences would leave it in val."""
    cfg = dict(sim_cfg, data=dict(sim_cfg["data"], drop_substrings=["hill_run_b"]))
    ds, _ = build_sim_datasets(cfg, **_NO_VIS)
    assert [r.name for r in ds["train"]._records] == ["hill_run_a__ep00"]
    assert [r.name for r in ds["val"]._records] == ["lake_run_12__ep00"]


def test_drop_beats_heldout_so_val_stays_bit_identical(sim_cfg):
    """The exact Isaac+excitation case: a name matching BOTH a held-out substring and a
    drop substring (`humji2__cov_02__speedvar` vs held-out group `humji2__cov_02`) must
    vanish, NOT join val — val composition is what the A/B holds fixed."""
    root = Path(sim_cfg["data"]["seq_root"])
    _write_episode(root, "lake_run_12__speedvar__ep00")
    base, _ = build_sim_datasets(sim_cfg, **_NO_VIS)
    assert sorted(r.name for r in base["val"]._records) == [       # leaks into val untreated
        "lake_run_12__ep00", "lake_run_12__speedvar__ep00"]

    cfg = dict(sim_cfg, data=dict(sim_cfg["data"], drop_substrings=["lake_run_12__speedvar"]))
    ds, _ = build_sim_datasets(cfg, **_NO_VIS)
    assert [r.name for r in ds["val"]._records] == ["lake_run_12__ep00"]
    assert "lake_run_12__speedvar__ep00" not in [r.name for r in ds["train"]._records]


def test_drop_substrings_empty_is_bit_identical_to_not_setting_it(sim_cfg):
    """Guarantee for every existing sim/isaac/real config: the knob's default changes
    nothing."""
    before, _ = build_sim_datasets(sim_cfg, **_NO_VIS)
    cfg = dict(sim_cfg, data=dict(sim_cfg["data"], drop_substrings=[]))
    after, _ = build_sim_datasets(cfg, **_NO_VIS)
    for split in ("train", "val"):
        assert _index_by_name(before[split]) == _index_by_name(after[split])


def test_drop_substrings_matching_nothing_is_an_error(sim_cfg):
    """A typo in a drop list is silent by nature — the sequences it was meant to remove just
    keep training. On the excitation arm that would mean training on near-duplicates of val, so it
    fails loudly instead. Doubles as the staged-corpus check: pointed at a tree without the
    excitation increment, the excitation-arm config refuses to build."""
    cfg = dict(sim_cfg, data=dict(sim_cfg["data"], drop_substrings=["hill_run_typo"]))
    with pytest.raises(ValueError, match="matched no episode"):
        build_sim_datasets(cfg, **_NO_VIS)


def test_norm_stats_from_config_honours_drop_substrings(sim_cfg, tmp_path):
    """Stats and loader read the same rules through sim_corpora — a dropped sequence must be
    out of the TRAIN-only statistics too, or the arm normalizes on anchors it never sees."""
    cns = importlib.import_module("tools.sim.compute_norm_stats")
    cfg = dict(sim_cfg, data=dict(sim_cfg["data"], drop_substrings=["hill_run_b"]))
    cfg_file = tmp_path / "ie.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))

    paths, counts = cns.paths_from_config(cfg_file)
    ds, _ = build_sim_datasets(cfg, splits=("train",), **_NO_VIS)
    assert sorted(paths) == sorted(str(r.spec.label_npz) for r in ds["train"]._records)
    assert counts["<single>"]["n_dropped_seqs"] == 1


def test_norm_stats_seq_root_override_reads_the_source_tree(sim_cfg, tmp_path):
    """--seq-root builds the stats from the on-disk corpus before it is tmpfs-staged (the
    excitation-arm case: staging waits on the DINO cache, the stats need only labels)."""
    cns = importlib.import_module("tools.sim.compute_norm_stats")
    cfg_file = tmp_path / "ie.yaml"
    cfg_file.write_text(yaml.safe_dump(sim_cfg))
    source = sim_cfg["data"]["seq_root"]
    cfg_elsewhere = dict(sim_cfg, data=dict(sim_cfg["data"], seq_root="/nonexistent/staging"))
    cfg_file.write_text(yaml.safe_dump(cfg_elsewhere))

    paths, counts = cns.paths_from_config(cfg_file, source)
    assert len(paths) == 2 and counts["<single>"]["seq_root"] == source
