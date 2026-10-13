"""Isaac-mode + zero-shot query routing for the legged open-loop CEM eval.

What is worth locking down, and why:
  - `legged_specs` isaac branch reproduces the TRAINER's split exactly (substring
    heldout, tag "isaac", single train-only stats file) — the prior and the in-domain
    query set both hang off it;
  - `external_query_specs` (zero-shot) keeps the requested sequence list verbatim,
    fails loudly on a missing name, and — the load-bearing part — maps the query tag
    to the MODEL's train stats (`model_train_stats` convention), so the loader's
    per-tag routing silently becomes "normalize everything as the model expects";
  - the convention is isaac-mode-only: a gt/tg-trained model has two stats files and
    no single "own" transform, so the guard must hold.

Fixtures are synthesized (portable, no corpora on disk), reusing the v2 test's
mission writer — the isaac corpus layout differs only in directory nesting, which
`discover_legged_specs` owns and is exercised here through real tmp trees.
"""
import numpy as np
import pytest

from feelwm.data.legged_dataset import discover_legged_specs
from feelwm.data.normalize import NormStats

from tools.eval.openloop_cem_legged import (
    _open_partial, _partial_path, build_context_dataset, external_query_specs,
    legged_specs,
)
from tests.planning.test_legged_cem_eval import ACTIVE, H, K, _write_mission, _write_stats

ACTIVE5 = ACTIVE[:3] + ["slip", "cost_of_transport"]


@pytest.fixture
def isaac_corpus(tmp_path):
    """Flat isaac layout: 3 path-groups × 1 seq, one heldout substring, isaac stats."""
    root = tmp_path / "anymal_isaac"
    for name, speed in [("Hill3__cov_03__v0.4__ep00", 0.4),
                        ("Hill3__cov_03__v1__ep00", 1.0),
                        ("Asphalt1__cov_02__v0.7__ep00", 0.7)]:
        _write_mission(root / name, n=200, speed=speed, seed=hash(name) % 100)
        # discover() keys on the label file; the feat version dir just has to exist
        feat = root / name / "features" / "anymal_isaac_dinov3_v1"
        feat.mkdir(parents=True, exist_ok=True)
        for src in (root / name / "features" / "gt_dinov2reg_v1").iterdir():
            (feat / src.name).write_bytes(src.read_bytes())
    stats = tmp_path / "isaac.json"
    _write_stats(stats, action_mean=(0.05, 0.0, 0.0), action_std=(0.5, 1.0, 1.0))
    wm_cfg = {
        "model": {"context_len": K, "rollout_horizon": H, "n_patch": 4, "d_vis": 8},
        "data": {"isaac_root": str(root), "feat_version": "anymal_isaac_dinov3_v1",
                 "norm_stats_path": str(stats),
                 "heldout_substrings": ["Hill3__cov_03"],
                 "active_proprio_channels": ACTIVE5, "proprio_horizon_s": 4.0},
    }
    return root, stats, wm_cfg


@pytest.fixture
def tg_corpus(tmp_path):
    """Nested TG layout: 2 envs × 2 trajectories."""
    root = tmp_path / "tartanground"
    for env in ("Gascola", "GreatMarsh"):
        for p in ("P2000", "P2001"):
            _write_mission(root / env / "Data_anymal" / p, n=120, speed=0.6, seed=7)
            feat = root / env / "Data_anymal" / p / "features" / "tg_dinov3_v1"
            feat.mkdir(parents=True, exist_ok=True)
            for src in (root / env / "Data_anymal" / p / "features" / "gt_dinov2reg_v1").iterdir():
                (feat / src.name).write_bytes(src.read_bytes())
    return root


# --------------------------------------------------------------------- isaac split


def test_isaac_split_mirrors_trainer(isaac_corpus):
    _root, stats, wm_cfg = isaac_corpus
    train, val, norm, mode = legged_specs(wm_cfg)
    assert mode == "isaac"
    assert {s.name for s in val} == {"Hill3__cov_03__v0.4__ep00", "Hill3__cov_03__v1__ep00"}
    assert {s.name for s in train} == {"Asphalt1__cov_02__v0.7__ep00"}
    assert all(s.tag == "isaac" for s in train + val)
    assert norm == {"isaac": str(stats)}


def test_isaac_split_requires_heldout(isaac_corpus):
    _root, _stats, wm_cfg = isaac_corpus
    wm_cfg["data"].pop("heldout_substrings")
    with pytest.raises(SystemExit, match="heldout_substrings"):
        legged_specs(wm_cfg)


def test_gt_tg_split_applies_train_tags(tmp_path):
    """v2_tgonly_zs contract: with data.train_tags the eval PRIOR must come from the
    model's literal own (TG-only) train split — not TG + GT-train (silent prior
    contamination); val stays the heldout GT missions either way."""
    gt = tmp_path / "grand_tour"
    for m in ("2024-11-01-train", "2024-12-03-heldout"):
        _write_mission(gt / m, n=120, speed=0.6, seed=3)
    tgr = tmp_path / "tartanground"
    _write_mission(tgr / "Gascola" / "Data_anymal" / "P2000", n=120, speed=0.6, seed=4)
    stats = tmp_path / "s.json"
    _write_stats(stats, action_mean=(0.05, 0.0, 0.0), action_std=(0.5, 1.0, 1.0))
    wm_cfg = {
        "model": {"context_len": K, "rollout_horizon": H, "n_patch": 4, "d_vis": 8},
        "data": {"grandtour_root": str(gt), "tartanground_root": str(tgr),
                 "norm_stats_path_gt": str(stats), "norm_stats_path_tg": str(stats),
                 "heldout_grandtour_missions": ["2024-12-03-heldout"]},
    }
    train, val, _norm, mode = legged_specs(wm_cfg)
    assert mode == "gt_tg"
    assert {s.tag for s in train} == {"gt", "tg"}      # default: GT-train included
    wm_cfg["data"]["train_tags"] = ["tg"]
    train_tg, val_tg, _n, _m = legged_specs(wm_cfg)
    assert {s.tag for s in train_tg} == {"tg"}          # prior = own train only
    assert [s.name for s in val_tg] == [s.name for s in val]
    wm_cfg["data"]["train_tags"] = ["nope"]
    with pytest.raises(SystemExit, match="train_tags"):
        legged_specs(wm_cfg)


# --------------------------------------------------------------------- zero-shot queries


def _qd(root, seqs, dataset="tg", feat="tg_dinov3_v1"):
    return {"dataset": dataset, "root": str(root), "feat_version": feat, "sequences": seqs}


def test_external_specs_tg_names_and_norm_mapping(isaac_corpus, tg_corpus):
    _root, stats, wm_cfg = isaac_corpus
    _train, _val, norm, mode = legged_specs(wm_cfg)
    wanted = ["Gascola_P2001", "GreatMarsh_P2000"]
    specs, norm2, tag = external_query_specs(_qd(tg_corpus, wanted), norm, mode)
    assert tag == "tg"
    assert [s.name for s in specs] == wanted                      # order preserved
    assert all(s.tag == "tg" for s in specs)
    assert specs[0].feat_dir.name == "tg_dinov3_v1"
    assert norm2["tg"] == str(stats)                              # model_train_stats
    assert norm2["isaac"] == str(stats)                           # prior routing intact


def test_external_specs_missing_sequence_fails(isaac_corpus, tg_corpus):
    _root, _stats, wm_cfg = isaac_corpus
    _train, _val, norm, mode = legged_specs(wm_cfg)
    with pytest.raises(SystemExit, match="Gascola_P9999"):
        external_query_specs(_qd(tg_corpus, ["Gascola_P9999"]), norm, mode)


def test_external_specs_isaac_mode_only(tg_corpus, tmp_path):
    stats = tmp_path / "gt.json"
    _write_stats(stats)
    with pytest.raises(SystemExit, match="isaac-mode"):
        external_query_specs(_qd(tg_corpus, ["Gascola_P2000"]),
                             {"gt": str(stats), "tg": str(stats)}, "gt_tg")


# --------------------------------------------------------------------- partial / resume


IDENT = {"wm_config": "a.yaml", "checkpoint": "b.pt", "arms": ["goalonly"],
         "shard": 0, "num_shards": 1, "n_queries_gated": 42}


def test_partial_fresh_then_resume_roundtrip(tmp_path):
    import json
    partial = tmp_path / "run.partial.jsonl"
    fh, rows = _open_partial(partial, IDENT, resume=False)
    assert rows == [] and partial.exists()
    fh.write(json.dumps({"qid": 7, "ade_goalonly": 0.1}) + "\n")
    fh.close()
    fh2, rows2 = _open_partial(partial, IDENT, resume=True)
    fh2.close()
    assert [r["qid"] for r in rows2] == [7]                       # finished row reused


def test_partial_exists_without_resume_refuses(tmp_path):
    partial = tmp_path / "run.partial.jsonl"
    _open_partial(partial, IDENT, resume=False)[0].close()
    with pytest.raises(SystemExit, match="--resume"):
        _open_partial(partial, IDENT, resume=False)


def test_partial_resume_identity_mismatch_refuses(tmp_path):
    partial = tmp_path / "run.partial.jsonl"
    _open_partial(partial, IDENT, resume=False)[0].close()
    with pytest.raises(SystemExit, match="mismatch"):
        _open_partial(partial, {**IDENT, "checkpoint": "OTHER.pt"}, resume=True)


def test_partial_path_shard_aware():
    from types import SimpleNamespace
    ns = SimpleNamespace(out="out/x.md", merge_shards=False, num_shards=2, shard=1)
    assert _partial_path(ns).name.endswith(".partial.jsonl")
    assert "shard1" in _partial_path(ns).name
    assert _partial_path(SimpleNamespace(out=None, merge_shards=False,
                                         num_shards=1, shard=0)) is None


def test_zero_shot_loader_normalizes_with_model_stats(isaac_corpus, tg_corpus):
    """End-to-end: a TG record inside the context dataset carries ISAAC action stats."""
    _root, stats, wm_cfg = isaac_corpus
    _train, _val, norm, mode = legged_specs(wm_cfg)
    specs, norm2, _tag = external_query_specs(_qd(tg_corpus, ["Gascola_P2000"]), norm, mode)
    ds = build_context_dataset(wm_cfg, specs, norm2)
    want = NormStats.from_json(str(stats))
    np.testing.assert_allclose(ds._stats["tg"].action_mean, want.action_mean)
    np.testing.assert_allclose(ds._stats["tg"].action_std, want.action_std)
