"""Integration smoke for plan_eval glue (metric helpers + the per-query scoring path).

Drives the *actual* functions used inside ``evaluate()`` with a fake WM and hand-built
dataset-shaped samples — no real data, checkpoint, or GPU — to catch shape/NaN glue bugs
that the building-block unit tests don't reach.

Run: python -m pytest tests/planning/test_plan_eval_helpers.py -v
"""
from __future__ import annotations

import importlib

import torch

from feelwm.planning.candidates import inject_expert, stack_samples
from feelwm.planning.cem import WorldModelScorer, expert_rank_percentile, rank_candidates
from feelwm.planning.ride_cost import RideCostConfig
from feelwm.planning.score import ScoreConfig

PE = importlib.import_module("tools.eval.plan_eval")
CHANNELS = ["roughness", "slip_ratio", "angular_instability", "slope_power"]


class _FakeWM:
    def forward_rollout(self, initial_batch, actions):
        N, H, _ = actions.shape
        z = torch.zeros(N, H, 4, 8)
        z[:, -1, :, 0] = 1.0
        f = torch.zeros(N, H, 4)
        f[..., 0] = actions[..., 0]  # roughness tracks action dim 0
        return {"z_vis_preds": z, "f_prop_preds": f, "c_preds": torch.full((N, H, 1), 0.3)}


def _sample(H, rough_fill, P=4, D=8):
    a = torch.zeros(H, 4)
    a[:, 3] = 0.1                       # Δt
    a[:, 0] = rough_fill               # action dim 0 → predicted roughness
    return {
        "action_sequence": a,
        "f_prop_target": torch.full((H, 4), float(rough_fill)),
        "valid_target_mask": torch.ones(H, dtype=torch.bool),
        "proprio_horizon_mask": torch.ones(H, dtype=torch.bool),
        "z_vis_target": torch.zeros(H, P, D),
    }


def test_masked_mean_and_pearson():
    x = torch.tensor([[1.0, 2.0, 9.0]])
    m = torch.tensor([[True, True, False]])
    assert torch.allclose(PE._masked_mean_rows(x, m), torch.tensor([1.5]))
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(PE._pearson(a, 2 * a) - 1.0) < 1e-6
    assert abs(PE._pearson(a, -a) + 1.0) < 1e-6


def test_per_query_scoring_path():
    H = 6
    lib_idx = [0, 1, 2]
    lib_samples = [_sample(H, r) for r in (0.0, 2.0, 4.0)]  # smooth..rough
    query = _sample(H, 1.0)

    lib_pool = stack_samples(lib_samples, horizon=H, source_index=lib_idx)
    pool = inject_expert(lib_pool, query["action_sequence"][:H], at=0)

    goal = query["z_vis_target"][H - 1]
    scorer = WorldModelScorer(
        _FakeWM(), PE._query_context(query, torch.device("cpu")), goal, CHANNELS,
        score_cfg=ScoreConfig(), ride_cfg=RideCostConfig(), proprio_horizon_s=1.0,
    )
    res = rank_candidates(scorer, pool.actions)
    pct = expert_rank_percentile(res["scores"], pool.expert_index)
    assert 0.0 <= pct <= 1.0

    # _best_real excludes the expert and prefers the smoothest real candidate (idx 1 = lib row 0).
    full_best = PE._best_real(
        res["goal_sim"], res["ride_cost"], res["trav_risk"], ScoreConfig(), pool.expert_index
    )
    assert full_best != pool.expert_index
    gt_rough = PE._masked_mean_rows(pool.f_prop_gt[..., 0], pool.horizon_mask)
    # goal-only ignores roughness; full planner should not pick a rougher plan than goal-only.
    goalonly_best = PE._best_real(
        res["goal_sim"], res["ride_cost"], res["trav_risk"],
        ScoreConfig(lambda_trav=0.0, lambda_ride=0.0), pool.expert_index,
    )
    assert PE._safe(gt_rough, full_best) <= PE._safe(gt_rough, goalonly_best) + 1e-6


def test_aggregate_handles_nan_rows():
    rows = [
        {"query": 0, "expert_pct": 1.0, "sel_goal_sim": 0.9,
         "realized_rough_full": 0.2, "realized_rough_goalonly": 0.5,
         "pred_ride": 0.1, "gt_rough": 0.2},
        {"query": 1, "expert_pct": 0.5, "sel_goal_sim": 0.8,
         "realized_rough_full": float("nan"), "realized_rough_goalonly": 0.4,
         "pred_ride": 0.3, "gt_rough": 0.5},
    ]
    out = PE._aggregate(rows, "val", do_cem=False)
    assert out["n_queries"] == 2
    assert abs(out["C1_expert_rank_percentile"] - 0.75) < 1e-6
    assert out["C2b_realized_roughness_full"] == 0.2  # NaN row skipped
    assert "C2b_roughness_reduction" in out


# --- NWM query pairing (NWM-symmetric CEM eval) ---



def _fake_row(qid: int) -> dict:
    return {"qid": qid, "pair_discrepancy_m": 0.1 + 0.01 * (qid % 5),
            "ade_goalonly": 3.0 + qid % 7, "fde_goalonly": 5.0 + qid % 7,
            "ade_full": 2.5 + qid % 5, "fde_full": 4.0 + qid % 5,
            "nwm_ade": 6.0 + qid % 3, "nwm_fde": 9.0 + qid % 3}


def _write_shards(tmp_path, n_shards: int, n_rows: int):
    """Emit what N sharded runs leave on disk, round-robin over qid as the runner does."""
    import json

    from tools.eval.openloop_cem import _shard_path

    base = tmp_path / "run.md"
    meta = {"split": "val", "stride": 4, "horizon_anchors": 32, "amp": "bfloat16",
            "cem_chunk": 64, "cem": {"n_candidates": 256, "iterations": 4, "elite_frac": 0.1,
                                     "epsilon": 0.05, "init": "train action mean/std"},
            "prior_mean": [2.55, 0.0, 0.0], "prior_std": [1.645, 0.26, 0.1]}
    for k in range(n_shards):
        rows = [_fake_row(q) for q in range(n_rows) if q % n_shards == k]
        p = _shard_path(base, k)
        p.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        p.with_suffix(".meta.json").write_text(
            json.dumps({**meta, "n_skipped": k, "s_per_query": 40.0 + k,
                        "peak_gpu_gib": 1.0 + k}))
    return base


def test_merge_shards_reproduces_the_unsharded_aggregate(tmp_path):
    """A sharded run must be the same experiment as a single-process one. Sharding is a
    wall-clock device only, so the merged statistics have to match row-for-row."""
    import numpy as np

    from tools.eval.openloop_cem import _aggregate, _merge_shards

    n_rows = 20
    base = _write_shards(tmp_path, n_shards=4, n_rows=n_rows)
    merged = _merge_shards(base)

    prior = {"mean": np.array([2.55, 0.0, 0.0]), "std": np.array([1.645, 0.26, 0.1])}
    direct = _aggregate([_fake_row(q) for q in range(n_rows)], "val", 6, prior,
                        {"n_candidates": 256, "iterations": 4, "elite_frac": 0.1,
                         "epsilon": 0.05}, 4, 8, 41.5, "bfloat16", 64, 4.0)

    assert merged["n_queries"] == direct["n_queries"] == n_rows
    assert merged["n_skipped"] == 0 + 1 + 2 + 3
    assert merged["n_shards"] == 4
    for key in ("ours_cem_ade_goalonly", "ours_cem_fde_goalonly",
                "pair_discrepancy_median_m", "n_clean"):
        assert merged[key] == direct[key], key
    assert [r["qid"] for r in merged["_rows"]] == list(range(n_rows))   # sorted, no gaps


def test_merge_shards_refuses_a_mismatched_budget(tmp_path):
    """Two shards run at different CEM budgets are two experiments; averaging them would
    silently produce a number belonging to neither."""
    import json

    import pytest

    from tools.eval.openloop_cem import _merge_shards, _shard_path

    base = _write_shards(tmp_path, n_shards=2, n_rows=8)
    bad = _shard_path(base, 1).with_suffix(".meta.json")
    meta = json.loads(bad.read_text())
    meta["cem"]["iterations"] = 15
    bad.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="disagree on 'cem'"):
        _merge_shards(base)


def test_merge_shards_refuses_overlapping_shards(tmp_path):
    """Guards the failure that would otherwise pass silently: two shards given the same
    --shard index, which double-counts queries instead of covering the set."""
    import json

    import pytest

    from tools.eval.openloop_cem import _merge_shards, _shard_path

    base = _write_shards(tmp_path, n_shards=2, n_rows=8)
    dup = _shard_path(base, 1).with_suffix(".jsonl")
    dup.write_text("\n".join(json.dumps(_fake_row(q)) for q in range(0, 8, 2)) + "\n")
    with pytest.raises(SystemExit, match="shards overlap"):
        _merge_shards(base)


def test_query_seed_is_shard_invariant_and_collision_free():
    """The CEM stream must be a function of (seed, qid) only. A single generator consumed
    across the query loop makes every plan depend on how many queries ran first, so
    --num-shards silently changes results — a 4-way sharded re-run under that scheme
    matched the aggregate to 0.3 % with not one query bit-identical."""
    from tools.eval.openloop_cem import _query_seed

    # order-independence: the value for a qid does not depend on what came before it
    assert _query_seed(0, 37) == _query_seed(0, 37)
    # a shard covering {0,4,8,...} and one covering every query agree on shared qids
    for qid in (0, 4, 8, 12):
        assert _query_seed(0, qid) == _query_seed(0, qid)
    # different queries and different base seeds must not collide
    assert len({_query_seed(0, q) for q in range(2000)}) == 2000
    assert _query_seed(0, 5) != _query_seed(1, 5)
    assert all(0 <= _query_seed(s, q) < 2**63 for s in (0, 1, 12345) for q in (0, 99, 10**6))
