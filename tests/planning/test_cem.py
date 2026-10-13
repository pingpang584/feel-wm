"""Unit tests for the planner scorer + CEM (sampling-based rollout ranking).

Uses a tiny fake world model (no checkpoint / GPU) whose roughness tracks action
dim 0, so lower-magnitude plans are cheaper — lets us assert ranking/optimization
direction without the real network.

Run: python -m pytest tests/planning/test_cem.py -v
"""
from __future__ import annotations

import torch

from feelwm.planning.cem import (
    CEMConfig,
    WorldModelScorer,
    cem_optimize,
    expert_rank_percentile,
    horizon_mask_from_actions,
    rank_candidates,
)

CHANNELS = ["roughness", "slip_ratio", "angular_instability", "slope_power"]


class _FakeWM:
    """forward_rollout where roughness == action dim 0; goal_sim / trav constant."""

    def __init__(self, with_heads: bool = True):
        self.with_heads = with_heads

    def forward_rollout(self, initial_batch, actions):
        N, H, _ = actions.shape
        P, D = 4, 8
        z = torch.zeros(N, H, P, D)
        z[:, -1, :, 0] = 1.0  # constant goal direction → goal_sim constant across pool
        out = {"z_vis_preds": z, "f_prop_preds": None, "c_preds": None}
        if self.with_heads:
            f = torch.zeros(N, H, 4)
            f[..., 0] = actions[..., 0]  # roughness tracks action dim 0
            out["f_prop_preds"] = f
            out["c_preds"] = torch.full((N, H, 1), 0.3)
        return out


def _scorer(with_heads=True, ph_s=1.0):
    ib = {
        "z_vis_context": torch.zeros(1, 3, 4, 8),
        "proprio_window_context": torch.zeros(1, 3, 5, 14),
    }
    goal = torch.zeros(4, 8)
    goal[:, 0] = 1.0
    return WorldModelScorer(_FakeWM(with_heads), ib, goal, CHANNELS, proprio_horizon_s=ph_s)


def test_horizon_mask_from_actions():
    a = torch.zeros(1, 4, 4)
    a[0, :, 3] = torch.tensor([0.5, 0.5, 0.5, 0.5])  # cum_dt = .5,1.,1.5,2.
    mask = horizon_mask_from_actions(a, proprio_horizon_s=1.0)
    assert mask.tolist() == [[True, True, False, False]]
    # legacy 3-dim → no Δt → None
    assert horizon_mask_from_actions(torch.zeros(1, 4, 3)) is None


def test_scorer_returns_pool_scores():
    scorer = _scorer()
    actions = torch.zeros(5, 4, 4)
    actions[:, :, 3] = 0.1
    actions[:, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])[:, None]  # increasing roughness
    scores, aux = scorer(actions)
    assert scores.shape == (5,)
    assert torch.isfinite(scores).all()
    # lower roughness → higher score (goal_sim & trav constant)
    assert torch.argmax(scores).item() == 0


def test_rank_candidates_orders_descending():
    scorer = _scorer()
    actions = torch.zeros(4, 4, 4)
    actions[:, :, 3] = 0.1
    actions[:, :, 0] = torch.tensor([3.0, 0.0, 2.0, 1.0])[:, None]
    res = rank_candidates(scorer, actions)
    assert res["best"] == 1  # the smoothest plan
    assert res["order"].tolist() == [1, 3, 2, 0]


def test_scorer_drops_missing_heads():
    scorer = _scorer(with_heads=False)
    actions = torch.zeros(3, 4, 4)
    actions[:, :, 3] = 0.1
    scores, aux = scorer(actions)
    assert aux["ride_cost"] is None and aux["trav_risk"] is None
    # only goal_sim remains; constant → standardized to ~0
    assert torch.allclose(scores, torch.zeros(3), atol=1e-5)


def test_expert_rank_percentile():
    scores = torch.tensor([0.1, 0.5, 0.9, 0.3])
    assert expert_rank_percentile(scores, 2) == 1.0          # top
    assert expert_rank_percentile(scores, 0) == 0.0          # worst
    assert abs(expert_rank_percentile(scores, 1) - 2 / 3) < 1e-6


def test_cem_drives_cost_down():
    scorer = _scorer()
    init = torch.zeros(4, 4)
    init[:, 3] = 0.1
    init[:, 0] = 2.0  # start rough
    g = torch.Generator().manual_seed(0)
    res = cem_optimize(scorer, init, CEMConfig(n_candidates=64, iterations=5), generator=g)
    assert res["plan"].shape == (4, 4)
    # CEM should lower roughness (action dim 0) and keep Δt fixed
    assert res["plan"][:, 0].mean() < init[:, 0].mean()
    assert torch.allclose(res["plan"][:, 3], init[:, 3])
    assert len(res["history"]) == 5


class _TailHotWM(_FakeWM):
    """c_preds hot ONLY at the last step — the far-tail saturation pattern."""

    def forward_rollout(self, initial_batch, actions):
        out = super().forward_rollout(initial_batch, actions)
        c = torch.full_like(out["c_preds"], 0.01)
        c[:, -1] = 0.99
        out["c_preds"] = c
        return out


def test_gate_horizon_excludes_far_tail():
    # 4 steps × 0.5 s: full horizon sees the hot tail (max 0.99); a 1.0 s gate
    # horizon covers only steps 0-1 (max 0.01) — commitment-window gating.
    ib = {"z_vis_context": torch.zeros(1, 3, 4, 8),
          "proprio_window_context": torch.zeros(1, 3, 5, 14)}
    goal = torch.zeros(4, 8)
    goal[:, 0] = 1.0
    actions = torch.zeros(3, 4, 4)
    actions[:, :, 3] = 0.5
    full = WorldModelScorer(_TailHotWM(), ib, goal, CHANNELS, proprio_horizon_s=2.0)
    _, aux_full = full(actions)
    assert torch.allclose(aux_full["trav_max"], torch.full((3,), 0.99))
    gated = WorldModelScorer(_TailHotWM(), ib, goal, CHANNELS, proprio_horizon_s=2.0,
                             gate_horizon_s=1.0)
    _, aux_g = gated(actions)
    assert torch.allclose(aux_g["trav_max"], torch.full((3,), 0.01))
    # the mean trav term (lookahead guidance) is IDENTICAL — only the gate stat moves
    assert torch.allclose(aux_full["trav_risk"], aux_g["trav_risk"])


def test_cem_returns_pool_trav_max():
    scorer = _scorer()
    init = torch.zeros(4, 4)
    init[:, 3] = 0.1
    g = torch.Generator().manual_seed(0)
    res = cem_optimize(scorer, init, CEMConfig(n_candidates=16, iterations=2), generator=g)
    assert res["pool_trav_max"] is not None and res["pool_trav_max"].shape == (16,)


class _MeanHotScorer:
    """Stub: pool candidates are cold (survivors); any 1-candidate call (the refit
    mean) reports HOT — the measured elite-mean gate violation."""

    device = torch.device("cpu")

    def __call__(self, actions):
        n = actions.shape[0]
        scores = -actions[:, :, 0].abs().mean(dim=-1)  # prefer dim0 near 0
        hot = 0.99 if n == 1 else 0.01
        return scores, {"trav_max": torch.full((n,), hot),
                        "goal_sim": scores, "trav_risk": None, "ride_cost": None,
                        "rollout": {}}


def test_veto_guard_replaces_gate_rejected_mean():
    init = torch.zeros(4, 4)
    init[:, 3] = 0.1
    g = torch.Generator().manual_seed(0)
    cfg = CEMConfig(n_candidates=16, iterations=2, gate_tau=0.3)
    res = cem_optimize(_MeanHotScorer(), init, cfg, generator=g)
    assert res["guard_fired"] is True
    # without the guard the mean is executed regardless
    res0 = cem_optimize(_MeanHotScorer(), init,
                        CEMConfig(n_candidates=16, iterations=2), generator=g)
    assert res0["guard_fired"] is False


# ── slip term + proprio fusion gate plumbing (safety terms of the score) ─────────────


class _SlipWM(_FakeWM):
    """slip_ratio (column 1) tracks action dim 1; roughness stays on dim 0."""

    def forward_rollout(self, initial_batch, actions):
        out = super().forward_rollout(initial_batch, actions)
        out["f_prop_preds"][..., 1] = actions[..., 1]
        return out


def _slip_scorer(score_cfg=None):
    from feelwm.planning.ride_cost import RideCostConfig

    ib = {
        "z_vis_context": torch.zeros(1, 3, 4, 8),
        "proprio_window_context": torch.zeros(1, 3, 5, 14),
    }
    goal = torch.zeros(4, 8)
    goal[:, 0] = 1.0
    return WorldModelScorer(
        _SlipWM(), ib, goal, CHANNELS, score_cfg=score_cfg,
        slip_cfg=RideCostConfig(ride_channels=["slip_ratio"], agg="max"),
    )


def test_scorer_emits_slip_and_ride_max_aux():
    scorer = _slip_scorer()
    actions = torch.zeros(3, 4, 3)
    actions[:, :, 1] = torch.tensor([0.0, 1.0, 2.0])[:, None]
    _, aux = scorer(actions)
    assert aux["slip_cost"] is not None and aux["ride_max"] is not None
    # slip aux = worst-step slip column per candidate
    assert torch.allclose(aux["slip_cost"], torch.tensor([0.0, 1.0, 2.0]))


def test_lambda_slip_ranks_low_slip_first():
    from feelwm.planning.score import ScoreConfig

    scorer = _slip_scorer(ScoreConfig(lambda_trav=0.0, lambda_ride=0.0, lambda_slip=1.0))
    actions = torch.zeros(3, 4, 3)
    actions[:, :, 1] = torch.tensor([2.0, 0.0, 1.0])[:, None]
    scores, _ = scorer(actions)
    assert int(scores.argmax()) == 1 and int(scores.argmin()) == 0


def test_cem_returns_pool_ride_max_and_fusion_guard():
    from feelwm.planning.score import ScoreConfig

    # roughness (ride channel, dim 0) is what ride_max sees; make the pool hot on it
    scorer = _slip_scorer(ScoreConfig(proprio_gate_tau=0.5))
    init = torch.zeros(4, 3)
    out = cem_optimize(scorer, init, CEMConfig(n_candidates=8, iterations=2,
                                               gate_tau_p=0.5))
    assert out["pool_ride_max"] is not None
    assert out["pool_ride_max"].shape == (8,)
    # the guard path ran without error; guard_fired is a bool either way
    assert isinstance(out["guard_fired"], bool)


def test_cem_returns_pool_terms_for_the_score_breakdown():
    """Last-iteration raw term vectors are always retained (logging is non-behavioral)."""
    from feelwm.planning.score import score_breakdown

    scorer = _scorer()
    init = torch.zeros(4, 3)
    out = cem_optimize(scorer, init, CEMConfig(n_candidates=8, iterations=2))
    pool = out["pool_terms"]
    assert set(pool) >= {"goal_sim", "trav_risk", "ride_cost"}
    assert all(v.shape == (8,) for v in pool.values())
    # the executed plan (elite mean) is outside the pool but still decomposable
    aux = out["aux"]
    bd = score_breakdown({k: aux[k][0] for k in ("goal_sim", "trav_risk", "ride_cost")},
                         pool)
    assert {"z_gsim", "w_gsim", "z_ride", "w_ride", "w_sum"} <= set(bd)


def test_trace_pool_carries_the_per_candidate_proprio_rollout():
    """`trace` keeps [N, H, C] proprio preds — the offline score-counterfactual input."""
    scorer = _scorer()
    init = torch.zeros(4, 4)
    init[:, 3] = 0.1
    g = torch.Generator().manual_seed(0)
    out = cem_optimize(scorer, init, CEMConfig(n_candidates=8, iterations=2),
                       generator=g, trace=True)
    fp = out["pool"]["f_prop_preds"]
    assert fp.shape == (8, 4, 4)
    # it IS the last iteration's pool: roughness (channel 0) tracks action dim 0
    assert torch.allclose(fp[..., 0], out["pool"]["actions"][..., 0])
    # a head-less checkpoint simply omits the key (backward/forward compatible)
    out0 = cem_optimize(_scorer(with_heads=False), init,
                        CEMConfig(n_candidates=8, iterations=2),
                        generator=torch.Generator().manual_seed(0), trace=True)
    assert "f_prop_preds" not in out0["pool"]


def test_trace_is_non_behavioral():
    """Same seed, trace on/off → bit-identical plan (dumping must not move the robot)."""
    init = torch.zeros(4, 4)
    init[:, 3] = 0.1
    outs = [cem_optimize(_scorer(), init, CEMConfig(n_candidates=8, iterations=2),
                         generator=torch.Generator().manual_seed(3), trace=t)
            for t in (False, True)]
    assert torch.equal(outs[0]["plan"], outs[1]["plan"])
    assert outs[0]["score"] == outs[1]["score"]


def test_cem_pool_terms_do_not_change_the_plan():
    """Same seed → identical plan/score with the term capture in place (bit-identical)."""
    init = torch.zeros(4, 3)
    plans = []
    for _ in range(2):
        g = torch.Generator().manual_seed(7)
        out = cem_optimize(_scorer(), init, CEMConfig(n_candidates=8, iterations=2),
                           generator=g)
        plans.append(out)
    assert torch.equal(plans[0]["plan"], plans[1]["plan"])
    assert plans[0]["score"] == plans[1]["score"]


# --- free-CEM knobs (NWM-symmetric CEM eval): train-prior init + clipping ------------


def test_cem_clips_candidates_to_the_action_range():
    """With clip bounds set, neither the sampled pool nor the returned plan may leave the
    box — the point of the knob is that CEM cannot wander outside the training action
    distribution the world model was fitted on."""
    scorer = _scorer()
    H, da = 4, 4
    init = torch.zeros(H, da)
    init[:, 3] = 0.4                                  # Δt column, untouched by CEM
    lo, hi = torch.full((3,), -0.2), torch.full((3,), 0.2)
    res = cem_optimize(
        scorer, init,
        CEMConfig(n_candidates=32, iterations=3, init_sigma=5.0, clip_min=lo, clip_max=hi),
        generator=torch.Generator().manual_seed(0),
    )
    plan = res["plan"]
    assert (plan[:, :3] >= lo - 1e-6).all() and (plan[:, :3] <= hi + 1e-6).all()
    assert torch.allclose(plan[:, 3], torch.full((H,), 0.4))   # Δt preserved


def test_cem_without_clipping_leaves_the_box():
    """Guard on the default: no bounds → legacy behavior, wide exploration allowed."""
    scorer = _scorer()
    init = torch.zeros(4, 4)
    res = cem_optimize(scorer, init,
                       CEMConfig(n_candidates=32, iterations=3, init_sigma=5.0),
                       generator=torch.Generator().manual_seed(0))
    assert res["plan"][:, :3].abs().max() > 0.2       # unconstrained search


def test_cem_init_std_is_per_dimension():
    """init_std replaces the scalar init_sigma per action dim: a dim given ~zero std must
    stay at its seed value after the first sampling round."""
    scorer = _scorer()
    init = torch.zeros(3, 4)
    std = torch.tensor([1.0, 1e-8, 1e-8])
    res = cem_optimize(scorer, init,
                       CEMConfig(n_candidates=16, iterations=1, epsilon=1e-9, init_std=std),
                       generator=torch.Generator().manual_seed(0))
    assert res["plan"][:, 1].abs().max() < 1e-5
    assert res["plan"][:, 2].abs().max() < 1e-5
