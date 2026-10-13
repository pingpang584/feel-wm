"""Unit tests for the planning score algebra (goal similarity − trav risk − ride cost).

Run: python -m pytest tests/planning/test_score.py -v
"""
from __future__ import annotations

import torch

from feelwm.planning.score import (
    ScoreConfig,
    combine_score,
    goal_similarity,
    standardize,
    traversability_gate_stat,
    traversability_risk,
)


def test_goal_similarity_identical_is_one():
    z = torch.randn(4, 256, 768)
    sim = goal_similarity(z, z.clone())
    assert torch.allclose(sim, torch.ones(4), atol=1e-5)


def test_goal_similarity_mean_over_patches():
    # patch 0 identical (cos=1), patch 1 anti-parallel (cos=-1) → mean 0.
    pred = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])      # [1,2,2]
    goal = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    assert torch.allclose(goal_similarity(pred, goal), torch.zeros(1), atol=1e-6)


def test_traversability_risk_masked_mean():
    c = torch.tensor([[[0.2], [0.4], [0.9], [0.9]]])  # [1,4,1]
    mask = torch.tensor([[True, True, False, False]])
    assert torch.allclose(traversability_risk(c, mask), torch.tensor([0.3]))
    # squeeze handles [N,H] too
    assert torch.allclose(traversability_risk(c.squeeze(-1)), torch.tensor([0.6]))


def test_standardize_zero_mean_unit_std():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    z = standardize(x)
    assert torch.allclose(z.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(z.std(unbiased=False), torch.tensor(1.0), atol=1e-5)


def test_standardize_degenerate_pool_is_zero():
    x = torch.full((5,), 3.0)
    assert torch.allclose(standardize(x), torch.zeros(5))


def test_combine_score_lambda_ride_zero_is_goal_only():
    goal = torch.randn(8)
    trav = torch.rand(8)
    ride = torch.randn(8)
    # λ_ride=0 must equal dropping ride entirely (the ride-cost ablation).
    cfg0 = ScoreConfig(lambda_trav=1.0, lambda_ride=0.0)
    s_zero = combine_score(goal, trav, ride, cfg0)
    s_none = combine_score(goal, trav, None, ScoreConfig(lambda_trav=1.0))
    assert torch.allclose(s_zero, s_none)


def test_combine_score_lambda_gsim_default_is_locked_spec():
    goal, trav, ride = torch.randn(8), torch.rand(8), torch.randn(8)
    # explicit λ_gsim=1 must be bit-identical to the default (no code path change).
    assert torch.equal(combine_score(goal, trav, ride, ScoreConfig(lambda_gsim=1.0)),
                       combine_score(goal, trav, ride))


def test_combine_score_lambda_gsim_zero_drops_visual_goal():
    goal, trav, ride = torch.randn(8), torch.rand(8), torch.randn(8)
    cfg0 = ScoreConfig(lambda_gsim=0.0)
    s0 = combine_score(goal, trav, ride, cfg0)
    # identical to scoring a constant (uninformative) goal_sim — the visual goal term
    # is gone, the risk terms are untouched.
    s_flat = combine_score(torch.zeros(8), trav, ride)
    assert torch.allclose(s0, s_flat)
    # and the visual goal no longer moves the ranking at all
    s0_other = combine_score(torch.randn(8), trav, ride, cfg0)
    assert torch.allclose(s0, s0_other)


def test_combine_score_lambda_gsim_scales_only_the_goal_term():
    goal, trav = torch.randn(8), torch.rand(8)
    cfg2 = ScoreConfig(lambda_gsim=2.0)
    expected = (2.0 * standardize(goal)) - standardize(trav)
    assert torch.allclose(combine_score(goal, trav, None, cfg2), expected, atol=1e-6)


def test_combine_score_ranking_prefers_smooth():
    # two candidates, identical goal_sim/trav; candidate 0 is smoother (lower ride).
    goal = torch.tensor([0.0, 0.0])
    trav = torch.tensor([0.5, 0.5])
    ride = torch.tensor([-1.0, 1.0])  # cand 0 smoother
    score = combine_score(goal, trav, ride, ScoreConfig(lambda_ride=1.0))
    assert score[0] > score[1]
    # with λ_ride=0 the two tie (goal-only planner is indifferent).
    score0 = combine_score(goal, trav, ride, ScoreConfig(lambda_ride=0.0))
    assert torch.allclose(score0[0], score0[1])


def test_combine_score_sign_directions():
    # higher goal_sim → higher score; higher trav/ride → lower score.
    goal = torch.tensor([0.0, 1.0])
    flat = torch.tensor([0.0, 0.0])
    assert combine_score(goal, flat, flat)[1] > combine_score(goal, flat, flat)[0]
    risk = torch.tensor([0.0, 1.0])
    assert combine_score(flat, risk, flat)[1] < combine_score(flat, risk, flat)[0]
    ride = torch.tensor([0.0, 1.0])
    assert combine_score(flat, flat, ride)[1] < combine_score(flat, flat, ride)[0]


def test_gate_stat_masked_max():
    # max, not mean — a single lethal step must survive aggregation.
    c = torch.tensor([[[0.02], [0.30], [0.05], [0.90]]])  # [1,4,1]
    assert torch.allclose(traversability_gate_stat(c), torch.tensor([0.90]))
    mask = torch.tensor([[True, True, False, False]])  # proprio horizon cuts the far steps
    assert torch.allclose(traversability_gate_stat(c, mask), torch.tensor([0.30]))
    # all-false mask falls back to the full horizon, not −inf.
    none_mask = torch.zeros(1, 4, dtype=torch.bool)
    assert torch.allclose(traversability_gate_stat(c, none_mask), torch.tensor([0.90]))


def test_gate_off_by_default_and_when_stat_missing():
    goal, trav, ride = torch.randn(8), torch.rand(8), torch.randn(8)
    gate_stat = torch.rand(8)
    base = combine_score(goal, trav, ride)
    # default cfg (tau=None) ignores the stat; tau set but stat missing is also a no-op.
    assert torch.allclose(combine_score(goal, trav, ride, trav_gate_stat=gate_stat), base)
    cfg = ScoreConfig(trav_gate_tau=0.1)
    assert torch.allclose(combine_score(goal, trav, ride, cfg, trav_gate_stat=None), base)


def test_gate_vetoes_goal_purchase():
    # the S5 pathology: cand 1 leads on goal_sim while its lethal step hides in the
    # horizon-MEAN trav (mid-pool after dilution) — the linear score buys it.
    goal = torch.tensor([0.0, 3.0, 0.1, 0.2])
    trav = torch.tensor([0.02, 0.025, 0.03, 0.02])
    gate_stat = torch.tensor([0.03, 0.25, 0.04, 0.03])
    cfg = ScoreConfig(trav_gate_tau=0.1)
    linear = combine_score(goal, trav, None)
    gated = combine_score(goal, trav, None, cfg, trav_gate_stat=gate_stat)
    assert linear.argmax() == 1          # linear buys the lethal candidate
    assert gated.argmax() != 1           # gate refuses the purchase
    # penalty form: non-vetoed candidates keep their exact linear scores.
    assert torch.allclose(gated[[0, 2, 3]], linear[[0, 2, 3]])
    assert torch.allclose(gated[1], linear[1] - cfg.trav_gate_penalty)


def test_gate_all_vetoed_degrades_to_linear_ranking():
    # every candidate over τ → uniform penalty → ordering identical to linear (the
    # strict-refinement guarantee; no empty-pool special case).
    goal, trav = torch.randn(6), torch.rand(6)
    gate_stat = torch.full((6,), 0.5)
    cfg = ScoreConfig(trav_gate_tau=0.1)
    linear = combine_score(goal, trav, None)
    gated = combine_score(goal, trav, None, cfg, trav_gate_stat=gate_stat)
    assert torch.equal(torch.argsort(gated), torch.argsort(linear))


def test_gate_leaves_ride_axis_linear():
    # Ride-ablation protection: with identical gate outcomes, the λ_ride 0-vs-1 contrast is
    # unchanged by the gate (z terms computed on the FULL pool, penalty is additive).
    g = torch.Generator().manual_seed(0)
    goal, trav, ride = torch.randn(8, generator=g), torch.rand(8, generator=g), torch.randn(8, generator=g)
    gate_stat = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    cfg_r1 = ScoreConfig(lambda_ride=1.0, trav_gate_tau=0.1)
    cfg_r0 = ScoreConfig(lambda_ride=0.0, trav_gate_tau=0.1)
    delta_gated = (combine_score(goal, trav, ride, cfg_r1, trav_gate_stat=gate_stat)
                   - combine_score(goal, trav, ride, cfg_r0, trav_gate_stat=gate_stat))
    delta_linear = (combine_score(goal, trav, ride, ScoreConfig(lambda_ride=1.0))
                    - combine_score(goal, trav, ride, ScoreConfig(lambda_ride=0.0)))
    assert torch.allclose(delta_gated, delta_linear, atol=1e-6)


# ── λ_slip term + proprio fusion gate (safety terms of the score) ────────────────────


def test_lambda_slip_default_zero_leaves_score_unchanged():
    g = torch.tensor([1.0, 2.0, 3.0])
    slip = torch.tensor([9.0, 0.0, 5.0])
    base = combine_score(g, None, None)
    with_slip = combine_score(g, None, None, slip_cost=slip)
    assert torch.equal(base, with_slip)


def test_lambda_slip_prefers_low_slip():
    g = torch.zeros(3)                       # goal indifferent
    slip = torch.tensor([0.0, 1.0, 2.0])
    s = combine_score(g, None, None, ScoreConfig(lambda_slip=1.0), slip_cost=slip)
    assert int(s.argmax()) == 0 and int(s.argmin()) == 2


def test_proprio_gate_vetoes_on_ride_stat():
    g = torch.tensor([3.0, 0.0])             # candidate 0 wins on goal alone
    rmax = torch.tensor([2.0, 0.1])          # …but is proprio-hot
    cfg = ScoreConfig(proprio_gate_tau=1.0)
    s = combine_score(g, None, None, cfg, proprio_gate_stat=rmax)
    assert int(s.argmax()) == 1


def test_fusion_gate_or_semantics_single_penalty():
    g = torch.zeros(4)
    tstat = torch.tensor([0.9, 0.0, 0.9, 0.0])   # trav-hot: 0, 2
    rstat = torch.tensor([2.0, 2.0, 0.0, 0.0])   # proprio-hot: 0, 1
    cfg = ScoreConfig(trav_gate_tau=0.5, proprio_gate_tau=1.0, trav_gate_penalty=100.0)
    s = combine_score(g, None, None, cfg, trav_gate_stat=tstat, proprio_gate_stat=rstat)
    # candidate 0 trips BOTH gates but takes ONE penalty (OR, not stacked)
    assert torch.allclose(s, torch.tensor([-100.0, -100.0, -100.0, 0.0]))


def test_proprio_gate_off_when_stat_missing():
    g = torch.tensor([1.0, 0.0])
    cfg = ScoreConfig(proprio_gate_tau=1.0)
    s = combine_score(g, None, None, cfg)    # no stat supplied → no veto
    assert int(s.argmax()) == 0


# ── score-term breakdown (per-term score logging) ────────────────────────────────────


def test_standardize_against_matches_standardize_for_a_pool_member():
    from feelwm.planning.score import standardize_against

    pool = torch.tensor([1.0, 2.0, 5.0, 9.0])
    z_all = standardize(pool)
    for i in range(pool.numel()):
        assert torch.allclose(standardize_against(pool[i], pool), z_all[i], atol=1e-6)


def test_score_breakdown_reproduces_combine_score_for_a_pool_member():
    from feelwm.planning.score import score_breakdown

    g = torch.tensor([0.2, 0.9, 0.4, 0.1])
    trav = torch.tensor([0.02, 0.30, 0.05, 0.10])
    ride = torch.tensor([1.0, -0.5, 0.3, 2.0])
    slip = torch.tensor([0.1, 0.4, 0.9, 0.2])
    cfg = ScoreConfig(lambda_gsim=1.0, lambda_trav=1.0, lambda_ride=1.0, lambda_slip=0.5)
    scores = combine_score(g, trav, ride, cfg, slip_cost=slip)
    pool = {"goal_sim": g, "trav_risk": trav, "ride_cost": ride, "slip_cost": slip}
    for i in range(4):
        chosen = {k: v[i] for k, v in pool.items()}
        bd = score_breakdown(chosen, pool, cfg)
        assert abs(bd["w_sum"] - float(scores[i])) < 1e-5
        # signs: goal_sim adds, risk terms subtract
        assert bd["w_gsim"] == cfg.lambda_gsim * bd["z_gsim"]
        assert bd["w_trav"] == -cfg.lambda_trav * bd["z_trav"]


def test_score_breakdown_skips_missing_terms_and_honors_lambdas():
    from feelwm.planning.score import score_breakdown

    g = torch.tensor([0.0, 1.0, 2.0])
    trav = torch.tensor([0.1, 0.2, 0.3])
    cfg = ScoreConfig(lambda_gsim=0.0, lambda_trav=2.0)
    bd = score_breakdown({"goal_sim": g[2], "trav_risk": trav[2]},
                         {"goal_sim": g, "trav_risk": trav}, cfg)
    assert "z_ride" not in bd and "w_ride" not in bd      # term absent → skipped
    assert bd["w_gsim"] == 0.0                            # λ_gsim=0 zeroes its weight
    assert bd["z_gsim"] != 0.0                            # …but the z is still reported
    assert abs(bd["w_trav"] + 2.0 * bd["z_trav"]) < 1e-6
