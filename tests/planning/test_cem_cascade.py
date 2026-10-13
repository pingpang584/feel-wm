"""Cascaded candidate pruning in CEM — a speed lever for embedded real-time planning.

Profiling on an embedded GPU measured a flat cost
per candidate-STEP from batch 1 to 128: the GPU is saturated by a single candidate's
rollout, so there is no batching economy and wall time is exactly `iters × N × H` steps.
That is what makes pruning worth anything — cutting candidates cuts time 1:1. The
cascade rolls all N candidates `H_pre` steps, keeps the best K on that partial score, and
continues only those to the full horizon: `N·H → N·H_pre + K·H` per iteration.

It is not free: the prefix score is a DIFFERENT score (goal_sim read at step H_pre, the
proprio/trav window cut by the prefix's cum_dt), so a candidate whose payoff is in the
tail can be pruned before it is ever rolled. Hence the two properties asserted here — it
prunes what it claims to prune, and it is EXACTLY the old search when asked to keep
everything, so the closed-loop A/B measures the lever and nothing else.

Run: python -m pytest tests/planning/test_cem_cascade.py -v
"""
from __future__ import annotations

import torch

from feelwm.planning.cem import CEMConfig, cem_optimize


class _PrefixTrapScorer:
    """Score = −|a0 at the LAST step| — so the ranking is decided by the tail alone.

    The prefix (all but the last step) is deliberately uninformative, which makes the
    cascade's failure mode visible: pruning on a prefix score can throw away the very
    candidate the full-horizon score would have picked.
    """

    device = torch.device("cpu")

    def __init__(self):
        self.batches = []           # (N, H) of every batch scored — N·H is the cost
        self.full = []              # the full-horizon batches, kept for identity checks

    @property
    def horizons(self):
        return [h for _, h in self.batches]

    @property
    def candidate_steps(self):
        """What the lever actually buys: rolled candidate-steps, 1:1 with wall time."""
        return sum(n * h for n, h in self.batches)

    def __call__(self, actions):
        self.batches.append((int(actions.shape[0]), int(actions.shape[1])))
        if actions.shape[1] == 8:
            self.full.append(actions.clone())
        scores = -actions[:, -1, 0].abs()
        return scores, {"goal_sim": scores, "trav_risk": None, "trav_max": None,
                        "ride_cost": None, "ride_max": None, "slip_cost": None,
                        "horizon_mask": None, "rollout": {}}


def _init(H=8):
    a = torch.zeros(H, 4)
    a[:, 3] = 0.1
    a[:, 0] = 2.0
    return a


def _run(cfg_kw, seed=0):
    scorer = _PrefixTrapScorer()
    g = torch.Generator().manual_seed(seed)
    res = cem_optimize(scorer, _init(), CEMConfig(n_candidates=32, iterations=3, **cfg_kw),
                       generator=g)
    return res, scorer


def test_keeping_every_candidate_reproduces_the_uncascaded_search() -> None:
    """K = n_candidates (or H_pre = H) is a no-op, bit-for-bit.

    This is the control arm of a cascade A/B: any difference a sweep measures
    must come from actual pruning, not from the cascade code path existing. The RNG is
    untouched by the prescreen (it consumes no draws), so the pools are identical.
    """
    plain, s_plain = _run({})
    for kw in ({"cascade_k": 32, "cascade_h_pre": 3},      # keeps everyone
               {"cascade_k": 8, "cascade_h_pre": 8},        # prescreen == full horizon
               {"cascade_k": None, "cascade_h_pre": 3}):    # half-configured = off
        res, scorer = _run(kw)
        assert torch.equal(res["plan"], plain["plan"])
        assert res["history"] == plain["history"]
        # ... and it did not pay for a prescreen it would not use
        assert scorer.horizons == s_plain.horizons


def test_the_prescreen_rolls_the_prefix_and_only_survivors_the_full_horizon() -> None:
    res, scorer = _run({"cascade_k": 8, "cascade_h_pre": 3})
    # per iteration: one 32-candidate prefix batch (H=3) + one 8-candidate full batch,
    # then the final rescore of the mean
    assert scorer.horizons == [3, 8] * 3 + [8]
    assert res["plan"].shape == (8, 4)
    # the payoff, in the unit that converts 1:1 into wall time: per iteration
    # 32·3 + 8·8 = 160 candidate-steps instead of 32·8 = 256, so 3 iterations plus the
    # final rescore of the mean go from 776 to 488 — a 1.6× lever at K/N = 1/4.
    plain, s_plain = _run({})
    assert (scorer.candidate_steps, s_plain.candidate_steps) == (488, 776)


def test_the_pool_diagnostics_describe_the_survivors_not_the_pruned_pool() -> None:
    """`pool_terms` is what the executed plan's score breakdown is standardized against
    (`score.standardize_against`), so it has to be the pool the elite mean was refit
    from — the K survivors, not the N sampled."""
    res, _ = _run({"cascade_k": 8, "cascade_h_pre": 3})
    assert res["pool_terms"]["goal_sim"].shape == (8,)
    plain, _ = _run({})
    assert plain["pool_terms"]["goal_sim"].shape == (32,)


def test_the_elite_mean_is_never_pruned() -> None:
    """Elitism (`keep_mean_elite`) is what keeps the CEM mean monotone across iterations.
    A partial-horizon score must not be able to evict it, so index 0 is exempt from the
    prescreen — asserted here with a scorer that ranks the mean LAST on the prefix."""

    class _MeanIsWorstOnThePrefix(_PrefixTrapScorer):
        def __call__(self, actions):
            scores, aux = super().__call__(actions)
            if actions.shape[1] == 2:            # the prescreen batch
                scores = scores.clone()
                scores[0] = -1e9                 # candidate 0 = the carried mean
                aux["goal_sim"] = scores
            return scores, aux

    scorer = _MeanIsWorstOnThePrefix()
    g = torch.Generator().manual_seed(0)
    init = _init()
    cem_optimize(scorer, init,
                 CEMConfig(n_candidates=16, iterations=1, cascade_k=4, cascade_h_pre=2),
                 generator=g)
    survivors = scorer.full[0]                       # the K rolled to the full horizon
    assert survivors.shape[0] == 4
    # iteration 0's mean IS the seed, so it is identifiable by value
    assert any(torch.equal(survivors[i], init) for i in range(4))
    assert torch.equal(survivors[0], init)           # and it keeps slot 0


def test_elite_count_is_clamped_to_the_survivor_count() -> None:
    """elite_k comes from elite_frac × n_candidates; with a small K that can exceed the
    survivor pool, and `topk` would raise. The elite set is then simply every survivor."""
    res, _ = _run({"cascade_k": 2, "cascade_h_pre": 3}, seed=1)   # elite_k = 3 > K = 2
    assert torch.isfinite(res["plan"]).all()
