"""Sampling-based planning over the terrain world model (offline open-loop eval).

Fusion happens at the score level: score = goal_sim − λ_trav·trav_risk − λ_ride·ride_cost.
"""
from __future__ import annotations

from feelwm.planning.candidates import (
    CandidateBatch,
    inject_expert,
    perturb_actions,
    stack_samples,
)
from feelwm.planning.cem import (
    CEMConfig,
    WorldModelScorer,
    cem_optimize,
    expert_rank_percentile,
    horizon_mask_from_actions,
    rank_candidates,
)
from feelwm.planning.ride_cost import RideCostConfig, ride_cost, ride_cost_physical
from feelwm.planning.score import (
    ScoreConfig,
    combine_score,
    goal_similarity,
    standardize,
    traversability_gate_stat,
    traversability_risk,
)

__all__ = [
    # ride_cost (ablation term, weighted by lambda_ride)
    "RideCostConfig",
    "ride_cost",
    "ride_cost_physical",
    # score
    "ScoreConfig",
    "combine_score",
    "goal_similarity",
    "standardize",
    "traversability_gate_stat",
    "traversability_risk",
    # candidates
    "CandidateBatch",
    "stack_samples",
    "inject_expert",
    "perturb_actions",
    # cem / planner
    "CEMConfig",
    "WorldModelScorer",
    "cem_optimize",
    "rank_candidates",
    "expert_rank_percentile",
    "horizon_mask_from_actions",
]
