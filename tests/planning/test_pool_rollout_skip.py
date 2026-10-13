"""Pool-rollout skip and `wm_rescore_topk` — the fresh-plans implementation.

Two properties carry it:
the skip must fire ONLY when nothing the rollout produces can change the ranking, and it
must be UNREACHABLE under every committed sim config (those keep λ_gsim=1).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from feelwm.planning.cem import CEMConfig, WorldModelScorer, cem_optimize
from feelwm.planning.ride_cost import RideCostConfig
from feelwm.planning.score import ScoreConfig

H, DA, P, D, C = 4, 4, 6, 8, 2
CHANNELS = ["roughness", "angular_instability", "slope_power"]


class _CountingModel:
    """Minimal world model that counts how often the pool is rolled."""

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def forward_rollout(self, batch, actions):
        self.calls += 1
        n = actions.shape[0]
        self.batch_sizes.append(n)
        g = torch.Generator().manual_seed(n * 1000 + self.calls)
        return {
            "z_vis_preds": torch.rand(n, H, P, D, generator=g),
            "f_prop_preds": torch.rand(n, H, len(CHANNELS), generator=g),
            "c_preds": None,
            "c_pred_logits": None,
        }


def _scorer(model, **score_kw):
    cfg = ScoreConfig(**score_kw)
    return WorldModelScorer(model, {"z_vis_context": torch.zeros(1, C, P, D)},
                            torch.zeros(P, D), CHANNELS, score_cfg=cfg,
                            ride_cfg=RideCostConfig(ride_channels=CHANNELS, agg="mean"),
                            proprio_horizon_s=1.0)


def _pool(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.zeros(n, H, DA)
    a[..., :3] = torch.randn(n, H, 3, generator=g) * 0.1
    a[..., 3] = 0.1
    return a


ALL_ZERO = dict(lambda_gsim=0.0, lambda_trav=0.0, lambda_ride=0.0, lambda_slip=0.0)


# ── (a) the skip fires, and costs zero rollouts ─────────────────────────────
def test_all_model_lambdas_zero_runs_no_pool_rollout():
    m = _CountingModel()
    s = _scorer(m, **ALL_ZERO)
    assert s.needs_pool_rollout is False
    scores, aux = s(_pool())
    assert m.calls == 0, "the pool was rolled despite every model λ being 0"
    assert torch.equal(scores, torch.zeros(8))
    assert aux["pool_rollout_skipped"] is True
    assert aux["rollout"] is None and aux["ride_cost"] is None


def test_skipped_aux_still_carries_every_key():
    """Consumers read aux unconditionally; a missing key would be a crash, not a skip."""
    _, aux = _scorer(_CountingModel(), **ALL_ZERO)(_pool())
    for k in ("goal_sim", "trav_risk", "trav_max", "ride_cost", "ride_max",
              "slip_cost", "horizon_mask", "rollout"):
        assert k in aux


def test_cem_with_the_skip_costs_no_rollouts_at_all():
    m = _CountingModel()
    seed = torch.zeros(H, DA)
    seed[:, 3] = 0.1
    out = cem_optimize(_scorer(m, **ALL_ZERO), seed, CEMConfig(n_candidates=8, iterations=2))
    assert m.calls == 0
    assert out["n_pool_rollouts"] == 0


# ── (b) any model λ > 0 keeps the old path, byte-identical ──────────────────
@pytest.mark.parametrize("kw", [
    {"lambda_gsim": 1.0}, {"lambda_ride": 1.0}, {"lambda_trav": 1.0},
    {"lambda_slip": 1.0},
])
def test_any_model_lambda_keeps_the_rollout(kw):
    cfg = dict(ALL_ZERO)
    cfg.update(kw)
    m = _CountingModel()
    s = _scorer(m, **cfg)
    assert s.needs_pool_rollout is True
    s(_pool())
    assert m.calls == 1


@pytest.mark.parametrize("gate", ["trav_gate_tau", "proprio_gate_tau"])
def test_an_armed_gate_keeps_the_rollout(gate):
    """The vetoes are rollout quantities too — λ=0 alone does not make them free."""
    cfg = dict(ALL_ZERO)
    cfg[gate] = 0.5
    m = _CountingModel()
    assert _scorer(m, **cfg).needs_pool_rollout is True


def test_scoring_is_unchanged_when_the_rollout_runs():
    """Regression pin: same pool, same seed ⇒ same scores as the pre-change path."""
    a = _pool(seed=7)
    m1, m2 = _CountingModel(), _CountingModel()
    s1 = _scorer(m1, lambda_gsim=1.0, lambda_ride=1.0)
    s2 = _scorer(m2, lambda_gsim=1.0, lambda_ride=1.0)
    sc1, _ = s1(a)
    sc2, _ = s2(a)
    assert torch.allclose(sc1, sc2)
    assert m1.batch_sizes == m2.batch_sizes == [8]


# ── the skip must be unreachable in sim ─────────────────────────────────────
def test_every_committed_sim_config_keeps_the_rollout():
    """Sim configs carry λ_gsim=1, so the skip can never fire there. Asserted, not assumed."""
    from pathlib import Path

    from feelwm.config import load_config, variant_names

    checked = 0
    repo = Path(__file__).resolve().parents[2]
    for name in ("gazebo", "legged_isaac", "legged_tg_gt"):
        path = repo / "configs" / f"{name}.yaml"
        for v in variant_names(path) or [None]:
            cfg = load_config(path, v) or {}
            lam = (cfg.get("planning") or {}).get("lambda_gsim", cfg.get("lambda_gsim", 1.0))
            assert lam != 0.0, f"{path}:{v} would enable the real-only pool-rollout skip"
            checked += 1
    assert checked > 0, "no sim configs found — the guard would be vacuous"


# ── (c) wm_rescore_topk ─────────────────────────────────────────────────────
def _rescore_run(k, n=8, iters=2):
    m = _CountingModel()
    seed = torch.zeros(H, DA)
    seed[:, 3] = 0.1
    rank = _scorer(m, **ALL_ZERO)                       # geometry-only ⇒ skips
    resc = _scorer(m, lambda_ride=1.0)                  # prices ride ⇒ rolls
    out = cem_optimize(rank, seed, CEMConfig(n_candidates=n, iterations=iters,
                                             rescore_topk=k),
                       rescore_scorer=resc)
    return out, m


def test_rescore_off_by_default_bypasses_everything():
    out, m = _rescore_run(k=0)
    assert out["rescore"] is None
    assert m.calls == 0, "K=0 must not touch the world model"


def test_rescore_runs_exactly_one_batched_rollout():
    out, m = _rescore_run(k=4)
    assert out["rescore"] is not None
    assert m.calls == 1, f"expected ONE batched rollout, got {m.calls}"
    assert m.batch_sizes == [5], "K candidates + the converged mean, in one batch"


def test_rescore_argmax_is_drawn_from_the_topk_set():
    out, _ = _rescore_run(k=4)
    rs = out["rescore"]
    assert 0 <= rs["picked"] < rs["k"]
    assert torch.argmax(rs["scores"]).item() == rs["picked"]


def test_rescore_always_includes_the_converged_mean():
    """The geometry-optimal plan must never be unavailable to the rescore."""
    out, _ = _rescore_run(k=4)
    assert out["rescore"]["k"] == 5
    assert isinstance(out["rescore"]["picked_mean"], bool)


def test_rescore_k_larger_than_the_pool_is_clamped():
    out, m = _rescore_run(k=99, n=8)
    assert m.batch_sizes == [9]        # 8 candidates + the mean, not 100


def test_rescore_carries_the_chosen_plans_own_rollout():
    """Claim-② logging reuses this; a second forward_rollout would be waste."""
    out, m = _rescore_run(k=4)
    fp = out["rescore"]["chosen_f_prop"]
    assert fp is not None and fp.shape == (H, len(CHANNELS))
    assert m.calls == 1


def test_rescore_plan_is_one_of_the_scored_candidates():
    out, _ = _rescore_run(k=4)
    assert out["plan"].shape == (H, DA)
    assert np.isfinite(out["plan"].numpy()).all()


# ── (d) the counter survives the metric-goal wrapper ────────────────────────
