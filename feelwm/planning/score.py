"""score — the planning objective.

Fusion happens here, at the **score level**, not inside any head:

    score = z(goal_sim) − λ_trav · z(trav_risk) − λ_ride · z(ride_cost)

The three terms have incompatible native scales (goal_sim cosine ∈ [−1, 1],
trav_risk ∈ [0, 1], ride_cost in z-units), so **each term is z-score standardized
within the candidate pool** before weighting — the λ then become pure preference
weights. Code default ``λ_trav = λ_ride = 1``; ``λ_ride`` is an ablation knob (0 vs 1;
the shipped planning config sets ``lambda_ride: 0`` for wheeled models).

**Trav gate (optional, OFF by default):** the linear
sum lets goal_sim buy out a lethal step (horizon-mean dilutes a single
hazardous step, and pool-relative z has no absolute scale). With ``trav_gate_tau`` set, a
candidate whose per-step trav probability exceeds τ anywhere in the proprio horizon takes
a flat penalty ``trav_gate_penalty`` on top of the linear score:

    score' = score − M · 1[max_k p_k > τ]

The penalty form (not a hard veto) makes the gate a strict refinement of the linear
planner: survivors outrank vetoed candidates, both groups keep their linear ordering, and
if every candidate trips the gate the ranking degrades to exactly today's behavior — no
empty-pool special case. τ must be chosen from the head's own output distribution (train
positive base rate is ~0.6%, so hazard-time p sits at 0.05–0.22, not near 0.5); the
λ_ride axis stays linear over the full pool.

**Slip term (optional):** ``λ_slip`` adds a fourth linear term
``− λ_slip · z(slip_cost)``. Slip is a traction-safety signal, not comfort, so it is a
SEPARATE term rather than a ride channel: sim slip val R² ≈ +0.70 (the TartanDrive-era
exclusion rationale — structural val R² ≈ 0 — does not apply to sim), and its natural
aggregation is tail-shaped (max/exceed via ``RideCostConfig.agg``), not the ride mean.
Default ``λ_slip = 0`` — every existing score is unchanged.

**Visual goal weight (ablation knob):** ``lambda_gsim`` scales the
``z(goal_sim)`` term; it defaults to 1, so every existing score is
unchanged. ``lambda_gsim = 0`` removes the visual goal term entirely — the arm that
asks whether the WM's *visual* goal preference (not a metric goal term, which a wrapping
scorer outside this package may add) is what shifts the driven line at a choke.

**Proprio fusion gate (optional, OFF by default):** the trav gate sees only the visual
head, which is measurably blind to unlabeled hazard classes (e.g. bushes, banks).
With ``proprio_gate_tau`` set, a candidate is ALSO vetoed when its worst-step ride value
(``ride_step_values`` max within the gate window) exceeds τ_p — the veto becomes
``trav OR proprio`` (one flat penalty, not stacked). τ_p must be picked from logged
``ride_max`` distributions, same discipline as τ.

This module owns the scoring algebra and the two visual/binary terms; ``ride_cost`` (the
ride term) lives in :mod:`feelwm.planning.ride_cost`. Term vectors are ``[..., N]`` over the
candidate pool (last dim = the pool that standardization and ranking run over).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ScoreConfig:
    """Preference weights for :func:`combine_score`. Defaults are the reference weights.

    ``trav_gate_tau=None`` (default) disables the gate — the linear score is
    unchanged. ``trav_gate_penalty`` only needs to dominate the z-score span (~±4).
    """

    lambda_gsim: float = 1.0            # visual goal term; 1 = default
    lambda_trav: float = 1.0
    lambda_ride: float = 1.0
    lambda_slip: float = 0.0            # traction-safety term; 0 = default (off)
    trav_gate_tau: float | None = None
    trav_gate_penalty: float = 100.0    # shared by the trav and proprio vetoes
    proprio_gate_tau: float | None = None  # fusion veto on worst-step ride value


def goal_similarity(z_vis_pred_final: Tensor, z_vis_goal: Tensor) -> Tensor:
    """Per-patch cosine, averaged over patches.

    Args:
        z_vis_pred_final: predicted visual features at the visual horizon ``[..., P, D]``
            (e.g. ``forward_rollout(...)["z_vis_preds"][:, -1]``).
        z_vis_goal: the goal frame's DINOv2reg features ``[..., P, D]``
            (``z_vis_target[-1]``), broadcast over the candidate pool.

    Returns:
        ``[...]`` mean per-patch cosine similarity (higher = closer to goal).
    """
    if z_vis_pred_final.shape[-2:] != z_vis_goal.shape[-2:]:
        raise ValueError(
            f"patch/feature dims must match: {tuple(z_vis_pred_final.shape)} vs {tuple(z_vis_goal.shape)}"
        )
    cos = F.cosine_similarity(z_vis_pred_final, z_vis_goal, dim=-1)  # [..., P]
    return cos.mean(dim=-1)


def traversability_risk(c_preds: Tensor, horizon_mask: Tensor | None = None) -> Tensor:
    """Binary cost head probability averaged over the (masked) horizon.

    Args:
        c_preds: ``[..., H, 1]`` or ``[..., H]`` traversability probabilities
            (``forward_rollout(...)["c_preds"]``). ``None``-producing modes are the
            caller's responsibility to handle (trav head disabled → no term).
        horizon_mask: optional ``[..., H]`` (or ``[H]``) mask of horizon steps to
            include. ``None`` averages over all steps.

    Returns:
        ``[...]`` mean traversability risk (higher = more hazardous).
    """
    c = c_preds.squeeze(-1) if c_preds.shape[-1] == 1 and c_preds.ndim >= 2 else c_preds
    if horizon_mask is None:
        return c.mean(dim=-1)
    m = horizon_mask.to(c.dtype)
    if m.ndim == 1:
        m = m.expand_as(c)
    denom = m.sum(dim=-1).clamp_min(torch.finfo(c.dtype).eps)
    return (c * m).sum(dim=-1) / denom


def traversability_gate_stat(c_preds: Tensor, horizon_mask: Tensor | None = None) -> Tensor:
    """Worst-step trav probability over the (masked) horizon — the gate statistic.

    ``max_k`` instead of the mean so a single lethal step cannot be diluted; the same
    ``horizon_mask`` as :func:`traversability_risk` (cum Δt ≤ proprio horizon) keeps the
    max off the far free-running steps where rollout drift inflates it.

    Args:
        c_preds: ``[..., H, 1]`` or ``[..., H]`` traversability probabilities.
        horizon_mask: optional ``[..., H]`` (or ``[H]``) mask. An all-false mask falls
            back to the full horizon rather than returning −inf.

    Returns:
        ``[...]`` max trav probability (higher = worse single step).
    """
    c = c_preds.squeeze(-1) if c_preds.shape[-1] == 1 and c_preds.ndim >= 2 else c_preds
    if horizon_mask is None:
        return c.max(dim=-1).values
    m = horizon_mask.to(torch.bool)
    if m.ndim == 1:
        m = m.expand_as(c)
    degenerate = ~m.any(dim=-1, keepdim=True)
    m = m | degenerate
    return c.masked_fill(~m, float("-inf")).max(dim=-1).values


def standardize(x: Tensor, dim: int = -1) -> Tensor:
    """Z-score standardize a term within the candidate pool (population std).

    A degenerate pool (≤1 candidate or zero variance) yields all-zeros — a neutral
    term that contributes nothing rather than NaN.
    """
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, unbiased=False, keepdim=True)
    return (x - mean) / std.clamp_min(torch.finfo(x.dtype).eps)


def standardize_against(x: Tensor, ref: Tensor, dim: int = -1) -> Tensor:
    """Z-score ``x`` using ANOTHER pool's statistics (``ref``).

    The executed plan is the CEM elite MEAN, which is never itself a pool member, so it
    cannot be standardized in isolation — a one-element pool degenerates to 0 (see
    :func:`standardize`). Scoring it against the pool it was refit from recovers the
    z-values it *would* have had inside that pool, which is what makes a per-replan term
    breakdown comparable across replans and across arms.
    """
    mean = ref.mean(dim=dim, keepdim=True)
    std = ref.std(dim=dim, unbiased=False, keepdim=True)
    return (x - mean) / std.clamp_min(torch.finfo(ref.dtype).eps)


# term → (short name used in logs, sign in the score, ScoreConfig weight attribute)
SCORE_TERMS = {
    "goal_sim": ("gsim", +1.0, "lambda_gsim"),
    "trav_risk": ("trav", -1.0, "lambda_trav"),
    "ride_cost": ("ride", -1.0, "lambda_ride"),
    "slip_cost": ("slip", -1.0, "lambda_slip"),
}


def score_breakdown(
    chosen: dict[str, Tensor | float | None],
    pool: dict[str, Tensor | None],
    cfg: ScoreConfig | None = None,
) -> dict[str, float]:
    """Per-term z values and λ-weighted contributions of ONE candidate against a pool.

    Answers "which term actually decided this plan?" — with the terms on incompatible
    native scales, only the standardized, λ-weighted contributions are comparable, and
    reading them off the score requires the pool the candidate was ranked in.

    Args:
        chosen: the candidate's RAW term values (scalars or 0-/1-element tensors), keyed
            as in :data:`SCORE_TERMS`. Missing/None terms are skipped.
        pool: the same terms as ``[N]`` vectors over the candidate pool.
        cfg: the :class:`ScoreConfig` that scored the pool (supplies the λ).

    Returns:
        ``{"z_<short>": …, "w_<short>": …, "w_sum": …}`` — ``w`` are signed λ-weighted
        contributions, ``w_sum`` their total. The gate penalty is NOT included: it is a
        flat veto, logged separately (``trav_max`` / veto fraction).
    """
    cfg = cfg or ScoreConfig()
    out: dict[str, float] = {}
    total = 0.0
    for name, (short, sign, weight_attr) in SCORE_TERMS.items():
        c, p = chosen.get(name), pool.get(name)
        if c is None or p is None:
            continue
        p = p.detach().float().flatten()
        if p.numel() == 0:
            continue
        c = torch.as_tensor(c, device=p.device).detach().float().flatten()[0]
        z = float(standardize_against(c, p))
        w = sign * float(getattr(cfg, weight_attr)) * z
        out[f"z_{short}"], out[f"w_{short}"] = z, w
        total += w
    out["w_sum"] = total
    return out


def combine_score(
    goal_sim: Tensor,
    trav_risk: Tensor | None,
    ride_cost: Tensor | None,
    cfg: ScoreConfig | None = None,
    trav_gate_stat: Tensor | None = None,
    slip_cost: Tensor | None = None,
    proprio_gate_stat: Tensor | None = None,
) -> Tensor:
    """Combine standardized terms into the planning score over the candidate pool.

    ``score = λ_gsim·z(goal_sim) − λ_trav·z(trav_risk) − λ_ride·z(ride_cost) −
    λ_slip·z(slip_cost)`` (``λ_gsim = 1`` = default; 0 drops the visual goal
    term). Terms that are ``None`` (head disabled, or ablated to 0)
    drop out — passing ``ride_cost=None`` (or ``λ_ride=0``) is exactly the
    ``λ_ride=0`` goal-only ablation; ``λ_slip`` defaults to 0.

    With ``cfg.trav_gate_tau`` set and ``trav_gate_stat`` given (see
    :func:`traversability_gate_stat`), candidates with ``gate_stat > τ`` take the flat
    ``trav_gate_penalty`` — applied AFTER standardization, so the full-pool z terms (and
    the λ_ride axis) are untouched. With ``cfg.proprio_gate_tau`` also set, the veto
    is the OR over the two stats (a candidate flagged by both still takes ONE penalty —
    survivors must clear both heads, and the all-vetoed pool still degrades to the
    linear ranking).

    All term vectors share shape ``[..., N]`` (last dim = the candidate pool). Returns
    ``[..., N]`` scores; higher = better plan.
    """
    cfg = cfg or ScoreConfig()
    score = standardize(goal_sim)
    if cfg.lambda_gsim != 1.0:
        score = cfg.lambda_gsim * score
    if trav_risk is not None and cfg.lambda_trav != 0:
        score = score - cfg.lambda_trav * standardize(trav_risk)
    if ride_cost is not None and cfg.lambda_ride != 0:
        score = score - cfg.lambda_ride * standardize(ride_cost)
    if slip_cost is not None and cfg.lambda_slip != 0:
        score = score - cfg.lambda_slip * standardize(slip_cost)
    vetoed = None
    if cfg.trav_gate_tau is not None and trav_gate_stat is not None:
        vetoed = trav_gate_stat > cfg.trav_gate_tau
    if cfg.proprio_gate_tau is not None and proprio_gate_stat is not None:
        pv = proprio_gate_stat > cfg.proprio_gate_tau
        vetoed = pv if vetoed is None else (vetoed | pv)
    if vetoed is not None:
        score = score - cfg.trav_gate_penalty * vetoed.to(score.dtype)
    return score
