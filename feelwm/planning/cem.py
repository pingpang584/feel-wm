"""cem — sampling-based planner over the terrain world model.

Two entry points share **one** score function (:class:`WorldModelScorer`):
  - :func:`rank_candidates` — single-shot sample-and-rank (the offline ranking metric).
  - :func:`cem_optimize` — CEM (Gaussian over action sequences, iterative elite refit).

The scorer rolls candidate action sequences from a fixed query context through
``TerrainWorldModel.forward_rollout`` (feedback rollout mode is baked into the trained
checkpoint — no oracle flags), then fuses goal_sim / trav_risk / ride_cost at the score
level. Multi-horizon: goal_sim is read at the visual horizon
(the rollout's final step), while ride/trav are confined to the ``proprio_horizon_mask``
(cum_dt ≤ 1 s), here derived directly from each candidate's Δt column so library
candidates with different stride schedules get the correct cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from feelwm.planning.candidates import POSE_DELTA_DIMS
from feelwm.planning.ride_cost import RideCostConfig, ride_cost, ride_step_values
from feelwm.planning.score import (
    ScoreConfig,
    combine_score,
    goal_similarity,
    traversability_gate_stat,
    traversability_risk,
)


# Raw per-candidate term vectors kept from the LAST CEM iteration (cheap: N floats
# each). The executed plan is the elite mean, outside the pool, so its score breakdown
# can only be read against these — see score.standardize_against. `goal_pos_dist*` come
# from an optional wrapping metric-goal scorer (not part of this package) when present.
POOL_TERM_KEYS = ("goal_sim", "trav_risk", "ride_cost", "slip_cost",
                  "goal_pos_dist", "goal_pos_dist_wp")


def horizon_mask_from_actions(actions: Tensor, proprio_horizon_s: float = 1.0) -> Tensor | None:
    """Proprio horizon mask ``[N, H]`` from the Δt column of multi-stride actions.

    The Δt column (action dim 3) is raw seconds (``dataset._action_row``); cumulative Δt
    ≤ ``proprio_horizon_s`` defines where proprio/trav are reliable. Returns ``None`` for
    legacy 3-dim actions (no Δt) — the caller then averages over the full horizon.
    """
    if actions.shape[-1] < 4:
        return None
    dt = actions[..., 3]
    cum_dt = torch.cumsum(dt, dim=-1)
    return cum_dt <= proprio_horizon_s + 1e-9


@dataclass
class CEMConfig:
    """Knobs for :func:`cem_optimize` (defaults are the planning-config defaults)."""

    n_candidates: int = 256
    iterations: int = 4
    elite_frac: float = 0.1
    init_sigma: float = 0.5      # initial exploration std (normalized action units)
    epsilon: float = 0.05        # ε: exploration std floor (min_sigma)
    dims: tuple[int, ...] = POSE_DELTA_DIMS  # action dims CEM moves (Δt stays fixed)
    keep_mean_elite: bool = True  # carry the current mean as a candidate (elitism)
    # World-model rescore of the top K candidates (0 = off). Only meaningful together
    # with `rescore_scorer` in cem_optimize; see that function for the stage choice.
    rescore_topk: int = 0
    # Free-CEM knobs (None = legacy behavior — seeded refinement with a scalar sigma and
    # no bounds). Set together to search the TRAINING action distribution instead of a
    # neighbourhood of a seed: `init_std` replaces `init_sigma` per action dim, and
    # clipping keeps every sampled candidate inside the range the WM was trained on
    # (actions outside the training distribution make the rollout, not the ranking, the
    # thing being measured). Shapes broadcast to [H, len(dims)].
    init_std: Tensor | None = None
    clip_min: Tensor | None = None
    clip_max: Tensor | None = None
    gate_tau: float | None = None  # veto-guard threshold on the FINAL mean (see below).
    gate_tau_p: float | None = None  # proprio fusion-guard twin of gate_tau (ride_max)
    # The executed plan is the elite MEAN, which is never itself pool-ranked — the mean
    # of individually-cold elites can roll out HOT (observed: a chosen mean scored 0.79 on
    # the gate statistic while the pool minimum was 0.000). With gate_tau set, a vetoed
    # final mean is replaced by the best gate-surviving CANDIDATE from the last iteration,
    # if one exists.
    # ── cascaded pruning (optional; OFF by default) ─────────────────────────────────
    # Roll EVERY candidate only `cascade_h_pre` steps, keep the `cascade_k` best on that
    # prefix, and continue only those to the full horizon. On a small embedded GPU,
    # profiling measured NO batching economy — candidate-steps convert 1:1 into wall
    # time — so the cost goes n·H → n·H_pre + K·H per iteration
    # (64×8 = 512 → 64×3 + 16×8 = 320, 1.6×).
    # It is a SPEED lever with a quality cost: the prefix ranking is a different score
    # (goal_sim read at step H_pre, ride/trav windowed by the prefix's cum_dt), so a
    # candidate whose payoff is in the tail can be pruned before it is ever seen.
    # Either field None — or a degenerate setting (K ≥ n_candidates, H_pre ≥ H) — is OFF
    # and leaves the search bit-identical to the uncascaded one.
    cascade_k: int | None = None
    cascade_h_pre: int | None = None


class WorldModelScorer:
    """Binds a checkpoint + a single query context into a callable score function.

    ``__call__(actions[N, H, d_action]) -> (scores[N], aux)``. Runs under ``no_grad``.
    A degenerate term (trav/proprio head disabled) drops out of the score.
    """

    def __init__(
        self,
        model,
        initial_batch: dict[str, Tensor],
        z_vis_goal: Tensor,
        channel_names: list[str],
        score_cfg: ScoreConfig | None = None,
        ride_cfg: RideCostConfig | None = None,
        proprio_horizon_s: float = 1.0,
        gate_horizon_s: float | None = None,
        slip_cfg: RideCostConfig | None = None,
    ) -> None:
        self.model = model
        self.initial_batch = initial_batch  # tensors with batch dim 1 (one query)
        self.z_vis_goal = z_vis_goal        # [P, D] or [1, P, D]
        self.channel_names = channel_names
        self.score_cfg = score_cfg or ScoreConfig()
        self.ride_cfg = ride_cfg or RideCostConfig()
        # Separate traction-safety term (slip channel, tail aggregation). None = no slip
        # aux/term. When given, slip_cost is always computed for logging/validity even
        # at λ_slip=0 (combine_score only prices it when λ_slip ≠ 0).
        self.slip_cfg = slip_cfg
        self.proprio_horizon_s = proprio_horizon_s
        # gate the COMMITMENT window, cost the lookahead: the mean trav/ride terms guide
        # over the full proprio horizon, but a veto should only bind on steps the robot
        # will actually traverse before replanning can correct (far-tail hazards are
        # routinely present in QUIET successful runs). None = same mask as trav/ride.
        self.gate_horizon_s = gate_horizon_s
        self.device = z_vis_goal.device
        # How many times the candidate POOL was rolled through the world model. The skip
        # below is only trustworthy if it is observable, so callers can log it.
        self.n_pool_rollouts = 0

    @property
    def needs_pool_rollout(self) -> bool:
        """False when nothing the rollout produces can change the ranking.

        Every term this scorer contributes — goal_sim, trav, ride, slip, and both gate
        vetoes — is derived from ``forward_rollout``. With all four λ at 0 and neither
        gate armed, ``combine_score`` returns a vector of zeros whatever the rollout says
        (see `score.combine_score`: the base term is scaled by λ_gsim, and each other
        term is added only when its own λ is non-zero). Ranking then rests entirely on
        any model-free term a wrapping scorer adds outside this class — so rolling the
        pool is pure cost and can be skipped. It never fires under the published
        configs: those keep λ_gsim=1.
        """
        c = self.score_cfg
        return bool(
            c.lambda_gsim != 0.0
            or c.lambda_trav != 0.0
            or c.lambda_ride != 0.0
            or c.lambda_slip != 0.0
            or c.trav_gate_tau is not None
            or c.proprio_gate_tau is not None
        )

    def _expand_context(self, n: int) -> dict[str, Tensor]:
        out = {}
        for k in ("z_vis_context", "proprio_window_context", "action_context"):
            v = self.initial_batch.get(k)
            if v is not None:
                out[k] = v.expand(n, *v.shape[1:]).contiguous() if v.shape[0] == 1 else v
        return out

    @torch.no_grad()
    def __call__(self, actions: Tensor) -> tuple[Tensor, dict]:
        n = actions.shape[0]
        actions = actions.to(self.device)
        if not self.needs_pool_rollout:
            # Skip the pool rollout entirely (see `needs_pool_rollout`). The zeros are
            # what combine_score would have returned anyway; aux keeps every key so
            # consumers that read it unconditionally are unaffected, with None marking
            # "not computed" exactly as a disabled head already does.
            return (
                torch.zeros(n, device=self.device),
                {"goal_sim": None, "trav_risk": None, "trav_max": None,
                 "ride_cost": None, "ride_max": None, "slip_cost": None,
                 "horizon_mask": horizon_mask_from_actions(actions, self.proprio_horizon_s),
                 "rollout": None, "pool_rollout_skipped": True},
            )
        self.n_pool_rollouts += 1
        rollout = self.model.forward_rollout(self._expand_context(n), actions)

        z_final = rollout["z_vis_preds"][:, -1]  # [N, P, D]
        goal = self.z_vis_goal
        if goal.ndim == 2:
            goal = goal[None]
        goal = goal.expand(n, *goal.shape[-2:])
        gsim = goal_similarity(z_final, goal)

        hmask = horizon_mask_from_actions(actions, self.proprio_horizon_s)
        trav = (
            traversability_risk(rollout["c_preds"], hmask)
            if rollout.get("c_preds") is not None
            else None
        )
        gmask = (
            horizon_mask_from_actions(actions, self.gate_horizon_s)
            if self.gate_horizon_s is not None
            else hmask
        )
        trav_max = (
            traversability_gate_stat(rollout["c_preds"], gmask)
            if rollout.get("c_preds") is not None
            else None
        )
        f_prop = rollout.get("f_prop_preds")
        ride = (
            ride_cost(f_prop, self.channel_names, hmask, self.ride_cfg)
            if f_prop is not None
            else None
        )
        # Worst-step ride value within the GATE window — the proprio fusion-gate stat
        # (score.py). Computed whenever the head exists so its distribution is logged
        # even in gate-off runs (τ_p selection needs it, same discipline as trav_max).
        ride_max = (
            traversability_gate_stat(
                ride_step_values(f_prop, self.channel_names, self.ride_cfg), gmask
            )
            if f_prop is not None
            else None
        )
        slip = (
            ride_cost(f_prop, self.channel_names, hmask, self.slip_cfg)
            if f_prop is not None and self.slip_cfg is not None
            else None
        )
        scores = combine_score(gsim, trav, ride, self.score_cfg, trav_gate_stat=trav_max,
                               slip_cost=slip, proprio_gate_stat=ride_max)
        aux = {
            "goal_sim": gsim,
            "trav_risk": trav,
            "trav_max": trav_max,
            "ride_cost": ride,
            "ride_max": ride_max,
            "slip_cost": slip,
            "horizon_mask": hmask,
            "rollout": rollout,
            "pool_rollout_skipped": False,
        }
        return scores, aux


def rank_candidates(scorer: WorldModelScorer, actions: Tensor) -> dict:
    """Single-shot sample-and-rank. Returns scores, descending order, and per-term aux."""
    scores, aux = scorer(actions)
    order = torch.argsort(scores, descending=True)
    return {"scores": scores, "order": order, "best": int(order[0]), **aux}


def expert_rank_percentile(scores: Tensor, expert_index: int) -> float:
    """Fraction of the pool the expert scores ≥ (1.0 = expert is the top-ranked plan)."""
    n = scores.shape[0]
    if n <= 1:
        return 1.0
    beaten = (scores <= scores[expert_index]).sum().item() - 1  # exclude self
    return beaten / (n - 1)


def cem_optimize(
    scorer: WorldModelScorer,
    init_actions: Tensor,
    cfg: CEMConfig | None = None,
    generator: torch.Generator | None = None,
    trace: bool = False,
    rescore_scorer: "WorldModelScorer | None" = None,
) -> dict:
    """CEM over the pose-delta dims of an action sequence; Δt held fixed from ``init_actions``.

    Args:
        scorer: the shared :class:`WorldModelScorer`.
        init_actions: ``[H, d_action]`` seed (e.g. the expert sequence) — its Δt column
            defines the (fixed) stride schedule for every candidate.
        cfg: :class:`CEMConfig`. With ``cascade_k``/``cascade_h_pre`` set, each iteration
            prescreens the pool on a horizon prefix and only the survivors are rolled to
            full ``H`` (speed lever — see the config's docstring).
        generator: optional torch RNG for reproducible sampling.
        trace: also return the LAST iteration's full candidate pool (``pool`` key:
            actions/scores + per-term aux minus the heavy ``rollout`` tensors, plus the
            per-candidate proprio rollout ``f_prop_preds`` ``[N, H, C]`` when the head
            exists) — the converged distribution the final plan was refit from, for
            visualization and offline score-counterfactuals.

    Returns the refined ``plan`` ``[H, d_action]``, its ``score``, the per-iteration
    elite-score ``history``, the scorer ``aux`` for the final plan, and ``pool_terms``
    (:data:`POOL_TERM_KEYS` vectors from the last iteration — the reference pool for a
    per-term score breakdown of the executed plan).
    """
    cfg = cfg or CEMConfig()
    H, da = init_actions.shape
    device = scorer.device
    dims = list(cfg.dims)
    mean = init_actions.to(device).clone()
    std = (cfg.init_std.to(device).expand(H, len(dims)).clone()
           if cfg.init_std is not None
           else torch.full((H, len(dims)), cfg.init_sigma, device=device))
    lo = cfg.clip_min.to(device).expand(H, len(dims)) if cfg.clip_min is not None else None
    hi = cfg.clip_max.to(device).expand(H, len(dims)) if cfg.clip_max is not None else None

    def _clip(a: Tensor) -> Tensor:
        if lo is None and hi is None:
            return a
        a[..., dims] = a[..., dims].clamp(
            min=lo if lo is None else lo.expand_as(a[..., dims]),
            max=hi if hi is None else hi.expand_as(a[..., dims]),
        )
        return a

    mean = _clip(mean)
    elite_k = max(1, int(round(cfg.elite_frac * cfg.n_candidates)))
    # Cascade is ON only when it actually prunes something in BOTH axes; a degenerate
    # request (K ≥ n, H_pre ≥ H) silently degrades to the plain full-horizon search
    # instead of paying for a prescreen that keeps everyone.
    cascade = (cfg.cascade_k is not None and cfg.cascade_h_pre is not None
               and 0 < int(cfg.cascade_k) < cfg.n_candidates
               and 0 < int(cfg.cascade_h_pre) < H)
    history: list[float] = []
    first_pool = first_scores = None
    pool = None
    pool_trav_max = None
    pool_ride_max = None
    pool_terms: dict[str, Tensor] = {}
    best_survivor = None
    # Fusion veto-guard: a survivor must clear BOTH gates (trav τ and the proprio τ_p)
    # — mirrors the OR veto in combine_score. τ_p rides on CEMConfig like gate_tau
    # (the scorer may be wrapped by a caller, so no attribute sniffing).
    tau_p = cfg.gate_tau_p

    for it in range(cfg.iterations):
        n = cfg.n_candidates
        noise = torch.empty(n, H, len(dims), device=device)
        noise.normal_(generator=generator)
        cand = mean[None].expand(n, H, da).clone()
        cand[..., dims] += noise * std[None]
        cand = _clip(cand)
        if cfg.keep_mean_elite:
            cand[0] = mean  # elitism: keep the current mean unperturbed
        if cascade:
            # Prescreen on the first `cascade_h_pre` steps, then drop the rest. The
            # elite mean (index 0) is exempt: elitism is what makes the CEM mean
            # monotone across iterations, and it must not be prunable by a partial
            # score. Everything downstream — elites, pool_terms, the gate survivor,
            # the trace pool — sees the SURVIVORS only; that is the pool the plan was
            # actually refit from, so the logged distributions stay honest.
            pre_scores, _ = scorer(cand[:, :int(cfg.cascade_h_pre)].contiguous())
            keep = torch.topk(pre_scores, int(cfg.cascade_k)).indices
            if cfg.keep_mean_elite and not bool((keep == 0).any()):
                keep = torch.cat([keep.new_zeros(1), keep[:-1]])
            cand = cand[keep].contiguous()
        scores, it_aux = scorer(cand)
        if it == 0:
            # The rescore set is drawn from HERE, not from the converged pool. Iteration 0
            # is sampled at init_sigma around the seed, so it is the widest spread of
            # genuinely different trajectories the run ever holds; by the last iteration
            # the elites have collapsed onto one another and K rollouts would price K
            # near-duplicates. Geometry has already ranked these, so the top-K are also
            # goal-plausible — the rescore spends the world model on options that differ
            # in shape AND are worth reaching.
            first_pool, first_scores = cand.detach(), scores.detach()
        if it == cfg.iterations - 1:
            for key in POOL_TERM_KEYS:
                v = it_aux.get(key)
                if torch.is_tensor(v):
                    pool_terms[key] = v.detach()
            tm = it_aux.get("trav_max")
            rm = it_aux.get("ride_max")
            pool_trav_max = tm.detach() if tm is not None else None
            pool_ride_max = rm.detach() if rm is not None else None
            if (cfg.gate_tau is not None and tm is not None) or (tau_p is not None and rm is not None):
                ok = torch.ones_like(scores, dtype=torch.bool)
                if cfg.gate_tau is not None and tm is not None:
                    ok &= tm <= cfg.gate_tau
                if tau_p is not None and rm is not None:
                    ok &= rm <= tau_p
                if ok.any():
                    masked = torch.where(ok, scores, torch.full_like(scores, float("-inf")))
                    best_survivor = cand[int(masked.argmax())].clone()
        if trace and it == cfg.iterations - 1:
            pool = {"actions": cand.detach(), "scores": scores.detach()}
            pool.update({k: v.detach() for k, v in it_aux.items()
                         if k != "rollout" and torch.is_tensor(v)})
            # The per-candidate proprio ROLLOUT [N, H, C] — the only pool quantity that
            # lives inside the (dropped) rollout dict. Offline score-counterfactuals over
            # dumped pools need per-channel, per-step values: the aggregated ride/ride_max
            # terms above cannot express a per-channel gate or a re-windowed one. Kept
            # under `trace` only, so nothing changes for a normal (non-dumping) run.
            fp = (it_aux.get("rollout") or {}).get("f_prop_preds")
            if fp is not None:
                pool["f_prop_preds"] = fp.detach()
        elite_idx = torch.topk(scores, min(elite_k, scores.shape[0])).indices
        elite = cand[elite_idx]
        mean = _clip(elite.mean(dim=0))
        std = elite[..., dims].std(dim=0, unbiased=False).clamp_min(cfg.epsilon)
        history.append(float(scores[elite_idx].mean()))

    rescore = None
    k = int(cfg.rescore_topk)
    if k > 0 and rescore_scorer is not None and first_pool is not None:
        # Geometry ranked the pool; the world model now gets a say over a SHORT LIST.
        # The converged mean is always in the set, so the geometry-optimal plan is never
        # unavailable — the rescore can only replace it when the full score (geometry +
        # ride) prefers something else. ONE batched rollout of K+1, never K separate ones.
        idx = torch.topk(first_scores, min(k, first_scores.shape[0])).indices
        cand_set = torch.cat([first_pool[idx], mean[None]], dim=0)
        rs, r_aux = rescore_scorer(cand_set)
        pick = int(rs.argmax())
        rescore = {"k": int(cand_set.shape[0]), "picked": pick,
                   "picked_mean": bool(pick == cand_set.shape[0] - 1),
                   "scores": rs.detach(),
                   "ride_cost": (r_aux.get("ride_cost").detach()
                                 if torch.is_tensor(r_aux.get("ride_cost")) else None),
                   "f_prop_preds": ((r_aux.get("rollout") or {}).get("f_prop_preds",
                                                                     None)),
                   "chosen_f_prop": None}
        fp = rescore["f_prop_preds"]
        if fp is not None:
            rescore["f_prop_preds"] = fp.detach()
            rescore["chosen_f_prop"] = fp[pick].detach()
        mean = cand_set[pick].clone()

    final_scores, aux = scorer(mean[None])
    guard_fired = False
    mean_hot = (cfg.gate_tau is not None and aux.get("trav_max") is not None
                and float(aux["trav_max"][0]) > cfg.gate_tau)
    mean_hot = mean_hot or (tau_p is not None and aux.get("ride_max") is not None
                            and float(aux["ride_max"][0]) > tau_p)
    if best_survivor is not None and mean_hot:
        mean = best_survivor  # veto-guard: never execute a gate-rejected mean when a
        guard_fired = True    # surviving candidate exists (rescore for honest aux/logs)
        final_scores, aux = scorer(mean[None])
    # last-iteration pool worst-step trav/ride — the planning-time distributions (gate
    # τ/τ_p selection + "quiet-time veto rate" logging) at zero extra rollout cost.
    out = {"plan": mean, "score": float(final_scores[0]), "history": history, "aux": aux,
           "pool_trav_max": pool_trav_max, "pool_ride_max": pool_ride_max,
           "pool_terms": pool_terms, "guard_fired": guard_fired,
           # None unless rescore_topk fired; carries the chosen candidate's own proprio
           # rollout so claim-② logging never needs a SECOND forward_rollout.
           "rescore": rescore,
           "n_pool_rollouts": getattr(scorer, "n_pool_rollouts", None)}
    if trace:
        out["pool"] = pool
    return out
