"""candidates — assemble candidate action sequences for offline trajectory ranking.

Protocol: the candidate
distribution is **a library of real trajectory ``action_sequence``s drawn from the
dataset pool, with the GT expert sequence always injected**. Real candidates are
kinematically valid by construction and each carries its own GT proprio
(``f_prop_target``) — the *actually-experienced* ride, which is the ground truth for
the ride-cost validity test. Light Gaussian perturbations are a secondary, optional source.

A candidate action sequence under multi-stride is ``[dx, dy, dyaw (normalized), Δt]``
(``d_action = 4``; ``include_dt=True``). The Δt column carries the per-step stride and
must **not** be perturbed — only the pose-delta dims ``[0:3]`` are optimization targets.
For the legacy ``include_dt=False`` path actions are 3-dim and all dims are pose-delta.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Pose-delta action dims that CEM/perturbation may move (Δt at index 3 stays fixed).
POSE_DELTA_DIMS = (0, 1, 2)


@dataclass
class CandidateBatch:
    """A pool of ``N`` candidate action sequences (+ optional aligned GT metadata).

    All tensors share the planning horizon ``H``. ``actions`` is what the planner rolls
    out; the remaining fields are the candidates' own ground truth (populated when the
    pool is built from real dataset samples), used by the ride-cost validity test.
    """

    actions: Tensor                  # [N, H, d_action]
    f_prop_gt: Tensor | None = None  # [N, H, C]  each candidate's experienced proprio (GT)
    valid: Tensor | None = None      # [N, H] bool  GT valid mask
    horizon_mask: Tensor | None = None  # [N, H] bool  proprio horizon (cum_dt ≤ 1s)
    z_vis_goal: Tensor | None = None    # [N, P, D]  each candidate's own goal frame
    source_index: list[int] | None = None  # dataset index per row (provenance)
    expert_index: int | None = None        # row of the injected expert, if any

    def __len__(self) -> int:
        return self.actions.shape[0]

    def to(self, device: torch.device | str) -> "CandidateBatch":
        move = lambda t: t.to(device) if isinstance(t, Tensor) else t
        return CandidateBatch(
            actions=move(self.actions),
            f_prop_gt=move(self.f_prop_gt),
            valid=move(self.valid),
            horizon_mask=move(self.horizon_mask),
            z_vis_goal=move(self.z_vis_goal),
            source_index=self.source_index,
            expert_index=self.expert_index,
        )


def _truncate(t: Tensor | None, horizon: int) -> Tensor | None:
    return t[:horizon] if t is not None else None


def stack_samples(
    samples: list[dict[str, Tensor]],
    horizon: int,
    source_index: list[int] | None = None,
) -> CandidateBatch:
    """Stack dataset samples into a :class:`CandidateBatch`, truncated to ``horizon`` steps.

    Each sample is a dataset ``__getitem__`` dict (``action_sequence`` ``[H_i, d_action]``,
    ``f_prop_target`` ``[H_i, C]``, ``valid_target_mask`` ``[H_i]``, optionally
    ``proprio_horizon_mask`` and ``z_vis_target`` ``[H_i, P, D]``). Samples with fewer
    than ``horizon`` steps are skipped (a real candidate must span the full planning
    horizon). Raises if none qualify.
    """
    keep, kept_src = [], []
    src = source_index if source_index is not None else [None] * len(samples)
    for s, idx in zip(samples, src):
        if s["action_sequence"].shape[0] >= horizon:
            keep.append(s)
            kept_src.append(idx)
    if not keep:
        raise ValueError(f"no sample reaches the planning horizon H={horizon}")

    actions = torch.stack([_truncate(s["action_sequence"], horizon) for s in keep])
    f_prop_gt = torch.stack([_truncate(s["f_prop_target"], horizon) for s in keep])
    valid = torch.stack([_truncate(s["valid_target_mask"], horizon) for s in keep])
    hmask = (
        torch.stack([_truncate(s["proprio_horizon_mask"], horizon) for s in keep])
        if "proprio_horizon_mask" in keep[0]
        else None
    )
    # Goal = each candidate's own final visual frame (z_vis_target[-1] at the horizon).
    z_goal = (
        torch.stack([s["z_vis_target"][horizon - 1] for s in keep])
        if "z_vis_target" in keep[0]
        else None
    )
    return CandidateBatch(
        actions=actions,
        f_prop_gt=f_prop_gt,
        valid=valid,
        horizon_mask=hmask,
        z_vis_goal=z_goal,
        source_index=None if source_index is None else kept_src,
    )


def inject_expert(
    library: CandidateBatch, expert_actions: Tensor, at: int = 0
) -> CandidateBatch:
    """Insert the query's own expert ``action_sequence`` ``[H, d_action]`` into the pool.

    The expert row carries no aligned GT metadata here (its experienced ride is the
    query's own and is tracked separately by the caller); metadata rows are padded with
    NaN/False so tensor shapes stay aligned. Returns a new batch with ``expert_index`` set.
    """
    H, da = library.actions.shape[1:]
    if expert_actions.shape != (H, da):
        raise ValueError(
            f"expert_actions must be [H={H}, d_action={da}], got {tuple(expert_actions.shape)}"
        )
    exp = expert_actions.to(library.actions)[None]  # [1, H, da]

    def _insert(t: Tensor | None, filler) -> Tensor | None:
        if t is None:
            return None
        pad = torch.full((1, *t.shape[1:]), filler, dtype=t.dtype, device=t.device)
        return torch.cat([t[:at], pad, t[at:]], dim=0)

    actions = torch.cat([library.actions[:at], exp, library.actions[at:]], dim=0)
    src = None
    if library.source_index is not None:
        src = library.source_index[:at] + [None] + library.source_index[at:]
    return CandidateBatch(
        actions=actions,
        f_prop_gt=_insert(library.f_prop_gt, float("nan")),
        valid=_insert(library.valid, False),
        horizon_mask=_insert(library.horizon_mask, False),
        z_vis_goal=_insert(library.z_vis_goal, float("nan")),
        source_index=src,
        expert_index=at,
    )


def perturb_actions(
    base: Tensor,
    n: int,
    sigma: float,
    generator: torch.Generator | None = None,
    dims: tuple[int, ...] = POSE_DELTA_DIMS,
) -> Tensor:
    """Generate ``n`` Gaussian perturbations of a base action sequence ``[H, d_action]``.

    Noise (std ``sigma``, in normalized action units) is added only to ``dims`` — the
    pose-delta channels — leaving Δt untouched so every candidate keeps a valid stride
    schedule. Returns ``[n, H, d_action]`` (the base itself is not included).
    """
    H, da = base.shape
    out = base[None].expand(n, H, da).clone()
    noise = torch.zeros(n, H, len(dims), device=base.device, dtype=base.dtype)
    noise.normal_(mean=0.0, std=sigma, generator=generator)
    out[..., list(dims)] += noise
    return out
