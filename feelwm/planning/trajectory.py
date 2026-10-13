"""trajectory — SE(2) integration of action sequences + NWM-style ADE/FDE.

The dataset's action is a **body-frame SE(2) pose delta** ``(Δx, Δy, Δyaw)`` per step
(ΔT = T_a⁻¹ · T_b; under multi-stride the stored 3-dim part is the *rate*
``pose_delta / stride`` and dim 3 carries Δt). Chaining the deltas from the query pose
reconstructs the metric trajectory each candidate would drive, which lets us report the
NWM / ReL-NWM planning metrics — **ADE/FDE between the selected plan's trajectory and
the expert's GT trajectory** — on top of our score-level ranking.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from feelwm.data.normalize import NormStats, denormalize_action


def actions_to_deltas(actions: Tensor, stats: NormStats) -> Tensor:
    """Normalized dataset actions ``[..., H, 3|4]`` → physical body-frame deltas ``[..., H, 3]``.

    The first 3 dims are denormalized with the action norm-stats. If a Δt column is
    present (multi-stride, ``d_action=4``) the denormalized rate (per 0.1 s anchor) is
    scaled by ``stride = Δt / 0.1`` to recover the full per-step delta.
    """
    norm3 = actions[..., :3]
    raw = torch.from_numpy(
        denormalize_action(norm3.detach().cpu().numpy().astype(np.float32), stats)
    ).to(actions.device)
    if actions.shape[-1] >= 4:
        stride = (actions[..., 3:4] / 0.1).to(raw.dtype)
        raw = raw * stride
    return raw


def integrate_se2(deltas: Tensor) -> Tensor:
    """Chain body-frame SE(2) deltas ``[..., H, 3]`` into world positions ``[..., H, 2]``.

    Starts at the origin with yaw 0 (the query pose frame — shared by every candidate,
    so relative geometry is exact). Step h: ``p ← p + R(yaw)·(Δx, Δy)``, ``yaw ← yaw + Δyaw``.
    """
    dx, dy, dyaw = deltas[..., 0], deltas[..., 1], deltas[..., 2]
    # yaw entering step h = sum of previous dyaws (yaw_0 = 0)
    yaw = torch.cumsum(dyaw, dim=-1) - dyaw
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    step_x = cos * dx - sin * dy
    step_y = sin * dx + cos * dy
    return torch.stack(
        [torch.cumsum(step_x, dim=-1), torch.cumsum(step_y, dim=-1)], dim=-1
    )


def ade_fde(traj: Tensor, ref: Tensor) -> tuple[Tensor, Tensor]:
    """Average / Final Displacement Error between trajectories ``[..., H, 2]``.

    The NWM-paper planning metrics: ADE = mean waypoint distance, FDE = final-waypoint
    distance, both in meters (broadcast over leading dims; ``ref`` is typically the
    expert trajectory).
    """
    d = (traj - ref).norm(dim=-1)  # [..., H]
    return d.mean(dim=-1), d[..., -1]


def ate_rte(traj: Tensor, ref: Tensor) -> tuple[Tensor, Tensor]:
    """Absolute / Relative Trajectory Error between trajectories ``[..., H, 2]``, in meters.

    The metrics the NWM paper reports (its Table 2 / appendix Table 7), so our rows can be
    read in the same units as its per-dataset columns. Computed to match what
    the official NWM `planning_eval.py` gets out of `evo` for a translation-only 2-D plan:

    ``ATE`` — `main_ape.ape(pose_relation=translation_part, align=False,
    correct_scale=False)`, whose `stats['rmse']` is the **RMS** over waypoints of the same
    per-waypoint distance ADE averages. So ATE and ADE differ ONLY in the aggregator:
    RMS >= mean always, with equality iff every waypoint error is equal, so ATE weights a
    trajectory that diverges late more heavily than ADE does.

    ``RTE`` — `main_rpe.rpe(..., delta=1, delta_unit=frames)`: RMS over the H-1 consecutive
    pose pairs of the error in the *step displacement*. It never accumulates, so it isolates
    per-step motion (speed and heading) from drift. The pair count is H-1, matching evo's
    treatment of a length-H trajectory; the anchor->first-waypoint step is not a pair
    because the anchor pose is not part of the stored trajectory.

    Reading them together is the point: for a straight plan whose only error is a constant
    speed offset, the per-waypoint error grows linearly and ATE/RTE is fixed by the horizon,
    whereas a non-accumulating error (heading wobble, lateral offset) lowers the ratio.
    """
    d = (traj - ref).norm(dim=-1)                                  # [..., H]
    ate = d.pow(2).mean(dim=-1).sqrt()
    step = (traj.diff(dim=-2) - ref.diff(dim=-2)).norm(dim=-1)     # [..., H-1]
    rte = step.pow(2).mean(dim=-1).sqrt()
    return ate, rte


def candidate_trajectories(actions: Tensor, stats: NormStats) -> Tensor:
    """Convenience: normalized candidate actions ``[N, H, d_action]`` → positions ``[N, H, 2]``."""
    return integrate_se2(actions_to_deltas(actions, stats))
