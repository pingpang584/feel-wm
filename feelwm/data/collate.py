"""Collate functions for TartanDriveDataset.

default_collate handles homogeneous batches (fixed H). variable_h_collate
pads along the H dimension for batches with different rollout horizons —
needed when H varies per sample (e.g. sequence-tail clips).
"""
from __future__ import annotations

import torch
from torch import Tensor


# Context-side keys are stacked as-is; everything else is treated as an H-indexed
# target (dim 0 = rollout step) and padded in variable_h_collate.
_FIXED_KEYS = {
    "z_vis_context",
    "proprio_window_context",
    "action_context",
    "action_sequence",
    "current_feature",
}


def collate_fn(batch: list[dict]) -> dict[str, Tensor]:
    """Standard collate for fixed-H batches. Stacks every field along dim 0.

    Key-driven so optional fields (e.g. ``proprio_horizon_mask``, present only when
    ``include_dt=True``) are handled without a hardcoded key list.
    """
    return {key: torch.stack([sample[key] for sample in batch], dim=0) for key in batch[0]}


def variable_h_collate(batch: list[dict]) -> dict[str, Tensor]:
    """Collate for variable-H batches.

    Context keys (`_FIXED_KEYS`) are stacked; all other keys are H-indexed targets
    (f_prop_target, z_vis_target, binary_label_target, valid_target_mask,
    binary_valid_target_mask, and proprio_horizon_mask when present) and are padded
    along the H dimension to the batch maximum. Padded positions are 0 / False and
    excluded by the masks.
    """
    max_h = max(sample["f_prop_target"].shape[0] for sample in batch)
    out: dict[str, Tensor] = {}

    for key in batch[0]:
        if key in _FIXED_KEYS:
            out[key] = torch.stack([sample[key] for sample in batch], dim=0)
            continue
        padded: list[Tensor] = []
        for sample in batch:
            t = sample[key]
            h = t.shape[0]
            if h < max_h:
                pad_shape = (max_h - h,) + tuple(t.shape[1:])
                t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
            padded.append(t)
        out[key] = torch.stack(padded, dim=0)

    return out
