from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CurriculumStage(Enum):
    VISUAL_ONLY = 1
    VISUAL_PROPRIO = 2
    FULL = 3


@dataclass
class CurriculumConfig:
    enabled: bool = True
    proprio_start_epoch: int = 2
    traversability_start_epoch: int = 5


def stage_for_epoch(epoch: int, cfg: CurriculumConfig) -> CurriculumStage:
    if not cfg.enabled:
        return CurriculumStage.FULL
    if epoch < cfg.proprio_start_epoch:
        return CurriculumStage.VISUAL_ONLY
    if epoch < cfg.traversability_start_epoch:
        return CurriculumStage.VISUAL_PROPRIO
    return CurriculumStage.FULL
