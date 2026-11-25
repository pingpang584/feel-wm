"""Read a model checkpoint: a training ``.pt`` or a weights-only ``.safetensors`` file."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def read_checkpoint(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """Return the checkpoint payload; the weights are always under ``"model_state"``.

    A ``.safetensors`` file holds nothing but the weights, so its payload has that one key.
    """
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file  # noqa: PLC0415 (only this format needs it)

        return {"model_state": load_file(str(path), device=str(map_location))}
    return torch.load(path, map_location=map_location, weights_only=False)
