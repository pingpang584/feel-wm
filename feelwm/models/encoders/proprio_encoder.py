from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor


@dataclass
class ProprioEncoderConfig:
    d_raw: int = 14
    d_prop: int = 128
    n_conv_layers: int = 2
    kernel_size: int = 5


class _ConvBlock(nn.Module):
    """Conv1d → LayerNorm (over channel dim) → GELU, stride 2."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2
        )
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:  # [B, in_ch, L] → [B, out_ch, L//2]
        x = self.conv(x)
        x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        return self.act(x)


class ProprioEncoder(nn.Module):
    """
    Encodes raw proprioceptive history into per-frame features.

    Input:  [B, T, T_win, d_raw]
    Output: [B, T, D_prop]
    """

    def __init__(self, cfg: ProprioEncoderConfig = ProprioEncoderConfig()) -> None:
        super().__init__()
        d = cfg.d_prop
        layers: list[nn.Module] = []
        in_ch = cfg.d_raw
        for _ in range(cfg.n_conv_layers):
            layers.append(_ConvBlock(in_ch, d, cfg.kernel_size))
            in_ch = d
        self.conv = nn.Sequential(*layers)
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, T, T_win, d_raw = x.shape
        # Flatten B×T, put d_raw as channel dim for Conv1d
        x = x.reshape(B * T, T_win, d_raw).permute(0, 2, 1)  # [B*T, d_raw, T_win]
        x = self.conv(x)    # [B*T, d_prop, L']
        x = x.mean(dim=-1)  # [B*T, d_prop]  — global avg pool
        x = self.mlp(x)     # [B*T, d_prop]
        return x.reshape(B, T, -1)  # [B, T, d_prop]
