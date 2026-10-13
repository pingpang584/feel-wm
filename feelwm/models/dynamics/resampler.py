from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PerceiverBottleneckConfig:
    """Perceiver-style token bottleneck around the dynamics core.

    The core carries `k` latents per frame instead of `n_patch` patch tokens, cutting
    the causal-attention sequence from T*n_patch to T*k. `n_patch` is the grid the
    EXIT restores, so everything downstream of the dynamics (visual head, proprio
    head, the rollout's own visual feedback) keeps the full-grid contract.
    """

    d_vis: int = 768
    k: int = 32            # latent queries per frame == core sequence length per frame
    n_patch: int = 256     # patch grid entering the ENTRY / restored at the EXIT
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    # Gated residual EXIT. False (default) = plain behaviour: the EXIT regenerates the grid
    # from the latents ALONE.
    # True: out = x_in_last + tanh(g) * decode(latents) with g zero-init, so at step 0 the
    # module is the copy-last-frame predictor — on-manifold for a FROZEN visual_head, and
    # the EXIT learns the core's "sum of layer updates" as a residual instead of having to
    # regenerate the whole residual stream. Precedent: proprio_token_gate (the ungated
    # token arm diverged, the gated one was stable), AdaLN-Zero, Flamingo's gated resampler.
    residual_exit: bool = False


class _QueryResampler(nn.Module):
    """Learned query bank cross-attending a token set: [B, N_kv, D] -> [B, n_query, D].

    Pre-norm cross-attention + pre-norm FFN, both residual on the queries — the
    Perceiver-resampler block used by the flamingo/VLA line. The query bank is shared
    across frames: frames are already distinguished by the temporal position embedding
    added upstream, so a per-frame bank would only duplicate parameters.
    """

    def __init__(self, n_query: int, cfg: PerceiverBottleneckConfig) -> None:
        super().__init__()
        d = cfg.d_vis
        self.queries = nn.Parameter(torch.empty(n_query, d))
        nn.init.normal_(self.queries, std=0.02)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_mult * d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_mult * d, d),
        )

    def forward(self, kv: Tensor) -> Tensor:
        if kv.ndim != 3:
            raise ValueError(f"kv must have shape [B, N, D], got {tuple(kv.shape)}")
        q = self.queries[None].expand(kv.shape[0], -1, -1)
        h = self.norm_kv(kv)
        out = q + self.attn(self.norm_q(q), h, h, need_weights=False)[0]
        return out + self.ffn(self.norm_ffn(out))


class PerceiverBottleneck(nn.Module):
    """ENTRY (n_patch -> k, per frame) + EXIT (k -> n_patch, last frame) resamplers.

    ENTRY sits at the dynamics-core entry, AFTER FiLM and the temporal position
    embedding — the same place a training-free token-merging probe acted. EXIT replaces
    that probe's un-merge (copy each group's value back to its members) with 256
    learned output queries cross-attending the core's latents.

    Proprio conditioning is never routed through here: FiLM acts before the ENTRY,
    and the interleaved proprio token (when enabled) is appended to the latents AFTER
    compression and dropped before the EXIT.
    """

    def __init__(self, cfg: PerceiverBottleneckConfig = PerceiverBottleneckConfig()) -> None:
        super().__init__()
        if not 1 <= cfg.k <= cfg.n_patch:
            raise ValueError(f"bottleneck k={cfg.k} must be in [1, n_patch={cfg.n_patch}]")
        self.cfg = cfg
        self.entry = _QueryResampler(cfg.k, cfg)
        self.exit = _QueryResampler(cfg.n_patch, cfg)
        # zero-init per-channel gate; tanh(0)=0 ⇒ the bottleneck path contributes exactly
        # nothing at step 0. Lives inside this module so its state-dict key carries the
        # `dynamics.bottleneck.` prefix (the two-group-lr selector keys off that prefix).
        self.exit_gate = nn.Parameter(torch.zeros(cfg.d_vis)) if cfg.residual_exit else None

    def encode(self, x: Tensor) -> Tensor:
        """[B, T, n_patch, D] -> [B, T, k, D] (each frame compressed independently)."""
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, T, N, D], got {tuple(x.shape)}")
        B, T, N, D = x.shape
        if N != self.cfg.n_patch:
            raise ValueError(
                f"token bottleneck built for n_patch={self.cfg.n_patch}, got N={N}"
            )
        latents = self.entry(x.reshape(B * T, N, D))
        return latents.reshape(B, T, self.cfg.k, D)

    def decode(self, latents: Tensor, x_in_last: Tensor | None = None) -> Tensor:
        """[B, k, D] -> [B, n_patch, D] (the last frame's full-grid reconstruction).

        In residual_exit mode `x_in_last` — the last frame's tokens exactly as they
        entered :meth:`encode` — is required: the reconstruction is added onto it
        through the zero-init tanh gate.
        """
        if latents.ndim != 3 or latents.shape[1] != self.cfg.k:
            raise ValueError(
                f"latents must have shape [B, {self.cfg.k}, D], got {tuple(latents.shape)}"
            )
        out = self.exit(latents)
        if self.exit_gate is None:
            return out
        if x_in_last is None:
            raise ValueError("residual_exit=True requires the last frame's input tokens")
        if x_in_last.shape != out.shape:
            raise ValueError(
                f"x_in_last must have shape {tuple(out.shape)}, got {tuple(x_in_last.shape)}"
            )
        return x_in_last + torch.tanh(self.exit_gate) * out

    def gate_abs_mean(self) -> float | None:
        """mean |tanh(g)| — how much of the EXIT the model actually uses. None when off."""
        if self.exit_gate is None:
            return None
        return float(torch.tanh(self.exit_gate.detach()).abs().mean())
