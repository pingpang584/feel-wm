from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch import Tensor

from feelwm.models.dynamics.resampler import (
    PerceiverBottleneck,
    PerceiverBottleneckConfig,
)


warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
)


@dataclass
class DynamicsTransformerConfig:
    d_vis: int = 768
    n_layers: int = 6
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    max_context_len: int = 32
    gradient_checkpointing: bool = False
    # Interleaved proprio-token injection (alternative to FiLM injection). When True one
    # proprio token per context frame is appended to that frame's patch block; the token
    # is `gate ⊙ proprio_token_proj(prop) + type_emb + temporal_pos[t]`. The projection is
    # zero-initialised so at step 0 the token carries no proprio signal (≈ type_emb +
    # pos) and training starts from the FiLM/no-proprio behaviour. d_prop is the
    # conditioning-embedding width (same vector the FiLM/feedback buffer consumes).
    proprio_token: bool = False
    d_prop: int = 128
    # Zero-init tanh gate on the token's proprio content (token mode only). Mirrors the
    # AdaLN-Zero tanh-gate fix below: the token-ONLY arm (FiLM proprio removed) diverges
    # under free-running rollout because the model's own predicted proprio is re-injected
    # as a high-gain per-frame token. tanh(gate) (zero-init ⇒ 0 at start) bounds and ramps
    # that injection. Default False preserves checkpoints trained without the gate.
    proprio_token_gate: bool = False
    # Perceiver-style token bottleneck around the core. OFF by default: when False no
    # module is built, no state_dict key appears, and the forward is bitwise the stock one.
    # When True the core runs on T*bottleneck_k latents instead of T*n_patch patch
    # tokens; the last frame's latents are expanded back to the full n_patch grid before
    # out_norm, so every consumer downstream keeps the 256-token contract.
    token_bottleneck: bool = False
    bottleneck_k: int = 32
    n_patch: int = 256
    # Gated residual skip around the bottleneck (see PerceiverBottleneckConfig).
    bottleneck_residual_exit: bool = False


class DynamicsTransformer(nn.Module):
    """Causal transformer over cached visual patch features."""

    def __init__(self, cfg: DynamicsTransformerConfig = DynamicsTransformerConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.temporal_pos = nn.Parameter(torch.zeros(cfg.max_context_len, cfg.d_vis))
        if cfg.proprio_token:
            # zero-init projection: token starts as type_emb + temporal_pos (no proprio
            # leakage at init -> stable start, identical-ish to the no-token trunk).
            self.proprio_token_proj = nn.Linear(cfg.d_prop, cfg.d_vis)
            nn.init.zeros_(self.proprio_token_proj.weight)
            nn.init.zeros_(self.proprio_token_proj.bias)
            self.proprio_type_emb = nn.Parameter(torch.zeros(cfg.d_vis))
            nn.init.normal_(self.proprio_type_emb, std=0.02)
            if cfg.proprio_token_gate:
                # tanh-bounded, zero-init: tanh(0)=0 ⇒ token proprio content starts at 0.
                self.proprio_gate = nn.Parameter(torch.zeros(cfg.d_vis))
        self.bottleneck = (
            PerceiverBottleneck(
                PerceiverBottleneckConfig(
                    d_vis=cfg.d_vis,
                    k=cfg.bottleneck_k,
                    n_patch=cfg.n_patch,
                    n_heads=cfg.n_heads,
                    ffn_mult=cfg.ffn_mult,
                    dropout=cfg.dropout,
                    residual_exit=cfg.bottleneck_residual_exit,
                )
            )
            if cfg.token_bottleneck
            else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_vis,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_mult * cfg.d_vis,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_norm = nn.LayerNorm(cfg.d_vis)

    def forward(
        self,
        z_vis: Tensor,
        gamma: Tensor | None = None,
        beta: Tensor | None = None,
        proprio_window: Tensor | None = None,
    ) -> Tensor:
        if z_vis.ndim != 4:
            raise ValueError(f"z_vis must have shape [B, T, N, D], got {tuple(z_vis.shape)}")
        B, T, N, D = z_vis.shape
        if T > self.cfg.max_context_len:
            raise ValueError(
                f"context length {T} exceeds max_context_len={self.cfg.max_context_len}"
            )
        if D != self.cfg.d_vis:
            raise ValueError(f"expected D={self.cfg.d_vis}, got {D}")

        x = z_vis
        if gamma is not None and beta is not None:
            last = gamma[:, None, :] * x[:, -1] + beta[:, None, :]
            x = torch.cat([x[:, :-1], last[:, None]], dim=1)

        x = x + self.temporal_pos[:T][None, :, None, :]

        # Token bottleneck ENTRY: compress each frame's patch grid to K latents. Sits
        # after FiLM + positional info (see PerceiverBottleneck) and BEFORE the
        # proprio token, which is appended to the latents so compression never touches
        # the proprio conditioning path.
        x_in_last = None
        if self.bottleneck is not None:
            x_in_last = x[:, -1]                                        # residual-exit skip
            x = self.bottleneck.encode(x)                               # [B, T, K, D]
            n_core = x.shape[2]
        else:
            n_core = N

        use_tokens = self.cfg.proprio_token and proprio_window is not None
        if use_tokens:
            if proprio_window.shape[:2] != (B, T) or proprio_window.shape[-1] != self.cfg.d_prop:
                raise ValueError(
                    f"proprio_window must have shape [B, T, d_prop]=({B}, {T}, {self.cfg.d_prop}), "
                    f"got {tuple(proprio_window.shape)}"
                )
            # one token per frame: project (optionally tanh-gated) -> mark with type emb
            # -> SAME per-frame temporal pos as that frame's patches (lives at frame t).
            content = self.proprio_token_proj(proprio_window)           # [B, T, D]
            if self.cfg.proprio_token_gate:
                content = torch.tanh(self.proprio_gate) * content
            tok = content + self.proprio_type_emb[None, None, :]
            tok = tok + self.temporal_pos[:T][None, :, :]
            x = torch.cat([x, tok[:, :, None, :]], dim=2)               # [B, T, n_core+1, D]
            M = n_core + 1
        else:
            M = n_core

        x = x.reshape(B, T * M, D)
        mask = self._causal_time_mask(T, M, z_vis.device)
        # Final layer runs only the last frame's M tokens as queries (K/V stay full):
        # only x[:, -M:] leaves this module, and causally those rows attend to every
        # token, so dropping the other query rows is exact (~10% of dynamics FLOPs).
        # Earlier layers cannot be trimmed — their full outputs are the next layer's K/V.
        layers = self.encoder.layers
        if self.gradient_checkpointing and self.training:
            for layer in layers[:-1]:
                x = checkpoint(
                    lambda src, module=layer: module(src, src_mask=mask),
                    x,
                    use_reentrant=False,
                )
            x = checkpoint(
                lambda src: self._trimmed_last_layer(layers[-1], src, M),
                x,
                use_reentrant=False,
            )
        else:
            for layer in layers[:-1]:
                x = layer(x, src_mask=mask)
            x = self._trimmed_last_layer(layers[-1], x, M)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        if use_tokens:
            # last frame's patch/latent rows only; drop the proprio token output.
            # LayerNorm is per-token, so dropping it before out_norm (rather than after,
            # as the pre-bottleneck code did) leaves the kept rows bitwise unchanged.
            x = x[:, :n_core, :]
        if self.bottleneck is not None:
            # EXIT: n_patch learned output queries read the core's latents, restoring the
            # full grid every downstream consumer expects. out_norm still has the last
            # word, so the head-facing statistics are the ones the heads were trained on.
            x = self.bottleneck.decode(x, x_in_last)                    # [B, N, D]
        return self.out_norm(x)

    @staticmethod
    def _trimmed_last_layer(
        layer: nn.TransformerEncoderLayer, x: Tensor, n_last: int
    ) -> Tensor:
        """Final-layer pass over the last n_last query rows only (K/V = all tokens).

        Reuses the stock layer's submodules (norm_first path), so parameters and
        state_dict layout are unchanged and the output equals
        layer(x, src_mask=causal_mask)[:, -n_last:] exactly (fp32 rounding aside);
        no mask is needed because the last frame's rows are never masked.
        Equivalence is locked in by tests/models/test_query_trim.py.
        """
        h = layer.norm1(x)
        q = h[:, -n_last:]
        attn = layer.self_attn(q, h, h, need_weights=False)[0]
        y = x[:, -n_last:] + layer.dropout1(attn)
        return y + layer._ff_block(layer.norm2(y))

    @staticmethod
    def _causal_time_mask(T: int, N: int, device: torch.device) -> Tensor:
        times = torch.arange(T, device=device).repeat_interleave(N)
        return times[None, :] > times[:, None]


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaLN affine: x * (1 + tanh(scale)) + shift. shift/scale already token-broadcast.

    The tanh bounds the multiplicative factor to (0, 2) so it cannot compound across
    the stacked blocks into a bf16 overflow when proprio/alignment gradients push the
    modulation up (FiLM has no such per-block product, hence it is stable without this).
    tanh(0)=0 keeps the zero-init block exactly identity.
    """
    return x * (1.0 + torch.tanh(scale)) + shift


@dataclass
class AdaLNDynamicsTransformerConfig:
    d_vis: int = 768
    n_layers: int = 6
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    max_context_len: int = 32
    gradient_checkpointing: bool = False
    cond_bottleneck: int = 64


class _DiTBlock(nn.Module):
    """Pre-norm transformer block with AdaLN-Zero modulation (DiT-style).

    LayerNorms are affine-free; the affine (shift, scale) and residual gates come
    from the conditioning vector via a zero-initialised bottleneck MLP
    (cond -> r -> 6*d_vis), so the block starts as identity for the modulated
    (last-frame) tokens. Past-frame tokens are processed with the standard
    (unmodulated) pre-norm residual so they remain valid attention keys/values.
    """

    def __init__(self, cfg: AdaLNDynamicsTransformerConfig) -> None:
        super().__init__()
        d = cfg.d_vis
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.ffn_mult * d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_mult * d, d),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, cfg.cond_bottleneck),
            nn.SiLU(),
            nn.Linear(cfg.cond_bottleneck, 6 * d),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self, x: Tensor, cond: Tensor, attn_mask: Tensor, last_mask: Tensor
    ) -> Tensor:
        # cond: [B, d_vis] -> 6 modulation vectors [B, d_vis]
        s_a, sc_a, g_a, s_f, sc_f, g_f = self.adaLN(cond).chunk(6, dim=-1)

        # restrict modulation to last-frame tokens; past tokens get the standard
        # pre-norm residual (shift=scale=0, gate=1).
        def shift(v: Tensor) -> Tensor:
            return v[:, None, :] * last_mask

        # Gates are tanh-bounded to (-1, 1): raw gates are the explosion amplifier
        # measured in an AdaLN training divergence (gate ~300 x mlp branch -> activations
        # ~3e6 -> bf16 backward overflow on every batch while forward stays finite
        # behind the per-block LN). tanh(0)=0 keeps the zero-init identity start.
        def gate(v: Tensor) -> Tensor:
            return torch.tanh(v)[:, None, :] * last_mask + (1.0 - last_mask)

        h = _modulate(self.ln1(x), shift(s_a), shift(sc_a))
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + gate(g_a) * attn_out
        h = _modulate(self.ln2(x), shift(s_f), shift(sc_f))
        x = x + gate(g_f) * self.mlp(h)
        return x


class AdaLNDynamicsTransformer(nn.Module):
    """Causal transformer conditioned via per-block AdaLN-Zero (last-frame-only).

    Drop-in alternative to `DynamicsTransformer` for `action_conditioning="adaln"`.
    Instead of a single input-level (gamma, beta) on the last frame, the action+
    proprio conditioning vector modulates every block's LayerNorm and residual
    gates, applied only to the last-frame tokens (the transition target).
    """

    def __init__(
        self, cfg: AdaLNDynamicsTransformerConfig = AdaLNDynamicsTransformerConfig()
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.temporal_pos = nn.Parameter(torch.zeros(cfg.max_context_len, cfg.d_vis))
        self.blocks = nn.ModuleList(_DiTBlock(cfg) for _ in range(cfg.n_layers))
        self.out_norm = nn.LayerNorm(cfg.d_vis)

    def forward(self, z_vis: Tensor, cond: Tensor) -> Tensor:
        if z_vis.ndim != 4:
            raise ValueError(f"z_vis must have shape [B, T, N, D], got {tuple(z_vis.shape)}")
        B, T, N, D = z_vis.shape
        if T > self.cfg.max_context_len:
            raise ValueError(
                f"context length {T} exceeds max_context_len={self.cfg.max_context_len}"
            )
        if D != self.cfg.d_vis:
            raise ValueError(f"expected D={self.cfg.d_vis}, got {D}")

        x = z_vis + self.temporal_pos[:T][None, :, None, :]
        x = x.reshape(B, T * N, D)
        mask = DynamicsTransformer._causal_time_mask(T, N, z_vis.device)
        last_mask = self._last_frame_mask(T, N, z_vis.device, x.dtype)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, cond, mask, last_mask, use_reentrant=False)
            else:
                x = block(x, cond, mask, last_mask)

        x = self.out_norm(x)
        return x[:, -N:, :]

    @staticmethod
    def _last_frame_mask(T: int, N: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        idx = torch.arange(T * N, device=device)
        return (idx >= (T - 1) * N).to(dtype).view(1, T * N, 1)
