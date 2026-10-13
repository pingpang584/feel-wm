from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from feelwm.models.dynamics.film import (
    AdaLNConditioning,
    AdaLNConditioningConfig,
    FiLMConditioning,
    FiLMConditioningConfig,
)
from feelwm.models.dynamics.transformer import (
    AdaLNDynamicsTransformer,
    AdaLNDynamicsTransformerConfig,
    DynamicsTransformer,
    DynamicsTransformerConfig,
)
from feelwm.models.encoders.action_embedding import ActionEmbedding, ActionEmbeddingConfig
from feelwm.models.encoders.proprio_encoder import ProprioEncoder, ProprioEncoderConfig
from feelwm.models.heads.next_proprio_head import NextProprioHead, NextProprioHeadConfig
from feelwm.models.heads.proprio_feature_head import (
    PROPRIO_FEATURE_DIM,
    ProprioFeatureHead,
    ProprioFeatureHeadConfig,
)
from feelwm.models.heads.traversability_head import (
    TraversabilityHead,
    TraversabilityHeadConfig,
)
from feelwm.models.heads.visual_head import VisualHead, VisualHeadConfig


@dataclass
class ModelConfig:
    context_len: int = 3
    rollout_horizon: int = 8
    n_patch: int = 256
    d_vis: int = 768
    d_prop: int = 128
    d_action: int = 3
    d_raw: int = 14
    n_layers: int = 6
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    use_proprio_input: bool = True
    use_proprio_feature_head: bool = True
    use_traversability_head: bool = True
    # Number of active proprio output channels (subset of the canonical 5).
    # Must equal len(LossConfig.channel_names) and len(DatasetConfig.active_proprio_channels).
    proprio_feature_dim: int = PROPRIO_FEATURE_DIM
    # Traversability-head input source (ablation axis):
    #   "proprio"  – hierarchical (default): cost = g(f_prop_pred[5]). Cost confined to
    #                the OOD-stable, proprio-anchored 5-dim manifold; blind to visual-only
    #                risk not reflected in the physical features.
    #   "parallel" – cost = g(pooled h_t[d_vis]). Full fused (visual+proprio) hidden;
    #                higher capacity, may overfit appearance / sparse positives.
    #   "hybrid"   – cost = g([pooled h_t, f_prop_pred]). Keeps proprio grounding while
    #                restoring visual access.
    traversability_input_mode: str = "proprio"
    # When True (proprio/hybrid modes), the cost head reads sg(f_prop_pred): the cost
    # gradient does NOT flow into the proprio feature head, so proprio prediction (the
    # primary world-model output) stays undistorted while cost still uses the physical
    # features (visual+proprio fusion belongs at the planning-score level, not inside the
    # cost head). Default False preserves older checkpoints; the published configs enable it.
    traversability_stop_grad_proprio: bool = False
    # Action/proprio conditioning of the dynamics:
    #   "film"   – single input-level (gamma, beta) on the last context frame (default).
    #   "adaln"  – AdaLN-Zero: action+proprio modulates every transformer block's
    #              LayerNorm + residual gates, applied to last-frame tokens only,
    #              via a zero-init bottleneck MLP (cond -> adaln_bottleneck -> 6*d_vis).
    #              Starts as identity for the modulated tokens.
    #   "concat" – NotImplemented stub.
    action_conditioning: str = "film"
    # Bottleneck width of the per-block AdaLN modulation MLP (adaln mode only).
    adaln_bottleneck: int = 64
    # How proprio is INJECTED into the FiLM dynamics (ablation axis). Orthogonal to
    # action_conditioning (only valid with "film"):
    #   "film"       – current canonical: proprio enters via the FiLM (gamma, beta) on
    #                  the last context frame only.
    #   "token"      – proprio enters ONLY as an interleaved per-frame token in the
    #                  transformer; FiLM sees action only (proprio removed). Question
    #                  tested: does a token bypass the single FiLM (gamma, beta) bottleneck
    #                  and rollout/OOD better?
    #   "token_film" – both: per-frame token AND FiLM proprio. Tests whether the token
    #                  adds anything over film alone.
    proprio_injection: str = "film"
    # Zero-init tanh gate on the proprio token's content (token/token_film only). Stabilises
    # the token-only arm, which diverges under free-running rollout without it. Default
    # False preserves checkpoints trained pre-gate.
    proprio_token_gate: bool = False
    # Perceiver-resampler token bottleneck around the dynamics core. None = OFF (default;
    # an ablation axis — every existing config/checkpoint is unaffected, no extra
    # state_dict keys). ON: {"type": "perceiver", "k": 32} — K learned latents per frame
    # replace the 256 patch tokens INSIDE the core; the exit resampler restores the full
    # n_patch grid, so heads / visual feedback / planner see the unchanged contract.
    # FiLM path only: the AdaLN core has no bottleneck route. Optional
    # `residual_exit: true` adds the zero-init gated skip described in
    # PerceiverBottleneckConfig; absent/false = the plain bottleneck forward, bit for bit.
    token_bottleneck: dict | None = None
    curriculum_enabled: bool = True
    visual_encoder: str = "dinov2"
    gradient_checkpointing: bool = False
    # How the proprio FiLM-conditioning evolves DURING rollout (raw proprio only ever seeds
    # step 0 via the encoder; this controls steps >=1). Diagnostic ablation axis:
    #   "feedback" (default) – prop_current <- rollout_prop_projection(f_prop_pred[4-dim]):
    #                          the predicted 4-dim feature, projected back to d_prop. Current
    #                          design. Note the conditioning space differs from the t=0 seed
    #                          encode(raw); there is no alignment loss tying them (yet).
    #   "freeze"             – prop_current held at the t=0 encode(raw) seed for all steps.
    #   "zero"               – prop_current=0 for steps>=1 (proprio influences step 0 only).
    #   "aligned"            – a dedicated next_proprio_head predicts the next-step
    #                          conditioning as a residual
    #                          over the frozen seed, ẑ_prop=seed+Δ(h_t), trained by an alignment
    #                          loss to track encode(true future raw window). Fixes the feedback
    #                          space-mismatch + error re-injection; zero-init Δ ⇒ starts == freeze.
    # freeze/zero quantify how much the 4-dim feedback actually contributes. Default preserves
    # all existing runs.
    rollout_proprio_mode: str = "feedback"
    # --- NADA-style feedback-shift noise ---
    # Path to a JSON table of measured free-running proprio-conditioning error (per
    # horizon, per channel, on a baseline checkpoint). When set, TRAINING-mode rollouts
    # add zero-mean Gaussian noise to the GT 4-dim proprio fed on teacher-forced steps:
    #   sigma[c] = noise_scale * table[noise_metric][min(h, H_table-1)][c]
    # Motivation (measured on a baseline checkpoint): teacher-forced steps condition on
    # error-free GT while deployment (free-running) conditioning carries err_std
    # 0.35–1.37; the model shortcuts on the
    # clean GT (TF err_std 2–5.5x below free-running). Noising the GT branch makes the
    # training conditioning distribution resemble deployment. The free-running (pred)
    # branch is NOT noised — it already carries real model error. model.eval() never
    # noises. None = off (default; preserves all existing runs/checkpoints — the sigma
    # table is a non-persistent buffer, absent from state_dict).
    rollout_proprio_noise_path: str | None = None
    rollout_proprio_noise_scale: float = 1.0
    rollout_proprio_noise_metric: str = "err_std"  # or "w1"

    def __post_init__(self) -> None:
        if self.action_conditioning not in {"film", "adaln", "concat"}:
            raise ValueError(f"unknown action_conditioning={self.action_conditioning!r}")
        if self.proprio_injection not in {"film", "token", "token_film"}:
            raise ValueError(f"unknown proprio_injection={self.proprio_injection!r}")
        if self.proprio_injection != "film":
            if self.action_conditioning != "film":
                raise ValueError(
                    "proprio_injection='token'/'token_film' requires action_conditioning='film'"
                )
            if not self.use_proprio_input:
                raise ValueError(
                    "proprio_injection='token'/'token_film' requires use_proprio_input=True"
                )
        if self.rollout_proprio_mode not in {"feedback", "freeze", "zero", "aligned"}:
            raise ValueError(f"unknown rollout_proprio_mode={self.rollout_proprio_mode!r}")
        if self.visual_encoder not in {"dinov2", "dinov3", "vae"}:
            raise ValueError(f"unknown visual_encoder={self.visual_encoder!r}")
        if self.rollout_proprio_noise_metric not in {"err_std", "w1"}:
            raise ValueError(
                f"unknown rollout_proprio_noise_metric={self.rollout_proprio_noise_metric!r}"
            )
        if self.rollout_proprio_noise_path is not None and not self.use_proprio_input:
            raise ValueError(
                "rollout_proprio_noise_path requires use_proprio_input=True "
                "(noise targets the fed-back proprio conditioning)"
            )
        if self.token_bottleneck is not None:
            tb = self.token_bottleneck
            if not isinstance(tb, dict):
                raise ValueError(
                    f"token_bottleneck must be a dict like {{'type': 'perceiver', 'k': 32}}, "
                    f"got {tb!r}"
                )
            unknown = set(tb) - {"type", "k", "residual_exit"}
            if unknown:
                raise ValueError(f"unknown token_bottleneck keys {sorted(unknown)}")
            if tb.get("type") != "perceiver":
                raise ValueError(
                    f"unknown token_bottleneck type={tb.get('type')!r} (expected 'perceiver')"
                )
            k = int(tb.get("k", 32))
            if not 1 <= k <= self.n_patch:
                raise ValueError(
                    f"token_bottleneck k={k} must be in [1, n_patch={self.n_patch}]"
                )
            if not isinstance(tb.get("residual_exit", False), bool):
                raise ValueError(
                    f"token_bottleneck residual_exit must be a bool, got {tb['residual_exit']!r}"
                )
            if self.action_conditioning != "film":
                raise ValueError(
                    "token_bottleneck requires action_conditioning='film' "
                    "(the AdaLN core has no bottleneck path)"
                )
        if self.traversability_input_mode not in {"proprio", "parallel", "hybrid"}:
            raise ValueError(
                f"unknown traversability_input_mode={self.traversability_input_mode!r}"
            )
        # proprio/hybrid modes consume f_prop_pred; when the proprio head is disabled the
        # cost output degrades to None (handled in _traversability_logit), matching the
        # pre-existing graceful-degradation contract. 'parallel' needs no proprio head.


class TerrainWorldModel(nn.Module):
    """Top-level B1 joint world model operating on cached visual features."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        if config.action_conditioning == "concat":
            raise NotImplementedError("action_conditioning='concat' is not implemented in B1")

        self.cfg = config
        self.proprio_encoder = (
            ProprioEncoder(
                ProprioEncoderConfig(d_raw=config.d_raw, d_prop=config.d_prop)
            )
            if config.use_proprio_input
            else None
        )
        self.action_embedding = ActionEmbedding(
            ActionEmbeddingConfig(d_action=config.d_action, d_vis=config.d_vis)
        )
        self._adaln = config.action_conditioning == "adaln"
        # proprio injection axis (FiLM path only; validated in ModelConfig.__post_init__)
        self._proprio_token = config.proprio_injection in {"token", "token_film"}
        self._film_proprio = config.proprio_injection in {"film", "token_film"}
        max_ctx = max(config.context_len + 1, config.rollout_horizon + 1)
        if self._adaln:
            self.cond = AdaLNConditioning(
                AdaLNConditioningConfig(d_vis=config.d_vis, d_prop=config.d_prop)
            )
            self.dynamics = AdaLNDynamicsTransformer(
                AdaLNDynamicsTransformerConfig(
                    d_vis=config.d_vis,
                    n_layers=config.n_layers,
                    n_heads=config.n_heads,
                    ffn_mult=config.ffn_mult,
                    dropout=config.dropout,
                    max_context_len=max_ctx,
                    gradient_checkpointing=config.gradient_checkpointing,
                    cond_bottleneck=config.adaln_bottleneck,
                )
            )
        else:
            self.film = FiLMConditioning(
                FiLMConditioningConfig(d_vis=config.d_vis, d_prop=config.d_prop)
            )
            self.dynamics = DynamicsTransformer(
                DynamicsTransformerConfig(
                    d_vis=config.d_vis,
                    n_layers=config.n_layers,
                    n_heads=config.n_heads,
                    ffn_mult=config.ffn_mult,
                    dropout=config.dropout,
                    max_context_len=max_ctx,
                    gradient_checkpointing=config.gradient_checkpointing,
                    proprio_token=self._proprio_token,
                    d_prop=config.d_prop,
                    proprio_token_gate=config.proprio_token_gate,
                    token_bottleneck=config.token_bottleneck is not None,
                    bottleneck_k=int((config.token_bottleneck or {}).get("k", 32)),
                    bottleneck_residual_exit=bool(
                        (config.token_bottleneck or {}).get("residual_exit", False)
                    ),
                    n_patch=config.n_patch,
                )
            )
        self.visual_head = VisualHead(VisualHeadConfig(d_vis=config.d_vis))
        self.proprio_feature_head = (
            ProprioFeatureHead(
                ProprioFeatureHeadConfig(
                    d_in=config.d_vis, pool_patches=True, out_dim=config.proprio_feature_dim
                )
            )
            if config.use_proprio_feature_head
            else None
        )
        self.traversability_head = (
            TraversabilityHead(
                TraversabilityHeadConfig(d_in=self._traversability_d_in(config))
            )
            if config.use_traversability_head
            else None
        )
        self.rollout_prop_projection = (
            nn.Linear(config.proprio_feature_dim, config.d_prop)
            if config.use_proprio_input
            else None
        )
        # NADA-style noise table [H_table, C] for the teacher-forced proprio branch.
        # Non-persistent: absent from state_dict, so checkpoints stay interchangeable
        # between noise-on and noise-off configs.
        if config.rollout_proprio_noise_path is not None:
            payload = json.loads(Path(config.rollout_proprio_noise_path).read_text())
            table = torch.as_tensor(
                payload[config.rollout_proprio_noise_metric], dtype=torch.float32
            )
            if table.ndim != 2 or table.shape[1] != config.proprio_feature_dim:
                raise ValueError(
                    f"noise table {config.rollout_proprio_noise_metric!r} must be "
                    f"[H, {config.proprio_feature_dim}], got {tuple(table.shape)} "
                    f"from {config.rollout_proprio_noise_path}"
                )
            if not torch.isfinite(table).all():
                raise ValueError(
                    f"non-finite entries in noise table {config.rollout_proprio_noise_path}"
                )
            self.register_buffer(
                "prop_noise_sigma",
                config.rollout_proprio_noise_scale * table,
                persistent=False,
            )
        else:
            self.prop_noise_sigma = None
        # aligned mode: residual head predicting the next-step conditioning embedding.
        # Built only in aligned mode (other modes carry no unused params / checkpoint keys).
        self.next_proprio_head = (
            NextProprioHead(
                NextProprioHeadConfig(d_vis=config.d_vis, d_prop=config.d_prop)
            )
            if (config.rollout_proprio_mode == "aligned" and config.use_proprio_input)
            else None
        )

    def forward(
        self,
        batch: dict[str, Tensor],
        action_sequence: Tensor | None = None,
        **rollout_kwargs,
    ) -> dict[str, Tensor | None]:
        # With action_sequence this dispatches to forward_rollout so DDP-wrapped
        # training can call the rollout THROUGH DDP.forward. Calling
        # model.module.forward_rollout(...) directly (the old trainer pattern)
        # silently skips DDP gradient synchronization entirely — the reducer is
        # armed only inside DDP.forward — so each rank trains independently.
        if action_sequence is not None:
            return self.forward_rollout(batch, action_sequence, **rollout_kwargs)
        z_vis_context = batch["z_vis_context"]
        action = self._current_action(batch)
        prop_context = self._encode_proprio(batch) if self.proprio_encoder is not None else None
        prop_current = prop_context[:, -1] if prop_context is not None else None
        return self._predict_from_context(
            z_vis_context, prop_current, action, prop_window=prop_context
        )

    def forward_rollout(
        self,
        initial_batch: dict[str, Tensor],
        action_sequence: Tensor,
        use_ground_truth_visual: Tensor | None = None,
        use_ground_truth_proprio: Tensor | None = None,
        teacher_force_mask: Tensor | None = None,
        proprio_window_target: Tensor | None = None,
        oracle_proprio_embedding: bool = False,
    ) -> dict[str, Tensor | None]:
        if action_sequence.ndim != 3:
            raise ValueError(
                f"action_sequence must have shape [B, H, d_action], got {tuple(action_sequence.shape)}"
            )

        z_context = initial_batch["z_vis_context"]
        prop_context = (
            self._encode_proprio(initial_batch)
            if self.proprio_encoder is not None
            else None
        )
        prop_current = prop_context[:, -1] if prop_context is not None else None
        prop_seed = prop_current  # frozen t=0 encoder embedding (residual base / freeze)
        # Sliding per-frame proprio window for token injection. Seeded with the encoded
        # context windows; slid in lockstep with z_context so its last entry is ALWAYS the
        # current FiLM/feedback `prop_current` (no token<->feedback desync). None unless
        # token injection is active.
        prop_window = (
            prop_context if (self._proprio_token and prop_context is not None) else None
        )

        # Encode future raw windows once (stop-grad): alignment targets AND the
        # encode-oracle conditioning source. Needs proprio input + supplied windows.
        align_target = (
            self.encode_proprio_targets(proprio_window_target)
            if (proprio_window_target is not None and self.proprio_encoder is not None)
            else None
        )
        oracle_embedding = align_target if oracle_proprio_embedding else None

        z_preds: list[Tensor] = []
        f_preds: list[Tensor] = []
        c_logits: list[Tensor] = []
        c_probs: list[Tensor] = []
        align_preds: list[Tensor] = []

        H = action_sequence.shape[1]
        for h in range(H):
            out = self._predict_from_context(
                z_context, prop_current, action_sequence[:, h], prop_window=prop_window
            )
            z_pred = out["z_vis_pred"]
            f_pred = out["f_prop_pred"]
            c_logit = out["c_pred_logit"]
            c_prob = out["c_pred"]

            z_preds.append(z_pred)
            if f_pred is not None:
                f_preds.append(f_pred)
            if c_logit is not None:
                c_logits.append(c_logit)
            if c_prob is not None:
                c_probs.append(c_prob)

            # aligned-mode prediction for this step: ẑ_prop = seed + Δ(h_t)
            prop_pred = None
            if out["prop_delta"] is not None and prop_seed is not None:
                prop_pred = prop_seed + out["prop_delta"]
                align_preds.append(prop_pred)

            z_next = self._next_rollout_visual(
                z_vis_pred=z_pred,
                use_ground_truth_visual=use_ground_truth_visual,
                teacher_force_mask=teacher_force_mask,
                step=h,
            )
            z_context = torch.cat([z_context[:, 1:], z_next[:, None]], dim=1)
            prop_current = self._next_rollout_proprio(
                prop_current=prop_current,
                prop_seed=prop_seed,
                prop_pred=prop_pred,
                f_prop_pred=f_pred,
                oracle_embedding=oracle_embedding,
                use_ground_truth_proprio=use_ground_truth_proprio,
                teacher_force_mask=teacher_force_mask,
                step=h,
            )
            # slide proprio window in lockstep with z_context; appended entry is the
            # updated prop_current, so prop_window[:, -1] == prop_current next step.
            if prop_window is not None and prop_current is not None:
                prop_window = torch.cat(
                    [prop_window[:, 1:], prop_current[:, None]], dim=1
                )

        return {
            "z_vis_preds": torch.stack(z_preds, dim=1),
            "f_prop_preds": torch.stack(f_preds, dim=1) if f_preds else None,
            "c_pred_logits": torch.stack(c_logits, dim=1) if c_logits else None,
            "c_preds": torch.stack(c_probs, dim=1) if c_probs else None,
            "z_prop_align_preds": torch.stack(align_preds, dim=1) if align_preds else None,
            "z_prop_align_target": align_target,
        }

    def _predict_from_context(
        self,
        z_vis_context: Tensor,
        prop_current: Tensor | None,
        action: Tensor,
        prop_window: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        action_emb = self.action_embedding(action)
        # proprio enters the FiLM (gamma, beta) only when injection mode keeps it there
        # (film / token_film); in pure "token" mode FiLM sees action only.
        film_prop = prop_current if self._film_proprio else None
        if self._adaln:
            cond = self.cond(action_emb, film_prop)
            h = self.dynamics(z_vis_context, cond=cond)
        else:
            gamma, beta = self.film(action_emb, film_prop)
            token_window = prop_window if self._proprio_token else None
            h = self.dynamics(
                z_vis_context, gamma=gamma, beta=beta, proprio_window=token_window
            )
        z_vis_pred = self.visual_head(h)

        f_prop_pred = (
            self.proprio_feature_head(h)
            if self.proprio_feature_head is not None
            else None
        )
        c_pred_logit = self._traversability_logit(h, f_prop_pred)
        c_pred = torch.sigmoid(c_pred_logit) if c_pred_logit is not None else None

        prop_delta = (
            self.next_proprio_head(h) if self.next_proprio_head is not None else None
        )

        return {
            "z_vis_pred": z_vis_pred,
            "f_prop_pred": f_prop_pred,
            "c_pred_logit": c_pred_logit,
            "c_pred": c_pred,
            "prop_delta": prop_delta,
        }

    def encode_proprio_targets(self, proprio_window_target: Tensor) -> Tensor:
        """Encode future raw proprio windows into the conditioning space (stop-grad).

        proprio_window_target: [B, H, T_win, d_raw] (the windows at the rollout
        target positions t+h*s). Returns [B, H, d_prop], detached — used as the
        alignment-loss target for `next_proprio_head` and as the conditioning for
        the encode-oracle eval (the clean ceiling for the aligned scheme).
        """
        if self.proprio_encoder is None:
            raise RuntimeError("encode_proprio_targets requires use_proprio_input=True")
        return self.proprio_encoder(proprio_window_target).detach()

    @staticmethod
    def _traversability_d_in(config: ModelConfig) -> int:
        mode = config.traversability_input_mode
        if mode == "proprio":
            return config.proprio_feature_dim
        if mode == "parallel":
            return config.d_vis
        return config.d_vis + config.proprio_feature_dim  # hybrid

    def _traversability_logit(
        self, h: Tensor, f_prop_pred: Tensor | None
    ) -> Tensor | None:
        """Cost logit from the configured input source. h is [B, N, d_vis]."""
        if self.traversability_head is None:
            return None
        mode = self.cfg.traversability_input_mode
        if mode == "proprio":
            if f_prop_pred is None:
                return None
            trav_in = self._cost_proprio(f_prop_pred)
        elif mode == "parallel":
            trav_in = h.mean(dim=-2)  # pool patches → [B, d_vis]
        else:  # hybrid
            if f_prop_pred is None:
                return None
            trav_in = torch.cat([h.mean(dim=-2), self._cost_proprio(f_prop_pred)], dim=-1)
        return self.traversability_head(trav_in)

    def _cost_proprio(self, f_prop_pred: Tensor) -> Tensor:
        """f_prop fed to the cost head; detached when stop-grad keeps proprio primary."""
        return f_prop_pred.detach() if self.cfg.traversability_stop_grad_proprio else f_prop_pred

    def _encode_proprio(self, batch: dict[str, Tensor]) -> Tensor:
        if "proprio_window_context" not in batch:
            raise KeyError("proprio_window_context is required when use_proprio_input=True")
        assert self.proprio_encoder is not None
        return self.proprio_encoder(batch["proprio_window_context"])

    def _current_action(self, batch: dict[str, Tensor]) -> Tensor:
        if "action" in batch:
            return batch["action"]
        if "action_context" not in batch:
            raise KeyError("batch must contain either action or action_context")
        action_context = batch["action_context"]
        if action_context.ndim == 3:
            return action_context[:, -1]
        if action_context.ndim == 2:
            return action_context
        raise ValueError(
            f"action_context must have shape [B, T, d_action] or [B, d_action], got {tuple(action_context.shape)}"
        )

    def _next_rollout_proprio(
        self,
        prop_current: Tensor | None,
        prop_seed: Tensor | None,
        prop_pred: Tensor | None,
        f_prop_pred: Tensor | None,
        oracle_embedding: Tensor | None,
        use_ground_truth_proprio: Tensor | None,
        teacher_force_mask: Tensor | None,
        step: int,
    ) -> Tensor | None:
        if self.rollout_prop_projection is None or prop_current is None:
            return prop_current

        # encode-oracle eval override (any mode): feed encode(true future raw window)
        # directly as the conditioning — the clean ceiling for the aligned scheme.
        if oracle_embedding is not None:
            return oracle_embedding[:, step]

        # how proprio conditioning evolves during rollout (steps >=1)
        mode = self.cfg.rollout_proprio_mode
        if mode == "freeze":
            return prop_current                      # hold the t=0 encode(raw) seed
        if mode == "zero":
            return torch.zeros_like(prop_current)    # proprio influences step 0 only
        if mode == "aligned":
            # ẑ_prop = seed + Δ(h_t) from next_proprio_head (see NextProprioHead)
            return prop_pred if prop_pred is not None else prop_seed

        pred_prop = (
            self.rollout_prop_projection(f_prop_pred)
            if f_prop_pred is not None
            else prop_current
        )
        if use_ground_truth_proprio is None:
            return pred_prop

        gt_step = use_ground_truth_proprio[:, step]
        if gt_step.shape[-1] == self.cfg.proprio_feature_dim:
            # NADA-style shift noise (training only): teacher-forced steps otherwise
            # condition on error-free GT, while deployment conditioning carries the
            # model's own free-running error.
            if self.training and self.prop_noise_sigma is not None:
                sigma = self.prop_noise_sigma[min(step, self.prop_noise_sigma.shape[0] - 1)]
                gt_step = gt_step + torch.randn_like(gt_step) * sigma
            gt_prop = self.rollout_prop_projection(gt_step)
        elif gt_step.shape[-1] == self.cfg.d_prop:
            gt_prop = gt_step
        else:
            raise ValueError(
                "use_ground_truth_proprio last dim must be "
                f"{self.cfg.proprio_feature_dim} or {self.cfg.d_prop}, got {gt_step.shape[-1]}"
            )

        if teacher_force_mask is None:
            return gt_prop

        mask_step = teacher_force_mask[:, step] if teacher_force_mask.ndim == 2 else teacher_force_mask[step]
        mask_step = mask_step.to(dtype=gt_prop.dtype, device=gt_prop.device)
        while mask_step.ndim < gt_prop.ndim:
            mask_step = mask_step.unsqueeze(-1)
        return mask_step * gt_prop + (1.0 - mask_step) * pred_prop

    def _next_rollout_visual(
        self,
        z_vis_pred: Tensor,
        use_ground_truth_visual: Tensor | None,
        teacher_force_mask: Tensor | None,
        step: int,
    ) -> Tensor:
        if use_ground_truth_visual is None:
            return z_vis_pred

        gt_vis = use_ground_truth_visual[:, step]
        if teacher_force_mask is None:
            return gt_vis

        mask_step = teacher_force_mask[:, step] if teacher_force_mask.ndim == 2 else teacher_force_mask[step]
        mask_step = mask_step.to(dtype=gt_vis.dtype, device=gt_vis.device)
        while mask_step.ndim < gt_vis.ndim:
            mask_step = mask_step.unsqueeze(-1)
        return mask_step * gt_vis + (1.0 - mask_step) * z_vis_pred
