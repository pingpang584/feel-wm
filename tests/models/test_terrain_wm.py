from __future__ import annotations

import pytest
import torch

from feelwm.models.dynamics.film import FiLMConditioning, FiLMConditioningConfig
from feelwm.models.dynamics.transformer import (
    DynamicsTransformer,
    DynamicsTransformerConfig,
)
from feelwm.models.encoders.action_embedding import ActionEmbedding, ActionEmbeddingConfig
from feelwm.models.heads.traversability_head import (
    TraversabilityHead,
    TraversabilityHeadConfig,
)
from feelwm.models.heads.visual_head import VisualHead, VisualHeadConfig
from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel


B = 2
T = 4
N = 8
D_VIS = 32
D_PROP = 16
D_ACTION = 3
T_WIN = 10
D_RAW = 14
H = 3


def _cfg(**overrides) -> ModelConfig:
    kwargs = dict(
        context_len=T - 1,
        rollout_horizon=H,
        n_patch=N,
        d_vis=D_VIS,
        d_prop=D_PROP,
        d_action=D_ACTION,
        d_raw=D_RAW,
        n_layers=1,
        n_heads=4,
        ffn_mult=2,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "z_vis_context": torch.randn(B, T, N, D_VIS),
        "proprio_window_context": torch.randn(B, T, T_WIN, D_RAW),
        "action_context": torch.randn(B, T, D_ACTION),
    }


def _assert_param_grads(
    module: torch.nn.Module,
    skip_prefixes: tuple[str, ...] = (),
) -> None:
    for name, param in module.named_parameters():
        if any(name.startswith(prefix) for prefix in skip_prefixes):
            continue
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.isfinite().all(), f"Non-finite gradient for {name}"


def test_action_embedding_forward_and_grad() -> None:
    module = ActionEmbedding(ActionEmbeddingConfig(d_action=D_ACTION, d_vis=D_VIS, hidden_dim=24))
    out = module(torch.randn(B, D_ACTION))
    assert out.shape == (B, D_VIS)
    out.sum().backward()
    _assert_param_grads(module)


def test_film_conditioning_forward_and_grad() -> None:
    module = FiLMConditioning(FiLMConditioningConfig(d_vis=D_VIS, d_prop=D_PROP, hidden_dim=24))
    action_emb = torch.randn(B, D_VIS)
    proprio = torch.randn(B, D_PROP)
    gamma, beta = module(action_emb, proprio)
    assert gamma.shape == (B, D_VIS)
    assert beta.shape == (B, D_VIS)
    (gamma + beta).sum().backward()
    _assert_param_grads(module)


def test_film_conditioning_action_only_shape() -> None:
    module = FiLMConditioning(FiLMConditioningConfig(d_vis=D_VIS, d_prop=D_PROP, hidden_dim=24))
    gamma, beta = module(torch.randn(B, D_VIS), None)
    assert gamma.shape == (B, D_VIS)
    assert beta.shape == (B, D_VIS)


def test_dynamics_transformer_forward_and_grad() -> None:
    module = DynamicsTransformer(
        DynamicsTransformerConfig(
            d_vis=D_VIS,
            n_layers=1,
            n_heads=4,
            ffn_mult=2,
            dropout=0.0,
            max_context_len=T,
        )
    )
    z_vis = torch.randn(B, T, N, D_VIS)
    gamma = torch.randn(B, D_VIS)
    beta = torch.randn(B, D_VIS)
    out = module(z_vis, gamma=gamma, beta=beta)
    assert out.shape == (B, N, D_VIS)
    out.sum().backward()
    _assert_param_grads(module)


def test_visual_head_forward_and_grad() -> None:
    module = VisualHead(VisualHeadConfig(d_vis=D_VIS, hidden_dim=24))
    out = module(torch.randn(B, N, D_VIS))
    assert out.shape == (B, N, D_VIS)
    out.sum().backward()
    _assert_param_grads(module)


def test_traversability_head_forward_and_grad() -> None:
    module = TraversabilityHead(TraversabilityHeadConfig(d_in=5, hidden_dim=12))
    f_prop = torch.randn(B, 5)
    logit = module(f_prop)
    prob = module.predict_proba(f_prop)
    assert logit.shape == (B, 1)
    assert prob.shape == (B, 1)
    assert torch.all((prob >= 0.0) & (prob <= 1.0))
    logit.sum().backward()
    _assert_param_grads(module)


def test_terrain_world_model_forward_all_heads() -> None:
    model = TerrainWorldModel(_cfg())
    out = model(_batch())
    assert out["z_vis_pred"].shape == (B, N, D_VIS)
    assert out["f_prop_pred"].shape == (B, 5)
    assert out["c_pred_logit"].shape == (B, 1)
    assert out["c_pred"].shape == (B, 1)


@pytest.mark.parametrize(
    "mode, expected_d_in",
    [("proprio", 5), ("parallel", D_VIS), ("hybrid", D_VIS + 5)],
)
def test_traversability_input_mode_shapes_and_grad(mode, expected_d_in) -> None:
    model = TerrainWorldModel(_cfg(traversability_input_mode=mode))
    assert model.traversability_head.net[0].in_features == expected_d_in
    out = model(_batch())
    assert out["c_pred_logit"].shape == (B, 1)
    assert out["c_pred"].shape == (B, 1)
    # cost gradient must reach the trunk; in proprio/hybrid it also reaches the proprio head.
    out["c_pred_logit"].sum().backward()
    assert model.traversability_head.net[0].weight.grad is not None
    if mode in {"proprio", "hybrid"}:
        assert model.proprio_feature_head.mlp[0].weight.grad is not None


@pytest.mark.parametrize("mode", ["proprio", "hybrid"])
def test_stop_grad_blocks_cost_into_proprio_head(mode) -> None:
    # With stop-grad on, the cost gradient must NOT reach the proprio feature head,
    # while the trav head itself still receives gradient (cost stays trainable).
    model = TerrainWorldModel(
        _cfg(traversability_input_mode=mode, traversability_stop_grad_proprio=True)
    )
    out = model(_batch())
    out["c_pred_logit"].sum().backward()
    assert model.traversability_head.net[0].weight.grad is not None
    assert model.proprio_feature_head.mlp[0].weight.grad is None


def test_parallel_mode_predicts_cost_without_proprio_head() -> None:
    model = TerrainWorldModel(
        _cfg(traversability_input_mode="parallel", use_proprio_feature_head=False)
    )
    out = model(_batch())
    assert out["f_prop_pred"] is None
    assert out["c_pred_logit"].shape == (B, 1)


def test_hybrid_mode_without_proprio_head_degrades_to_none() -> None:
    # hybrid consumes f_prop_pred; with the proprio head off, cost degrades to None
    # (graceful-degradation contract), it does not error.
    model = TerrainWorldModel(
        _cfg(traversability_input_mode="hybrid", use_proprio_feature_head=False)
    )
    out = model(_batch())
    assert out["f_prop_pred"] is None
    assert out["c_pred_logit"] is None


def test_terrain_world_model_forward_grad_flow() -> None:
    model = TerrainWorldModel(_cfg())
    out = model(_batch())
    loss = (
        out["z_vis_pred"].sum()
        + out["f_prop_pred"].sum()
        + out["c_pred_logit"].sum()
    )
    loss.backward()
    _assert_param_grads(model, skip_prefixes=("rollout_prop_projection.",))


def test_forward_accepts_single_action_key() -> None:
    model = TerrainWorldModel(_cfg())
    batch = _batch()
    batch["action"] = torch.randn(B, D_ACTION)
    del batch["action_context"]
    out = model(batch)
    assert out["z_vis_pred"].shape == (B, N, D_VIS)


def test_forward_rollout_shapes() -> None:
    model = TerrainWorldModel(_cfg())
    action_sequence = torch.randn(B, H, D_ACTION)
    out = model.forward_rollout(_batch(), action_sequence)
    assert out["z_vis_preds"].shape == (B, H, N, D_VIS)
    assert out["f_prop_preds"].shape == (B, H, 5)
    assert out["c_pred_logits"].shape == (B, H, 1)
    assert out["c_preds"].shape == (B, H, 1)


def test_forward_rollout_grad_flow_includes_prop_projection() -> None:
    model = TerrainWorldModel(_cfg())
    out = model.forward_rollout(_batch(), torch.randn(B, H, D_ACTION))
    loss = (
        out["z_vis_preds"].sum()
        + out["f_prop_preds"].sum()
        + out["c_pred_logits"].sum()
    )
    loss.backward()
    assert model.rollout_prop_projection is not None
    for name, param in model.rollout_prop_projection.named_parameters():
        assert param.grad is not None, f"No gradient for rollout_prop_projection.{name}"
        assert param.grad.isfinite().all()


def test_forward_rollout_with_teacher_force_interface() -> None:
    model = TerrainWorldModel(_cfg())
    action_sequence = torch.randn(B, H, D_ACTION)
    gt_prop = torch.randn(B, H, 5)
    teacher_mask = torch.tensor([[True, False, True], [False, True, False]])
    out = model.forward_rollout(
        _batch(),
        action_sequence,
        use_ground_truth_proprio=gt_prop,
        teacher_force_mask=teacher_mask,
    )
    assert out["z_vis_preds"].shape == (B, H, N, D_VIS)


def test_ablation_no_proprio_input() -> None:
    model = TerrainWorldModel(_cfg(use_proprio_input=False))
    assert model.proprio_encoder is None
    batch = _batch()
    del batch["proprio_window_context"]
    out = model(batch)
    assert out["z_vis_pred"].shape == (B, N, D_VIS)
    assert out["f_prop_pred"].shape == (B, 5)


def test_ablation_no_proprio_feature_head() -> None:
    model = TerrainWorldModel(_cfg(use_proprio_feature_head=False))
    assert model.proprio_feature_head is None
    out = model(_batch())
    assert out["z_vis_pred"].shape == (B, N, D_VIS)
    assert out["f_prop_pred"] is None
    assert out["c_pred_logit"] is None
    assert out["c_pred"] is None


def test_ablation_no_traversability_head() -> None:
    model = TerrainWorldModel(_cfg(use_traversability_head=False))
    assert model.traversability_head is None
    out = model(_batch())
    assert out["z_vis_pred"].shape == (B, N, D_VIS)
    assert out["f_prop_pred"].shape == (B, 5)
    assert out["c_pred_logit"] is None
    assert out["c_pred"] is None


def test_no_proprio_head_with_traversability_requested_returns_none() -> None:
    model = TerrainWorldModel(
        _cfg(use_proprio_feature_head=False, use_traversability_head=True)
    )
    assert model.traversability_head is not None
    out = model(_batch())
    assert out["f_prop_pred"] is None
    assert out["c_pred_logit"] is None
    assert out["c_pred"] is None


def test_concat_action_conditioning_explicitly_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="concat"):
        TerrainWorldModel(_cfg(action_conditioning="concat"))


def test_adaln_conditioning_forward_grad_flow_and_rollout() -> None:
    model = TerrainWorldModel(_cfg(action_conditioning="adaln"))
    assert not hasattr(model, "film")
    out = model(_batch())
    assert out["z_vis_pred"].shape == (B, N, D_VIS)
    loss = out["z_vis_pred"].sum() + out["f_prop_pred"].sum() + out["c_pred_logit"].sum()
    loss.backward()
    _assert_param_grads(model, skip_prefixes=("rollout_prop_projection.",))
    # At init the zero-init final AdaLN Linear gets gradient (cond feeds the modulation);
    # bottleneck layers are zero-grad until the final weight leaves zero (DiT-Zero). One
    # optimizer step + a second backward must then make the WHOLE modulation MLP live.
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    final_grads = [
        p.grad for n, p in model.dynamics.named_parameters()
        if n.endswith("adaLN.3.weight")
    ]
    assert final_grads and all(g.abs().sum() > 0 for g in final_grads)
    opt.step()
    opt.zero_grad()
    out2 = model(_batch())
    (out2["z_vis_pred"].sum() + out2["f_prop_pred"].sum()).backward()
    live = [p.grad for n, p in model.dynamics.named_parameters() if "adaLN" in n]
    assert all(g is not None and g.abs().sum() > 0 for g in live)
    roll = model.forward_rollout(_batch(), torch.randn(B, H, D_ACTION))
    assert roll["z_vis_preds"].shape == (B, H, N, D_VIS)


def test_adaln_zero_init_is_identity_on_last_frame() -> None:
    """Zero-init: at init the modulated (last-frame) tokens pass through each
    block unchanged (gate=0), so the pre-out-norm last frame equals its input."""
    model = TerrainWorldModel(_cfg(action_conditioning="adaln")).eval()
    dyn = model.dynamics
    z = torch.randn(B, T - 1, N, D_VIS)
    cond = torch.randn(B, D_VIS)
    Tctx = z.shape[1]
    x = (z + dyn.temporal_pos[:Tctx][None, :, None, :]).reshape(B, Tctx * N, D_VIS)
    amask = DynamicsTransformer._causal_time_mask(Tctx, N, z.device)
    lmask = dyn._last_frame_mask(Tctx, N, z.device, x.dtype)
    xb = x.clone()
    for blk in dyn.blocks:
        xb = blk(xb, cond, amask, lmask)
    assert torch.allclose(xb[:, -N:], x[:, -N:], atol=1e-6)


def test_visual_encoder_config_validation_accepts_declared_values() -> None:
    for value in ["dinov2", "dinov3", "vae"]:
        model = TerrainWorldModel(_cfg(visual_encoder=value))
        out = model(_batch())
        assert out["z_vis_pred"].shape == (B, N, D_VIS)


def test_invalid_config_values_raise() -> None:
    with pytest.raises(ValueError, match="action_conditioning"):
        _cfg(action_conditioning="bad")
    with pytest.raises(ValueError, match="visual_encoder"):
        _cfg(visual_encoder="bad")


@pytest.mark.parametrize("mode", ["proprio", "parallel", "hybrid"])
def test_reduced_proprio_feature_dim_shapes(mode: str) -> None:
    """ke_rate-dropped (4-channel) proprio path: heads emit the reduced dim and
    forward_rollout / scheduled-sampling accept 4-dim ground-truth proprio."""
    k = 4
    model = TerrainWorldModel(_cfg(proprio_feature_dim=k, traversability_input_mode=mode))
    out = model(_batch())
    assert out["f_prop_pred"].shape == (B, k)
    assert out["c_pred_logit"].shape == (B, 1)

    action_seq = torch.randn(B, H, D_ACTION)
    gt_prop = torch.randn(B, H, k)
    teacher = torch.ones(B, H)
    roll = model.forward_rollout(
        _batch(),
        action_seq,
        use_ground_truth_proprio=gt_prop,
        teacher_force_mask=teacher,
    )
    assert roll["f_prop_preds"].shape == (B, H, k)
    roll["f_prop_preds"].sum().backward()


def test_aligned_mode_builds_head_and_emits_alignment_outputs() -> None:
    model = TerrainWorldModel(_cfg(rollout_proprio_mode="aligned"))
    assert model.next_proprio_head is not None
    window_target = torch.randn(B, H, T_WIN, D_RAW)
    out = model.forward_rollout(
        _batch(), torch.randn(B, H, D_ACTION), proprio_window_target=window_target
    )
    assert out["z_prop_align_preds"].shape == (B, H, D_PROP)
    assert out["z_prop_align_target"].shape == (B, H, D_PROP)
    # target is stop-grad (detached encoder output)
    assert out["z_prop_align_target"].requires_grad is False


def test_non_aligned_modes_have_no_next_proprio_head() -> None:
    for mode in ["feedback", "freeze", "zero"]:
        model = TerrainWorldModel(_cfg(rollout_proprio_mode=mode))
        assert model.next_proprio_head is None
        out = model.forward_rollout(_batch(), torch.randn(B, H, D_ACTION))
        assert out["z_prop_align_preds"] is None


def test_aligned_residual_is_freeze_at_init() -> None:
    """Zero-init residual ⇒ ẑ_prop = seed for every step at init (== freeze conditioning)."""
    model = TerrainWorldModel(_cfg(rollout_proprio_mode="aligned"))
    out = model.forward_rollout(
        _batch(), torch.randn(B, H, D_ACTION),
        proprio_window_target=torch.randn(B, H, T_WIN, D_RAW),
    )
    preds = out["z_prop_align_preds"]  # [B, H, D_PROP]
    seed = preds[:, 0]
    for h in range(H):
        assert torch.allclose(preds[:, h], seed, atol=1e-6), f"step {h} departs from seed at init"


def test_aligned_alignment_loss_reduces_gap_and_flows_to_head() -> None:
    torch.manual_seed(0)
    model = TerrainWorldModel(_cfg(rollout_proprio_mode="aligned"))
    batch = _batch()
    action_sequence = torch.randn(B, H, D_ACTION)
    window_target = torch.randn(B, H, T_WIN, D_RAW)

    def gap() -> float:
        out = model.forward_rollout(batch, action_sequence, proprio_window_target=window_target)
        return float(((out["z_prop_align_preds"] - out["z_prop_align_target"]) ** 2).mean())

    g0 = gap()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(40):
        opt.zero_grad()
        out = model.forward_rollout(batch, action_sequence, proprio_window_target=window_target)
        loss = ((out["z_prop_align_preds"] - out["z_prop_align_target"]) ** 2).mean()
        loss.backward()
        # gradient reaches the new head
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.next_proprio_head.parameters())
        opt.step()
    assert gap() < g0


def test_encode_oracle_overrides_conditioning() -> None:
    """encode-oracle feeds encode(GT window) as conditioning; differs from the head path."""
    model = TerrainWorldModel(_cfg(rollout_proprio_mode="aligned"))
    batch = _batch()
    action_sequence = torch.randn(B, H, D_ACTION)
    window_target = torch.randn(B, H, T_WIN, D_RAW)
    base = model.forward_rollout(batch, action_sequence, proprio_window_target=window_target)
    oracle = model.forward_rollout(
        batch, action_sequence, proprio_window_target=window_target,
        oracle_proprio_embedding=True,
    )
    # step 0 conditioning is the seed in both (oracle only changes steps>=1) → identical,
    # but later steps diverge once the oracle embedding replaces the head prediction.
    assert torch.allclose(base["z_vis_preds"][:, 0], oracle["z_vis_preds"][:, 0], atol=1e-6)
    assert not torch.allclose(base["z_vis_preds"][:, -1], oracle["z_vis_preds"][:, -1])


def test_denormalize_feature_subset_matches_full_columns() -> None:
    """denormalize_feature with an index subset equals the matching columns of the
    full denormalization (and inverts ke_rate's signed_log1p only when present)."""
    import numpy as np

    from feelwm.data.normalize import NormStats, denormalize_feature

    stats = NormStats(
        feature_mean=np.arange(5, dtype=np.float32),
        feature_std=np.arange(1, 6, dtype=np.float32),
        window_mean=np.zeros(14, np.float32),
        window_std=np.ones(14, np.float32),
        action_mean=np.zeros(3, np.float32),
        action_std=np.ones(3, np.float32),
    )
    full = denormalize_feature(np.random.randn(7, 5).astype(np.float32), stats)
    arr = np.random.randn(7, 5).astype(np.float32)
    idx = [0, 1, 2, 4]  # drop ke_rate (index 3)
    sub = denormalize_feature(arr[:, idx], stats, idx)
    ref = denormalize_feature(arr, stats)[:, idx]
    np.testing.assert_allclose(sub, ref, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# proprio_injection ablation (film / token / token_film) — sanity
# ---------------------------------------------------------------------------


def test_proprio_token_transformer_sequence_shape_and_output() -> None:
    # (a) token mode runs the interleaved [N patches + 1 token] per frame and the
    #     causal mask is built over T*(N+1) positions; output is last-frame patches only.
    cfg = DynamicsTransformerConfig(
        d_vis=D_VIS, n_layers=1, n_heads=4, ffn_mult=2, dropout=0.0,
        max_context_len=T, proprio_token=True, d_prop=D_PROP,
    )
    module = DynamicsTransformer(cfg)
    z_vis = torch.randn(B, T, N, D_VIS)
    prop_window = torch.randn(B, T, D_PROP)
    out = module(z_vis, gamma=torch.randn(B, D_VIS), beta=torch.randn(B, D_VIS),
                 proprio_window=prop_window)
    assert out.shape == (B, N, D_VIS)
    # the interleaved sequence has T*(N+1) positions; mask is square over it.
    mask = DynamicsTransformer._causal_time_mask(T, N + 1, z_vis.device)
    assert mask.shape == (T * (N + 1), T * (N + 1))
    # zero-init projection: at init the token is proprio-independent (stable start).
    g0, b0 = torch.zeros(B, D_VIS), torch.zeros(B, D_VIS)
    assert torch.allclose(
        module(z_vis, gamma=g0, beta=b0, proprio_window=prop_window),
        module(z_vis, gamma=g0, beta=b0, proprio_window=prop_window + 5.0),
    )
    # once the projection is non-zero (i.e. trained), the token is actually wired in:
    # perturbing the proprio changes the last-frame output.
    torch.nn.init.normal_(module.proprio_token_proj.weight, std=0.5)
    out0 = module(z_vis, gamma=g0, beta=b0, proprio_window=prop_window)
    out2 = module(z_vis, gamma=g0, beta=b0, proprio_window=prop_window + 5.0)
    assert not torch.allclose(out2, out0)


def test_proprio_token_mask_is_causal() -> None:
    # (b) a proprio token at frame t must NOT be visible to any query at frame < t
    #     (proprio cannot leak into the past). Positions are ordered per frame as
    #     [N patches, 1 token]; the token of frame j sits at index j*(N+1) + N.
    M = N + 1
    mask = DynamicsTransformer._causal_time_mask(T, M, torch.device("cpu"))  # True == blocked
    for j in range(T):
        token_pos = j * M + N
        for q in range(T * M):
            q_frame = q // M
            if q_frame < j:
                assert mask[q, token_pos], f"frame {q_frame} query sees future proprio @frame {j}"
            else:
                assert not mask[q, token_pos]


def test_proprio_token_rollout_no_desync_with_the_feedback() -> None:
    # (c) at every rollout step the proprio token window's last entry must equal the
    #     FiLM/feedback conditioning vector prop_current that the same step uses.
    model = TerrainWorldModel(_cfg(proprio_injection="token")).eval()
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig = model._predict_from_context

    def spy(z, prop_current, action, prop_window=None):
        assert prop_window is not None
        seen.append((prop_current.detach().clone(), prop_window.detach().clone()))
        return orig(z, prop_current, action, prop_window=prop_window)

    model._predict_from_context = spy  # type: ignore[assignment]
    with torch.no_grad():
        model.forward_rollout(_batch(), torch.randn(B, H, D_ACTION))
    assert len(seen) == H
    for prop_current, prop_window in seen:
        assert prop_window.shape == (B, T, D_PROP)
        assert torch.allclose(prop_window[:, -1], prop_current, atol=1e-6)


def test_film_mode_unaffected_by_proprio_window_plumbing() -> None:
    # (d) regression guard: in the default "film" mode no proprio-token params exist
    #     (checkpoint-compatible with pre-change runs) and the transformer ignores any
    #     proprio_window passed, so behaviour is bit-identical to before.
    model = TerrainWorldModel(_cfg())  # proprio_injection defaults to "film"
    keys = dict(model.named_parameters())
    assert not any("proprio_token_proj" in k or "proprio_type_emb" in k for k in keys)

    module = DynamicsTransformer(
        DynamicsTransformerConfig(
            d_vis=D_VIS, n_layers=1, n_heads=4, ffn_mult=2, dropout=0.0,
            max_context_len=T, proprio_token=False,
        )
    ).eval()
    z_vis = torch.randn(B, T, N, D_VIS)
    gamma, beta = torch.randn(B, D_VIS), torch.randn(B, D_VIS)
    with torch.no_grad():
        a = module(z_vis, gamma=gamma, beta=beta)
        b = module(z_vis, gamma=gamma, beta=beta, proprio_window=torch.randn(B, T, D_PROP))
    assert torch.equal(a, b)


def test_token_film_mode_builds_both_paths_and_runs() -> None:
    model = TerrainWorldModel(_cfg(proprio_injection="token_film"))
    assert model._proprio_token and model._film_proprio
    assert hasattr(model.dynamics, "proprio_token_proj")
    out = model.forward_rollout(_batch(), torch.randn(B, H, D_ACTION))
    assert out["z_vis_preds"].shape == (B, H, N, D_VIS)
    out["z_vis_preds"].sum().backward()
    for name, p in model.dynamics.proprio_token_proj.named_parameters():
        assert p.grad is not None, f"no grad for proprio_token_proj.{name}"


def test_token_mode_leaves_film_proprio_proj_unused() -> None:
    # In pure "token" mode FiLM sees action only, so film.proprio_proj never runs and
    # receives no grad. This is WHY the trainer must set find_unused_parameters=True for
    # token mode (DDP otherwise crashes). token_film uses both paths -> all params used.
    def full_loss(out):  # exercise every head so only injection-specific params differ
        return (
            out["z_vis_preds"].sum()
            + out["f_prop_preds"].sum()
            + out["c_pred_logits"].sum()
        )

    model = TerrainWorldModel(_cfg(proprio_injection="token"))
    full_loss(model.forward_rollout(_batch(), torch.randn(B, H, D_ACTION))).backward()
    no_grad = {n for n, p in model.named_parameters() if p.requires_grad and p.grad is None}
    assert no_grad == {"film.proprio_proj.weight", "film.proprio_proj.bias"}, no_grad
    # the token projection itself IS trained.
    assert model.dynamics.proprio_token_proj.weight.grad is not None

    model2 = TerrainWorldModel(_cfg(proprio_injection="token_film"))
    full_loss(model2.forward_rollout(_batch(), torch.randn(B, H, D_ACTION))).backward()
    assert not [n for n, p in model2.named_parameters() if p.requires_grad and p.grad is None]


def test_proprio_token_gate_default_off_and_on() -> None:
    # default off: no gate param -> checkpoints trained pre-gate (P2/token_film) load fine.
    off = TerrainWorldModel(_cfg(proprio_injection="token"))
    assert not hasattr(off.dynamics, "proprio_gate")

    # on: zero-init gate param exists and is trained; rollout runs and backprops.
    on = TerrainWorldModel(_cfg(proprio_injection="token", proprio_token_gate=True))
    assert torch.equal(on.dynamics.proprio_gate, torch.zeros(D_VIS))
    out = on.forward_rollout(_batch(), torch.randn(B, H, D_ACTION))
    out["z_vis_preds"].sum().backward()
    assert on.dynamics.proprio_gate.grad is not None
    # gate is checkpoint-incompatible across the flag (extra param) — intended.
    assert set(dict(on.named_parameters())) - set(dict(off.named_parameters())) == {
        "dynamics.proprio_gate"
    }


def test_proprio_injection_validation() -> None:
    with pytest.raises(ValueError):
        _cfg(proprio_injection="bogus")
    with pytest.raises(ValueError):  # token requires film conditioning
        _cfg(proprio_injection="token", action_conditioning="adaln")
    with pytest.raises(ValueError):  # token requires proprio input
        _cfg(proprio_injection="token", use_proprio_input=False)


# --- NADA-style feedback-shift noise (rollout_proprio_noise_*) ---


def _noise_table_json(tmp_path, C: int = 4, H_table: int = 3, value: float = 0.5) -> str:
    import json as _json

    payload = {
        "channels": [f"ch{i}" for i in range(C)],
        "err_std": [[value] * C for _ in range(H_table)],
        "w1": [[value * 0.2] * C for _ in range(H_table)],
    }
    p = tmp_path / "shift.json"
    p.write_text(_json.dumps(payload))
    return str(p)


def test_prop_noise_off_by_default() -> None:
    model = TerrainWorldModel(_cfg())
    assert model.prop_noise_sigma is None


def test_prop_noise_table_loaded_scaled_and_nonpersistent(tmp_path) -> None:
    path = _noise_table_json(tmp_path, C=4, value=0.5)
    model = TerrainWorldModel(
        _cfg(
            proprio_feature_dim=4,
            rollout_proprio_noise_path=path,
            rollout_proprio_noise_scale=2.0,
        )
    )
    assert model.prop_noise_sigma is not None
    assert model.prop_noise_sigma.shape == (3, 4)
    assert torch.allclose(model.prop_noise_sigma, torch.full((3, 4), 1.0))
    # non-persistent buffer: checkpoints stay interchangeable with noise-off configs
    assert "prop_noise_sigma" not in model.state_dict()
    plain = TerrainWorldModel(_cfg(proprio_feature_dim=4))
    model.load_state_dict(plain.state_dict())  # strict load must pass


def test_prop_noise_applied_only_in_training_mode(tmp_path) -> None:
    path = _noise_table_json(tmp_path, C=4, value=5.0)  # large so the effect is visible
    model = TerrainWorldModel(
        _cfg(proprio_feature_dim=4, rollout_proprio_noise_path=path)
    )
    batch = _batch()
    actions = torch.randn(B, H, D_ACTION)
    gt_vis = torch.randn(B, H, N, D_VIS)
    gt_prop = torch.randn(B, H, 4)

    def _run() -> torch.Tensor:
        out = model.forward_rollout(
            batch,
            actions,
            use_ground_truth_visual=gt_vis,
            use_ground_truth_proprio=gt_prop,
        )
        return out["f_prop_preds"]

    model.eval()
    with torch.no_grad():
        a, b_run = _run(), _run()
    assert torch.equal(a, b_run)  # eval: no noise, deterministic

    model.train()
    with torch.no_grad():
        c, d = _run(), _run()
    # step-0 prediction is pre-feedback: unaffected by the noised conditioning
    assert torch.allclose(c[:, 0], d[:, 0])
    assert not torch.allclose(c[:, 1:], d[:, 1:])  # noised GT conditioning differs
    # and train-mode differs from eval-mode outputs beyond step 0
    assert not torch.allclose(a[:, 1:], c[:, 1:])


def test_prop_noise_config_validation(tmp_path) -> None:
    path = _noise_table_json(tmp_path, C=4)
    with pytest.raises(ValueError):  # unknown metric
        _cfg(rollout_proprio_noise_path=path, rollout_proprio_noise_metric="rmse")
    with pytest.raises(ValueError):  # noise targets the proprio input path
        _cfg(rollout_proprio_noise_path=path, use_proprio_input=False)
    with pytest.raises(ValueError):  # channel-count mismatch vs proprio_feature_dim
        TerrainWorldModel(
            _cfg(proprio_feature_dim=5, rollout_proprio_noise_path=path)
        )
