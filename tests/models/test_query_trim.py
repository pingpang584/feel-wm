"""Last-layer query-trim equivalence + noprop grad-coverage guards.

The trimmed final layer (DynamicsTransformer._trimmed_last_layer) must equal the
untrimmed stack sliced to the last frame, on every execution path the trainer uses
(eval, train + gradient checkpointing, token injection). The grad-coverage test
pins the exact orphan-parameter set of the no-proprio-input (noprop) arm, which licenses
find_unused_parameters=False + freeze in FullModelTrainer.
"""
from __future__ import annotations

import torch

from feelwm.models.dynamics.transformer import (
    DynamicsTransformer,
    DynamicsTransformerConfig,
)
from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel
from feelwm.training.curriculum import CurriculumStage
from feelwm.training.losses import LossConfig, compute_total_loss


B = 2
T = 4
N = 16
D_VIS = 64
D_PROP = 16


def _reference_forward(model: DynamicsTransformer, z_vis, gamma, beta, proprio_window=None):
    """Pre-trim behaviour: every layer over all T*M tokens, then slice."""
    cfg = model.cfg
    B_, T_, N_, D = z_vis.shape
    last = gamma[:, None, :] * z_vis[:, -1] + beta[:, None, :]
    x = torch.cat([z_vis[:, :-1], last[:, None]], dim=1)
    x = x + model.temporal_pos[:T_][None, :, None, :]
    use_tokens = cfg.proprio_token and proprio_window is not None
    if use_tokens:
        content = model.proprio_token_proj(proprio_window)
        if cfg.proprio_token_gate:
            content = torch.tanh(model.proprio_gate) * content
        tok = content + model.proprio_type_emb[None, None, :]
        tok = tok + model.temporal_pos[:T_][None, :, :]
        x = torch.cat([x, tok[:, :, None, :]], dim=2)
        M = N_ + 1
    else:
        M = N_
    x = x.reshape(B_, T_ * M, D)
    mask = model._causal_time_mask(T_, M, z_vis.device)
    for layer in model.encoder.layers:
        x = layer(x, src_mask=mask)
    x = model.out_norm(x)
    if use_tokens:
        return x.reshape(B_, T_, M, D)[:, -1, :N_, :]
    return x[:, -N_:, :]


def _model(**overrides) -> DynamicsTransformer:
    kwargs = dict(
        d_vis=D_VIS,
        n_layers=3,
        n_heads=4,
        dropout=0.0,
        max_context_len=T + 1,
    )
    kwargs.update(overrides)
    return DynamicsTransformer(DynamicsTransformerConfig(**kwargs))


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, T, N, D_VIS, generator=g)
    gamma = 1.0 + 0.1 * torch.randn(B, D_VIS, generator=g)
    beta = 0.1 * torch.randn(B, D_VIS, generator=g)
    return z, gamma, beta


def test_trim_matches_reference_eval() -> None:
    model = _model().eval()
    z, gamma, beta = _inputs()
    with torch.no_grad():
        out = model(z, gamma=gamma, beta=beta)
        ref = _reference_forward(model, z, gamma, beta)
    assert out.shape == (B, N, D_VIS)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_trim_matches_reference_train_checkpointed() -> None:
    model = _model(gradient_checkpointing=True).train()
    z, gamma, beta = _inputs(seed=1)

    out = model(z, gamma=gamma, beta=beta)
    (out**2).sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    assert all(g is not None for g in grads.values()), "trim must not orphan any param"

    model.zero_grad()
    ref = _reference_forward(model, z, gamma, beta)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)
    (ref**2).sum().backward()
    for name, p in model.named_parameters():
        assert torch.allclose(grads[name], p.grad, rtol=1e-4, atol=1e-6), name


def test_trim_matches_reference_token_mode() -> None:
    model = _model(proprio_token=True, d_prop=D_PROP, proprio_token_gate=True).eval()
    # zero-init token projection/gate would hide token-path bugs — randomize them
    with torch.no_grad():
        model.proprio_token_proj.weight.normal_()
        model.proprio_token_proj.bias.normal_()
        model.proprio_gate.normal_()
    z, gamma, beta = _inputs(seed=2)
    window = torch.randn(B, T, D_PROP)
    with torch.no_grad():
        out = model(z, gamma=gamma, beta=beta, proprio_window=window)
        ref = _reference_forward(model, z, gamma, beta, proprio_window=window)
    assert out.shape == (B, N, D_VIS)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_noprop_orphan_params_are_exactly_film_proprio_proj() -> None:
    """Licenses find_unused_parameters=False for the noprop arm (trainer freezes
    film.proprio_proj): after a FULL-stage rollout loss backward, those two tensors
    must be the ONLY parameters without grad."""
    cfg = ModelConfig(
        context_len=3,
        rollout_horizon=8,
        n_patch=8,
        d_vis=D_VIS,
        d_prop=D_PROP,
        d_action=4,
        d_raw=13,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        use_proprio_input=False,
        use_proprio_feature_head=True,
        use_traversability_head=True,
        proprio_feature_dim=4,
        traversability_input_mode="parallel",
        gradient_checkpointing=True,
    )
    model = TerrainWorldModel(cfg).train()
    H = cfg.rollout_horizon
    batch = {
        "z_vis_context": torch.randn(B, cfg.context_len + 1, cfg.n_patch, cfg.d_vis),
        "action_sequence": torch.randn(B, H, cfg.d_action),
        "z_vis_target": torch.randn(B, H, cfg.n_patch, cfg.d_vis),
        "f_prop_target": torch.randn(B, H, cfg.proprio_feature_dim),
        "binary_label_target": torch.randint(0, 2, (B, H)).float(),
        "binary_valid_target_mask": torch.ones(B, H, dtype=torch.bool),
        "valid_target_mask": torch.ones(B, H, dtype=torch.bool),
        "proprio_horizon_mask": torch.tensor([[True] * 4 + [False] * 4] * B),
    }
    preds = model(
        batch,
        batch["action_sequence"],
        use_ground_truth_visual=batch["z_vis_target"],
        use_ground_truth_proprio=batch["f_prop_target"],
        teacher_force_mask=(torch.rand(B, H) < 0.5).float(),
    )
    loss, _ = compute_total_loss(
        preds=preds,
        targets=batch,
        cfg=LossConfig(
            channel_names=["roughness", "slip_ratio", "angular_instability", "slope_power"],
            proprio_weights=[1.0, 1.0, 1.0, 1.0],
        ),
        stage=CurriculumStage.FULL,
        pos_weight=torch.tensor([3.0]),
    )
    loss.backward()
    orphans = sorted(n for n, p in model.named_parameters() if p.grad is None)
    assert orphans == ["film.proprio_proj.bias", "film.proprio_proj.weight"]


def test_freeze_mode_orphan_params_are_exactly_rollout_prop_projection() -> None:
    """Licenses find_unused_parameters=False for the freeze ablation arm (trainer freezes
    rollout_prop_projection): with the proprio input and both heads ON, a FULL-stage
    rollout loss backward must leave that Linear's two tensors as the ONLY parameters
    without grad. Teacher forcing is on, because freeze must bypass the GT projection
    call site too — if it did not, this Linear would receive grad and the trainer's
    freeze would silently delete a real gradient path."""
    cfg = ModelConfig(
        context_len=3,
        rollout_horizon=8,
        n_patch=8,
        d_vis=D_VIS,
        d_prop=D_PROP,
        d_action=4,
        d_raw=13,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        use_proprio_input=True,
        use_proprio_feature_head=True,
        use_traversability_head=True,
        proprio_feature_dim=4,
        traversability_input_mode="parallel",
        rollout_proprio_mode="freeze",
        gradient_checkpointing=True,
    )
    model = TerrainWorldModel(cfg).train()
    H = cfg.rollout_horizon
    batch = {
        "z_vis_context": torch.randn(B, cfg.context_len + 1, cfg.n_patch, cfg.d_vis),
        "proprio_window_context": torch.randn(B, cfg.context_len + 1, 50, cfg.d_raw),
        "action_sequence": torch.randn(B, H, cfg.d_action),
        "z_vis_target": torch.randn(B, H, cfg.n_patch, cfg.d_vis),
        "f_prop_target": torch.randn(B, H, cfg.proprio_feature_dim),
        "binary_label_target": torch.randint(0, 2, (B, H)).float(),
        "binary_valid_target_mask": torch.ones(B, H, dtype=torch.bool),
        "valid_target_mask": torch.ones(B, H, dtype=torch.bool),
        "proprio_horizon_mask": torch.tensor([[True] * 4 + [False] * 4] * B),
    }
    preds = model(
        batch,
        batch["action_sequence"],
        use_ground_truth_visual=batch["z_vis_target"],
        use_ground_truth_proprio=batch["f_prop_target"],
        teacher_force_mask=(torch.rand(B, H) < 0.5).float(),
    )
    loss, _ = compute_total_loss(
        preds=preds,
        targets=batch,
        cfg=LossConfig(
            channel_names=["roughness", "slip_ratio", "angular_instability", "slope_power"],
            proprio_weights=[1.0, 1.0, 1.0, 1.0],
        ),
        stage=CurriculumStage.FULL,
        pos_weight=torch.tensor([3.0]),
    )
    loss.backward()
    orphans = sorted(n for n, p in model.named_parameters() if p.grad is None)
    assert orphans == [
        "rollout_prop_projection.bias",
        "rollout_prop_projection.weight",
    ]


def test_rollout_prop_projection_is_used_in_feedback() -> None:
    """The control for the test above: on the DEFAULT feedback mode the same Linear DOES
    receive grad. Without this, a bug that made the projection dead in every mode would
    pass the freeze test and quietly turn the ON arm into the freeze arm."""
    cfg = ModelConfig(
        context_len=3,
        rollout_horizon=8,
        n_patch=8,
        d_vis=D_VIS,
        d_prop=D_PROP,
        d_action=4,
        d_raw=13,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        use_proprio_input=True,
        use_proprio_feature_head=True,
        use_traversability_head=True,
        proprio_feature_dim=4,
        traversability_input_mode="parallel",
        rollout_proprio_mode="feedback",
        gradient_checkpointing=True,
    )
    model = TerrainWorldModel(cfg).train()
    H = cfg.rollout_horizon
    batch = {
        "z_vis_context": torch.randn(B, cfg.context_len + 1, cfg.n_patch, cfg.d_vis),
        "proprio_window_context": torch.randn(B, cfg.context_len + 1, 50, cfg.d_raw),
        "action_sequence": torch.randn(B, H, cfg.d_action),
        "z_vis_target": torch.randn(B, H, cfg.n_patch, cfg.d_vis),
        "f_prop_target": torch.randn(B, H, cfg.proprio_feature_dim),
        "binary_label_target": torch.randint(0, 2, (B, H)).float(),
        "binary_valid_target_mask": torch.ones(B, H, dtype=torch.bool),
        "valid_target_mask": torch.ones(B, H, dtype=torch.bool),
        "proprio_horizon_mask": torch.tensor([[True] * 4 + [False] * 4] * B),
    }
    preds = model(batch, batch["action_sequence"])
    loss, _ = compute_total_loss(
        preds=preds,
        targets=batch,
        cfg=LossConfig(
            channel_names=["roughness", "slip_ratio", "angular_instability", "slope_power"],
            proprio_weights=[1.0, 1.0, 1.0, 1.0],
        ),
        stage=CurriculumStage.FULL,
        pos_weight=torch.tensor([3.0]),
    )
    loss.backward()
    orphans = sorted(n for n, p in model.named_parameters() if p.grad is None)
    assert orphans == []
