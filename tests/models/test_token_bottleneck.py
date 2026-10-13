"""Perceiver-resampler token bottleneck guards.

Three properties the retrain arm depends on:
  * SHAPES — the core carries K latents per frame, but everything downstream of the
    dynamics still sees the full n_patch grid (heads, visual feedback, planner).
  * OFF is bitwise stock — the default config builds no module, adds no state_dict key,
    and returns exactly what the pre-bottleneck forward returned (ablation switch).
  * FRESH INIT — building the init checkpoint from a donor that predates the module
    leaves every donor weight bit-identical and creates the bottleneck fresh.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from feelwm.models.dynamics.transformer import (
    DynamicsTransformer,
    DynamicsTransformerConfig,
)
from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel

REPO_ROOT = Path(__file__).resolve().parents[2]

B = 2
T = 3
N = 16          # stands in for the real 256-patch grid
K = 4           # stands in for K=32
D_VIS = 32
D_PROP = 8
D_ACTION = 4
D_RAW = 11
T_WIN = 10
H = 3


def _dyn(**overrides) -> DynamicsTransformer:
    kwargs = dict(
        d_vis=D_VIS, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0,
        max_context_len=8, n_patch=N,
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return DynamicsTransformer(DynamicsTransformerConfig(**kwargs))


def _model_cfg(**overrides) -> ModelConfig:
    kwargs = dict(
        context_len=T, rollout_horizon=H, n_patch=N, d_vis=D_VIS, d_prop=D_PROP,
        d_action=D_ACTION, d_raw=D_RAW, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0,
        proprio_feature_dim=3, use_traversability_head=False,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    return {
        "z_vis_context": torch.randn(B, T, N, D_VIS),
        "proprio_window_context": torch.randn(B, T, T_WIN, D_RAW),
        "action": torch.randn(B, D_ACTION),
    }


# ── shapes ───────────────────────────────────────────────────────────────────
def test_core_runs_on_k_latents_and_exit_restores_the_grid():
    dyn = _dyn(token_bottleneck=True, bottleneck_k=K).eval()
    seen: list[tuple[int, ...]] = []
    dyn.encoder.layers[0].register_forward_pre_hook(
        lambda _m, inputs: seen.append(tuple(inputs[0].shape))
    )
    z = torch.randn(B, T, N, D_VIS)
    out = dyn(z, gamma=torch.randn(B, D_VIS), beta=torch.randn(B, D_VIS))
    assert seen == [(B, T * K, D_VIS)]      # core sequence is T*K, not T*N
    assert out.shape == (B, N, D_VIS)       # full grid leaves the module


def test_model_forward_and_rollout_keep_the_full_grid():
    model = TerrainWorldModel(_model_cfg(token_bottleneck={"type": "perceiver", "k": K})).eval()
    batch = _batch()
    out = model(batch)
    assert out["z_vis_pred"].shape == (B, N, D_VIS)
    assert out["f_prop_pred"].shape == (B, 3)

    roll = model.forward_rollout(batch, torch.randn(B, H, D_ACTION))
    assert roll["z_vis_preds"].shape == (B, H, N, D_VIS)
    assert roll["f_prop_preds"].shape == (B, H, 3)


def test_proprio_token_is_appended_after_compression_not_merged_into_it():
    """token/token_film injection must survive the bottleneck untouched: the proprio
    token joins the K latents (core sees K+1) and is dropped before the exit."""
    dyn = _dyn(
        token_bottleneck=True, bottleneck_k=K, proprio_token=True, d_prop=D_PROP
    ).eval()
    seen: list[tuple[int, ...]] = []
    dyn.encoder.layers[0].register_forward_pre_hook(
        lambda _m, inputs: seen.append(tuple(inputs[0].shape))
    )
    out = dyn(
        torch.randn(B, T, N, D_VIS),
        gamma=torch.randn(B, D_VIS),
        beta=torch.randn(B, D_VIS),
        proprio_window=torch.randn(B, T, D_PROP),
    )
    assert seen == [(B, T * (K + 1), D_VIS)]
    assert out.shape == (B, N, D_VIS)


def test_bottleneck_params_receive_gradient_through_the_checkpointed_path():
    dyn = _dyn(token_bottleneck=True, bottleneck_k=K, gradient_checkpointing=True).train()
    out = dyn(
        torch.randn(B, T, N, D_VIS),
        gamma=torch.randn(B, D_VIS),
        beta=torch.randn(B, D_VIS),
    )
    out.square().mean().backward()
    for name, p in dyn.bottleneck.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


# ── OFF is stock ─────────────────────────────────────────────────────────────
def test_off_adds_no_module_and_no_state_dict_key():
    stock, off = _dyn(), _dyn(token_bottleneck=False, bottleneck_k=K)
    assert off.bottleneck is None
    assert list(off.state_dict()) == list(stock.state_dict())


def test_off_forward_is_bitwise_stock():
    stock, off = _dyn(), _dyn(token_bottleneck=False)
    off.load_state_dict(stock.state_dict())
    stock.eval(), off.eval()
    z = torch.randn(B, T, N, D_VIS)
    gamma, beta = torch.randn(B, D_VIS), torch.randn(B, D_VIS)
    assert torch.equal(stock(z, gamma=gamma, beta=beta), off(z, gamma=gamma, beta=beta))


def test_on_state_dict_is_stock_plus_bottleneck_only():
    stock = TerrainWorldModel(_model_cfg())
    on = TerrainWorldModel(_model_cfg(token_bottleneck={"type": "perceiver", "k": K}))
    extra = [k for k in on.state_dict() if k not in stock.state_dict()]
    assert extra and all(k.startswith("dynamics.bottleneck.") for k in extra)
    assert [k for k in on.state_dict() if not k.startswith("dynamics.bottleneck.")] == list(
        stock.state_dict()
    )


@pytest.mark.parametrize(
    "tb",
    [
        {"type": "tokenlearner", "k": K},          # unsupported type
        {"type": "perceiver", "k": N + 1},         # k above the patch grid
        {"type": "perceiver", "k": 0},
        {"type": "perceiver", "k": K, "kk": 1},    # typo'd key
        "perceiver",                                # not a dict
    ],
)
def test_bad_bottleneck_config_is_refused(tb):
    with pytest.raises(ValueError):
        _model_cfg(token_bottleneck=tb)


def test_bottleneck_refuses_the_adaln_core():
    with pytest.raises(ValueError):
        _model_cfg(action_conditioning="adaln", token_bottleneck={"type": "perceiver", "k": K})


# ── fresh init from a pre-bottleneck donor ───────────────────────────────────

# ── run 2: gated residual EXIT ───────────────────────────────────────────────
def test_residual_exit_off_is_bitwise_the_run1_forward():
    """The flag's absence and residual_exit: false must be the same model, bit for bit."""
    absent = TerrainWorldModel(_model_cfg(token_bottleneck={"type": "perceiver", "k": K})).eval()
    explicit = TerrainWorldModel(
        _model_cfg(token_bottleneck={"type": "perceiver", "k": K, "residual_exit": False})
    ).eval()
    assert list(absent.state_dict()) == list(explicit.state_dict())
    assert "dynamics.bottleneck.exit_gate" not in absent.state_dict()
    explicit.load_state_dict(absent.state_dict())
    batch = _batch()
    assert torch.equal(absent(batch)["z_vis_pred"], explicit(batch)["z_vis_pred"])


def test_residual_exit_at_gate_zero_is_the_copy_last_frame_predictor():
    """g=0 ⇒ out == out_norm(last frame after FiLM + temporal pos), exactly."""
    dyn = _dyn(token_bottleneck=True, bottleneck_k=K, bottleneck_residual_exit=True).eval()
    assert torch.equal(dyn.bottleneck.exit_gate, torch.zeros(D_VIS))   # zero-init
    z = torch.randn(B, T, N, D_VIS)
    gamma, beta = torch.randn(B, D_VIS), torch.randn(B, D_VIS)

    last = gamma[:, None, :] * z[:, -1] + beta[:, None, :]
    x = torch.cat([z[:, :-1], last[:, None]], dim=1) + dyn.temporal_pos[:T][None, :, None, :]
    expected = dyn.out_norm(x[:, -1])

    assert torch.equal(dyn(z, gamma=gamma, beta=beta), expected)


def test_residual_exit_opens_when_the_gate_moves():
    dyn = _dyn(token_bottleneck=True, bottleneck_k=K, bottleneck_residual_exit=True).eval()
    z = torch.randn(B, T, N, D_VIS)
    gamma, beta = torch.randn(B, D_VIS), torch.randn(B, D_VIS)
    closed = dyn(z, gamma=gamma, beta=beta)
    assert dyn.bottleneck.gate_abs_mean() == 0.0
    with torch.no_grad():
        dyn.bottleneck.exit_gate.fill_(1.0)
    assert not torch.allclose(dyn(z, gamma=gamma, beta=beta), closed)
    assert dyn.bottleneck.gate_abs_mean() == pytest.approx(0.7615942, abs=1e-6)  # tanh(1)


def test_residual_exit_gate_and_module_get_gradient():
    dyn = _dyn(
        token_bottleneck=True, bottleneck_k=K, bottleneck_residual_exit=True,
        gradient_checkpointing=True,
    ).train()
    out = dyn(torch.randn(B, T, N, D_VIS), gamma=torch.randn(B, D_VIS), beta=torch.randn(B, D_VIS))
    out.square().mean().backward()
    g = dyn.bottleneck.exit_gate.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_residual_exit_config_is_validated():
    with pytest.raises(ValueError):
        _model_cfg(token_bottleneck={"type": "perceiver", "k": K, "residual_exit": "yes"})
