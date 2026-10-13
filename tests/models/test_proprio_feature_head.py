from __future__ import annotations

import torch
import pytest

from feelwm.models.heads.proprio_feature_head import (
    ProprioFeatureHead,
    ProprioFeatureHeadConfig,
    PROPRIO_FEATURE_DIM,
)


def test_full_model_mode():
    """Full model path: patch features [B, N, D_vis] → pool → [B, 5]."""
    cfg = ProprioFeatureHeadConfig(d_in=384, pool_patches=True)
    head = ProprioFeatureHead(cfg)
    x = torch.randn(2, 196, 384)  # [B, N_patch, D_vis]
    y = head(x)
    assert y.shape == (2, PROPRIO_FEATURE_DIM), f"Expected (2, 5), got {y.shape}"


def test_baseline_1a_mode():
    """Baseline 1a path: proprio encoding [B, D_prop] (no patch dim) → [B, 5]."""
    cfg = ProprioFeatureHeadConfig(d_in=128, pool_patches=False)
    head = ProprioFeatureHead(cfg)
    x = torch.randn(4, 128)  # [B, D_prop]
    y = head(x)
    assert y.shape == (4, PROPRIO_FEATURE_DIM)


def test_gradient_flow_full_model():
    cfg = ProprioFeatureHeadConfig(d_in=384, pool_patches=True)
    head = ProprioFeatureHead(cfg)
    x = torch.randn(2, 196, 384)
    head(x).sum().backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.isfinite().all(), f"Non-finite gradient for {name}"


def test_gradient_flow_baseline():
    cfg = ProprioFeatureHeadConfig(d_in=128, pool_patches=False)
    head = ProprioFeatureHead(cfg)
    x = torch.randn(4, 128)
    head(x).sum().backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_no_output_activation():
    """Output must be unbounded (no sigmoid/tanh at the end)."""
    cfg = ProprioFeatureHeadConfig(d_in=128, pool_patches=False)
    head = ProprioFeatureHead(cfg)
    # Large-magnitude inputs would saturate to ±1 if a bounding activation existed
    x = torch.randn(200, 128) * 20
    y = head(x)
    assert y.abs().max().item() > 1.0, "Output appears saturated — unexpected output activation"


def test_output_dim_fixed_across_input_sizes():
    """Output is always 5-dim regardless of d_in."""
    for d_in in [64, 128, 256, 384, 768]:
        head = ProprioFeatureHead(ProprioFeatureHeadConfig(d_in=d_in, pool_patches=False))
        y = head(torch.randn(2, d_in))
        assert y.shape == (2, 5), f"Failed for d_in={d_in}: got {y.shape}"


def test_baseline_1a_end_to_end():
    """Baseline 1a pipeline: ProprioEncoder → squeeze T → ProprioFeatureHead."""
    from feelwm.models.encoders.proprio_encoder import ProprioEncoder, ProprioEncoderConfig

    enc = ProprioEncoder(ProprioEncoderConfig(d_prop=128))
    head = ProprioFeatureHead(ProprioFeatureHeadConfig(d_in=128, pool_patches=False))

    window = torch.randn(4, 1, 50, 14)  # [B=4, T=1, T_win=50, d_raw=14]
    f_prop = enc(window).squeeze(1)     # [B, D_prop]
    out = head(f_prop)                  # [B, 5]
    assert out.shape == (4, 5)

    out.sum().backward()
    for name, p in list(enc.named_parameters()) + list(head.named_parameters()):
        assert p.grad is not None, f"No gradient for {name}"
