from __future__ import annotations

import torch
import pytest

from feelwm.models.encoders.proprio_encoder import ProprioEncoder, ProprioEncoderConfig


def test_output_shape_default():
    enc = ProprioEncoder()
    x = torch.randn(2, 4, 50, 14)  # [B=2, T=4, T_win=50, d_raw=14]
    y = enc(x)
    assert y.shape == (2, 4, 128), f"Expected (2, 4, 128), got {y.shape}"


def test_output_shape_custom_config():
    cfg = ProprioEncoderConfig(d_raw=14, d_prop=64, n_conv_layers=2, kernel_size=5)
    enc = ProprioEncoder(cfg)
    x = torch.randn(1, 3, 50, 14)
    y = enc(x)
    assert y.shape == (1, 3, 64)


def test_single_frame_baseline_scenario():
    """Baseline 1a uses T=1 (single anchor frame)."""
    enc = ProprioEncoder()
    x = torch.randn(4, 1, 50, 14)  # [B=4, T=1, T_win=50, d_raw=14]
    y = enc(x)
    assert y.shape == (4, 1, 128)


def test_gradient_flow():
    enc = ProprioEncoder()
    x = torch.randn(2, 4, 50, 14)
    y = enc(x)
    y.sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.isfinite().all(), f"Non-finite gradient for {name}"


def test_output_is_finite():
    enc = ProprioEncoder()
    x = torch.randn(2, 4, 50, 14)
    y = enc(x)
    assert y.isfinite().all(), "Output contains NaN or Inf"


def test_batch_independence():
    """Each sample in the batch should be processed independently."""
    enc = ProprioEncoder()
    enc.eval()
    x = torch.randn(4, 4, 50, 14)
    y_full = enc(x)
    y_single = enc(x[:1])
    assert torch.allclose(y_full[:1], y_single, atol=1e-5)
