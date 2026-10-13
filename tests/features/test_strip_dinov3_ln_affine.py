"""tools/features/strip_dinov3_ln_affine.py — the loss-scale-parity cache transform.

This script rewrites feature caches that training then consumes, so a wrong result is
silent twice over: no shape changes, no NaNs, just a differently-scaled visual loss and a
run whose numbers look plausible. The tests pin the inverse itself (exactness, and that it
lands at the variance dinov2reg_v1 uses), the resume/atomicity behaviour, and the naming
tripwires. LayerNorm parameters are synthetic — the point is the arithmetic, not the
checkpoint, so nothing here downloads the gated DINOv3 weights.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

STRIP = importlib.import_module("tools.features.strip_dinov3_ln_affine")
SCRIPT = Path(STRIP.__file__)

N_FRAME, N_PATCH, D = 40, 16, 32


@pytest.fixture
def affine():
    """A deliberately awkward affine: w spans two orders of magnitude, as DINOv3's does
    (0.247 .. 11.19), which is what makes the float16 round trip worth measuring."""
    rng = np.random.default_rng(0)
    w = np.exp(rng.uniform(np.log(0.25), np.log(11.0), D)).astype(np.float32)
    b = rng.normal(0, 0.25, D).astype(np.float32)
    return w, b


def _write_cache(root: Path, name: str, w, b, *, n_frame: int = N_FRAME):
    """A cache holding EXACTLY what the encoder would emit: LN output put back through
    the affine. The stripped result therefore has a known ground truth."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (n_frame, N_PATCH, D)).astype(np.float32)
    x = (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)   # unit-variance LN output
    y = (x * w + b).astype(np.float16)
    d = root / name
    d.mkdir(parents=True)
    np.save(d / "features.npy", y)
    np.save(d / "features_valid.npy", np.ones(n_frame, bool))
    return x


def test_inverse_recovers_the_layernorm_output(tmp_path, affine):
    w, b = affine
    src = tmp_path / "src"
    truth = _write_cache(src, "seq0", w, b)

    n, var, rms = STRIP.strip_sequence(
        src / "seq0" / "features.npy", tmp_path / "dst" / "seq0" / "features.npy",
        w, b, overwrite=False, verify=True,
    )

    assert n == N_FRAME
    got = np.load(tmp_path / "dst" / "seq0" / "features.npy").astype(np.float32)
    # the whole point: back at the variance dinov2reg_v1 is stored at
    assert var == pytest.approx(1.0, abs=0.02)
    assert got.std() == pytest.approx(1.0, abs=0.02)
    # and it is the SAME signal, not merely a rescaled one
    assert np.abs(got - truth).max() < 0.02
    # float16 storage should cost far less than a percent of a sigma
    assert rms < 1e-3


def test_valid_mask_travels_with_the_features(tmp_path, affine):
    w, b = affine
    src = tmp_path / "src"
    _write_cache(src, "seq0", w, b)
    dst = tmp_path / "dst" / "seq0" / "features.npy"
    STRIP.strip_sequence(src / "seq0" / "features.npy", dst, w, b, overwrite=False, verify=False)
    # sim_dataset.py reads features_valid.npy alongside; a cache without it silently
    # resolves zero usable frames
    assert (dst.parent / "features_valid.npy").exists()
    assert np.load(dst.parent / "features_valid.npy").sum() == N_FRAME


def test_existing_destination_is_skipped_and_overwrite_forces(tmp_path, affine):
    w, b = affine
    src = tmp_path / "src"
    _write_cache(src, "seq0", w, b)
    dst = tmp_path / "dst" / "seq0" / "features.npy"
    STRIP.strip_sequence(src / "seq0" / "features.npy", dst, w, b, overwrite=False, verify=False)

    n, _, _ = STRIP.strip_sequence(src / "seq0" / "features.npy", dst, w, b,
                                   overwrite=False, verify=False)
    assert n == 0, "a resumed run must not redo completed sequences"

    n, _, _ = STRIP.strip_sequence(src / "seq0" / "features.npy", dst, w, b,
                                   overwrite=True, verify=False)
    assert n == N_FRAME


def test_no_partial_file_survives_a_completed_run(tmp_path, affine):
    """The write goes to .npy.partial and is renamed, so an interrupted job cannot leave
    a truncated cache that the loader would happily mmap."""
    w, b = affine
    src = tmp_path / "src"
    _write_cache(src, "seq0", w, b)
    dst = tmp_path / "dst" / "seq0" / "features.npy"
    STRIP.strip_sequence(src / "seq0" / "features.npy", dst, w, b, overwrite=False, verify=False)
    assert not list(dst.parent.glob("*.partial"))


def test_width_mismatch_is_refused(tmp_path, affine):
    w, b = affine
    src = tmp_path / "src"
    _write_cache(src, "seq0", w, b)
    with pytest.raises(SystemExit, match="LayerNorm width"):
        STRIP.strip_sequence(src / "seq0" / "features.npy", tmp_path / "dst" / "s" / "features.npy",
                             w[:-1], b[:-1], overwrite=False, verify=False)


def test_chunking_does_not_change_the_result(tmp_path, affine, monkeypatch):
    """CHUNK is a memory knob; the output must not depend on it."""
    w, b = affine
    src = tmp_path / "src"
    _write_cache(src, "seq0", w, b, n_frame=37)      # deliberately not a multiple of CHUNK
    a = tmp_path / "a" / "seq0" / "features.npy"
    c = tmp_path / "c" / "seq0" / "features.npy"
    STRIP.strip_sequence(src / "seq0" / "features.npy", a, w, b, overwrite=False, verify=False)
    monkeypatch.setattr(STRIP, "CHUNK", 7)
    STRIP.strip_sequence(src / "seq0" / "features.npy", c, w, b, overwrite=False, verify=False)
    assert np.array_equal(np.load(a), np.load(c))


@pytest.mark.parametrize(
    "args,expect",
    [
        (["--src", "X/dinov3_v1", "--dst", "X/dinov3_v1"], "refusing to overwrite in place"),
        (["--src", "X/dinov3_v1", "--dst", "X/dinov2reg_v1"], "refusing to write DINOv3"),
        (["--src", "X/dinov3_v1"], "either --seq-root"),
    ],
)
def test_naming_tripwires(tmp_path, args, expect):
    """Mirrors the tripwires in tools/sim/cache_dino_features.py: writing one encoder's
    features under the other's version name only surfaces as a bad training run."""
    argv = [sys.executable, str(SCRIPT)] + [a.replace("X", str(tmp_path)) for a in args]
    r = subprocess.run(argv, capture_output=True, text=True, cwd=SCRIPT.parents[2])
    assert r.returncode != 0
    assert expect in (r.stderr + r.stdout), r.stderr
