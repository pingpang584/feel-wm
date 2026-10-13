"""tools/eval/horizon_sweep.py — the persistence-control helpers, pinned.

The stride sweep's persistence baseline must be computed
on exactly the anchors/steps/masks the model rows use. These tests pin the two pieces the
sweep loop cannot check for itself: the anchor-time label lookup walks the dataset's own
index, and the per-step AUC has the trainer's tie handling (so the pooled row can be
reconciled against eval_full_model's `binary_safety.auc`).
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

HS = importlib.import_module("tools.eval.horizon_sweep")


def _fake_ds(with_labels: bool = True):
    rec0 = SimpleNamespace(binary_unsafe=np.array([0, 0, 0, 1, 0, 1, 0, 0], np.uint8),
                           binary_valid=np.array([1, 1, 1, 1, 0, 1, 1, 1], np.uint8))
    rec1 = SimpleNamespace(binary_unsafe=np.array([1, 0, 0], np.uint8),
                           binary_valid=np.array([1, 1, 1], np.uint8))
    if not with_labels:
        rec1 = SimpleNamespace()
    # (sid, anchor t, stride) — the same triples the loader walks, in loader order
    return SimpleNamespace(_index=[(0, 3, 1), (0, 4, 1), (1, 0, 5), (0, 7, 2)], _records=[rec0, rec1])


def test_current_binary_labels_follow_the_dataset_index():
    unsafe, valid = HS.current_binary_labels(_fake_ds())
    assert unsafe.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert valid.tolist() == [True, False, True, True]


def test_current_binary_labels_none_when_dataset_does_not_expose_them():
    assert HS.current_binary_labels(SimpleNamespace()) is None
    assert HS.current_binary_labels(_fake_ds(with_labels=False)) is None


def _pairwise_auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    tot = 0.0
    for p in pos:
        tot += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return tot / (len(pos) * len(neg))


def test_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300), 1)                 # coarse → many ties
    chunks = np.array_split(np.arange(300), 4)       # the sweep appends per batch
    got = HS._auc([s[c] for c in chunks], [y[c].astype(np.float32) for c in chunks])
    assert got == pytest.approx(_pairwise_auc(s, y), abs=1e-12)


def test_auc_is_nan_without_both_classes_or_data():
    assert np.isnan(HS._auc([], []))
    assert np.isnan(HS._auc([np.array([0.2, 0.9])], [np.array([1.0, 1.0])]))


def test_persistence_r2_from_shared_sufficient_stats():
    """Persistence R² reuses the model rows' sy/sy2/n (same targets) with its own SSE."""
    rng = np.random.default_rng(1)
    tgt = rng.normal(0, 1, (500, 3))
    sy, sy2, n = tgt.sum(0), (tgt ** 2).sum(0), np.full(3, 500.0)
    sse_mean = ((tgt - tgt.mean(0)) ** 2).sum(0)      # predictor = target mean → R² 0
    assert np.allclose(HS._r2(sse_mean, sy, sy2, n), 0.0, atol=1e-9)
    assert np.allclose(HS._r2(np.zeros(3), sy, sy2, n), 1.0)
    worse = ((tgt - (tgt.mean(0) + 2.0)) ** 2).sum(0)  # biased constant → negative
    assert (HS._r2(worse, sy, sy2, n) < 0).all()
