"""feelwm.config: root expansion and named config variants."""
from __future__ import annotations

from pathlib import Path

import pytest

from feelwm.config import deep_merge, load_config, variant_names

REPO_ROOT = Path(__file__).resolve().parents[1]

DOC = """
default_variant: a
model: {width: 8, depth: 2, heads: [1, 2]}
training: {lr: 0.1, checkpoint_path: "${OUTPUT_ROOT}/demo/${VARIANT}/checkpoint.pt"}
variants:
  a: null
  b:
    model: {depth: 4, heads: [3]}
    extra: {only_in_b: true}
"""


@pytest.fixture
def cfg_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_ROOT", "/out")
    p = tmp_path / "demo.yaml"
    p.write_text(DOC)
    return p


def test_deep_merge_recurses_into_dicts_and_replaces_everything_else():
    out = deep_merge({"a": {"x": 1, "y": [1, 2]}, "b": 1}, {"a": {"y": [9]}, "c": 3})
    assert out == {"a": {"x": 1, "y": [9]}, "b": 1, "c": 3}


def test_default_variant_is_used_and_bookkeeping_keys_are_stripped(cfg_file):
    cfg = load_config(cfg_file)
    assert "variants" not in cfg and "default_variant" not in cfg
    assert cfg["model"] == {"width": 8, "depth": 2, "heads": [1, 2]}
    assert cfg["training"]["checkpoint_path"] == "/out/demo/a/checkpoint.pt"


def test_named_variant_merges_over_the_shared_keys(cfg_file):
    cfg = load_config(cfg_file, "b")
    assert cfg["model"] == {"width": 8, "depth": 4, "heads": [3]}
    assert cfg["extra"] == {"only_in_b": True}
    assert cfg["training"]["checkpoint_path"] == "/out/demo/b/checkpoint.pt"


def test_unknown_variant_lists_the_valid_names(cfg_file):
    with pytest.raises(ValueError, match=r"unknown variant 'nope'.*\['a', 'b'\]"):
        load_config(cfg_file, "nope")


def test_variant_on_a_file_without_variants_is_refused(tmp_path):
    p = tmp_path / "plain.yaml"
    p.write_text("model: {width: 8}\n")
    assert load_config(p) == {"model": {"width": 8}}
    with pytest.raises(ValueError, match="defines no variants"):
        load_config(p, "a")


def test_every_shipped_config_variant_resolves():
    files = sorted((REPO_ROOT / "configs").glob("*.yaml"))
    assert len(files) <= 6
    for f in files:
        for v in variant_names(f) or [None]:
            cfg = load_config(f, v)
            assert "${" not in repr(cfg), f"{f.name}:{v} left a placeholder unexpanded"
