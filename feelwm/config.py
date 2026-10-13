"""Config loading and the three filesystem roots every path in this repo hangs off.

    DATA_ROOT      datasets: raw bags, processed sequences, DINO feature caches
    OUTPUT_ROOT    training runs: checkpoints, metrics, per-sequence evals
    EXTERNAL_ROOT  third-party clones (optional, e.g. the NWM repo for baseline rows)

Configs refer to them as ``${DATA_ROOT}/...`` etc. Each root is resolved, in order, from
the config's own ``paths:`` block (``paths: {data_root: /data/datasets}``), then the environment
variable of the same name, then the default below (relative to the working directory).
Paths without a placeholder are used as written.

A config file can hold several named variants of one recipe (``variants:`` +
``default_variant:``, see ``resolve_variant``); entry points select one with ``--variant``.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ROOT_DEFAULTS = {"DATA_ROOT": "datasets", "OUTPUT_ROOT": "runs", "EXTERNAL_ROOT": "third_party"}
_PLACEHOLDER = re.compile(r"\$\{(DATA_ROOT|OUTPUT_ROOT|EXTERNAL_ROOT)\}")


def root(name: str, overrides: dict[str, Any] | None = None) -> str:
    """Resolve one root: config override > environment variable > default."""
    if name not in ROOT_DEFAULTS:
        raise KeyError(f"unknown root {name!r}; expected one of {sorted(ROOT_DEFAULTS)}")
    overrides = overrides or {}
    value = overrides.get(name.lower()) or os.environ.get(name) or ROOT_DEFAULTS[name]
    return str(value).rstrip("/") or "/"


def data_path(*parts: str) -> Path:
    """``${DATA_ROOT}/<parts...>`` — used for command-line defaults."""
    return Path(root("DATA_ROOT"), *parts)


def external_path(*parts: str) -> Path:
    """``${EXTERNAL_ROOT}/<parts...>``."""
    return Path(root("EXTERNAL_ROOT"), *parts)


def expand(obj: Any, overrides: dict[str, Any] | None = None,
           variant: str | None = None) -> Any:
    """Recursively substitute the root placeholders (and ``${VARIANT}``) in every string."""
    if isinstance(obj, str):
        out = _PLACEHOLDER.sub(lambda m: root(m.group(1), overrides), obj)
        if "${VARIANT}" in out:
            if variant is None:
                raise ValueError(f"${{VARIANT}} used in a config without variants: {obj!r}")
            out = out.replace("${VARIANT}", variant)
        return out
    if isinstance(obj, dict):
        return {k: expand(v, overrides, variant) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v, overrides, variant) for v in obj]
    return obj


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """``update`` over ``base``: dicts merge recursively, every other value replaces."""
    out = dict(base)
    for k, v in (update or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) \
            else v
    return out


def variant_names(path: str | Path) -> list[str]:
    """Variant names declared in a config file ([] if it has none)."""
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    return list((cfg.get("variants") or {}).keys())


def resolve_variant(cfg: dict[str, Any], variant: str | None = None,
                    source: str = "<config>") -> tuple[dict[str, Any], str | None]:
    """Apply a named variant to a parsed config.

    A file may declare ``variants: {name: {...overrides...}}`` and ``default_variant``. The
    chosen variant (``variant``, else ``default_variant``) is deep-merged over the top-level
    keys and both bookkeeping keys are removed. Returns (config, variant name or None).
    """
    cfg = dict(cfg or {})
    variants = cfg.pop("variants", None)
    default = cfg.pop("default_variant", None)
    if variants is None:
        if variant is not None:
            raise ValueError(f"{source} defines no variants; drop --variant {variant!r}")
        return cfg, None
    name = variant if variant is not None else default
    if name is None:
        raise ValueError(f"{source}: choose a variant with --variant; valid: {sorted(variants)}")
    if name not in variants:
        raise ValueError(f"{source}: unknown variant {name!r}; valid: {sorted(variants)}")
    return deep_merge(cfg, variants[name] or {}), name


def load_config(path: str | Path, variant: str | None = None) -> dict[str, Any]:
    """``yaml.safe_load`` + variant selection + root expansion.

    Every entry point reads configs through this. ``${VARIANT}`` in a string expands to the
    selected variant's name, so per-variant output directories never collide.
    """
    cfg, name = resolve_variant(yaml.safe_load(Path(path).read_text()), variant, str(path))
    overrides = cfg.get("paths") or {}
    return expand(cfg, overrides, name)


def add_variant_arg(parser, flag: str = "--variant", help_for: str = "the config") -> None:
    """The ``--variant NAME`` option every config-taking entry point exposes."""
    parser.add_argument(flag, default=None,
                        help=f"named variant of {help_for} (default: its default_variant)")
