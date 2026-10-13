"""Build the TartanDrive 1.0 train/val/test split (bag-disjoint).

Mirrors `data/splits/v2.json` in shape so `compute_norm_stats_v2.py` and
`feelwm/data/dataset.py` consume it unchanged.

## Why not reuse NWM's TD1 split

The NWM release ships a TartanDrive split (1000 train / 251 test
trajectories over 570 bags) and adopting it would have matched their training
corpus exactly. It is **segment-level, not run-level**: 124 of the 570 bags have
segments on both sides. `20210826_47_bigpuddle` is cut into 55 segments with
0, 2, 3, 4 in test and 1, 5, 6, 7 in train — temporally adjacent slices of one
continuous 30-40 s drive, sharing terrain, lighting and field of view.

For a world model that is leakage, and our TD2 protocol holds out whole sequences
(plus a whole terrain class as OOD). Copying a leaky split into TD1 would break the
protocol this experiment exists to keep symmetric, so the split below is built at
**bag granularity**: one bag is one continuous run, and a run is never split.

## What TD1 cannot support

TD2's split holds out `slag_heap` as an OOD terrain class. **TD1 has no terrain
signal at all** — folder names carry only the collection date (a handful of bags
have descriptive names like `bigpuddle`, too few to build a class from). So there
is no TD1 OOD-terrain split. That costs nothing here: for this experiment the
out-of-domain evaluation *is* TD2. The `test` split emitted below is an in-domain
diagnostic, not the main cross-domain result.

Usage (after `td1_fetch_build_loop.sh` has finished):
    python -m tools.tartandrive.make_td1_split
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from feelwm.config import data_path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = data_path("tartandrive1_processed")
DEFAULT_OUT = REPO_ROOT / "data" / "splits" / "td1_v1.json"

# A sequence shorter than this contributes almost nothing: the training config uses
# context_len 3 with rollout_horizon 8 over strides [1, 2, 5, 10], so a stride-10
# sample alone spans 3 + 8*10 + 1 = 84 anchors. Short bags are kept only if they can
# still serve the smaller strides with some margin.
MIN_ANCHORS = 40

# Deterministic, seed-free assignment: within each archive, bags are sorted by name
# and every 10th goes to val, every 20th (offset) to test. Reproducible from the
# file list alone, and spread across each collection day rather than clustered at
# one end of it.
VAL_MODULUS, VAL_RESIDUE = 10, 5
TEST_MODULUS, TEST_RESIDUE = 20, 12


def archive_of(seq_name: str) -> str:
    """`20210903_heightmaps_1_20210903_26` -> `20210903_heightmaps_1`."""
    parts = seq_name.split("_heightmaps_")
    return f"{parts[0]}_heightmaps_{parts[1].split('_')[0]}" if len(parts) > 1 else seq_name


def scan(root: Path, min_anchors: int = MIN_ANCHORS) -> tuple[dict, dict]:
    """Return (usable {name: meta}, excluded {name: reason})."""
    usable, excluded = {}, {}
    for meta_path in sorted(root.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        name = meta_path.parent.name
        if meta.get("dataset") != "td1":
            excluded[name] = f"dataset={meta.get('dataset')!r}, not td1"
        elif meta.get("stage") == "extract-only":
            excluded[name] = "extract-only, labels not built"
        elif not meta.get("has_proprio", False):
            excluded[name] = "has_proprio=False"
        elif float(meta.get("valid_fraction", 0.0)) <= 0.0:
            excluded[name] = "valid_fraction=0"
        elif int(meta.get("n_anchors", 0)) < min_anchors:
            excluded[name] = f"n_anchors={meta.get('n_anchors')} < {min_anchors}"
        else:
            usable[name] = meta
    return usable, excluded


def partition(usable: dict) -> tuple[list, list, list]:
    by_archive: dict[str, list[str]] = {}
    for name in usable:
        by_archive.setdefault(archive_of(name), []).append(name)

    train, val, test = [], [], []
    for archive in sorted(by_archive):
        for i, name in enumerate(sorted(by_archive[archive])):
            if i % TEST_MODULUS == TEST_RESIDUE:
                test.append(name)
            elif i % VAL_MODULUS == VAL_RESIDUE:
                val.append(name)
            else:
                train.append(name)
    return sorted(train), sorted(val), sorted(test)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-anchors", type=int, default=MIN_ANCHORS)
    args = ap.parse_args()

    usable, excluded = scan(args.processed_root, args.min_anchors)
    if not usable:
        raise SystemExit(f"no usable TD1 sequences under {args.processed_root}")
    train, val, test = partition(usable)

    anchors = {s: int(usable[s]["n_anchors"] * usable[s]["valid_fraction"])
               for s in usable}
    split = {
        "version": "td1_v1",
        "pipeline_version": next(iter(usable.values())).get("pipeline_version"),
        "created_at": date.today().isoformat(),
        "strategy": (
            "Bag-disjoint: one TD1 bag is one continuous 30-40 s run and is never "
            "split across sets. Within each archive, bags sorted by name go to test "
            f"at index%{TEST_MODULUS}=={TEST_RESIDUE}, val at "
            f"index%{VAL_MODULUS}=={VAL_RESIDUE}, train otherwise — deterministic and "
            "spread across each collection day. NWM's own TD1 split is NOT reused: it "
            "is segment-level and 124 of 570 bags have segments in both train and "
            "test. TD1 carries no terrain labels, so there is no OOD-terrain split; "
            "for this experiment the out-of-domain evaluation is TD2."
        ),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_excluded": len(excluded),
        "train": train, "val": val, "test": test,
        "excluded": excluded,
        "notes": {
            "min_anchors": args.min_anchors,
            "approx_valid_anchors": {
                "train": sum(anchors[s] for s in train),
                "val": sum(anchors[s] for s in val),
                "test": sum(anchors[s] for s in test),
            },
            "test_rationale": (
                "TD1 in-domain diagnostic only. The headline evaluation for the "
                "TD1-train experiment is zero-shot on the TD2 val/test splits."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2))

    print(f"wrote {args.output}")
    print(f"  usable   {len(usable)} bags over "
          f"{len({archive_of(s) for s in usable})} archives")
    print(f"  train {len(train)}  val {len(val)}  test {len(test)}  "
          f"excluded {len(excluded)}")
    for name, n in split["notes"]["approx_valid_anchors"].items():
        print(f"  valid anchors {name:5s} {n}")
    if excluded:
        reasons: dict[str, int] = {}
        for reason in excluded.values():
            key = reason.split("=")[0]
            reasons[key] = reasons.get(key, 0) + 1
        print(f"  excluded by reason: {reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
