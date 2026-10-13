#!/usr/bin/env python3
"""Derive binary unsafe proxy labels from TartanDrive v2 processed labels.

This script intentionally keeps binary safety labels downstream of the raw
label-generation pipeline. The resulting label is a teacher-derived proxy from
pitch/roll/traversability cost, not a human failure ground truth.

Default label:
    unsafe = valid_mask & isfinite(trav_cost_teacher) & (trav_cost_teacher > 0.5)

Outputs:
    data/binary_labels/v2/<sequence>/safety.npz
    data/binary_labels/v2/meta.json
    a markdown threshold-sweep report (path set by --report)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from feelwm.config import data_path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = REPO_ROOT / "data" / "splits" / "v2.json"
DEFAULT_PROCESSED_ROOT = data_path("tartandrive2_processed")
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "binary_labels" / "v2"
DEFAULT_REPORT = REPO_ROOT / "out" / "binary_label" / "v2_threshold_report.md"

LABEL_VERSION = "v2"
SELECTED_THRESHOLDS = {
    "pitch_abs_deg": None,
    "roll_abs_deg": None,
    "trav_cost": 0.5,
}
DIAGNOSTIC_THRESHOLDS = {
    "pitch_abs_deg": 10.0,
    "roll_abs_deg": 8.0,
    "trav_cost": 0.5,
}

PITCH_OPTIONS = [None, 10.0, 15.0, 20.0]
ROLL_OPTIONS = [None, 8.0, 10.0, 12.0]
TRAV_OPTIONS = [None, 0.5, 0.8, 1.0]


@dataclass(frozen=True)
class Combo:
    pitch_abs_deg: float | None
    roll_abs_deg: float | None
    trav_cost: float | None

    def label(self) -> str:
        parts = []
        if self.pitch_abs_deg is not None:
            parts.append(f"|pitch|>{self.pitch_abs_deg:g}")
        if self.roll_abs_deg is not None:
            parts.append(f"|roll|>{self.roll_abs_deg:g}")
        if self.trav_cost is not None:
            parts.append(f"trav>{self.trav_cost:g}")
        return " OR ".join(parts)


@dataclass
class SeqData:
    seq: str
    anchor_idx: np.ndarray
    valid_mask: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
    trav_cost: np.ndarray

    @property
    def finite_mask(self) -> np.ndarray:
        return (
            np.isfinite(self.pitch_deg)
            & np.isfinite(self.roll_deg)
            & np.isfinite(self.trav_cost)
        )

    @property
    def eval_mask(self) -> np.ndarray:
        return self.valid_mask.astype(bool) & self.finite_mask


def load_split(path: Path) -> dict[str, Any]:
    split = json.loads(path.read_text())
    for key in ("train", "val", "test"):
        if key not in split:
            raise KeyError(f"Missing split key: {key}")
    return split


def split_sequences(split: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in ("train", "val", "test"):
        for seq in split[part]:
            if seq not in seen:
                seen.add(seq)
                out.append(seq)
    return out


def load_seq(processed_root: Path, seq: str) -> SeqData:
    labels_path = processed_root / seq / "labels.npz"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    z = np.load(labels_path)
    required = ["anchor_idx", "valid_mask", "pitch_deg", "roll_deg", "trav_cost_teacher"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"{labels_path} missing keys: {missing}")

    return SeqData(
        seq=seq,
        anchor_idx=z["anchor_idx"].astype(np.int32),
        valid_mask=z["valid_mask"].astype(np.uint8),
        pitch_deg=z["pitch_deg"].astype(np.float32),
        roll_deg=z["roll_deg"].astype(np.float32),
        trav_cost=z["trav_cost_teacher"].astype(np.float32),
    )


def all_combos() -> list[Combo]:
    combos = []
    for pitch, roll, trav in itertools.product(PITCH_OPTIONS, ROLL_OPTIONS, TRAV_OPTIONS):
        if pitch is None and roll is None and trav is None:
            continue
        combos.append(Combo(pitch, roll, trav))
    return combos


def component_masks(data: SeqData, combo: Combo, valid_gated: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = data.finite_mask
    base = data.valid_mask.astype(bool) & finite if valid_gated else finite

    pitch = np.zeros_like(base, dtype=bool)
    roll = np.zeros_like(base, dtype=bool)
    trav = np.zeros_like(base, dtype=bool)

    if combo.pitch_abs_deg is not None:
        pitch = base & (np.abs(data.pitch_deg) > combo.pitch_abs_deg)
    if combo.roll_abs_deg is not None:
        roll = base & (np.abs(data.roll_deg) > combo.roll_abs_deg)
    if combo.trav_cost is not None:
        trav = base & (data.trav_cost > combo.trav_cost)
    return pitch, roll, trav


def unsafe_mask(data: SeqData, combo: Combo, valid_gated: bool = True) -> np.ndarray:
    pitch, roll, trav = component_masks(data, combo, valid_gated=valid_gated)
    return pitch | roll | trav


def count_for_combo(datas: list[SeqData], combo: Combo) -> dict[str, Any]:
    pos = 0
    total = 0
    per_seq = []

    for data in datas:
        eval_mask = data.eval_mask
        unsafe = unsafe_mask(data, combo, valid_gated=True)
        n = int(eval_mask.sum())
        k = int((unsafe & eval_mask).sum())
        pos += k
        total += n
        per_seq.append({
            "seq": data.seq,
            "positive": k,
            "total": n,
            "positive_ratio": float(k / n) if n else 0.0,
        })

    neg = total - pos
    return {
        "positive": pos,
        "negative": neg,
        "total": total,
        "positive_ratio": float(pos / total) if total else 0.0,
        "pos_weight": float(neg / pos) if pos else None,
        "per_seq": per_seq,
    }


def sweep(split: dict[str, Any], loaded: dict[str, SeqData]) -> list[dict[str, Any]]:
    rows = []
    for combo in all_combos():
        row: dict[str, Any] = {
            "pitch_abs_deg": combo.pitch_abs_deg,
            "roll_abs_deg": combo.roll_abs_deg,
            "trav_cost": combo.trav_cost,
            "label": combo.label(),
        }
        for part in ("train", "val", "test"):
            stats = count_for_combo([loaded[s] for s in split[part]], combo)
            row[f"{part}_positive"] = stats["positive"]
            row[f"{part}_negative"] = stats["negative"]
            row[f"{part}_total"] = stats["total"]
            row[f"{part}_positive_ratio"] = stats["positive_ratio"]
            row[f"{part}_pos_weight"] = stats["pos_weight"]
            seq_ratios = [x["positive_ratio"] for x in stats["per_seq"] if x["total"] > 0]
            row[f"{part}_seq_max_ratio"] = max(seq_ratios) if seq_ratios else 0.0
            row[f"{part}_seq_nonzero"] = sum(x > 0 for x in seq_ratios)
        rows.append(row)
    return rows


def selected_combo() -> Combo:
    return Combo(
        pitch_abs_deg=SELECTED_THRESHOLDS["pitch_abs_deg"],
        roll_abs_deg=SELECTED_THRESHOLDS["roll_abs_deg"],
        trav_cost=SELECTED_THRESHOLDS["trav_cost"],
    )


def diagnostic_combo() -> Combo:
    return Combo(
        pitch_abs_deg=DIAGNOSTIC_THRESHOLDS["pitch_abs_deg"],
        roll_abs_deg=DIAGNOSTIC_THRESHOLDS["roll_abs_deg"],
        trav_cost=DIAGNOSTIC_THRESHOLDS["trav_cost"],
    )


def write_labels(out_root: Path, split: dict[str, Any], loaded: dict[str, SeqData]) -> dict[str, Any]:
    combo = selected_combo()
    diag = diagnostic_combo()
    out_root.mkdir(parents=True, exist_ok=True)

    written = []
    for seq in split_sequences(split):
        data = loaded[seq]
        unsafe = unsafe_mask(data, combo, valid_gated=True).astype(np.uint8)
        trigger_pitch, trigger_roll, trigger_trav = component_masks(data, diag, valid_gated=True)

        seq_out = out_root / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            seq_out / "safety.npz",
            anchor_idx=data.anchor_idx.astype(np.int32),
            unsafe=unsafe,
            valid_mask=data.valid_mask.astype(np.uint8),
            trigger_trav=trigger_trav.astype(np.uint8),
            trigger_pitch=trigger_pitch.astype(np.uint8),
            trigger_roll=trigger_roll.astype(np.uint8),
        )
        written.append(seq)

    return {"n_sequences": len(written), "sequences": written}


def validate_outputs(out_root: Path, split: dict[str, Any], loaded: dict[str, SeqData],
                     expected_train_positive: int | None = 7647,
                     expected_train_total: int | None = 99311) -> list[str]:
    failures = []
    combo = selected_combo()

    for seq in split_sequences(split):
        path = out_root / seq / "safety.npz"
        if not path.exists():
            failures.append(f"missing {path}")
            continue

        z = np.load(path)
        data = loaded[seq]
        for key in ("anchor_idx", "unsafe", "valid_mask", "trigger_trav", "trigger_pitch", "trigger_roll"):
            if key not in z.files:
                failures.append(f"{path} missing key {key}")
        if failures:
            continue

        shape = data.valid_mask.shape
        for key in ("anchor_idx", "unsafe", "valid_mask", "trigger_trav", "trigger_pitch", "trigger_roll"):
            if z[key].shape != shape:
                failures.append(f"{path}:{key} shape {z[key].shape} != {shape}")

        expected = unsafe_mask(data, combo, valid_gated=True).astype(np.uint8)
        if not np.array_equal(z["unsafe"], expected):
            failures.append(f"{path}: unsafe does not match selected threshold")

    train_stats = count_for_combo([loaded[s] for s in split["train"]], combo)
    if expected_train_positive is not None and train_stats["positive"] != expected_train_positive:
        failures.append(
            f"train positive {train_stats['positive']} != expected {expected_train_positive}"
        )
    if expected_train_total is not None and train_stats["total"] != expected_train_total:
        failures.append(f"train total {train_stats['total']} != expected {expected_train_total}")

    return failures


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_weight(x: float | None) -> str:
    return "inf" if x is None else f"{x:.6g}"


def markdown_table(rows: list[list[str]], header: list[str]) -> str:
    def esc(cell: str) -> str:
        return cell.replace("|", "\\|")

    lines = [
        "| " + " | ".join(esc(h) for h in header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(esc(c) for c in r) + " |" for r in rows)
    return "\n".join(lines)


def make_report(split: dict[str, Any], loaded: dict[str, SeqData], sweep_rows: list[dict[str, Any]]) -> str:
    combo = selected_combo()
    selected_train = count_for_combo([loaded[s] for s in split["train"]], combo)
    selected_val = count_for_combo([loaded[s] for s in split["val"]], combo)
    selected_test = count_for_combo([loaded[s] for s in split["test"]], combo)

    target_rows = sorted(
        [r for r in sweep_rows if 0.05 <= r["train_positive_ratio"] <= 0.15],
        key=lambda r: r["train_positive_ratio"],
    )
    all_rows = sorted(sweep_rows, key=lambda r: abs(r["train_positive_ratio"] - 0.10))

    sweep_table_rows = []
    for r in target_rows:
        sweep_table_rows.append([
            r["label"],
            f'{r["train_positive"]}/{r["train_total"]}',
            fmt_pct(r["train_positive_ratio"]),
            fmt_pct(r["val_positive_ratio"]),
            fmt_pct(r["test_positive_ratio"]),
            fmt_pct(r["train_seq_max_ratio"]),
            str(r["train_seq_nonzero"]),
        ])

    nearest_rows = []
    for r in all_rows[:12]:
        nearest_rows.append([
            r["label"],
            fmt_pct(r["train_positive_ratio"]),
            fmt_pct(r["val_positive_ratio"]),
            fmt_pct(r["test_positive_ratio"]),
            fmt_pct(r["train_seq_max_ratio"]),
            str(r["train_seq_nonzero"]),
        ])

    per_seq_rows = []
    for part in ("train", "val", "test"):
        stats = count_for_combo([loaded[s] for s in split[part]], combo)
        for item in sorted(stats["per_seq"], key=lambda x: x["positive_ratio"], reverse=True):
            per_seq_rows.append([
                part,
                item["seq"],
                str(item["positive"]),
                str(item["total"]),
                fmt_pct(item["positive_ratio"]),
            ])

    single_combos = [
        Combo(10.0, None, None), Combo(15.0, None, None), Combo(20.0, None, None),
        Combo(None, 8.0, None), Combo(None, 10.0, None), Combo(None, 12.0, None),
        Combo(None, None, 0.5), Combo(None, None, 0.8), Combo(None, None, 1.0),
    ]
    single_rows = []
    row_by_key = {
        (r["pitch_abs_deg"], r["roll_abs_deg"], r["trav_cost"]): r for r in sweep_rows
    }
    for c in single_combos:
        r = row_by_key[(c.pitch_abs_deg, c.roll_abs_deg, c.trav_cost)]
        single_rows.append([
            c.label(),
            fmt_pct(r["train_positive_ratio"]),
            fmt_pct(r["val_positive_ratio"]),
            fmt_pct(r["test_positive_ratio"]),
            fmt_pct(r["train_seq_max_ratio"]),
            str(r["train_seq_nonzero"]),
        ])

    meta = [
        f"# Binary Safety Label {LABEL_VERSION} Threshold Report",
        "",
        "**Dataset**: TartanDrive 2.0 processed",
        "**Split**: `data/splits/v2.json`",
        f"**Label version**: `{LABEL_VERSION}`",
        "",
        "## 1. Label Semantics",
        "",
        "This label is a **teacher-derived unsafe proxy**, not a human failure "
        "ground-truth label. It is derived from raw `pitch_deg`, `roll_deg`, "
        "and `trav_cost_teacher` saved by the label-generation pipeline.",
        "",
        f"The selected {LABEL_VERSION} label is:",
        "",
        "```",
        "unsafe = valid_mask & isfinite(trav_cost_teacher) & (trav_cost_teacher > 0.5)",
        "```",
        "",
        "`trav_cost_teacher` is the TartanDrive rig's internal traversability "
        "estimator. It is useful as a safety/risk teacher, but should not be "
        "described as collision, stuck, or human intervention ground truth.",
        "",
        "## 2. Selected Threshold",
        "",
        markdown_table(
            [
                ["train", str(selected_train["positive"]), str(selected_train["total"]),
                 fmt_pct(selected_train["positive_ratio"]), fmt_weight(selected_train["pos_weight"])],
                ["val", str(selected_val["positive"]), str(selected_val["total"]),
                 fmt_pct(selected_val["positive_ratio"]), fmt_weight(selected_val["pos_weight"])],
                ["test", str(selected_test["positive"]), str(selected_test["total"]),
                 fmt_pct(selected_test["positive_ratio"]), fmt_weight(selected_test["pos_weight"])],
            ],
            ["split", "positive", "valid total", "positive ratio", "pos_weight"],
        ),
        "",
        "Rationale: `trav_cost_teacher > 0.5` gives a stable train/val class "
        "balance in the target 5-15% range while avoiding attitude-only labels "
        "that are sparse and sequence-sensitive. The higher test positive ratio "
        "is expected because the v2 test split is intentionally OOD "
        "(`slag_heap`/`trash`).",
        "",
        "Pitch/roll thresholds are retained as diagnostic component triggers in "
        f"`safety.npz`, but they are not part of the {LABEL_VERSION} training label.",
        "",
        "## 3. Single-Signal Baselines",
        "",
        markdown_table(
            single_rows,
            ["threshold", "train pos", "val pos", "test pos", "max train seq pos", "nonzero train seqs"],
        ),
        "",
        "## 4. OR Threshold Sweep: Target Train Positive Ratio 5-15%",
        "",
        markdown_table(
            sweep_table_rows,
            ["threshold", "train pos/total", "train pos", "val pos", "test pos", "max train seq pos", "nonzero train seqs"],
        ),
        "",
        "## 5. Nearest Combinations To 10% Train Positive Ratio",
        "",
        markdown_table(
            nearest_rows,
            ["threshold", "train pos", "val pos", "test pos", "max train seq pos", "nonzero train seqs"],
        ),
        "",
        f"## 6. Per-Sequence Positive Ratio For Selected {LABEL_VERSION} Label",
        "",
        markdown_table(
            per_seq_rows,
            ["split", "sequence", "positive", "valid total", "positive ratio"],
        ),
        "",
    ]
    return "\n".join(meta)


def write_meta(out_root: Path, split: dict[str, Any], loaded: dict[str, SeqData],
               sweep_rows: list[dict[str, Any]], write_result: dict[str, Any]) -> dict[str, Any]:
    combo = selected_combo()
    meta: dict[str, Any] = {
        "label_version": LABEL_VERSION,
        "split_version": split.get("version", "v2"),
        "source_split": str(DEFAULT_SPLIT.relative_to(REPO_ROOT)),
        "source_processed_root": str(DEFAULT_PROCESSED_ROOT),
        "semantics": "teacher-derived unsafe proxy; not human failure ground truth",
        "selected_thresholds": SELECTED_THRESHOLDS,
        "diagnostic_thresholds": DIAGNOSTIC_THRESHOLDS,
        "outputs": write_result,
        "splits": {},
    }

    for part in ("train", "val", "test"):
        stats = count_for_combo([loaded[s] for s in split[part]], combo)
        meta["splits"][part] = {
            "positive": stats["positive"],
            "negative": stats["negative"],
            "valid_total": stats["total"],
            "positive_ratio": stats["positive_ratio"],
            "pos_weight": stats["pos_weight"],
            "n_sequences": len(split[part]),
        }

    meta["train_positive_ratio"] = meta["splits"]["train"]["positive_ratio"]
    meta["pos_weight"] = meta["splits"]["train"]["pos_weight"]
    meta["threshold_sweep"] = [
        {
            "pitch_abs_deg": r["pitch_abs_deg"],
            "roll_abs_deg": r["roll_abs_deg"],
            "trav_cost": r["trav_cost"],
            "label": r["label"],
            "train_positive_ratio": r["train_positive_ratio"],
            "val_positive_ratio": r["val_positive_ratio"],
            "test_positive_ratio": r["test_positive_ratio"],
            "train_positive": r["train_positive"],
            "train_total": r["train_total"],
        }
        for r in sweep_rows
    ]

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def print_summary(split: dict[str, Any], loaded: dict[str, SeqData], sweep_rows: list[dict[str, Any]]) -> None:
    combo = selected_combo()
    print(f"Selected {LABEL_VERSION} label:", combo.label())
    for part in ("train", "val", "test"):
        stats = count_for_combo([loaded[s] for s in split[part]], combo)
        print(
            f"  {part:5s}: {stats['positive']}/{stats['total']} "
            f"({fmt_pct(stats['positive_ratio'])}), pos_weight={fmt_weight(stats['pos_weight'])}"
        )

    print("\nCombinations in train target range [5%, 15%]:")
    for r in sorted(
        [x for x in sweep_rows if 0.05 <= x["train_positive_ratio"] <= 0.15],
        key=lambda x: x["train_positive_ratio"],
    ):
        print(
            f"  {r['label']:<38s} train={fmt_pct(r['train_positive_ratio']):>7s} "
            f"val={fmt_pct(r['val_positive_ratio']):>7s} "
            f"test={fmt_pct(r['test_positive_ratio']):>7s}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute sweep and print summary without writing labels/report.")
    args = ap.parse_args()

    split_path = Path(args.split)
    processed_root = Path(args.processed_root)
    out_root = Path(args.out_root)
    report_path = Path(args.report)

    split = load_split(split_path)
    loaded = {seq: load_seq(processed_root, seq) for seq in split_sequences(split)}
    sweep_rows = sweep(split, loaded)
    print_summary(split, loaded, sweep_rows)

    if args.dry_run:
        print("\nDry run: no files written.")
        return 0

    write_result = write_labels(out_root, split, loaded)
    meta = write_meta(out_root, split, loaded, sweep_rows, write_result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(make_report(split, loaded, sweep_rows))

    failures = validate_outputs(out_root, split, loaded)
    if failures:
        print("\nValidation failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    train = meta["splits"]["train"]
    print(f"\nWrote {write_result['n_sequences']} sequence label files to {out_root}")
    print(f"Wrote meta: {out_root / 'meta.json'}")
    print(f"Wrote report: {report_path}")
    print(
        "Validation passed: "
        f"train positives {train['positive']}/{train['valid_total']}, "
        f"pos_weight={train['pos_weight']:.6g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
