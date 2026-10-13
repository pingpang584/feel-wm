"""Run build_sequence() across every usable TartanDrive 2.0 sequence in parallel.

Filters out the sequences found broken / partial in a TartanDrive 2.0 corpus inventory:
  - satellite_views/                             (0 bags)
  - turnpike_flat_2023-09-12-13-40-02            (14 s, broken)
  - turnpike_flat_2023-09-12-13-41-45            (1 s, broken)
  - turnpike_flat_2023-09-12-13-42-26            (2 / 17 bags OK, aria2 partial)

Usage: see --help.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from feelwm.config import data_path  # noqa: E402
from tools.tartandrive.build import build_sequence, PIPELINE_VERSION  # noqa: E402


EXCLUDE_SEQS = {
    "satellite_views",
    "turnpike_flat_2023-09-12-13-40-02",
    "turnpike_flat_2023-09-12-13-41-45",
    "turnpike_flat_2023-09-12-13-42-26",
}


def list_sequences(raw_root: Path) -> list[Path]:
    """All sequence dirs with at least one .bag, excluding known-broken ones."""
    out = []
    for d in sorted(raw_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in EXCLUDE_SEQS:
            continue
        bags = list(d.glob("*.bag"))
        if not bags:
            continue
        out.append(d)
    return out


def _worker(seq_dir: str, out_root: str, mode: str, force: bool) -> dict:
    """Top-level function so it's picklable for ProcessPoolExecutor."""
    try:
        meta = build_sequence(Path(seq_dir), Path(out_root), mode=mode, force=force)
        return {"ok": True, "seq": Path(seq_dir).name, "meta": meta}
    except Exception as e:
        return {
            "ok": False,
            "seq": Path(seq_dir).name,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(data_path("tartandrive")))
    ap.add_argument("--out-root",
                    default=str(data_path("tartandrive2_processed")))
    ap.add_argument("--mode", choices=["full", "extract-only", "features-only"],
                    default="full")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel processes; 1 = sequential. Bag I/O is disk-bound, "
                         "so 4-8 is a typical range.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Optional explicit list of sequence names; overrides scan.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the number of sequences (useful for a second smoke pass).")
    ap.add_argument("--log", default=None,
                    help="Write a JSONL per-sequence result log to this path.")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.only:
        seqs = [raw_root / name for name in args.only]
        missing = [p for p in seqs if not p.is_dir()]
        if missing:
            print("Missing sequence dirs:", [p.name for p in missing], file=sys.stderr)
            return 1
    else:
        seqs = list_sequences(raw_root)
    if args.limit is not None:
        seqs = seqs[: args.limit]

    print(f"Scheduling {len(seqs)} sequences with mode={args.mode}, "
          f"workers={args.workers}, force={args.force}", flush=True)
    print(f"Output root: {out_root}", flush=True)
    print(f"Pipeline version: {PIPELINE_VERSION}", flush=True)

    log_fp = open(args.log, "w") if args.log else None
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    skip_count = 0
    try:
        if args.workers == 1:
            results_iter = (
                _worker(str(s), str(out_root), args.mode, args.force) for s in seqs
            )
        else:
            ex = ProcessPoolExecutor(max_workers=args.workers)
            futures = [
                ex.submit(_worker, str(s), str(out_root), args.mode, args.force)
                for s in seqs
            ]
            results_iter = (f.result() for f in as_completed(futures))
        for i, res in enumerate(results_iter, start=1):
            elapsed = time.time() - t0
            if res["ok"]:
                meta = res["meta"]
                was_skip = meta.get("stage") not in ("extract-only",) and \
                           meta.get("build_seconds") is None
                if was_skip:
                    skip_count += 1
                    tag = "SKIP"
                else:
                    ok_count += 1
                    tag = "OK"
                print(f"[{i}/{len(seqs)}] {tag:4s} {res['seq']:55s} "
                      f"(elapsed total {elapsed/60:.1f} min)", flush=True)
            else:
                fail_count += 1
                print(f"[{i}/{len(seqs)}] FAIL {res['seq']:55s} "
                      f"-- {res['error']}", flush=True)
            if log_fp:
                log_fp.write(json.dumps(res) + "\n")
                log_fp.flush()
    finally:
        if log_fp:
            log_fp.close()

    total = time.time() - t0
    print(f"\nDone. ok={ok_count} skip={skip_count} fail={fail_count} "
          f"total_elapsed={total/60:.1f} min", flush=True)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
