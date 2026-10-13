"""Compute sim Husky norm-stats from labels.npz.

Per-channel z-score over VALID anchors only, written with feature_set="sim" so
feelwm/data/normalize.py applies pure z-score (no wheeled signed_log1p / window clips —
the 13-dim sim window has different columns than the 14-dim TartanDrive window).
TRAIN split only — pass the held-out (val) substrings to --exclude. When a config also
holds out the TAIL of training sequences (`data.tail_val_*`), pass the same substrings and
fraction to --tail-exclude/--tail-fraction so those anchors stay out of the stats too; the
cut is imported from the loader, so the two cannot drift.

Usage:
  python tools/sim/compute_norm_stats.py \
    '${DATA_ROOT}/negs-ugv/sim_husky_v3/sequences/all/*/labels.npz' \
    --exclude n1990 --out data/norm_stats/sim_husky_v3.json

JOINT stats over several corpora (combined Gazebo+Isaac training) — derive the corpus roots
AND the per-corpus held-out substrings from the training config instead of restating them,
so the stats and the loader cannot drift (`--exclude` is a GLOBAL substring list: with two
corpora it would also have to be proven collision-free across their naming schemes, which
this mode does per corpus by construction):
  python tools/sim/compute_norm_stats.py --from-config <combined-corpus config.yaml> \
    --out data/norm_stats/sim_husky_gazebo_isaac.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feelwm.config import add_variant_arg, load_config  # noqa: E402
from feelwm.data.sim_dataset import (  # noqa: E402
    jump_cum_from_pose, sim_corpora, tail_cut_index,
)


def paths_from_config(
    cfg_path: Path, seq_root_override: str | None = None, variant: str | None = None
) -> tuple[list[str], dict[str, dict]]:
    """TRAIN-side labels.npz paths for every corpus in a sim training config.

    Uses the loader's own corpus normalization (sim_corpora) and its own split rule (a
    sequence whose DIR NAME contains one of that corpus's heldout substrings is val,
    a sequence matching `drop_substrings` is in neither split), so the stats cover exactly
    the anchors the training run will see — including the case where the two corpora have
    different held-out lists. Returns (paths, per-corpus provenance).

    A config's seq_root may point at a STAGING copy (e.g. on faster storage). Labels are
    identical in the staging copy and the on-disk source (staging only drops `image`), so
    building from either is equivalent — the resolved root is recorded in the output's _meta either way.
    `seq_root_override` reads the SOURCE root instead, for the one case where that identity
    does not hold yet: a new run whose corpus grew on disk but has not been re-staged
    (staging waits on the DINO cache, while the stats do not — labels are all they need).
    Single-corpus configs only; the split/drop rules still come from the config.
    """
    d = load_config(cfg_path, variant)["data"]
    corpora = sim_corpora(d)
    if seq_root_override is not None and len(corpora) > 1:
        raise SystemExit(
            "--seq-root overrides one root; this config has "
            f"{len(corpora)} corpora — stage them and drop the flag")
    paths: list[str] = []
    meta: dict[str, dict] = {}
    for c in corpora:
        root = Path(seq_root_override if seq_root_override is not None else c["seq_root"])
        root = root if root.is_absolute() else REPO_ROOT / root
        found = sorted(str(p) for p in root.glob("*/labels.npz"))
        if not found:
            raise SystemExit(f"corpus {c['name'] or '<single>'}: no labels under {root}")
        kept = [p for p in found
                if not any(sub in Path(p).parent.name for sub in c["drop_substrings"])]
        keep = [p for p in kept
                if not any(sub in Path(p).parent.name for sub in c["heldout_substrings"])]
        meta[c["name"] or "<single>"] = {
            "n_train_seqs": len(keep), "n_seqs": len(found), "seq_root": str(root),
            "feat_version": c["feat_version"], "n_dropped_seqs": len(found) - len(kept),
        }
        paths += keep
    return paths, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*")
    ap.add_argument("--from-config", type=Path, default=None,
                    help="sim training config; derive corpus roots + per-corpus val "
                         "exclusion from data.corpora (joint multi-corpus stats)")
    ap.add_argument("--seq-root", default=None,
                    help="with --from-config (single corpus): read labels from this root "
                         "instead of the config's, for a corpus not yet staged")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="substrings; drop matching paths (held-out val seqs — TRAIN-only norm)")
    ap.add_argument("--tail-exclude", nargs="*", default=[],
                    help="substrings whose TAIL is held out (data.tail_val_substrings)")
    ap.add_argument("--tail-fraction", type=float, default=0.0,
                    help="matching data.tail_val_fraction; the tail anchors are dropped")
    ap.add_argument("--max-pose-step-m", type=float, default=None,
                    help="matching data.max_pose_step_m — the tail cut is taken after it")
    add_variant_arg(ap, help_for="the --from-config file")
    args = ap.parse_args()

    corpus_counts: dict[str, dict] = {}
    if args.from_config:
        if args.globs or args.exclude:
            raise SystemExit("--from-config derives the paths and the val exclusion itself; "
                             "do not also pass globs/--exclude")
        paths, corpus_counts = paths_from_config(args.from_config, args.seq_root, args.variant)
    else:
        if args.seq_root:
            raise SystemExit("--seq-root only applies with --from-config")
        if not args.globs:
            raise SystemExit("pass label globs, or --from-config <training config>")
        paths = sorted({p for g in args.globs for p in glob.glob(g)})
        paths = [p for p in paths if not any(x in p for x in args.exclude)]
    if not paths:
        raise SystemExit(f"no labels matched: {args.globs or args.from_config}")
    if bool(args.tail_exclude) != (args.tail_fraction > 0.0):
        raise SystemExit("--tail-exclude and --tail-fraction must be given together")

    feats, wins, acts = [], [], []
    n_total = n_valid = n_tail = 0
    for p in paths:
        z = np.load(p)
        v = z["valid_mask"].astype(bool)
        n_total += len(v)
        if any(x in p for x in args.tail_exclude):
            cut = tail_cut_index(z["valid_mask"].astype(np.uint8),
                                 jump_cum_from_pose(z["pose"], args.max_pose_step_m),
                                 args.tail_fraction)
            n_tail += int(v[cut:].sum())
            v[cut:] = False                      # tail anchors are val — keep them out
        n_valid += int(v.sum())
        if v.sum() == 0:
            continue
        feats.append(z["proprio_feature"][v].astype(np.float64))
        w = z["proprio_window"][v].astype(np.float64)        # (n, T, 13)
        wins.append(w.reshape(-1, w.shape[-1]))
        acts.append(z["action"][v].astype(np.float64))

    F = np.concatenate(feats); W = np.concatenate(wins); A = np.concatenate(acts)

    def stats(x):
        m = np.nanmean(x, axis=0)
        s = np.nanstd(x, axis=0)
        s[s < 1e-6] = 1.0                # guard degenerate channels
        return m.tolist(), s.tolist()

    fm, fs = stats(F); wm, ws = stats(W); am, as_ = stats(A)
    out = {
        "feature_set": "sim",
        "proprio_feature": {"mean": fm, "std": fs},
        "proprio_window": {"mean": wm, "std": ws},
        "action": {"mean": am, "std": as_},
        "_meta": {"n_seqs": len(paths), "n_total": n_total, "n_valid": n_valid,
                  "n_tail_excluded": n_tail,
                  "channels": ["roughness", "slip_ratio", "angular_instability", "ke_rate", "slope_power"]},
    }
    if corpus_counts:
        # provenance for a joint (pooled) stats file: which corpora, how many TRAIN seqs each
        out["_meta"]["corpora"] = corpus_counts
        out["_meta"]["from_config"] = str(args.from_config)
        if args.seq_root:
            out["_meta"]["seq_root_override"] = args.seq_root
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}  seqs={len(paths)}  valid={n_valid}/{n_total}"
          + (f"  (tail-excluded {n_tail:,})" if n_tail else ""))
    if corpus_counts:
        print("  corpora (train seqs): "
              + ", ".join(f"{k}={v['n_train_seqs']}/{v['n_seqs']}"
                          for k, v in corpus_counts.items()))
    print(f"  feature mean={np.round(fm,3)}")
    print(f"  window dims={len(wm)}  action mean={np.round(am,3)}")


if __name__ == "__main__":
    main()
