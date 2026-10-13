#!/usr/bin/env python
"""Legged open-loop free-CEM characterization — GrandTour val (also Isaac ANYmal val).

**DESCRIPTIVE, NOT A COMPARISON.** GrandTour has no NWM baseline and no legged
no-proprio-input counterpart, so no "vs" row exists and none is produced. Nothing here is
comparable to the wheeled TD1/TD2 CEM tables (different robot, ~0.6-0.8 m/s regime,
different domain); every output carries the label. The question is narrower and prior to
any comparison: can the legged multistride WM produce and rank action plans on REAL legged
data at all?

Why a sibling script rather than a `--legged` branch in `openloop_cem.py`. That script's
query population, action prior and aggregation are all wheeled-specific (TartanDrive query
manifests, a single-stride TRAIN prior, `pair_discrepancy_m`); a branch would fork
`load_manifest_queries`, `train_action_prior` and `_aggregate` alike. Only the CEM core
(`ChunkedScorer`, `_query_seed`) is imported, so the wheeled path is untouched.

Why the query is assembled by hand instead of read off `LeggedDataset`. The multistride
schedule `1,1,1,1,5,5,10,10` gives each rollout step its OWN stride, while the loader binds
one `val_stride` to all H steps: at stride 1 its goal frame sits at t+8 anchors (0.8 s), not
the t+34 (3.4 s) the protocol asks for. So the loader supplies the context (k+1 DINO frames,
the 53-dim proprio window, per-dataset normalization) and the goal / expert actions /
realized proprio labels are built from the record at the cumulative anchor offsets
[1,2,3,4,9,14,24,34]. This is sound only because the stored 3-dim action is a per-anchor
RATE (`legged_dataset._action_row`: pose_delta/stride): one set of action norm-stats covers
every stride in the schedule.

Two query modes. By default the script derives its own query set from the schedule
(`build_queries`) — the descriptive mode above. With `--query-manifest` it instead PINS the
set to an external manifest: the schedule is an
input to the gate and the seeded cap, so re-deriving it at a different Δt schedule silently
selects different anchors, and any paired delta against a baseline scored on the manifest's
qids would then be joining two different query sets. Pinning is what makes a same-grid
ours-vs-baseline comparison possible at all; it changes nothing else about the run.

    python tools/eval/openloop_cem_legged.py \
        configs/planning.yaml --variant legged_tg_gt --amp \
        --out out/legged_cem_val.md
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feelwm.config import add_variant_arg, load_config  # noqa: E402
from feelwm.data.dataset import _pose_delta  # noqa: E402
from feelwm.data.legged_dataset import (  # noqa: E402
    LeggedDataset, LeggedDatasetConfig, LeggedSeqSpec, discover_legged_specs,
)
from feelwm.data.normalize import NormStats, denormalize_action, normalize_action  # noqa: E402
from feelwm.planning.cem import CEMConfig, WorldModelScorer, cem_optimize  # noqa: E402
from feelwm.planning.ride_cost import RideCostConfig  # noqa: E402
from feelwm.planning.score import ScoreConfig  # noqa: E402
from feelwm.planning.trajectory import ade_fde, ate_rte, candidate_trajectories  # noqa: E402
from tools.eval.plan_eval import _load_model, _query_context  # noqa: E402
from tools.eval.openloop_cem import ChunkedScorer, _query_seed  # noqa: E402

ANCHOR_DT_S = 0.1          # 10 Hz camera anchor grid, verified on BOTH legged extractions
BANNER = "descriptive / non-comparison"


# --------------------------------------------------------------------- schedule helpers


def cum_offsets(schedule: list[int]) -> np.ndarray:
    """Anchor offsets of each rollout step's END, from the query anchor.

    `[1,1,1,1,5,5,10,10]` → `[1,2,3,4,9,14,24,34]`; the last entry is the goal frame.
    """
    return np.cumsum(np.asarray(schedule, dtype=int))


def cum_seconds(schedule: list[int]) -> np.ndarray:
    return cum_offsets(schedule).astype(np.float64) * ANCHOR_DT_S


def n_supervised_steps(schedule: list[int], proprio_horizon_s: float) -> int:
    """Steps inside the proprio supervision window the WM was trained with (e.g. 1.0 s)."""
    return int((cum_seconds(schedule) <= proprio_horizon_s + 1e-9).sum())


# --------------------------------------------------------------------- specs / datasets


def _repo_path(x: str) -> str:
    p = Path(x)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def legged_specs(wm_cfg: dict) -> tuple[list[LeggedSeqSpec], list[LeggedSeqSpec],
                                        dict[str, str], str]:
    """(train_specs, val_specs, norm_stats, mode) — the SAME split the legged trainer builds.

    Mirrors `trainer.build_legged_trainer_from_config`, BOTH modes:
      - GrandTour/TartanGround (`mode="gt_tg"`): val = the held-out GrandTour missions
        (real temporal OOD, the December block), train = everything else.
      - Isaac ANYmal single-corpus (`mode="isaac"`, `data.isaac_root` set): val = the
        `heldout_substrings` path-groups (the 9 wheeled-isaac groups), train = the rest,
        one train-only norm-stats file under tag "isaac".
    """
    d = wm_cfg["data"]
    if d.get("isaac_root"):
        norm = {"isaac": _repo_path(d["norm_stats_path"])}
        specs = discover_legged_specs(
            isaac_root=d["isaac_root"],
            feat_version_isaac=d.get("feat_version", "anymal_isaac_dinov3_v1"),
        )
        held = d.get("heldout_substrings")
        if not held:
            raise SystemExit("isaac mode requires data.heldout_substrings (val path-groups)")
        val = [s for s in specs if any(h in s.name for h in held)]
        if not val:
            raise SystemExit("no sequences matched data.heldout_substrings — check data.isaac_root")
        return [s for s in specs if not any(h in s.name for h in held)], val, norm, "isaac"
    norm = {"gt": _repo_path(d["norm_stats_path_gt"]), "tg": _repo_path(d["norm_stats_path_tg"])}
    specs = discover_legged_specs(
        grandtour_root=d["grandtour_root"], tartanground_root=d["tartanground_root"],
        feat_version_gt=d.get("feat_version_gt", "gt_dinov2reg_v1"),
        feat_version_tg=d.get("feat_version_tg", "tg_dinov2reg_v1"),
    )
    heldout = set(d.get("heldout_grandtour_missions", []))
    val = [s for s in specs if s.tag == "gt" and s.name in heldout]
    if not val:
        raise SystemExit("no held-out GrandTour missions found — check data.grandtour_root")
    train = [s for s in specs if not (s.tag == "gt" and s.name in heldout)]
    # Mirror the builder's optional TRAIN-side tag filter (`data.train_tags`): the action
    # PRIOR must come from the model's literal own train split — without this, a
    # tgonly eval would silently build its prior from GT-train anchors too.
    train_tags = d.get("train_tags")
    if train_tags:
        train = [s for s in train if s.tag in set(train_tags)]
        if not train:
            raise SystemExit(f"data.train_tags {train_tags} left no train sequences")
    return train, val, norm, "gt_tg"


def external_query_specs(qd: dict, norm: dict[str, str],
                         mode: str) -> tuple[list[LeggedSeqSpec], dict[str, str], str]:
    """Zero-shot query set: sequences from ANOTHER legged corpus, model unchanged.

    Config section (`query_dataset`):
        dataset: gt | tg
        root: <corpus root>
        feat_version: <DINO cache matching the MODEL's encoder, e.g. gt_dinov3_v1>
        sequences: [<name>, ...]        # gt: mission dirs; tg: "<Env>_P####"

    Normalization convention (`norm_convention: model_train_stats`, the
    only one defined): the model is a fixed function — every input transform (proprio
    window z-scoring, action rate normalization) and every output denormalization uses
    the MODEL's own train stats, never the query corpus's. Implemented by mapping the
    query tag to the model's stats file in the norm dict, so the loader's tag routing
    is reused untouched. Only isaac-mode models are supported: a gt/tg-trained model
    has two stats files and no single "own" transform.
    """
    if mode != "isaac":
        raise SystemExit("query_dataset (zero-shot) is only defined for isaac-mode models")
    conv = qd.get("norm_convention", "model_train_stats")
    if conv != "model_train_stats":
        raise SystemExit(f"unknown norm_convention {conv!r} (model_train_stats only)")
    tag = qd["dataset"]
    if tag == "gt":
        specs = discover_legged_specs(grandtour_root=qd["root"],
                                      feat_version_gt=qd["feat_version"])
    elif tag == "tg":
        specs = discover_legged_specs(tartanground_root=qd["root"],
                                      feat_version_tg=qd["feat_version"])
    else:
        raise SystemExit(f"query_dataset.dataset must be gt|tg, got {tag!r}")
    wanted = list(qd["sequences"])
    by_name = {s.name: s for s in specs}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise SystemExit(f"query_dataset sequences not found under {qd['root']}: {missing}")
    return [by_name[n] for n in wanted], {**norm, tag: norm["isaac"]}, tag


def build_context_dataset(wm_cfg: dict, val_specs: list[LeggedSeqSpec],
                          norm: dict[str, str]) -> LeggedDataset:
    """The loader used ONLY for query context (+ its validity/normalization machinery).

    `rollout_horizon=1` on purpose: this dataset's own targets are never read (the goal and
    the proprio labels are taken at the schedule's offsets instead), and every unread target
    step would mmap another 256x768 DINO frame off a 1.5-2 GB feature file per query. On a
    host whose page cache is already under pressure (e.g. concurrent training on mmap'd
    features) that traffic is not free.
    """
    m, d = wm_cfg["model"], wm_cfg["data"]
    return LeggedDataset(val_specs, LeggedDatasetConfig(
        context_len=m["context_len"], rollout_horizon=1,
        n_patch=m["n_patch"], d_vis=m["d_vis"],
        require_vis_context=True, require_vis_target=False, norm_stats=norm,
        split="val", include_dt=True, val_stride=1,
        proprio_horizon_s=d.get("proprio_horizon_s", 1.0),
        stride_sampling="all",
        active_proprio_channels=d.get("active_proprio_channels"),
    ))


# --------------------------------------------------------------------- queries


def _path_cumsum(pose: np.ndarray) -> np.ndarray:
    """Cumulative XY path length over the anchor grid — the movement gate's raw material."""
    xy = pose[:, :2, 3].astype(np.float64)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def build_queries(ds: LeggedDataset, schedule: list[int], anchor_stride: int,
                  speed_floor: float, max_per_mission: int | None, seed: int,
                  missions: set[str] | None = None) -> list[dict]:
    """Default query set: stride-`anchor_stride` anchors, movement-gated, capped.

    A query is usable when the whole 3.4 s plan window is real data: every anchor it touches
    is `valid_mask`-valid, the GOAL frame has a cached DINO feature (it is the score's
    reference), and no dropped-frame gap falls inside [context start, goal] — a gap would
    make the emitted Δt claim an elapsed time the robot did not have (`legged_dataset._grid_ok`
    only screens the loader's own stride-1 horizon, so the check is redone here at 34 anchors).

    Gating and capping happen per MISSION and the cap subsample is seeded, so the realized
    query count per mission is a reported quantity, not a run-to-run accident.
    """
    offs = cum_offsets(schedule)
    goal_off, span_s = int(offs[-1]), float(offs[-1]) * ANCHOR_DT_S
    k = ds.cfg.context_len
    pos = {(rid, t): i for i, (rid, t, _s) in enumerate(ds._index)}
    per_mission: dict[str, list[dict]] = {}
    cumlen: dict[int, np.ndarray] = {}

    for (rid, t), i in pos.items():
        if t % anchor_stride:
            continue
        rec = ds._records[rid]
        if missions is not None and rec.name not in missions:
            continue
        N = len(rec.valid_mask)
        if t + goal_off >= N:
            continue
        if any(rec.valid_mask[t + int(o)] != 1 for o in offs):
            continue
        if (t + goal_off) not in rec.cached_set:      # goal feature must exist
            continue
        if rec.gap_cum is not None and rec.gap_cum[t + goal_off] != rec.gap_cum[t - k]:
            continue
        if rid not in cumlen:
            cumlen[rid] = _path_cumsum(rec.pose)
        speed = float(cumlen[rid][t + goal_off] - cumlen[rid][t]) / span_s
        if speed < speed_floor:
            continue
        per_mission.setdefault(rec.name, []).append(
            {"mission": rec.name, "rid": rid, "anchor": int(t), "ds_index": i,
             "window_speed_mps": round(speed, 4)})

    out: list[dict] = []
    for name in sorted(per_mission):
        pool = sorted(per_mission[name], key=lambda q: q["anchor"])
        n_gated = len(pool)
        if max_per_mission and n_gated > max_per_mission:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(n_gated, size=max_per_mission, replace=False))
            pool = [pool[j] for j in keep]
        for q in pool:
            q["n_gated_in_mission"] = n_gated
        out += pool
    for qid, q in enumerate(out):
        q["qid"] = qid
    return out


def queries_from_manifest(ds: LeggedDataset, schedule: list[int], manifest_path: Path,
                          missions: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Query set PINNED to an external manifest — the exact instances a baseline was scored on.

    `build_queries` derives its own pool FROM the schedule, so changing the schedule moves
    the query set: at goal offset 32 (uniform stride 4) instead of 34 (the multistride
    schedule) both the movement gate and the seeded cap subsample select a DIFFERENT 600
    anchors, and no paired ours-vs-baseline delta over them would be valid. Pinning removes
    both degrees of freedom. The manifest's `qid` is KEPT VERBATIM because it is the join
    key any paired ours-vs-baseline comparison joins on — a re-numbered qid silently
    mis-joins every row.

    The manifest is the query SELECTION, never a validity certificate: it was built for a
    3.4 s horizon and says nothing about anchors +4..+32 being real data. So `build_queries`'
    checks are re-applied at THIS schedule's offsets, and each dropped row is returned with
    a reason rather than vanishing — a pinned run whose n differs from the manifest's must
    say which queries it lost and why.

    The config's gate/cap are NOT applied: they would re-introduce exactly the selection the
    pin exists to remove. Returns `(queries, skipped)`.
    """
    offs = cum_offsets(schedule)
    goal_off, span_s = int(offs[-1]), float(offs[-1]) * ANCHOR_DT_S
    k = ds.cfg.context_len
    pos = {(rid, t): i for i, (rid, t, _s) in enumerate(ds._index)}
    rid_of = {rec.name: rid for rid, rec in enumerate(ds._records)}
    cumlen: dict[int, np.ndarray] = {}

    rows = [json.loads(x) for x in Path(manifest_path).read_text().splitlines() if x.strip()]
    per_mission_total: dict[str, int] = {}
    for m in rows:
        name = m.get("mission", m.get("seq"))
        per_mission_total[name] = per_mission_total.get(name, 0) + 1

    out: list[dict] = []
    skipped: list[dict] = []
    for m in rows:
        name = m.get("mission", m.get("seq"))
        if missions is not None and name not in missions:
            continue                       # a --missions SELECTION, not a validity skip
        qid, t = int(m["qid"]), int(m["our_anchor"])
        rid = rid_of.get(name)

        def drop(reason: str) -> None:
            skipped.append({"qid": qid, "mission": name, "anchor": t, "reason": reason})

        if rid is None:
            drop("mission not in the dataset split")
            continue
        rec = ds._records[rid]
        if (rid, t) not in pos:
            drop("anchor not in the dataset index (context or validity)")
            continue
        if t + goal_off >= len(rec.valid_mask):
            drop(f"goal anchor t+{goal_off} past sequence end")
            continue
        if any(rec.valid_mask[t + int(o)] != 1 for o in offs):
            drop("invalid anchor inside the plan window")
            continue
        if (t + goal_off) not in rec.cached_set:
            drop("goal frame has no cached DINO feature")
            continue
        if rec.gap_cum is not None and rec.gap_cum[t + goal_off] != rec.gap_cum[t - k]:
            drop("dropped-frame gap inside [context start, goal]")
            continue
        if rid not in cumlen:
            cumlen[rid] = _path_cumsum(rec.pose)
        speed = float(cumlen[rid][t + goal_off] - cumlen[rid][t]) / span_s
        out.append({"qid": qid, "mission": name, "rid": rid, "anchor": t,
                    "ds_index": pos[(rid, t)],
                    # recomputed over THIS schedule's window, not copied from the manifest
                    # (whose number is the 3.4 s one) — the report's speed column has to
                    # describe the window the plan actually spans.
                    "window_speed_mps": round(speed, 4),
                    # the pinned pool IS the manifest, so the report's "gated pool" column
                    # reads as the manifest's per-mission count
                    "n_gated_in_mission": per_mission_total[name]})
    out.sort(key=lambda q: q["qid"])
    return out, skipped


# --------------------------------------------------------------------- action prior


def legged_action_prior(specs: list[LeggedSeqSpec], norm: dict[str, str], strides: list[int],
                        n_samples: int, seed: int, speed_floor: float = 0.0) -> dict[int, dict]:
    """Per-stride mean/std/p1/p99 of the normalized action RATE over the TRAIN split.

    This is the free-CEM search distribution, and it is computed PER STRIDE because the
    schedule mixes them: a 1.0 s step's rate distribution is a time-average of the 0.1 s
    one, so it is materially narrower. `CEMConfig.init_std/clip_min/clip_max` accept `[H, 3]`
    arrays, so each step searches the distribution of rates the WM actually saw at ITS Δt
    instead of a single pooled band. Bounds are p1-p99 (by protocol), not min/max: a single
    odometry glitch would otherwise set the search range.

    Sampling is over anchors with a valid (a, a+s) pair, drawn uniformly across the pool with
    a seeded RNG — the full pool is ~200k anchors per stride and only the draw is inverted.
    `speed_floor > 0` restricts the prior to moving anchors (an ARM; 0 = no floor, default).
    """
    stats = {tag: NormStats.from_json(p) for tag, p in norm.items()}
    rng = np.random.default_rng(seed)
    out: dict[int, dict] = {}
    loaded = [(s, np.load(s.label_npz, allow_pickle=False)) for s in specs]
    poses = [(spec, z["pose"].astype(np.float32), z["valid_mask"].astype(np.uint8))
             for spec, z in loaded]

    for s in sorted(set(strides)):
        pool: list[tuple[int, int]] = []          # (spec_index, anchor)
        for si, (_spec, pose, vm) in enumerate(poses):
            N = len(vm)
            if N <= s:
                continue
            a = np.arange(0, N - s)
            ok = (vm[a] == 1) & (vm[a + s] == 1)
            if speed_floor > 0:
                step = np.linalg.norm(pose[a + s, :2, 3] - pose[a, :2, 3], axis=1)
                ok &= step / (s * ANCHOR_DT_S) >= speed_floor
            pool += [(si, int(x)) for x in a[ok]]
        if not pool:
            raise SystemExit(f"empty action prior pool at stride {s}")
        take = rng.choice(len(pool), size=min(n_samples, len(pool)), replace=False)
        rows = []
        for j in take:
            si, a = pool[int(j)]
            spec, pose, _vm = poses[si]
            rate = _pose_delta(pose[a], pose[a + s]) / s
            rows.append(normalize_action(rate[None, :], stats[spec.tag])[0])
        A = np.stack(rows).astype(np.float64)
        out[s] = {"mean": A.mean(0), "std": A.std(0),
                  "p1": np.percentile(A, 1, axis=0), "p99": np.percentile(A, 99, axis=0),
                  "n": len(A)}
    return out


def schedule_prior(prior: dict[int, dict], schedule: list[int]) -> dict[str, torch.Tensor]:
    """Per-stride prior → per-STEP `[H, 3]` tensors for CEMConfig."""
    def stack(key: str) -> torch.Tensor:
        return torch.tensor(np.stack([prior[s][key] for s in schedule]), dtype=torch.float32)
    return {k: stack(k) for k in ("mean", "std", "p1", "p99")}


def forward_speed_mps(mean_norm: torch.Tensor, stats: NormStats, schedule: list[int]) -> float:
    """Physical forward speed of the constant-prior init plan, in m/s.

    The number that makes CEM init PLACEMENT readable: the prior is an average over the train
    split (standing included unless `prior_speed_floor_mps` is set) while every query is
    movement-gated, so the init can sit well below the query regime. Reported, never tuned
    against — `ade_prior_mean` measures the same thing in the metric's own units.
    """
    raw = denormalize_action(mean_norm.detach().cpu().numpy().astype(np.float32), stats)
    dist = float(np.abs(raw[:, 0] * np.asarray(schedule, dtype=np.float32)).sum())
    return dist / (float(sum(schedule)) * ANCHOR_DT_S)


# --------------------------------------------------------------------- per-query assembly


def expert_actions(ds: LeggedDataset, rid: int, anchor: int,
                   schedule: list[int]) -> torch.Tensor:
    """The executed action sequence under the schedule, `[H, 4]` = [rate(3), Δt].

    Reuses the loader's own row builder so the encoding (rate = pose_delta/stride, per-tag
    normalization, Δt in seconds) is defined in exactly one place.
    """
    rec = ds._records[rid]
    N = len(rec.valid_mask)
    starts = np.concatenate([[0], cum_offsets(schedule)[:-1]])
    rows = [ds._action_row(rec, anchor + int(a0), int(s), float(s) * ANCHOR_DT_S, N)
            for a0, s in zip(starts, schedule)]
    return torch.from_numpy(np.stack(rows))


def odom_waypoints(pose: np.ndarray, anchor: int, schedule: list[int]) -> np.ndarray:
    """Executed path in the query's body frame, `[H, 2]` metres — straight off the poses.

    Kept ALONGSIDE the SE(2)-chained expert (the primary ADE reference, as on wheeled data):
    chaining planar body-frame deltas is not the same curve as the projected 3-D odometry
    once the robot pitches and rolls, which a legged platform on slopes does constantly. The
    gap between the two is reported as `se2_chain_discrepancy_m` so the reader knows how
    much of any ADE is integration-model, not planning.
    """
    T0 = np.linalg.inv(pose[anchor].astype(np.float64))
    return np.stack([(T0 @ pose[anchor + int(o)].astype(np.float64))[:2, 3]
                     for o in cum_offsets(schedule)])


def realized_proprio(ds: LeggedDataset, rid: int, anchor: int,
                     schedule: list[int]) -> np.ndarray:
    """Normalized proprio labels at the schedule's anchors, `[H, C]` (active channels only)."""
    rec = ds._records[rid]
    return np.stack([rec.feature[anchor + int(o)][ds._chan_idx] for o in cum_offsets(schedule)])


def plan_proprio(scorer: WorldModelScorer, plan: torch.Tensor,
                 amp_dtype: torch.dtype | None) -> np.ndarray | None:
    """Predicted proprio rollout of ONE plan, `[H, C]`.

    A separate 1-candidate call because `ChunkedScorer` drops the heavy `rollout` entry that
    holds it (it is the reason chunking is affordable at all).
    """
    ctx = (torch.autocast(device_type=scorer.device.type, dtype=amp_dtype) if amp_dtype
           else contextlib.nullcontext())
    with ctx:
        _s, aux = scorer(plan[None].to(scorer.device))
    f = (aux.get("rollout") or {}).get("f_prop_preds")
    return None if f is None else f[0].float().cpu().numpy()


# --------------------------------------------------------------------- partial / resume


def _partial_path(args) -> Path | None:
    """Sidecar the loop appends every finished row to (only when --out is set).

    Removed on successful completion, so its existence MEANS an interrupted run.
    Shard-aware: each shard owns its own partial.
    """
    if not getattr(args, "out", None) or getattr(args, "merge_shards", False):
        return None
    out = Path(args.out) if args.num_shards == 1 else _shard_path(Path(args.out), args.shard)
    return out.with_suffix(".partial.jsonl")


def _open_partial(partial: Path, ident: dict, resume: bool):
    """(append-handle, previously-finished rows) for the incremental sidecar.

    Line 1 is a header identifying the run (config content + shard layout + gated query
    count). Resume REFUSES on any mismatch: per-query results are (seed, qid)-seeded, so
    rows are reusable only when the qid assignment and every result-shaping knob are
    identical — a silent mix of two protocols is worse than a lost run.
    """
    ident_s = json.dumps(ident, sort_keys=True, default=str)
    if partial.exists():
        if not resume:
            raise SystemExit(f"{partial} exists (an interrupted run) — pass --resume to "
                             f"continue it, or delete it to start over")
        lines = partial.read_text().splitlines()
        head = json.loads(lines[0]).get("__header__") if lines else None
        if head is None or json.dumps(head, sort_keys=True, default=str) != ident_s:
            raise SystemExit(f"--resume identity mismatch against {partial} — the partial "
                             f"was produced by a different config/shard layout; delete it "
                             f"or rerun the original command")
        rows = [json.loads(ln) for ln in lines[1:] if ln.strip()]
        return partial.open("a"), rows
    if resume:
        print(f"[legged-cem] --resume: no partial at {partial}, starting fresh", flush=True)
    partial.parent.mkdir(parents=True, exist_ok=True)
    fh = partial.open("w")
    fh.write(json.dumps({"__header__": json.loads(ident_s)}) + "\n")
    fh.flush()
    return fh, []


# --------------------------------------------------------------------- eval


def evaluate(cfg: dict, device: torch.device, args) -> dict:
    schedule = [int(s) for s in cfg["dt_schedule"]]
    H = int(cfg["planning_horizon"])
    if len(schedule) != H:
        raise SystemExit(f"dt_schedule has {len(schedule)} steps, planning_horizon={H}")

    model, model_cfg, channel_names, wm_cfg = _load_model(
        REPO_ROOT / cfg["wm_config"], REPO_ROOT / cfg["checkpoint"], device,
        cfg.get("wm_variant"))
    if model_cfg.rollout_horizon != H:
        raise SystemExit(f"model rollout_horizon {model_cfg.rollout_horizon} != horizon {H}")
    train_specs, val_specs, norm, mode = legged_specs(wm_cfg)
    qd = cfg.get("query_dataset")
    if qd:                                        # zero-shot: queries from another corpus
        query_specs, norm, query_tag = external_query_specs(qd, norm, mode)
    else:                                         # in-domain: the WM's own val split
        query_specs = val_specs
        query_tag = "isaac" if mode == "isaac" else "gt"
    ds = build_context_dataset(wm_cfg, query_specs, norm)
    stats = NormStats.from_json(norm[query_tag])  # action encode/decode stats of the queries
                                                  # (zero-shot: = the model's train stats)

    qcfg = cfg["query"]
    missions = set(args.missions.split(",")) if args.missions else None
    manifest_info = None
    if args.query_manifest:
        mp = Path(args.query_manifest)
        mp = mp if mp.is_absolute() else REPO_ROOT / mp
        queries, skipped = queries_from_manifest(ds, schedule, mp, missions=missions)
        manifest_info = {"path": str(args.query_manifest),
                         "n_rows": len(queries) + len(skipped),
                         "n_used": len(queries), "n_skipped": len(skipped),
                         "skips": skipped}
        print(f"[legged-cem] query set PINNED to {args.query_manifest}: "
              f"{len(queries)} used, {len(skipped)} skipped "
              f"(config gate/cap NOT applied)", flush=True)
        for sk in skipped:
            print(f"  skip qid {sk['qid']} ({sk['mission']} @ {sk['anchor']}): "
                  f"{sk['reason']}", flush=True)
    else:
        queries = build_queries(
            ds, schedule, int(qcfg["anchor_stride"]), float(qcfg["speed_floor_mps"]),
            int(qcfg["max_per_mission"]) if qcfg.get("max_per_mission") else None,
            int(qcfg.get("seed", 0)), missions=missions)
    if not queries:
        raise SystemExit("no queries survived the gate — check speed_floor / missions")

    # Δt-conditioning disclosure: a schedule stride the WM never trained on is a claim about
    # generalization, not a measurement, so the report says so in its own header.
    train_strides = wm_cfg["data"].get("train_strides")
    untrained = (sorted(set(schedule) - set(int(x) for x in train_strides))
                 if train_strides else [])
    if untrained:
        print(f"[legged-cem] ⚠ schedule strides {untrained} are OUTSIDE the WM's "
              f"train_strides {list(train_strides)} — Δt-conditioning is extrapolating",
              flush=True)

    ecfg = cfg["eval"]
    if mode == "isaac":
        # single-corpus model: the only coherent prior is its own train split
        # (--prior-source gt/tg would select zero isaac specs)
        prior_source, prior_specs = "isaac-train", train_specs
    else:
        prior_source = args.prior_source
        prior_specs = ([s for s in train_specs if s.tag == "gt"]
                       if prior_source == "gt" else train_specs)
    prior = legged_action_prior(
        prior_specs, norm, schedule, int(ecfg["prior_samples"]), int(ecfg["seed"]),
        speed_floor=float(args.prior_speed_floor
                          if args.prior_speed_floor is not None
                          else ecfg.get("prior_speed_floor_mps", 0.0)))
    sp = schedule_prior(prior, schedule)
    init_speed = forward_speed_mps(sp["mean"], stats, schedule)
    print(f"[legged-cem] {len(queries)} queries over {len({q['mission'] for q in queries})} "
          f"missions | action prior: source={prior_source}, "
          f"{ {s: prior[s]['n'] for s in prior} } samples/stride, "
          f"init forward speed {init_speed:.3f} m/s "
          f"(query window speed median "
          f"{np.median([q['window_speed_mps'] for q in queries]):.3f} m/s)", flush=True)

    ride_cfg = RideCostConfig(ride_channels=cfg["ride_cost"]["ride_channels"],
                              weights=cfg["ride_cost"].get("weights"),
                              discount=cfg["ride_cost"].get("discount", 1.0))
    arms = {}
    for name in args.arms.split(","):
        if name == "goalonly":
            arms[name] = ScoreConfig(lambda_trav=0.0, lambda_ride=0.0)
        elif name == "ride":
            arms[name] = ScoreConfig(lambda_trav=0.0,
                                     lambda_ride=float(cfg["score"]["lambda_ride"]))
        else:
            raise SystemExit(f"unknown arm {name!r} (goalonly | ride)")
    primary = "goalonly" if "goalonly" in arms else next(iter(arms))

    cem_base = {k: cfg["cem"][k] for k in ("n_candidates", "iterations", "elite_frac", "epsilon")}
    if args.cem_candidates:
        cem_base["n_candidates"] = args.cem_candidates
    if args.cem_iterations:
        cem_base["iterations"] = args.cem_iterations
    amp_dtype = torch.bfloat16 if args.amp else None      # bf16 only (register feature space)
    base_seed = int(ecfg["seed"])
    ph_s = float(ecfg["proprio_horizon_s"])
    n_random = int(args.n_random if args.n_random is not None else ecfg.get("n_random", 64))
    gen = torch.Generator(device=device).manual_seed(base_seed)

    partial = _partial_path(args)
    pf, done_rows = (None, [])
    if partial is not None:
        ident = {"wm_config": str(cfg["wm_config"]), "checkpoint": str(cfg["checkpoint"]),
                 "arms": sorted(arms), "dt_schedule": schedule, "query": dict(qcfg),
                 "query_dataset": ({"dataset": qd["dataset"], "root": str(qd["root"]),
                                    "feat_version": qd["feat_version"],
                                    "n_sequences": len(qd["sequences"])} if qd else None),
                 "cem": dict(cem_base), "eval_seed": base_seed, "n_random": n_random,
                 "shard": args.shard, "num_shards": args.num_shards,
                 "query_manifest": ({k: manifest_info[k] for k in
                                     ("path", "n_rows", "n_used")} if manifest_info else None),
                 "n_queries_gated": len(queries)}
        pf, done_rows = _open_partial(partial, ident, getattr(args, "resume", False))
        if done_rows:
            print(f"[legged-cem] resume: {len(done_rows)} finished rows loaded from "
                  f"{partial}", flush=True)
    done_qids = {r["qid"] for r in done_rows}

    rows, t0 = list(done_rows), time.time()
    for qi, q in enumerate(queries):
        if qi % args.num_shards != args.shard:
            continue
        if q["qid"] in done_qids:
            continue
        if args.max_queries and len(rows) >= args.max_queries:
            break
        rec = ds._records[q["rid"]]
        sample = ds[q["ds_index"]]
        goal = torch.from_numpy(ds._load_vis(rec, q["anchor"] + int(cum_offsets(schedule)[-1])))
        expert = expert_actions(ds, q["rid"], q["anchor"], schedule)
        expert_traj = candidate_trajectories(expert[None], stats)[0]
        odom = odom_waypoints(rec.pose, q["anchor"], schedule)
        context = _query_context(sample, device)

        init = expert.clone()
        init[:, :3] = sp["mean"]                  # free-CEM seed: the train prior, never the expert
        row = {**{k: q[k] for k in ("qid", "mission", "anchor", "window_speed_mps",
                                    "n_gated_in_mission")},
               "expert_m": expert_traj.tolist(), "odom_m": odom.tolist(),
               "se2_chain_discrepancy_m": float(
                   np.linalg.norm(expert_traj.numpy() - odom, axis=1).mean())}

        cem_cfg = CEMConfig(**cem_base, init_std=sp["std"],
                            clip_min=sp["p1"], clip_max=sp["p99"])
        for name, score_cfg in arms.items():
            scorer = WorldModelScorer(model, context, goal.to(device), channel_names,
                                      score_cfg=score_cfg, ride_cfg=ride_cfg,
                                      proprio_horizon_s=ph_s)
            chunked = ChunkedScorer(scorer, args.cem_chunk, amp_dtype=amp_dtype)
            # per-query seeding (never position-in-loop) so --num-shards cannot move a plan
            gen.manual_seed(_query_seed(base_seed, int(q["qid"])))
            res = cem_optimize(chunked, init, cem_cfg, generator=gen)
            traj = candidate_trajectories(res["plan"][None].cpu(), stats)[0]
            ade, fde = ade_fde(traj[None], expert_traj[None])
            ate, rte = ate_rte(traj[None], expert_traj[None])
            row[f"ade_{name}"] = float(ade[0])
            row[f"fde_{name}"] = float(fde[0])
            row[f"ate_{name}"] = float(ate[0])
            row[f"rte_{name}"] = float(rte[0])
            row[f"ade_odom_{name}"] = float(
                np.linalg.norm(traj.numpy() - odom, axis=1).mean())
            row[f"goal_sim_{name}"] = float(res["aux"]["goal_sim"][0])
            row[f"waypoints_m_{name}"] = traj.tolist()
            if name == primary:
                pred = plan_proprio(scorer, res["plan"], amp_dtype)
                if pred is not None:
                    row["prop_pred"] = pred.tolist()
                    row["prop_real"] = realized_proprio(
                        ds, q["rid"], q["anchor"], schedule).tolist()
                # ② score-discrimination diagnostic: does goal-sim separate the chosen plan
                # from draws of the same prior? Uses the PRIMARY arm's scorer, and the same
                # (seed, qid) stream, so it is shard-invariant like the plan itself.
                row.update(_random_diagnostic(chunked, init, sp, cem_cfg, stats, expert_traj,
                                              base_seed, int(q["qid"]), n_random, device))
                row["goal_sim_prior_mean"], row["ade_prior_mean"] = _init_reference(
                    chunked, init, stats, expert_traj)
        rows.append(row)
        if pf is not None:
            pf.write(json.dumps(row) + "\n")
            pf.flush()
        n_new = len(rows) - len(done_rows)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)} queries  ({(time.time() - t0) / max(n_new, 1):.2f} s/query)",
                  flush=True)

    if pf is not None:
        pf.close()
    peak = (torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else float("nan"))
    meta = {
        "n_queries_gated": len(queries), "arms": list(arms), "primary_arm": primary,
        "dt_schedule": schedule, "cum_offsets": cum_offsets(schedule).tolist(),
        "lookahead_s": round(float(cum_seconds(schedule)[-1]), 2),
        "n_supervised_steps": n_supervised_steps(schedule, ph_s),
        "proprio_horizon_s": ph_s, "channel_names": channel_names,
        "ride_channels": list(ride_cfg.ride_channels),
        "query": dict(qcfg), "cem": dict(cem_base),
        "query_manifest": manifest_info,
        "train_strides": ([int(x) for x in train_strides] if train_strides else None),
        "untrained_strides": untrained,
        "wm_mode": mode, "query_tag": query_tag,
        "query_dataset": ({"dataset": qd["dataset"], "root": str(qd["root"]),
                           "feat_version": qd["feat_version"],
                           "n_sequences": len(qd["sequences"]),
                           "norm_convention": qd.get("norm_convention",
                                                     "model_train_stats")}
                          if qd else None),
        "prior": {"source": prior_source,
                  "speed_floor_mps": float(args.prior_speed_floor
                                           if args.prior_speed_floor is not None
                                           else ecfg.get("prior_speed_floor_mps", 0.0)),
                  "n_per_stride": {str(s): int(prior[s]["n"]) for s in sorted(prior)},
                  "init_forward_speed_mps": round(init_speed, 4),
                  "mean_by_stride": {str(s): [round(float(x), 5) for x in prior[s]["mean"]]
                                     for s in sorted(prior)},
                  "std_by_stride": {str(s): [round(float(x), 5) for x in prior[s]["std"]]
                                    for s in sorted(prior)}},
        "amp": "bfloat16" if amp_dtype is not None else "off",
        "cem_chunk": int(args.cem_chunk), "n_random": n_random,
        "peak_gpu_gib": round(peak, 2),
        "s_per_query": round((time.time() - t0) / max(len(rows) - len(done_rows), 1), 2),
        "n_resumed_rows": len(done_rows),
    }
    return _aggregate(rows, meta)


def _random_diagnostic(chunked, init: torch.Tensor, sp: dict, cem_cfg: CEMConfig,
                       stats: NormStats, expert_traj: torch.Tensor, base_seed: int,
                       qid: int, n_random: int, device: torch.device) -> dict:
    """Random draws from the SAME prior, scored by the same scorer (a random-pick control).

    If the chosen plan's goal-sim does not separate from this pool, the score is not
    discriminating and the ADE is prior placement, not planning.
    """
    if n_random <= 0:
        return {}
    g = torch.Generator(device=device).manual_seed(_query_seed(base_seed, qid) ^ 0x5EED)
    H = init.shape[0]
    cand = init[None].expand(n_random, *init.shape).clone().to(device)
    noise = torch.empty(n_random, H, 3, device=device)
    noise.normal_(generator=g)
    cand[..., :3] += noise * sp["std"].to(device)[None]
    cand[..., :3] = cand[..., :3].clamp(min=sp["p1"].to(device).expand(H, 3),
                                        max=sp["p99"].to(device).expand(H, 3))
    scores, aux = chunked(cand)
    gs = aux["goal_sim"].float().cpu()
    trajs = candidate_trajectories(cand.cpu(), stats)
    ades = (trajs - expert_traj[None]).norm(dim=-1).mean(dim=-1)
    return {"goal_sim_random_mean": float(gs.mean()), "goal_sim_random_std": float(gs.std()),
            "goal_sim_random_max": float(gs.max()),
            "ade_random_mean": float(ades.mean()), "ade_random_min": float(ades.min()),
            "n_random": int(n_random)}


def _init_reference(chunked, init: torch.Tensor, stats: NormStats,
                    expert_traj: torch.Tensor) -> tuple[float, float]:
    """Goal-sim and ADE of the UNOPTIMIZED prior-mean plan — the Δ-over-own-prior control."""
    _s, aux = chunked(init[None])
    traj = candidate_trajectories(init[None].cpu(), stats)[0]
    ade, _fde = ade_fde(traj[None], expert_traj[None])
    return float(aux["goal_sim"][0]), float(ade[0])


# --------------------------------------------------------------------- aggregation


def _mean(rows: list[dict], key: str) -> float:
    v = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
    return float(np.mean(v)) if v else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-12 else float("nan")


def _proprio_alignment(rows: list[dict], channels: list[str], n_sup: int) -> dict:
    """Predicted-vs-realized proprio, per channel, inside and outside the supervised window.

    Both sides are in normalized z-space (the labels the head was trained on), so MAE is in
    z-units and directly comparable across channels. `beyond` covers the steps past
    `proprio_horizon_s`: the head was never supervised there, so it is extrapolation, not a
    fair error — reported separately, never pooled with `within`.
    """
    have = [r for r in rows if "prop_pred" in r and "prop_real" in r]
    if not have:
        return {}
    P = np.asarray([r["prop_pred"] for r in have], dtype=np.float64)   # [Q, H, C]
    R = np.asarray([r["prop_real"] for r in have], dtype=np.float64)
    out: dict = {"n_queries": len(have), "n_supervised_steps": n_sup}
    for label, sl in (("within", slice(0, n_sup)), ("beyond", slice(n_sup, P.shape[1]))):
        if P[:, sl].shape[1] == 0:
            continue
        block = {}
        for ci, name in enumerate(channels):
            p, r = P[:, sl, ci].ravel(), R[:, sl, ci].ravel()
            block[name] = {"mae_z": round(float(np.abs(p - r).mean()), 4),
                           "pearson_r": round(_pearson(p, r), 4),
                           "pred_mean_z": round(float(p.mean()), 4),
                           "real_mean_z": round(float(r.mean()), 4)}
        out[label] = block
    return out


def _arm_block(rows: list[dict], arm: str) -> dict:
    return {"ade_m": round(_mean(rows, f"ade_{arm}"), 4),
            "fde_m": round(_mean(rows, f"fde_{arm}"), 4),
            "ate_m": round(_mean(rows, f"ate_{arm}"), 4),
            "rte_m": round(_mean(rows, f"rte_{arm}"), 4),
            "ade_vs_odom_m": round(_mean(rows, f"ade_odom_{arm}"), 4),
            "goal_sim": round(_mean(rows, f"goal_sim_{arm}"), 5)}


def _discrimination(rows: list[dict], arm: str) -> dict:
    """Chosen-vs-random and chosen-vs-prior-mean, the two "did planning do anything" columns."""
    have = [r for r in rows if "goal_sim_random_mean" in r]
    if not have:
        return {}
    d_gs = [r[f"goal_sim_{arm}"] - r["goal_sim_random_mean"] for r in have]
    d_ade = [r[f"ade_{arm}"] - r["ade_prior_mean"] for r in have
             if np.isfinite(r.get("ade_prior_mean", np.nan))]
    beats = [r[f"goal_sim_{arm}"] > r["goal_sim_random_max"] for r in have]
    return {"goal_sim_chosen_minus_random_mean": round(float(np.mean(d_gs)), 5),
            "frac_chosen_beats_random_max": round(float(np.mean(beats)), 4),
            "goal_sim_random_mean": round(_mean(have, "goal_sim_random_mean"), 5),
            "goal_sim_prior_mean": round(_mean(have, "goal_sim_prior_mean"), 5),
            "ade_prior_mean_m": round(_mean(have, "ade_prior_mean"), 4),
            "ade_random_mean_m": round(_mean(have, "ade_random_mean"), 4),
            "ade_chosen_minus_prior_m": round(float(np.mean(d_ade)), 4) if d_ade else float("nan")}


def _aggregate(rows: list[dict], meta: dict) -> dict:
    arms, primary = meta["arms"], meta["primary_arm"]
    channels, n_sup = meta["channel_names"], meta["n_supervised_steps"]

    def block(sel: list[dict]) -> dict:
        return {"n_queries": len(sel),
                "window_speed_mps": round(_mean(sel, "window_speed_mps"), 4),
                "se2_chain_discrepancy_m": round(_mean(sel, "se2_chain_discrepancy_m"), 4),
                **{arm: _arm_block(sel, arm) for arm in arms},
                "discrimination": _discrimination(sel, primary),
                "proprio_alignment": _proprio_alignment(sel, channels, n_sup)}

    per_mission = {name: block([r for r in rows if r["mission"] == name])
                   for name in sorted({r["mission"] for r in rows})}
    for name, b in per_mission.items():
        got = [r for r in rows if r["mission"] == name]
        b["n_gated_in_mission"] = int(got[0]["n_gated_in_mission"]) if got else 0
    return {"label": BANNER, **meta, "n_queries": len(rows),
            "pooled": block(rows), "per_mission": per_mission, "_rows": rows}


# --------------------------------------------------------------------- shards / report


def _shard_path(base: Path, shard: int) -> Path:
    return base.with_name(f"{base.stem}_shard{shard}{base.suffix}")


def _merge_shards(base: Path) -> dict:
    """Recombine `<stem>_shard*.jsonl` — statistics are RECOMPUTED by the same `_aggregate`.

    Everything not derivable from the rows (schedule, prior, CEM budget) is read from shard 0
    and asserted identical across shards: silently averaging two different budgets would be
    worse than crashing.
    """
    metas = sorted(base.parent.glob(f"{base.stem}_shard*.meta.json"))
    if not metas:
        raise SystemExit(f"no {base.stem}_shard*.meta.json beside {base} — nothing to merge")
    loaded = [json.loads(p.read_text()) for p in metas]
    for key in ("dt_schedule", "cem", "prior", "query", "query_manifest", "arms",
                "primary_arm", "amp", "channel_names", "proprio_horizon_s", "n_random"):
        vals = {json.dumps(mt.get(key), sort_keys=True) for mt in loaded}
        if len(vals) > 1:
            raise SystemExit(f"shards disagree on '{key}': {sorted(vals)} — refusing to merge")
    rows: list[dict] = []
    for p in metas:
        jl = p.with_name(p.name.replace(".meta.json", ".jsonl"))
        rows += [json.loads(x) for x in jl.read_text().splitlines() if x.strip()]
    qids = [r["qid"] for r in rows]
    if len(set(qids)) != len(qids):
        raise SystemExit(f"duplicate qids across shards ({len(qids)} rows, {len(set(qids))} "
                         "unique) — shards overlap")
    rows.sort(key=lambda r: r["qid"])
    print(f"merging {len(metas)} shards -> {len(rows)} rows")
    meta = {k: v for k, v in loaded[0].items() if not k.startswith("_")}
    meta["s_per_query"] = round(float(np.mean([mt["s_per_query"] for mt in loaded])), 2)
    meta["peak_gpu_gib"] = max(mt.get("peak_gpu_gib", 0) or 0 for mt in loaded)
    meta["n_shards"] = len(metas)
    for k in ("pooled", "per_mission", "n_queries", "label"):
        meta.pop(k, None)
    return _aggregate(rows, meta)


def _fmt_arm_table(res: dict, arm: str) -> list[str]:
    head = ["| mission | n | gated pool | speed (m/s) | ADE (m) | FDE (m) | ATE | RTE | "
            "ADE vs odom | goal-sim |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    lines = []
    for name, b in res["per_mission"].items():
        a = b[arm]
        lines.append(f"| {name} | {b['n_queries']} | {b['n_gated_in_mission']} | "
                     f"{b['window_speed_mps']:.2f} | {a['ade_m']:.3f} | {a['fde_m']:.3f} | "
                     f"{a['ate_m']:.3f} | {a['rte_m']:.3f} | {a['ade_vs_odom_m']:.3f} | "
                     f"{a['goal_sim']:.4f} |")
    p = res["pooled"][arm]
    lines.append(f"| **pooled** | {res['pooled']['n_queries']} | — | "
                 f"{res['pooled']['window_speed_mps']:.2f} | {p['ade_m']:.3f} | "
                 f"{p['fde_m']:.3f} | {p['ate_m']:.3f} | {p['rte_m']:.3f} | "
                 f"{p['ade_vs_odom_m']:.3f} | {p['goal_sim']:.4f} |")
    return head + lines


def _write_report(res: dict, out_path: Path, cfg: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qd = res.get("query_dataset")
    if qd:
        corpus = {"gt": "GrandTour", "tg": "TartanGround"}.get(qd["dataset"], qd["dataset"])
        title_scope = f"{corpus} ZERO-SHOT (isaac-trained model)"
        caveat = (f"> **{BANNER.upper()} — ZERO-SHOT.** The model never saw {corpus}; every "
                  f"input/output transform uses the MODEL's train stats "
                  f"(`norm_convention: {qd['norm_convention']}`). Read Δ-over-own-prior, "
                  f"never absolute ADE (absolute ADE does not transfer across domains); "
                  f"no valid \"vs\" row exists "
                  f"in this file.")
    elif res.get("wm_mode") == "isaac":
        title_scope = "Isaac ANYmal val (in-domain)"
        caveat = (f"> **{BANNER.upper()}.** No NWM baseline and no legged no-proprio counterpart "
                  f"exists on this corpus, so there is no valid \"vs\" row. Do NOT compare "
                  f"these ADE numbers against wheeled or GrandTour tables — different robot/"
                  f"domain, ~{res['pooled']['window_speed_mps']:.1f} m/s regime here.")
    else:
        title_scope = "GrandTour val"
        caveat = (f"> **{BANNER.upper()}.** GrandTour has no NWM baseline and no legged no-proprio "
                  f"counterpart, so there is no valid \"vs\" row. Do NOT compare these ADE "
                  f"numbers against the wheeled TD1/TD2 tables — different robot, different "
                  f"speed regime (~{res['pooled']['window_speed_mps']:.1f} m/s here), "
                  f"different domain.")
    L = [f"# Legged open-loop free-CEM characterization — {title_scope} "
         f"(**{BANNER}**)", "", caveat, "",
         f"- model: `{cfg['checkpoint']}` (config `{cfg['wm_config']}`)",
         *([f"- query corpus: {qd['dataset']} @ `{qd['root']}` "
            f"(feat `{qd['feat_version']}`, {qd['n_sequences']} sequences, "
            f"norm = {qd['norm_convention']})"] if qd else []),
         f"- protocol: free CEM over the legged action space, seeded from the TRAIN action "
         f"distribution (never the expert); Δt schedule {res['dt_schedule']} = "
         f"{res['lookahead_s']} s lookahead at anchors {res['cum_offsets']}",
         f"- goal: DINO feature of the executed frame at t + {res['lookahead_s']} s; "
         f"score = goal similarity only in the `goalonly` arm",
         *([f"- queries: **PINNED** to `{res['query_manifest']['path']}` "
            f"({res['query_manifest']['n_rows']} manifest rows → "
            f"{res['query_manifest']['n_used']} used, "
            f"{res['query_manifest']['n_skipped']} skipped) — the same (mission, anchor) "
            f"instances the external rows were scored on, with the manifest's `qid` kept as "
            f"the pairing key. The config's own gate/cap are NOT applied: at this schedule's "
            f"goal offset they would select a different query set and void every paired "
            f"delta."]
           if res.get("query_manifest") else
           [f"- queries: every {res['query']['anchor_stride']}th anchor, movement gate "
            f"≥ {res['query']['speed_floor_mps']} m/s over the plan window, cap "
            f"{res['query']['max_per_mission']}/mission (uniform subsample, seed "
            f"{res['query'].get('seed', 0)})"]),
         *([f"- ⚠ **stride disclosure**: schedule strides {res['untrained_strides']} are "
            f"OUTSIDE the WM's trained set {res['train_strides']} "
            f"(`data.train_strides`) — every number here rests on Δt-conditioning "
            f"GENERALIZING to an unseen step length, which this run is the first legged "
            f"measurement of. The wheeled stride sweep is favourable but is a "
            f"different model and corpus."]
           if res.get("untrained_strides") else []),
         f"- CEM: {res['cem']}; action bounds = train p1-p99 per dim PER STRIDE",
         f"- prior: source={res['prior']['source']}, speed floor "
         f"{res['prior']['speed_floor_mps']} m/s, init forward speed "
         f"**{res['prior']['init_forward_speed_mps']} m/s**",
         f"- amp: {res['amp']}, cem_chunk {res['cem_chunk']}, peak "
         f"{res['peak_gpu_gib']} GiB, {res['s_per_query']} s/query", ""]

    for arm in res["arms"]:
        L += [f"## ① Plan-vs-executed ({arm} arm)", ""] + _fmt_arm_table(res, arm) + [""]

    d = res["pooled"].get("discrimination") or {}
    if d:
        L += ["## ② Score discrimination (pooled, primary arm)", "",
              "| quantity | value |", "|---|---|",
              f"| goal-sim, chosen | {res['pooled'][res['primary_arm']]['goal_sim']:.5f} |",
              f"| goal-sim, random-draw mean (n={res['n_random']}/query) | "
              f"{d['goal_sim_random_mean']:.5f} |",
              f"| goal-sim, prior-mean plan | {d['goal_sim_prior_mean']:.5f} |",
              f"| Δ chosen − random mean | {d['goal_sim_chosen_minus_random_mean']:.5f} |",
              f"| fraction chosen > random-pool max | {d['frac_chosen_beats_random_max']:.3f} |",
              f"| ADE, prior-mean plan (m) | {d['ade_prior_mean_m']:.3f} |",
              f"| ADE, random-draw mean (m) | {d['ade_random_mean_m']:.3f} |",
              f"| Δ ADE chosen − prior-mean (m) | {d['ade_chosen_minus_prior_m']:.3f} |", "",
              "A chosen plan whose goal-sim sits inside the random pool means the score is "
              "not discriminating; a Δ-ADE near zero means the number is prior PLACEMENT, "
              "not planning (the same Δ-over-own-prior control as on wheeled data).", ""]

    pa = res["pooled"].get("proprio_alignment") or {}
    if pa:
        L += [f"## ③ Predicted vs realized proprio along the chosen plan "
              f"(z-space, {pa['n_queries']} queries)", ""]
        n_sup, n_all = pa["n_supervised_steps"], len(res["dt_schedule"])
        headers = {
            "within": f"**Within the supervised window** (steps 1–{n_sup} = cum Δt ≤ "
                      f"{res['proprio_horizon_s']} s — what the head was trained on)",
            "beyond": f"**Beyond the supervised window** (steps {n_sup + 1}–{n_all} — the "
                      f"head was never trained there; extrapolation, not a fair error)",
        }
        for label in ("within", "beyond"):
            if label not in pa:
                continue
            L += [headers[label], "",
                  "| channel | MAE (z) | Pearson r | pred mean z | real mean z |",
                  "|---|---|---|---|---|"]
            for name, v in pa[label].items():
                L += [f"| {name} | {v['mae_z']:.3f} | {v['pearson_r']:.3f} | "
                      f"{v['pred_mean_z']:.3f} | {v['real_mean_z']:.3f} |"]
            L += [""]

    qm = res.get("query_manifest") or {}
    if qm.get("skips"):
        L += ["## Skipped manifest queries", "",
              "Rows the manifest names that this schedule cannot plan on. The manifest is a "
              "query SELECTION, not a validity certificate — it was built for a different "
              "horizon, so the plan window is re-checked here.", "",
              "| qid | mission | anchor | reason |", "|---|---|---|---|"]
        L += [f"| {k['qid']} | {k['mission']} | {k['anchor']} | {k['reason']} |"
              for k in qm["skips"]]
        L += [""]

    L += ["## Notes", "",
          f"- `ADE`/`FDE`/`ATE`/`RTE` are plan-vs-expert in metres, where the expert is the "
          f"executed action sequence chained through SE(2) (as on wheeled data — plan and "
          f"reference share one integration model). `ADE vs odom` uses the raw pose path "
          f"instead; the two differ by "
          f"{res['pooled']['se2_chain_discrepancy_m']:.3f} m on average "
          f"(`se2_chain_discrepancy_m`) because a legged robot pitches and rolls.",
          "- Per-mission split is mandatory here: GrandTour missions 12-03 vs 12-09 "
          "show mission-concentrated roughness structure.",
          "- No closed-loop, no comparison rows, no model changes. Nothing here is "
          "ride-cost evidence — a free-CEM plan is never executed, so no realized proprio "
          "exists.", ""]
    out_path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------- cli


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--out", default=None)
    p.add_argument("--wm-config", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--arms", default="goalonly",
                   help="comma list of goalonly|ride (ride = + λ_ride, descriptive secondary)")
    p.add_argument("--missions", default=None, help="comma list; default = all 6 held-out")
    p.add_argument("--query-manifest", default=None,
                   help="pin the query set to an external manifest jsonl (rows carrying "
                        "`qid` + `mission`/`seq` + `our_anchor`, e.g. a baseline's "
                        "query export) instead of "
                        "re-deriving it from the schedule. The manifest's qid is kept — it "
                        "is the key the baseline layer pairs on — and the config's "
                        "gate/cap are bypassed. Default off: existing behaviour unchanged.")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--prior-source", default="gt", choices=["gt", "all"],
                   help="gt = GrandTour-train only (domain- and normalization-matched to the "
                        "queries); all = + TartanGround, whose action norm-stats differ")
    p.add_argument("--prior-speed-floor", type=float, default=None,
                   help="override eval.prior_speed_floor_mps (movement-gated prior ARM)")
    p.add_argument("--n-random", type=int, default=None,
                   help="random draws per query for the score-discrimination diagnostic")
    p.add_argument("--cem-chunk", type=int, default=64,
                   help="candidates per forward pass (bounds GPU memory; result unchanged)")
    p.add_argument("--cem-candidates", type=int, default=None, help="override cem.n_candidates")
    p.add_argument("--cem-iterations", type=int, default=None, help="override cem.iterations")
    p.add_argument("--amp", action="store_true",
                   help="run the rollout in bfloat16 autocast (bf16 ONLY — fp16 overflows "
                        "the autoregressive rollout in register feature space)")
    p.add_argument("--device", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1,
                   help="run query i only when i %% num_shards == shard")
    p.add_argument("--merge-shards", action="store_true",
                   help="no GPU: read <out>_shard*.jsonl and write the combined report")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run from <out>.partial.jsonl (rows are "
                        "(seed, qid)-seeded, so finished queries are reused verbatim; the "
                        "partial's header must match this run's config/shard layout). "
                        "Row order in the final jsonl = resumed rows first, then new ones.")
    add_variant_arg(p, help_for="the planning config")
    add_variant_arg(p, "--wm-variant", "--wm-config (with --wm-config)")
    args = p.parse_args(argv)
    if not 0 <= args.shard < args.num_shards:
        p.error(f"--shard {args.shard} outside 0..{args.num_shards - 1}")

    cfg_path = Path(args.config)
    cfg = load_config((cfg_path if cfg_path.is_absolute() else REPO_ROOT / cfg_path),
                      args.variant)
    if args.wm_config:
        cfg["wm_config"] = args.wm_config
        cfg["wm_variant"] = args.wm_variant
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.merge_shards:
        if not args.out:
            p.error("--merge-shards needs --out (the base path the shards were derived from)")
        res = _merge_shards(Path(args.out))
    else:
        res = evaluate(cfg, device, args)

    print(json.dumps({k: v for k, v in res.items() if k not in ("_rows", "per_mission")},
                     indent=2, default=str))
    if args.out:
        out = Path(args.out) if args.num_shards == 1 or args.merge_shards \
            else _shard_path(Path(args.out), args.shard)
        _write_report(res, out, cfg)
        out.with_suffix(".jsonl").write_text(
            "\n".join(json.dumps(r) for r in res["_rows"]) + "\n")
        if args.num_shards > 1 and not args.merge_shards:
            out.with_suffix(".meta.json").write_text(json.dumps(
                {k: v for k, v in res.items()
                 if k not in ("_rows", "pooled", "per_mission")}, indent=2, default=str))
            print(f"after every shard finishes:\n  python {sys.argv[0]} {args.config} "
                  f"--out {args.out} --merge-shards")
        partial = _partial_path(args)
        if partial is not None and partial.exists():
            partial.unlink()          # success — the final .jsonl above is the record
        print(f"\nwrote {out} (+ .jsonl, {len(res['_rows'])} rows)")


if __name__ == "__main__":
    main(sys.argv[1:])
