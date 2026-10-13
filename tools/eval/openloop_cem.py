#!/usr/bin/env python
"""Open-loop free-CEM planning eval on a frozen query manifest — ADE/FDE vs the expert.

The world model plans by **free CEM over its own action space**, seeded from the TRAIN
action distribution (never from the expert), and each plan is scored against the logged
expert trajectory (ADE/FDE/ATE/RTE, metres). The query population is read from a query
manifest (jsonl): one row per query with `qid`, `split`, `seq`, `nwm_frame`, `our_anchor`
(the anchor in the processed sequence), `our_stride`, `horizon_steps` and
`pair_discrepancy_m` (mean distance between the expert waypoints of the image export the
query was defined on and the same waypoints rebuilt from our poses). A fixed manifest lets
several planners be evaluated on identical instances.

Scope: ADE/goal-reaching only. A free-CEM plan is never executed, so no realized-terrain
metric is produced offline.

    python tools/eval/openloop_cem.py configs/planning.yaml --variant wheeled \
        --split val --query-manifest <queries.jsonl> \
        --wm-config configs/tartandrive.yaml --wm-variant td1 \
        --checkpoint ${OUTPUT_ROOT}/tartandrive/td1/checkpoint_best.pt \
        --out out/td1_val_cem.md
"""
from __future__ import annotations

import argparse
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
from feelwm.data.dataset import DatasetConfig, TartanDriveDataset  # noqa: E402
from feelwm.data.normalize import NormStats, denormalize_action, normalize_action  # noqa: E402
from feelwm.planning.cem import CEMConfig, WorldModelScorer, cem_optimize  # noqa: E402
from feelwm.planning.ride_cost import RideCostConfig  # noqa: E402
from feelwm.planning.score import ScoreConfig  # noqa: E402
from feelwm.planning.trajectory import ade_fde, ate_rte, candidate_trajectories  # noqa: E402
from feelwm.training.trainer import (  # noqa: E402
    _dataclass_from_dict, _replace_dataclass, _resolve_paths,
)
from tools.eval.plan_eval import _load_dataset, _load_model, _query_context  # noqa: E402

SPACING_M = 0.72           # waypoint unit of the query manifests (metres per spacing unit)


# --------------------------------------------------------------------- query manifest


def load_manifest_queries(path: Path, split: str, horizon: int, stride: int,
                          include_unpaired: bool = False) -> list[dict]:
    """Frozen query manifest -> the per-query rows `evaluate` iterates over.

    Rows flagged `"nwm_pred_saved": false` are queries a reference baseline run produced no
    prediction for; they are skipped unless `include_unpaired` is set, so paired
    comparisons stay on the common set. The manifest records the stride/horizon it was built
    for, and a mismatch is an error rather than a silent re-pairing.
    """
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("split", split) != split:
            raise SystemExit(f"{path}: row qid={r['qid']} is split {r['split']!r}, not {split!r}")
        if int(r["our_stride"]) != stride or int(r["horizon_steps"]) != horizon:
            raise SystemExit(f"{path}: built for stride {r['our_stride']} / horizon "
                             f"{r['horizon_steps']}, this run uses {stride} / {horizon}")
        if not include_unpaired and r.get("nwm_pred_saved") is False:
            continue
        out.append({"qid": int(r["qid"]), "seq": r["seq"], "nwm_frame": int(r["nwm_frame"]),
                    "anchor": int(r["our_anchor"]),
                    "pair_discrepancy_m": float(r["pair_discrepancy_m"])})
    return out


# --------------------------------------------------------------------- action prior


def train_action_prior(wm_cfg: dict, stride: int, horizon: int, n_samples: int,
                       seed: int) -> dict[str, torch.Tensor]:
    """mean/std/min/max of the normalized pose-delta dims over the TRAIN split.

    This is the free-CEM search distribution: init the Gaussian at the train mean with the
    train std, and clip samples to the train range so CEM cannot leave the action
    distribution the world model was fitted on. The train split is indexed at the QUERY's
    stride (single stride, every anchor) so the prior describes the same Δt the plans use.
    """
    data_cfg = _dataclass_from_dict(DatasetConfig, _resolve_paths(wm_cfg["data"], Path.cwd()))
    ds = TartanDriveDataset(_replace_dataclass(
        data_cfg, split="train", train_strides=[stride], stride_sampling="all"))
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ds), generator=gen).tolist()[:n_samples]
    acts = []
    for i in order:
        a = ds[i]["action_sequence"][:horizon, :3]
        if torch.isfinite(a).all():
            acts.append(a)
    A = torch.stack(acts)                                  # [n, H, 3]
    flat = A.reshape(-1, 3)
    return {"mean": flat.mean(0), "std": flat.std(0),
            "min": flat.min(0).values, "max": flat.max(0).values, "n": len(A)}


def forward_u_per_step(mean_norm: torch.Tensor, stats: NormStats, stride: int) -> float:
    """Forward step of a constant plan at `mean_norm`, in NWM spacing units per export step.

    The unit the CEM priors/controls are compared in (NWM's waypoint spacing convention):
    the dataset action is a per-0.1 s RATE (`trajectory.actions_to_deltas`), so the physical
    per-step displacement is `denorm(mean)[0] * stride`, and one spacing unit is 0.72 m.
    """
    raw = denormalize_action(mean_norm.detach().cpu().numpy().astype(np.float32), stats)
    return float(raw[0] * stride / SPACING_M)


def scale_init_mean(mean_norm: torch.Tensor, stats: NormStats, factor: float) -> torch.Tensor:
    """Scale the LINEAR (dx, dy) components of a CEM init mean by `factor`.

    **The scaling is applied in PHYSICAL space, then re-normalized** — not to the normalized
    numbers directly. The action norm-stats are z-score with a non-zero offset, and the train
    prior sits (by construction) at ~0 in normalized space: on the TD2 model
    `prior["mean"] = [-0.0158, ...]`, so multiplying the normalized value by 0.3922 moves the
    physical init from 2.501 -> 2.516 spacing units, i.e. nothing. The arm this flag exists
    for is defined physically (init = forward mean at 1.000 u/step, the spacing
    convention), so physical scaling is what implements it: 0.3922 x 2.501 u = 0.981 u.

    `factor == 1.0` short-circuits: the float32 denorm/renorm round trip is not guaranteed
    bit-exact, and the default must reproduce existing runs bit-for-bit.
    """
    if factor == 1.0:
        return mean_norm
    raw = denormalize_action(mean_norm.detach().cpu().numpy().astype(np.float32), stats).copy()
    raw[:2] *= factor
    out = torch.from_numpy(normalize_action(raw, stats)).to(mean_norm.dtype)
    out[2] = mean_norm[2]                   # dyaw carried across, not round-tripped
    return out


# --------------------------------------------------------------------- chunked scorer


class ChunkedScorer:
    """Evaluate a CEM population in chunks — bounds peak GPU memory for large populations.

    Mirrors `WorldModelScorer.__call__`'s contract (scores + aux tensors); the heavy
    `rollout` entry is dropped, which `cem_optimize` does not use. `amp_dtype` runs the
    rollout under autocast — use bfloat16 only (fp16 overflows the autoregressive rollout
    in register feature space).
    """

    def __init__(self, scorer: WorldModelScorer, chunk: int,
                 amp_dtype: torch.dtype | None = None) -> None:
        self.scorer, self.chunk, self.device = scorer, chunk, scorer.device
        self.amp_dtype = amp_dtype

    def _score(self, actions: torch.Tensor):
        if self.amp_dtype is None:
            return self.scorer(actions)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            s, aux = self.scorer(actions)
        return s.float(), {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                           for k, v in aux.items() if k != "rollout"}

    def __call__(self, actions: torch.Tensor):
        if actions.shape[0] <= self.chunk:
            s, aux = self._score(actions)
            return s, {k: v for k, v in aux.items() if k != "rollout"}
        parts = [self._score(actions[i:i + self.chunk])
                 for i in range(0, actions.shape[0], self.chunk)]
        scores = torch.cat([p[0] for p in parts])
        keys = [k for k, v in parts[0][1].items() if torch.is_tensor(v) and k != "rollout"]
        return scores, {k: torch.cat([p[1][k] for p in parts]) for k in keys}


# --------------------------------------------------------------------- eval


def evaluate(cfg: dict, split: str, device: torch.device, args) -> dict:
    H = int(cfg["planning_horizon"])
    stride = int(args.val_stride)
    model, _model_cfg, channel_names, wm_cfg = _load_model(
        REPO_ROOT / cfg["wm_config"], REPO_ROOT / cfg["checkpoint"], device,
        cfg.get("wm_variant"))
    dataset = _load_dataset(wm_cfg, split, keep_stationary=True, val_stride=stride)
    stats = NormStats.from_json(_resolve_paths(wm_cfg["data"], Path.cwd())["norm_stats_path"])

    anchor_index = {(dataset._records[sid].name, t): i
                    for i, (sid, t, _s) in enumerate(dataset._index)}
    queries = load_manifest_queries(Path(args.query_manifest), split, H, stride,
                                    include_unpaired=args.include_unpaired)
    prior = train_action_prior(wm_cfg, stride, H, args.prior_samples, int(cfg["eval"]["seed"]))
    init_mean = scale_init_mean(prior["mean"], stats, float(args.init_mean_scale))
    init_u = forward_u_per_step(init_mean, stats, stride)
    print(f"[cem] {split}: {len(queries)} queries from the manifest | train action prior over "
          f"{prior['n']} sequences: mean={prior['mean'].tolist()} std={prior['std'].tolist()}")
    print(f"[cem] init mean scale {args.init_mean_scale} -> init {init_mean.tolist()} "
          f"= {init_u:.4f} spacing units/step "
          f"(unscaled prior: {forward_u_per_step(prior['mean'], stats, stride):.4f} u/step)")

    ride_cfg = RideCostConfig(ride_channels=cfg["ride_cost"]["ride_channels"],
                              weights=cfg["ride_cost"].get("weights"),
                              discount=cfg["ride_cost"].get("discount", 1.0))
    arms = {"goalonly": ScoreConfig(lambda_trav=0.0, lambda_ride=0.0),
            "full": ScoreConfig(lambda_trav=cfg["score"]["lambda_trav"],
                                lambda_ride=cfg["score"]["lambda_ride"])}
    cem_base = {k: cfg["cem"][k] for k in ("n_candidates", "iterations", "elite_frac", "epsilon")}
    if args.cem_candidates:
        cem_base["n_candidates"] = args.cem_candidates
    if args.cem_iterations:
        cem_base["iterations"] = args.cem_iterations
    amp_dtype = torch.bfloat16 if args.amp else None
    base_seed = int(cfg["eval"]["seed"])
    gen = torch.Generator(device=device).manual_seed(base_seed)

    rows, skipped, t0 = [], 0, time.time()
    for qi, q in enumerate(queries):
        # Shard on the query's POSITION in the manifest, before any skip test, so the
        # shards partition the same query set no matter which queries turn out unusable.
        if qi % args.num_shards != args.shard:
            continue
        if args.max_queries and len(rows) >= args.max_queries:
            break
        key = (q["seq"], q["anchor"])
        if key not in anchor_index:
            skipped += 1                  # anchor lacks valid context/features at this stride
            continue
        sample = dataset[anchor_index[key]]
        if sample["action_sequence"].shape[0] < H or not bool(sample["valid_target_mask"][:H].all()):
            skipped += 1
            continue
        goal = sample["z_vis_target"][H - 1].to(device)
        expert = sample["action_sequence"][:H].to(device)
        expert_traj = candidate_trajectories(expert[None].cpu(), stats)[0]
        # free-CEM seed: train mean everywhere, the query's own Δt schedule kept intact
        init = expert.clone()
        init[:, :3] = init_mean.to(device)[None]
        # Re-seed PER QUERY. One generator consumed sequentially across queries makes the
        # CEM population a function of how many queries ran before this one, so --num-shards
        # silently changes every plan: a 4-way sharded re-run with a sequential generator
        # matched the unsharded aggregate to within 0.3 % but not one query bit-for-bit
        # (|Δ| median 0.19-0.31 m). Deriving the stream from (seed, qid) makes a
        # run reproducible whatever the shard count.
        gen.manual_seed(_query_seed(base_seed, int(q["qid"])))
        row = {**q, "n_anchor_used": int(q["anchor"])}
        # Store the trajectories, not only the scalars they reduce to. The external
        # baseline (NWM) rows carry their `waypoints_m` too, so every row can be
        # re-analysed after the fact: step-length audit (is an ADE a speed offset or a
        # shape error?), prefix tiers, recomputing a metric not thought of at run time.
        # The arrays are (H, 2) metres — 100 queries x 8 x 2 floats per arm, i.e. free.
        row["expert_m"] = expert_traj.tolist()
        for name, score_cfg in arms.items():
            scorer = WorldModelScorer(model, _query_context(sample, device), goal, channel_names,
                                      score_cfg=score_cfg, ride_cfg=ride_cfg,
                                      proprio_horizon_s=float(cfg["eval"]["proprio_horizon_s"]))
            res = cem_optimize(
                ChunkedScorer(scorer, args.cem_chunk, amp_dtype=amp_dtype), init,
                CEMConfig(**cem_base, init_std=prior["std"], clip_min=prior["min"],
                          clip_max=prior["max"]),
                generator=gen)
            traj = candidate_trajectories(res["plan"][None].cpu(), stats)[0]
            ade, fde = ade_fde(traj[None], expert_traj[None])
            ate, rte = ate_rte(traj[None], expert_traj[None])
            row[f"ade_{name}"] = float(ade[0])
            row[f"fde_{name}"] = float(fde[0])
            # ATE/RTE put our rows in the NWM paper's units (its Table 2 / Table 7):
            # ATE is the RMS of the same per-waypoint errors ADE averages, RTE is the
            # per-step displacement error, which does not accumulate.
            row[f"ate_{name}"] = float(ate[0])
            row[f"rte_{name}"] = float(rte[0])
            row[f"goal_sim_{name}"] = float(res["aux"]["goal_sim"][0])
            row[f"waypoints_m_{name}"] = traj.tolist()
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}/{len(queries)} queries  "
                  f"({(time.time() - t0) / len(rows):.2f} s/query)", flush=True)

    # Record the peak so the next person does not have to estimate it. --cem-chunk is the
    # knob that fits the run into whatever GPU headroom is available, and its cost is
    # wall-clock only: ChunkedScorer splits the population along dim 0 and the candidates
    # are scored independently, so the chunk size does not change the result.
    peak = (torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda" else float("nan"))
    # The init is an ARM of this experiment (fairness symmetry with NWM's corrected prior),
    # so the factor and the physical init it produces travel with the run and across the
    # shard merge — an unlabelled report from a scaled-init run would be unreadable.
    cem_meta = {**cem_base, "init_mean_scale": float(args.init_mean_scale),
                "init_forward_u_per_step": round(init_u, 4)}
    return _aggregate(rows, split, skipped, prior, cem_meta, stride, H,
                      (time.time() - t0) / max(len(rows), 1),
                      "bfloat16" if amp_dtype is not None else "off",
                      int(args.cem_chunk), peak)


def _aggregate(rows, split, skipped, prior, cem_base, stride, H, s_per_query, amp,
               cem_chunk=None, peak_gib=float("nan")) -> dict:
    def m(key, sel=None):
        v = [r[key] for r in (rows if sel is None else sel) if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(v)) if v else float("nan")

    clean = [r for r in rows if r["pair_discrepancy_m"] <= 0.5]
    scale = float(cem_base.get("init_mean_scale", 1.0))
    out = {
        "split": split, "n_queries": len(rows), "n_skipped": skipped,
        "horizon_anchors": H * stride, "stride": stride, "s_per_query": round(s_per_query, 2),
        "cem": {**cem_base,
                "init": "train action mean/std, clipped to train min/max" if scale == 1.0 else
                        f"train action mean with LINEAR dims x{scale} in physical space "
                        f"(std/clip bounds unchanged)"},
        "amp": amp,
        "cem_chunk": cem_chunk, "peak_gpu_gib": round(peak_gib, 2),
        "prior_mean": [round(x, 4) for x in prior["mean"].tolist()],
        "prior_std": [round(x, 4) for x in prior["std"].tolist()],
        "pair_discrepancy_median_m": float(np.median([r["pair_discrepancy_m"] for r in rows])),
        "ours_cem_ade_goalonly": m("ade_goalonly"), "ours_cem_fde_goalonly": m("fde_goalonly"),
        "ours_cem_ade_full": m("ade_full"), "ours_cem_fde_full": m("fde_full"),
        # NWM-paper units (Table 2 / Table 7). Reported alongside ADE/FDE, never instead:
        # the primary comparison is in ADE, and RMS aggregation is tail-sensitive, so the
        # two are not interchangeable.
        "ours_cem_ate_goalonly": m("ate_goalonly"), "ours_cem_rte_goalonly": m("rte_goalonly"),
        "ours_cem_ate_full": m("ate_full"), "ours_cem_rte_full": m("rte_full"),
        "n_clean": len(clean),
        "ours_cem_ade_goalonly_clean": m("ade_goalonly", clean),
    }
    out["_rows"] = rows
    return out


def _query_seed(base_seed: int, qid: int) -> int:
    """Per-query CEM seed. Must depend only on (base_seed, qid) — never on position in the
    loop — so sharding cannot change a plan. Kept inside int64 for torch's Generator."""
    return (base_seed * 1_000_003 + qid * 2_654_435_761) % (2**62)


def _shard_path(base: Path, shard: int) -> Path:
    return base.with_name(f"{base.stem}_shard{shard}{base.suffix}")


def _merge_shards(base: Path) -> dict:
    """Recombine `<base stem>_shard*.jsonl` into one report.

    The per-query rows are self-contained, so the statistics are recomputed by the SAME
    `_aggregate` the single-process path uses. Row-for-row equality with an unsharded run
    additionally needs the per-query CEM seeding (`_query_seed`); rows produced with one
    sequential generator are a valid re-draw, not a reproduction — a 4-way sharded run of
    that kind matched the unsharded aggregate to 0.3 % with no query bit-identical.
    Everything that is not derivable from the rows (prior, CEM budget,
    stride/horizon) is read from shard 0's sidecar and asserted identical across shards,
    because silently averaging two different budgets would be worse than crashing.
    """
    metas = sorted(base.parent.glob(f"{base.stem}_shard*.meta.json"))
    if not metas:
        raise SystemExit(f"no {base.stem}_shard*.meta.json beside {base} — nothing to merge")
    loaded = [json.loads(p.read_text()) for p in metas]
    for key in ("split", "stride", "horizon_anchors", "cem", "amp", "cem_chunk",
                "prior_mean", "prior_std"):
        vals = {json.dumps(mt.get(key), sort_keys=True) for mt in loaded}
        if len(vals) > 1:
            raise SystemExit(f"shards disagree on '{key}': {sorted(vals)} — refusing to merge")
    m0 = loaded[0]

    rows: list[dict] = []
    for p in metas:
        jl = p.with_name(p.name.replace(".meta.json", ".jsonl"))
        rows += [json.loads(x) for x in jl.read_text().splitlines() if x.strip()]
    seen = [r["qid"] for r in rows]
    if len(set(seen)) != len(seen):
        raise SystemExit(f"duplicate qids across shards ({len(seen)} rows, "
                         f"{len(set(seen))} unique) — shards overlap")
    rows.sort(key=lambda r: r["qid"])

    print(f"merging {len(metas)} shards -> {len(rows)} rows")
    prior = {"mean": np.asarray(m0["prior_mean"]), "std": np.asarray(m0["prior_std"])}
    stride = int(m0["stride"])
    res = _aggregate(rows, m0["split"], sum(int(mt["n_skipped"]) for mt in loaded), prior,
                     dict(m0["cem"]), stride, int(m0["horizon_anchors"]) // stride,
                     float(np.mean([mt["s_per_query"] for mt in loaded])), m0["amp"],
                     m0.get("cem_chunk"), max(mt.get("peak_gpu_gib", 0) or 0 for mt in loaded))
    res["n_shards"] = len(metas)
    return res


def _write_report(res: dict, out_path: Path, cfg: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in res if not k.startswith("_")]
    lines = [f"# Free-CEM planning eval — split `{res['split']}`", "",
             f"- model: `{cfg['checkpoint']}`",
             f"- protocol: free CEM over our action space, seeded from the TRAIN action "
             f"distribution (never the expert), on the frozen query manifest; "
             f"horizon {res['horizon_anchors']} anchors ≈ {res['horizon_anchors'] * 0.1:.1f} s.",
             ""]
    lines += [f"- **{k}**: {res[k]}" for k in keys]
    lines += ["",
              "`goalonly` is the NWM-comparable row (score = goal similarity only, as NWM's "
              "score has only a goal term); `full` adds our λ_ride/λ_trav terms and is "
              "ours-internal — never the comparison row.",
              "`pair_discrepancy_m` = mean distance between the export's expert waypoints and "
              "the same waypoints rebuilt from our poses; `*_clean` restricts to ≤ 0.5 m.",
              "`cem.init_mean_scale` / `cem.init_forward_u_per_step` record the CEM "
              "INITIALISATION arm (1.0 = the measured train prior).",
              "ADE/FDE are plan-vs-expert (m). No realized-roughness GT exists offline, so "
              "nothing here is evidence of terrain-aware (ride-cost) plan selection."]
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--out", default=None)
    p.add_argument("--wm-config", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--val-stride", type=int, default=4, help="4 = NWM's 32-anchor horizon")
    p.add_argument("--query-manifest", default=None,
                   help="query manifest (jsonl), one row per query (see module docstring)")
    p.add_argument("--include-unpaired", action="store_true",
                   help="also run manifest rows flagged nwm_pred_saved=false "
                        "(skipped by default)")
    # Fairness arm, not a tuning knob: NWM's prior was corrected and re-run while our init
    # was left at the measured train mean (2.550 u/step, ~2x the eval queries' own step
    # scale). 0.3922 = 1.000 / 2.550 puts the init at the spacing convention. Applied
    # in PHYSICAL space; search std and clip bounds are untouched.
    p.add_argument("--init-mean-scale", type=float, default=1.0,
                   help="multiply the linear (dx, dy) part of the CEM init mean by this "
                        "factor, in physical units (1.0 = the measured train prior)")
    p.add_argument("--prior-samples", type=int, default=2000)
    p.add_argument("--cem-chunk", type=int, default=64,
                   help="candidates scored per forward pass (bounds GPU memory; result unchanged)")
    p.add_argument("--cem-candidates", type=int, default=None,
                   help="override cem.n_candidates (smoke runs; the budget used is reported)")
    p.add_argument("--cem-iterations", type=int, default=None, help="override cem.iterations")
    p.add_argument("--amp", action="store_true",
                   help="run the rollout in bfloat16 autocast (halves activation memory)")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    # Sharding is a WALL-CLOCK knob, not a memory one: each shard is a separate process
    # holding its own copy of the model, so per-GPU peak is unchanged and total usage is
    # N x. Use --cem-chunk for memory. One shard per free card turns ~78 min/split into
    # ~20 min on four.
    p.add_argument("--shard", type=int, default=0, help="this shard's index, 0..num-shards-1")
    p.add_argument("--num-shards", type=int, default=1,
                   help="run query i only when i %% num_shards == shard")
    p.add_argument("--merge-shards", action="store_true",
                   help="no GPU: read <out>_shard*.jsonl written by the shards and write "
                        "the combined report at <out>")
    add_variant_arg(p, help_for="the planning config")
    add_variant_arg(p, "--wm-variant", "--wm-config (with --wm-config)")
    args = p.parse_args(argv)
    if not 0 <= args.shard < args.num_shards:
        p.error(f"--shard {args.shard} outside 0..{args.num_shards - 1}")
    if not args.merge_shards and not args.query_manifest:
        p.error("--query-manifest is required (except with --merge-shards)")

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
        res = evaluate(cfg, args.split, device, args)
    for k in [k for k in res if not k.startswith("_")]:
        print(f"  {k}: {res[k]}")
    if args.out:
        out = Path(args.out) if args.num_shards == 1 or args.merge_shards \
            else _shard_path(Path(args.out), args.shard)
        _write_report(res, out, cfg)
        out.with_suffix(".jsonl").write_text(
            "\n".join(json.dumps(r) for r in res["_rows"]) + "\n")
        # The merge needs the summary fields that are not derivable from the rows
        # (prior, cem budget, stride/H, peak memory), so each shard leaves them beside it.
        if args.num_shards > 1 and not args.merge_shards:
            out.with_suffix(".meta.json").write_text(json.dumps(
                {k: v for k, v in res.items() if not k.startswith("_")}, indent=2))
        print(f"\nwrote {out} (+ .jsonl, {len(res['_rows'])} rows)")
        if args.num_shards > 1 and not args.merge_shards:
            print(f"after every shard finishes:\n"
                  f"  python {sys.argv[0]} {args.config} --split {args.split} "
                  f"--out {args.out} --merge-shards")


if __name__ == "__main__":
    main(sys.argv[1:])
