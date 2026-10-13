"""Offline planning evaluation — library-ranking test of the WM inside the planning score.

Open-loop trajectory ranking on TartanDrive (NWM/ReL-NWM protocol; true closed-loop is the
Gazebo sim track). For each query anchor we roll a pool of real candidate action sequences
(+ the query's own expert) from the query's visual context through the WM and score them at
the score level (goal_sim − λ_trav·trav_risk − λ_ride·ride_cost).

Metrics (the output-key prefixes C1_ / C2a_ / C2b_ name these families):
  C1_  — goal reaching: expert-rank percentile + goal-feature distance of the selected plan
         (compare against an optional no-proprio-input or visual-only checkpoint).
  C2a_ — ride_cost validity: per-anchor predicted ride_cost vs actually-experienced GT
         roughness/instability (Pearson r, own-context rollout).
  C2b_ — λ_ride 0-vs-1: realized GT roughness of the selected library plan — does adding the
         ride term make the planner pick smoother real trajectories?

Single-GPU inference only — do NOT launch with torchrun.

Usage:
  python tools/eval/plan_eval.py configs/planning.yaml --variant wheeled \
      --split val --out out/plan_eval_val.md [--cem] [--max-queries 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feelwm.config import add_variant_arg, load_config
from feelwm.data.dataset import DatasetConfig, TartanDriveDataset
from feelwm.data.normalize import NormStats
from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel
from feelwm.planning.candidates import inject_expert, stack_samples
from feelwm.planning.cem import (
    CEMConfig,
    WorldModelScorer,
    cem_optimize,
    expert_rank_percentile,
    rank_candidates,
)
from feelwm.planning.ride_cost import RideCostConfig
from feelwm.planning.score import ScoreConfig, combine_score
from feelwm.planning.trajectory import ade_fde, candidate_trajectories
from feelwm.training.trainer import (
    _dataclass_from_dict,
    _replace_dataclass,
    _resolve_paths,
)
from tools.eval_offline import _stationary_seqs


# --------------------------------------------------------------------------- loading


def _load_model(wm_cfg_path: Path, ckpt_path: Path, device: torch.device,
                variant: str | None = None):
    cfg = load_config(wm_cfg_path, variant)
    model_cfg: ModelConfig = _dataclass_from_dict(ModelConfig, cfg["model"])
    model = TerrainWorldModel(model_cfg).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    channel_names = list(cfg["loss"]["channel_names"])
    return model, model_cfg, channel_names, cfg


def _load_dataset(wm_cfg: dict, split: str, keep_stationary: bool = False,
                  val_stride: int | None = None) -> TartanDriveDataset:
    data_cfg: DatasetConfig = _dataclass_from_dict(
        DatasetConfig, _resolve_paths(wm_cfg["data"], Path.cwd())
    )
    excluded: list[str] = []
    if not keep_stationary:
        # parked recordings (sam_loop, trash) have near-zero GT variance: they pollute
        # the ride-cost validity correlation and seed degenerate library candidates. Same
        # criterion as tools/eval_offline.py (<5% of anchors moving ≥0.5 m/s).
        excluded = _stationary_seqs(data_cfg, split, speed_floor=0.5, max_moving_frac=0.05)
        if excluded:
            print(f"[plan_eval] dropping stationary sequence(s) from {split}: {excluded}")
    overrides = dict(split=split, exclude_sequences=excluded)
    if val_stride is not None:
        # Each of the H rollout steps spans `val_stride` anchors (Δt = val_stride·0.1 s),
        # so the planning horizon = H·val_stride anchors. Use to MATCH an external
        # baseline's goal distance (e.g. NWM export stride 4 → 8·4 = 32 anchors ≈ 3.2 s).
        overrides["val_stride"] = val_stride
        print(f"[plan_eval] val_stride={val_stride} → horizon "
              f"{val_stride}·H anchors ({val_stride * 0.1:.1f} s/step)")
    return TartanDriveDataset(_replace_dataclass(data_cfg, **overrides))


def _query_context(sample: dict, device: torch.device) -> dict:
    """Single-query initial_batch (batch dim 1) for forward_rollout."""
    out = {}
    for k in ("z_vis_context", "proprio_window_context", "action_context"):
        if k in sample:
            out[k] = sample[k][None].to(device)
    return out


# --------------------------------------------------------------------------- metrics


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp_min(1e-12)
    return float((x @ y) / denom)


def evaluate(cfg: dict, split: str, device: torch.device, do_cem: bool,
             keep_stationary: bool = False, val_stride: int | None = None,
             eps_sweep: list[float] | None = None) -> dict:
    H = int(cfg["planning_horizon"])
    ecfg = cfg["eval"]
    ride_cfg = RideCostConfig(
        ride_channels=cfg["ride_cost"]["ride_channels"],
        weights=cfg["ride_cost"].get("weights"),
        discount=cfg["ride_cost"].get("discount", 1.0),
    )
    ph_s = float(ecfg["proprio_horizon_s"])

    model, model_cfg, channel_names, wm_cfg = _load_model(
        REPO_ROOT / cfg["wm_config"], REPO_ROOT / cfg["checkpoint"], device,
        cfg.get("wm_variant"))
    dataset = _load_dataset(wm_cfg, split, keep_stationary=keep_stationary,
                            val_stride=val_stride)
    stats = NormStats.from_json(
        _resolve_paths(wm_cfg["data"], Path.cwd())["norm_stats_path"]
    )

    # Randomized index order (avoid scanning/loading the whole split). A usable anchor
    # spans the full planning horizon with no padding inside it (valid_target_mask[:H]).
    gen = torch.Generator().manual_seed(int(ecfg["seed"]))
    order = torch.randperm(len(dataset), generator=gen).tolist()

    def _usable(s) -> bool:
        return s["action_sequence"].shape[0] >= H and bool(s["valid_target_mask"][:H].all())

    # a fixed candidate library, drawn once from the front of the shuffled index.
    lib_size = int(ecfg["library_size"])
    lib_samples, lib_idx, cursor = [], [], 0
    while len(lib_samples) < lib_size and cursor < len(order):
        s = dataset[order[cursor]]
        if _usable(s):
            lib_samples.append(s)
            lib_idx.append(order[cursor])
        cursor += 1
    if not lib_samples:
        raise ValueError("no usable anchor for the candidate library")

    # GT-roughness column resolved against the ACTIVE channel order (channel_names),
    # which is what f_prop_gt / f_prop_target are laid out in. Indexing via
    # ride_channels would be a latent bug — correct only while both lists happen to
    # put roughness at 0.
    roughness_col = channel_names.index("roughness") if "roughness" in channel_names else 0
    goal_eps = float(ecfg.get("goal_sim_epsilon") or 0.0)
    score_full = ScoreConfig(lambda_trav=cfg["score"]["lambda_trav"], lambda_ride=cfg["score"]["lambda_ride"])
    score_goalonly = ScoreConfig(lambda_trav=0.0, lambda_ride=0.0)

    # library pool stacked once (query-independent); expert is injected per query.
    lib_pool = stack_samples(lib_samples, horizon=H, source_index=lib_idx)

    max_q = ecfg.get("max_queries")
    rows = []
    for qi in order[cursor:]:
        if max_q and len(rows) >= max_q:
            break
        q = dataset[qi]
        if not _usable(q):
            continue
        goal = q["z_vis_target"][H - 1].to(device)  # [P, D]
        pool = inject_expert(lib_pool, q["action_sequence"][:H], at=0).to(device)

        scorer = WorldModelScorer(
            model, _query_context(q, device), goal, channel_names,
            score_cfg=score_full, ride_cfg=ride_cfg, proprio_horizon_s=ph_s,
        )
        res = rank_candidates(scorer, pool.actions)
        scores, aux = res["scores"], res
        pct = expert_rank_percentile(scores, pool.expert_index)
        # Goal-reaching diagnostic: expert rank on goal_sim ALONE — separates "goal_sim
        # cannot find the expert" (visual drift) from "expert legitimately penalized for
        # roughness by the full score" (the ride term working as designed).
        pct_goal = expert_rank_percentile(aux["goal_sim"], pool.expert_index)

        # selected-plan goal distance (1 − goal_sim of the top plan).
        best = res["best"]
        sel_goal_sim = float(aux["goal_sim"][best])

        # C2b_: realized GT roughness of the selected library plan under λ_ride 0 vs 1.
        # exclude the expert row (NaN GT). roughness from each candidate's own f_prop_gt.
        gt_rough = pool.f_prop_gt[..., roughness_col]  # [N, H]
        hm = pool.horizon_mask
        gt_rough_mean = _masked_mean_rows(gt_rough, hm)  # [N], NaN for expert
        full_best = _best_real(aux["goal_sim"], aux["ride_cost"], aux["trav_risk"], score_full, pool.expert_index)
        goalonly_best = _best_real(aux["goal_sim"], aux["ride_cost"], aux["trav_risk"], score_goalonly, pool.expert_index)
        # Goal-matched variant (eval.goal_sim_epsilon): the ride-aware pick is restricted
        # to candidates whose goal_sim is within ε of the best real candidate's, so a
        # "reduction" means smoother AMONG comparably-goal-reaching plans — the
        # "just pick the slowest" strategy (a speed confound) is cut off.
        gs_real = _nan_expert(aux["goal_sim"], pool.expert_index)
        gs_max = torch.nan_to_num(gs_real, nan=float("-inf")).max()
        band = aux["goal_sim"] >= (gs_max - goal_eps)
        full_best_eps = _best_real(aux["goal_sim"], aux["ride_cost"], aux["trav_risk"],
                                   score_full, pool.expert_index, allowed=band)
        # ε-sensitivity sweep (selection is post-scoring CPU math → free): on OOD the
        # goal_sim spread can be smaller than ε, saturating the band — the sweep shows
        # where the goal constraint actually starts to bite.
        sweep_fields = {}
        for e in (eps_sweep or []):
            b = aux["goal_sim"] >= (gs_max - e)
            j = _best_real(aux["goal_sim"], aux["ride_cost"], aux["trav_risk"],
                           score_full, pool.expert_index, allowed=b)
            sweep_fields[f"realized_rough_full_eps@{e:g}"] = _safe(gt_rough_mean, j)
            sweep_fields[f"band_n@{e:g}"] = int(b.sum()) - int(bool(b[pool.expert_index]))

        # NWM-protocol metrics: ADE/FDE (meters) between the selected real plan's SE(2)
        # trajectory and the expert's GT trajectory, in the shared query frame.
        trajs = candidate_trajectories(pool.actions.cpu(), stats)  # [N, H, 2]
        expert_traj = trajs[pool.expert_index]
        ade_all, fde_all = ade_fde(trajs, expert_traj[None])       # [N]

        row = {
            "query": int(qi),
            "expert_pct": pct,
            "expert_pct_goalonly": pct_goal,
            "sel_goal_sim": sel_goal_sim,
            "realized_rough_full": _safe(gt_rough_mean, full_best),
            "realized_rough_goalonly": _safe(gt_rough_mean, goalonly_best),
            "realized_rough_full_eps": _safe(gt_rough_mean, full_best_eps),
            "band_n": int(band.sum()) - int(bool(band[pool.expert_index])),
            "ade_full_eps": float(ade_all[full_best_eps]),
            "fde_full_eps": float(fde_all[full_best_eps]),
            **sweep_fields,
            "ade_full": float(ade_all[full_best]),
            "fde_full": float(fde_all[full_best]),
            "ade_goalonly": float(ade_all[goalonly_best]),
            "fde_goalonly": float(fde_all[goalonly_best]),
            # expected ADE of picking a random real candidate (scale reference)
            "ade_random": float(_safe(_nan_expert(ade_all, pool.expert_index).nanmean()[None], 0)),
        }
        # C2a_: own-context ride_cost validity. The expert row of the scored pool IS
        # the query's own action sequence rolled from its own context — reuse it.
        row["pred_ride"] = (
            float(aux["ride_cost"][pool.expert_index]) if aux["ride_cost"] is not None else float("nan")
        )
        row["gt_rough"] = _safe(_masked_mean_rows(
            q["f_prop_target"][:H][..., roughness_col][None],
            q.get("proprio_horizon_mask", torch.ones(H, dtype=torch.bool))[:H][None]), 0)

        if do_cem:
            cem_cfg = CEMConfig(**{k: cfg["cem"][k] for k in
                                   ("n_candidates", "iterations", "elite_frac", "init_sigma", "epsilon")})
            cem_res = cem_optimize(scorer, q["action_sequence"][:H].to(device), cem_cfg, generator=gen)
            row["cem_score"] = cem_res["score"]
            row["cem_goal_sim"] = float(cem_res["aux"]["goal_sim"][0])
            cem_traj = candidate_trajectories(cem_res["plan"][None].cpu(), stats)[0]
            cem_ade, cem_fde = ade_fde(cem_traj[None], expert_traj[None])
            row["cem_ade"], row["cem_fde"] = float(cem_ade[0]), float(cem_fde[0])
        rows.append(row)

    return _aggregate(rows, split, do_cem)


# --------------------------------------------------------------------------- helpers


def _masked_mean_rows(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=-1)
    m = mask.to(device=x.device, dtype=x.dtype)
    denom = m.sum(dim=-1).clamp_min(1e-12)
    return (x * m).sum(dim=-1) / denom


def _best_real(goal_sim, ride, trav, score_cfg, expert_index, allowed=None):
    """argmax score over real candidates only (expert excluded).

    `allowed` (bool [N], optional) further restricts the argmax — used for the
    goal-matched ride-aware band. Restricting AFTER combine_score is exact: the pool-level
    z-standardization is affine, so the argmax within any subset is unchanged.
    """
    s = combine_score(goal_sim, trav, ride, score_cfg).clone()
    s[expert_index] = float("-inf")
    if allowed is not None:
        s[~allowed] = float("-inf")
    return int(torch.argmax(s))


def _safe(vec: torch.Tensor, idx: int) -> float:
    v = float(vec[idx])
    return v if v == v else float("nan")  # NaN guard


def _nan_expert(vec: torch.Tensor, expert_index: int) -> torch.Tensor:
    """Copy with the expert row NaN'd out (stats over real candidates only)."""
    out = vec.clone().float()
    out[expert_index] = float("nan")
    return out


def _aggregate(rows: list[dict], split: str, do_cem: bool) -> dict:
    import math

    def _mean(key):
        vals = [r[key] for r in rows if not math.isnan(r.get(key, float("nan")))]
        return sum(vals) / len(vals) if vals else float("nan")

    pred = torch.tensor([r["pred_ride"] for r in rows])
    gtr = torch.tensor([r["gt_rough"] for r in rows])
    finite = torch.isfinite(pred) & torch.isfinite(gtr)
    validity_r = _pearson(pred[finite], gtr[finite]) if finite.sum() > 2 else float("nan")

    out = {
        "split": split,
        "n_queries": len(rows),
        "C1_expert_rank_percentile": _mean("expert_pct"),
        "C1_expert_rank_goalonly": _mean("expert_pct_goalonly"),
        "C1_selected_goal_sim": _mean("sel_goal_sim"),
        "C2a_ride_validity_pearson": validity_r,
        "C2b_realized_roughness_full": _mean("realized_rough_full"),
        "C2b_realized_roughness_goalonly": _mean("realized_rough_goalonly"),
        "C2b_realized_roughness_full_goalmatched": _mean("realized_rough_full_eps"),
        "C2b_goalmatch_band_n": _mean("band_n"),
        "C2b_goalmatch_active_share": (
            sum(1 for r in rows if r.get("band_n", 0) > 1) / len(rows) if rows else float("nan")
        ),
        "NWM_ade_full_goalmatched": _mean("ade_full_eps"),
        # NWM-protocol plan-vs-expert trajectory errors (meters)
        "NWM_ade_full": _mean("ade_full"),
        "NWM_fde_full": _mean("fde_full"),
        "NWM_ade_goalonly": _mean("ade_goalonly"),
        "NWM_fde_goalonly": _mean("fde_goalonly"),
        "NWM_ade_random": _mean("ade_random"),
    }
    out["C2b_roughness_reduction"] = out["C2b_realized_roughness_goalonly"] - out["C2b_realized_roughness_full"]
    # goal-matched variant: reduction achievable WITHOUT sacrificing goal progress.
    # band_n==1 queries contribute 0 by construction (only the goal-only pick is in
    # band) — that is the honest reading, not a bug; active_share reports how often
    # the band actually offers a choice.
    out["C2b_roughness_reduction_goalmatched"] = (
        out["C2b_realized_roughness_goalonly"] - out["C2b_realized_roughness_full_goalmatched"]
    )
    for k in sorted({k for r in rows for k in r if k.startswith("realized_rough_full_eps@")}):
        e = k.split("@", 1)[1]
        out[f"C2b_roughness_reduction_goalmatched@{e}"] = (
            out["C2b_realized_roughness_goalonly"] - _mean(k)
        )
        out[f"C2b_goalmatch_band_n@{e}"] = _mean(f"band_n@{e}")
    if do_cem:
        out["cem_goal_sim"] = _mean("cem_goal_sim")
        out["NWM_ade_cem"] = _mean("cem_ade")
        out["NWM_fde_cem"] = _mean("cem_fde")
    out["_rows"] = rows
    return out


def _write_report(res: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in res if not k.startswith("_")]
    lines = [f"# Planning eval — split `{res['split']}`", ""]
    lines += [f"- **{k}**: {res[k]}" for k in keys]
    lines += ["", "Higher expert-rank percentile + lower realized roughness (full vs goal-only) = "
              "better goal reaching / smoother picks.",
              "C2b_*_goalmatched: ride-aware pick restricted to the goal_sim ε-band "
              "(eval.goal_sim_epsilon) — reduction without sacrificing goal progress; "
              "robust to the speed confound (picking slower plans).",
              "C2a_ride_validity_pearson > 0 = predicted ride_cost tracks experienced roughness.",
              "NWM_ade/fde: selected-plan vs expert SE(2) trajectory error (m); _random = "
              "expected error of a random library pick (scale reference).",
              "expert_rank_goalonly: expert ranked by goal_sim alone — low = visual-rollout "
              "drift; gap vs full-score rank = roughness penalty on the expert (ride term acting)."]
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--split", default="val")
    p.add_argument("--out", default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--cem", action="store_true", help="also run CEM refinement per query")
    p.add_argument("--keep-stationary", action="store_true",
                   help="keep fully-parked sequences (sam_loop/trash) in the pool")
    p.add_argument("--wm-config", default=None,
                   help="override cfg.wm_config (e.g. the no-proprio-input baseline)")
    p.add_argument("--checkpoint", default=None, help="override cfg.checkpoint")
    p.add_argument("--val-stride", type=int, default=None,
                   help="override dataset val_stride → planning horizon = H·stride anchors; "
                        "use 4 to match the NWM export (32 anchors ≈ 3.2 s)")
    p.add_argument("--eps-sweep", type=float, nargs="*", default=None,
                   help="extra goal_sim_epsilon values for the goal-matched ride-aware "
                        "sensitivity sweep (free — selection is post-scoring math)")
    add_variant_arg(p, help_for="the planning config")
    add_variant_arg(p, "--wm-variant", "--wm-config (with --wm-config)")
    args = p.parse_args(argv)

    cfg = load_config((REPO_ROOT / args.config) if not Path(args.config).is_absolute()
                      else Path(args.config), args.variant)
    if args.max_queries is not None:
        cfg["eval"]["max_queries"] = args.max_queries
    if args.wm_config:
        cfg["wm_config"] = args.wm_config
        cfg["wm_variant"] = args.wm_variant
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = evaluate(cfg, args.split, device, args.cem, keep_stationary=args.keep_stationary,
                   val_stride=args.val_stride, eps_sweep=args.eps_sweep)
    for k in [k for k in res if not k.startswith("_")]:
        print(f"  {k}: {res[k]}")
    if args.out:
        _write_report(res, Path(args.out))
        print(f"\nwrote {args.out}  (+ {len(res['_rows'])} per-query rows)")
        Path(args.out).with_suffix(".jsonl").write_text(
            "\n".join(json.dumps(r) for r in res["_rows"]) + "\n"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
