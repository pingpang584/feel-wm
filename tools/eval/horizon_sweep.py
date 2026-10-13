#!/usr/bin/env python
"""Per-horizon decay sweep: visual feature prediction vs proprio prediction vs failure risk.

The visual analog of a single-step proprio-R²-vs-offset check. Here we run the full
multi-step rollout and report, **per horizon h**, how well the world model predicts
(a) the visual DINO features, (b) each proprio channel and (c) the binary failure-risk
label (trav head AUC) — to test the hypothesis that the smooth/geometric **visual latent
decays slower** than the fast, contact-driven **proprio** signals.

Two rollout modes (run both for the full picture):
  --teacher-force : GT visual+proprio context fed each step -> pure 1-step prediction
                    (intrinsic per-step predictability, no rollout drift).
  (default)       : free-running -> the model's own predictions feed back
                    (real rollout; includes autoregressive compounding).

Metrics per horizon (over all valid steps in the split):
  * visual:  feature R² (pooled over patches×dims) and mean per-patch cosine(pred, GT)
  * proprio: per-channel R²  (R² is invariant to the shared de-norm affine, so these
             match tools/eval_offline.py's denormalized R²)
  * failure risk: AUC of the trav head's probability vs the step's binary label, masked by
             valid_target_mask & binary_valid_target_mask (== trainer._binary_metric_mask)
  * a "pooled 1..H" row = the same statistics pooled over ALL steps — this is exactly what
    tools/eval_offline.py reports (its R²/AUC pool anchors×steps and apply NO
    proprio_horizon cutoff), so at the base stride it must reproduce those numbers.

--persistence adds the persistence control computed in the SAME loop
with the SAME anchors/masks/steps: the anchor's own GT proprio feature (`current_feature`,
cum. Δt = 0) held constant as the prediction for every step h, and the anchor's own
binary failure-risk label as the score for every step h. Persistence has no seed
dimension — it is a property of the data, identical across arms/seeds of one split+stride.

Steps whose cumulative Δt exceeds `data.proprio_horizon_s` are flagged `*` (beyond proprio
/ failure-risk supervision — the heads were never trained there; extrapolation).

--max-horizon K may exceed the trained rollout_horizon (8): the dynamics uses a fixed
context window, so rolling further just extrapolates (expected to degrade — informative).

`--sim` runs the same sweep on the sim Husky dataset (mirrors `tools/train.py --sim`):
the val split comes from `build_sim_datasets` so the held-out substrings and normalization
are byte-identical to training. Everything downstream (batch keys, rollout, metrics) is shared.

Run (GPU). config and checkpoint are POSITIONAL (in that order):
    # intrinsic per-step decay:
    python tools/eval/horizon_sweep.py configs/tartandrive.yaml --variant td2 \
        ${OUTPUT_ROOT}/tartandrive/td2/checkpoint_best.pt \
        --split val --max-horizon 16 --teacher-force \
        --out out/horizon_sweep_tf.md
    # real rollout decay:
    python tools/eval/horizon_sweep.py configs/tartandrive.yaml --variant td2 \
        ${OUTPUT_ROOT}/tartandrive/td2/checkpoint_best.pt \
        --split val --max-horizon 16 \
        --out out/horizon_sweep_free.md
    # stride sweep with the persistence control (one model x split x stride):
    python tools/eval/horizon_sweep.py configs/tartandrive.yaml --variant td2 \
        ${OUTPUT_ROOT}/tartandrive/td2/epoch_ckpts/epoch_13.pt \
        --split val --max-horizon 8 --val-stride 5 --persistence \
        --out out/stride_persistence/td2_val_s5.md
    # Gazebo sim (teacher-forced; compare against a free-running run for the TF↔free gap):
    python tools/eval/horizon_sweep.py configs/gazebo.yaml --variant feelwm \
        ${OUTPUT_ROOT}/gazebo/feelwm/checkpoint_best.pt \
        --sim --split val --max-horizon 8 --teacher-force \
        --out out/gazebo_horizon_tf.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feelwm.config import add_variant_arg, load_config  # noqa: E402
from feelwm.data.dataset import DatasetConfig, TartanDriveDataset  # noqa: E402
from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel  # noqa: E402
from feelwm.training.trainer import FullModelTrainer, _IndexedDataset  # noqa: E402


def _mk(dc, d: dict):
    return dc(**{k: v for k, v in d.items() if k in dc.__dataclass_fields__})


def _r2(sse: np.ndarray, sy: np.ndarray, sy2: np.ndarray, n: np.ndarray) -> np.ndarray:
    sst = sy2 - (sy * sy) / np.maximum(n, 1)
    return 1.0 - sse / np.maximum(sst, 1e-12)


def _auc(scores: list[np.ndarray], labels: list[np.ndarray]) -> float:
    """Rank AUC with the trainer's tie handling (== eval_full_model's binary_safety.auc)."""
    if not scores:
        return float("nan")
    s = np.concatenate(scores).astype(np.float64)
    y = np.concatenate(labels).astype(np.int64)
    return float(FullModelTrainer._auc_np(s, y)) if len(y) else float("nan")


def current_binary_labels(ds) -> tuple[np.ndarray, np.ndarray] | None:
    """Anchor-time failure-risk label per dataset index: (unsafe[N] float, valid[N] bool).

    The sample only carries the FUTURE labels (`binary_label_target` at t+h·s); the
    persistence control needs the anchor's own label at t. Read it from the dataset's
    records along the same `_index` the loader walks, so loader index i == row i here.
    Returns None when the dataset does not expose these (then persistence AUC is NaN).
    """
    index = getattr(ds, "_index", None)
    records = getattr(ds, "_records", None)
    if index is None or records is None:
        return None
    unsafe = np.zeros(len(index), np.float32)
    valid = np.zeros(len(index), bool)
    for i, (sid, t, _s) in enumerate(index):
        rec = records[sid]
        bu, bv = getattr(rec, "binary_unsafe", None), getattr(rec, "binary_valid", None)
        if bu is None or bv is None:
            return None
        valid[i] = bool(bv[t])
        unsafe[i] = float(bu[t])
    return unsafe, valid


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-horizon", type=int, default=None,
                    help="rollout length (default = config rollout_horizon; may exceed it)")
    ap.add_argument("--teacher-force", action="store_true")
    ap.add_argument("--sim", action="store_true",
                    help="sim Husky dataset (mirrors tools/train.py --sim)")
    ap.add_argument("--val-stride", type=int, default=None,
                    help="override data.val_stride (Δt-conditioned runs; step Δt = 0.1s x stride)")
    ap.add_argument("--persistence", action="store_true",
                    help="also report the persistence control (anchor's current proprio "
                         "feature / current failure-risk label held constant over h)")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="optional markdown output path")
    add_variant_arg(ap)
    args = ap.parse_args()

    cfg = load_config(Path(args.config), args.variant)
    model = TerrainWorldModel(_mk(ModelConfig, cfg["model"])).to(args.device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model_state"])

    overrides = {} if args.val_stride is None else {"val_stride": args.val_stride}
    if args.sim:
        # same split/normalization as training — see feelwm/training/trainer.py
        from feelwm.training.trainer import build_sim_datasets  # noqa: PLC0415 (torch-heavy import)
        H = args.max_horizon or int(cfg["model"].get("rollout_horizon", 8))
        datasets, data_cfg = build_sim_datasets(
            cfg, splits=(args.split,), rollout_horizon=H, **overrides)
        ds = datasets[args.split]
    else:
        H = args.max_horizon or int(cfg["data"].get("rollout_horizon", 8))
        data_cfg = _mk(DatasetConfig, {**cfg["data"], **overrides,
                                       "split": args.split, "rollout_horizon": H})
        ds = TartanDriveDataset(data_cfg)
    # anchors are a fixed 10 Hz grid on both datasets; a Δt-conditioned run advances
    # `val_stride` anchors per rollout step (include_dt=False → always 0.1 s).
    stride = int(data_cfg.val_stride if data_cfg.include_dt else 1)
    step_dt = 0.1 * stride
    prop_horizon_s = float(getattr(data_cfg, "proprio_horizon_s", 1.0))
    cum_dt = step_dt * np.arange(1, H + 1)
    beyond = cum_dt > prop_horizon_s + 1e-9          # heads unsupervised there
    chans = (cfg.get("loss", {}).get("channel_names")
             or cfg["data"].get("active_proprio_channels")
             or [f"ch{i}" for i in range(model.cfg.proprio_feature_dim)])
    C = len(chans)
    mode = "teacher-forced" if args.teacher_force else "free-running"
    print(f"[sweep] {args.config} | split={args.split} | H={H} | mode={mode} | stride={stride} | "
          f"Δt={step_dt:.1f}s/step | N={len(ds):,} | channels={chans} | "
          f"proprio_horizon_s={prop_horizon_s} | persistence={args.persistence}", flush=True)

    cur_bin = current_binary_labels(ds) if args.persistence else None
    if args.persistence and cur_bin is None:
        print("[sweep] dataset exposes no anchor-time binary label -> persistence AUC = NaN")

    loader = DataLoader(_IndexedDataset(ds), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)

    # per-horizon accumulators (float64)
    v_sse = np.zeros(H); v_sy = np.zeros(H); v_sy2 = np.zeros(H); v_n = np.zeros(H)
    v_cos = np.zeros(H); v_cosn = np.zeros(H)
    p_sse = np.zeros((H, C)); p_sy = np.zeros((H, C)); p_sy2 = np.zeros((H, C)); p_n = np.zeros((H, C))
    q_sse = np.zeros((H, C))                          # persistence proprio (same sy/sy2/n)
    # per-SEQUENCE accumulators: pooled R² is 1 - SSE/(SY2 - SY^2/N) over a single global
    # target mean, so keeping (SSE, SY, SY2, N) per sequence lets ANY subset of sequences be
    # re-pooled exactly offline (leave-one-out sequence analysis) from one pass.
    seq_names = [r.name for r in ds._records]
    sid_of = np.asarray([e[0] for e in ds._index], dtype=np.int64)   # loader row -> sequence id
    S = len(seq_names)
    ps_sse = np.zeros((S, H, C)); ps_sy = np.zeros((S, H, C))
    ps_sy2 = np.zeros((S, H, C)); ps_n = np.zeros((S, H))
    qs_sse = np.zeros((S, H, C))                      # persistence, per sequence
    t_sid: list[list[np.ndarray]] = [[] for _ in range(H)]     # sequence id per trav sample
    u_sid: list[list[np.ndarray]] = [[] for _ in range(H)]
    t_score: list[list[np.ndarray]] = [[] for _ in range(H)]   # model failure-risk prob
    t_label: list[list[np.ndarray]] = [[] for _ in range(H)]
    u_score: list[list[np.ndarray]] = [[] for _ in range(H)]   # persistence failure-risk
    u_label: list[list[np.ndarray]] = [[] for _ in range(H)]
    has_trav = False
    has_prop = False      # model has a proprio feature head (the visual-only model has none)

    t0 = time.time()
    n_batches = len(loader) if args.max_batches is None else min(len(loader), args.max_batches)
    for bi, batch in enumerate(loader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        b = {k: (v.to(args.device).float() if v.is_floating_point() else v.to(args.device))
             for k, v in batch.items()}
        kw = {}
        if args.teacher_force:
            kw["use_ground_truth_visual"] = b["z_vis_target"]
            kw["use_ground_truth_proprio"] = b["f_prop_target"]
        out = model.forward_rollout(b, b["action_sequence"], **kw)

        z_pred = out["z_vis_preds"]          # [B,H,P,D]
        f_pred = out.get("f_prop_preds")     # [B,H,C] or None (no proprio head)
        c_pred = out.get("c_preds")          # [B,H] prob or None (no trav head)
        z_tgt = b["z_vis_target"]            # [B,H,P,D]
        f_tgt = b["f_prop_target"]           # [B,H,C]
        vmask = b["valid_target_mask"]       # [B,H] bool
        bmask = vmask & b["binary_valid_target_mask"].bool() if "binary_valid_target_mask" in b else vmask
        f_cur = b.get("current_feature")     # [B,C] anchor's own GT feature (cum. Δt = 0)
        idx = batch["_sample_idx"].numpy()
        if f_pred is not None:
            has_prop = True
        if c_pred is not None:
            has_trav = True
            c_pred = c_pred.reshape(c_pred.shape[0], H)

        for h in range(H):
            m = vmask[:, h]
            if m.sum() == 0:
                continue
            zp, zt = z_pred[m, h].double(), z_tgt[m, h].double()      # [nv,P,D]
            cos = F.cosine_similarity(zp, zt, dim=-1)                 # [nv,P]
            v_cos[h] += cos.sum().item(); v_cosn[h] += cos.numel()
            v_sse[h] += ((zp - zt) ** 2).sum().item()
            v_sy[h] += zt.sum().item(); v_sy2[h] += (zt ** 2).sum().item(); v_n[h] += zt.numel()

            ft = f_tgt[m, h].double()                                 # [nv,C]
            m_np = m.cpu().numpy()
            sid_b = sid_of[idx[m_np]]                                  # [nv]
            ft_np = ft.cpu().numpy()
            if f_pred is not None:
                fp = f_pred[m, h].double()
                p_sse[h] += ((fp - ft) ** 2).sum(0).cpu().numpy()
                np.add.at(ps_sse, (sid_b, h), (fp.cpu().numpy() - ft_np) ** 2)
            p_sy[h] += ft.sum(0).cpu().numpy()
            p_sy2[h] += (ft ** 2).sum(0).cpu().numpy()
            p_n[h] += ft.shape[0]
            np.add.at(ps_sy, (sid_b, h), ft_np)
            np.add.at(ps_sy2, (sid_b, h), ft_np ** 2)
            np.add.at(ps_n, (sid_b, h), 1.0)
            if args.persistence and f_cur is not None:
                d_cur = (f_cur[m].double() - ft)
                q_sse[h] += (d_cur ** 2).sum(0).cpu().numpy()
                np.add.at(qs_sse, (sid_b, h), (d_cur ** 2).cpu().numpy())

            bm = bmask[:, h]
            if c_pred is not None and bm.any():
                t_score[h].append(c_pred[bm, h].float().cpu().numpy())
                t_label[h].append(b["binary_label_target"][bm, h].cpu().numpy())
                t_sid[h].append(sid_of[idx[bm.cpu().numpy()]])
            if cur_bin is not None:
                unsafe, valid = cur_bin
                um = bm.cpu().numpy() & valid[idx]
                if um.any():
                    u_score[h].append(unsafe[idx][um])
                    u_label[h].append(batch["binary_label_target"][um, h].numpy())
                    u_sid[h].append(sid_of[idx[um]])
        if (bi + 1) % 50 == 0 or bi + 1 == n_batches:
            print(f"[sweep] batch {bi+1}/{n_batches} | {time.time()-t0:6.0f}s", flush=True)

    if args.device.startswith("cuda"):
        print(f"[sweep] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    vis_r2 = _r2(v_sse, v_sy, v_sy2, v_n)
    vis_cos = v_cos / np.maximum(v_cosn, 1)
    pro_r2 = _r2(p_sse, p_sy, p_sy2, p_n) if has_prop else None   # [H,C] or None
    per_r2 = _r2(q_sse, p_sy, p_sy2, p_n) if args.persistence else None
    trav_auc = np.array([_auc(t_score[h], t_label[h]) for h in range(H)]) if has_trav else None
    n_trav = [int(sum(len(a) for a in t_label[h])) for h in range(H)]
    per_auc = np.array([_auc(u_score[h], u_label[h]) for h in range(H)]) if cur_bin is not None else None
    # pooled over ALL steps 1..H == tools/eval_offline.py's statistic (no horizon cutoff)
    pooled = {
        "proprio_r2": (_r2(p_sse.sum(0), p_sy.sum(0), p_sy2.sum(0), p_n.sum(0)).tolist()
                       if has_prop else None),
        "trav_auc": _auc(sum(t_score, []), sum(t_label, [])) if has_trav else None,
        "n_steps": int(p_n[:, 0].sum()),
        "n_trav": int(sum(n_trav)),
    }
    pooled_pers = None
    if args.persistence:
        pooled_pers = {
            "proprio_r2": _r2(q_sse.sum(0), p_sy.sum(0), p_sy2.sum(0), p_n.sum(0)).tolist(),
            "trav_auc": _auc(sum(u_score, []), sum(u_label, [])) if cur_bin is not None else None,
        }

    # ---- report ----
    def _prow(tag: str, sec: float, r2row: np.ndarray, auc: float | None, flag: str = " ") -> str:
        aucs = f"{auc:6.3f}" if auc is not None and np.isfinite(auc) else "   n/a"
        return (f"{tag}({sec:.1f}s){flag}| " + "  ".join(f"{r2row[c]:+9.3f}" for c in range(C))
                + f" | {aucs}")

    def _procols(r2row) -> str:   # "—" columns for a model with no proprio head
        if r2row is None:
            return "  ".join(f"{chr(0x2014):>9}" for _ in range(C))
        return "  ".join(f"{r2row[c]:+9.3f}" for c in range(C))

    header = (f"horizon    |  vis_R²  vis_cos | " + "  ".join(f"{c[:9]:>9}" for c in chans)
              + " | trav_AUC")
    lines = [f"# Horizon sweep — {Path(args.config).stem} ({mode}, split={args.split})", "",
             f"- checkpoint: `{args.checkpoint}`  | N_anchors={len(ds):,} | H={H}"
             f" | stride={stride} | Δt={step_dt:.1f}s/step",
             "- proprio R² in normalized space (== denormalized R², affine-invariant).",
             "- trav_AUC = failure-risk head AUC, mask = valid_target_mask & binary_valid_target_mask.",
             f"- `*` = cumulative Δt > proprio_horizon_s={prop_horizon_s} s: proprio/failure-risk heads "
             "were NOT supervised there (extrapolation).",
             "- `pooled 1..H` = anchors×steps pooled with no horizon cutoff (== eval_full_model.py).",
             "", "```", header, "-" * len(header)]
    print("\n" + header); print("-" * len(header))
    for h in range(H):
        sec = cum_dt[h]
        flag = "*" if beyond[h] else " "
        aucs = f"{trav_auc[h]:6.3f}" if trav_auc is not None and np.isfinite(trav_auc[h]) else "   n/a"
        row = (f"h={h+1:>2}({sec:.1f}s){flag}| {vis_r2[h]:+6.3f}  {vis_cos[h]:6.3f} | "
               + _procols(None if pro_r2 is None else pro_r2[h]) + f" | {aucs}")
        print(row); lines.append(row)
    paucs = f"{pooled['trav_auc']:6.3f}" if pooled["trav_auc"] is not None else "   n/a"
    row = (f"pooled 1..{H:<2} |   —        —    | "
           + _procols(pooled["proprio_r2"]) + f" | {paucs}")
    print(row); lines.append(row)
    lines.append("```")

    if args.persistence:
        header2 = ("persistence |  " + "  ".join(f"{c[:9]:>9}" for c in chans) + " | trav_AUC")
        lines += ["", "## Persistence control (anchor's current value held constant; no seed dimension)",
                  "", "- proprio: prediction = `current_feature` (GT at cum. Δt = 0) for every h.",
                  "- failure risk: score = anchor's current binary label; anchors with an invalid "
                  "current label are dropped from the persistence AUC only.",
                  "", "```", header2, "-" * len(header2)]
        print("\n" + header2); print("-" * len(header2))
        for h in range(H):
            auc_h = None if per_auc is None else per_auc[h]
            row = _prow(f"h={h+1:>2}", cum_dt[h], per_r2[h], auc_h, "*" if beyond[h] else " ")
            print(row); lines.append(row)
        pa = pooled_pers["trav_auc"]
        paucs = f"{pa:6.3f}" if pa is not None and np.isfinite(pa) else "   n/a"
        row = (f"pooled 1..{H:<2}  | " + "  ".join(f"{v:+9.3f}" for v in pooled_pers["proprio_r2"])
               + f" | {paucs}")
        print(row); lines.append(row); lines.append("```")

    if args.out:
        op = Path(args.out)
        if not op.is_absolute():
            op = REPO_ROOT / op
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text("\n".join(lines) + "\n")
        # machine-readable sidecar: two of these (TF vs free) can be diffed, and
        # downstream aggregation stacks them; existing keys are kept as-is.
        sidecar = op.with_suffix(".json")
        payload = {
            "config": args.config, "checkpoint": args.checkpoint, "split": args.split,
            "sim": args.sim, "mode": mode, "H": H, "step_dt": step_dt, "stride": stride,
            "proprio_horizon_s": prop_horizon_s, "cum_dt": cum_dt.tolist(),
            "beyond_supervision": beyond.tolist(),
            "n_anchors": len(ds), "channels": list(chans),
            "vis_r2": vis_r2.tolist(), "vis_cos": vis_cos.tolist(),
            "proprio_r2": None if pro_r2 is None else pro_r2.tolist(),
            "n_steps": p_n[:, 0].astype(int).tolist(),
            "trav_auc": None if trav_auc is None else trav_auc.tolist(),
            "n_trav": n_trav,
            "pooled": pooled,
        }
        if args.persistence:
            payload["persistence"] = {
                "note": "data property; no seed dimension — never attach error bars",
                "proprio_r2": per_r2.tolist(),
                "trav_auc": None if per_auc is None else per_auc.tolist(),
                "pooled": pooled_pers,
            }
        # Sufficient statistics for re-pooling ANY subset of sequences offline. R² is
        # 1 - SSE/(SY2 - SY^2/N) against one global target mean, so a subset's pooled R² is
        # exact from these sums; AUC is not additive, hence the raw score/label arrays.
        payload["per_sequence"] = {
            "note": "re-pool offline: R2 = 1 - sum(sse)/(sum(sy2) - sum(sy)^2/sum(n)) over the "
                    "kept sequences; identical to running the sweep with those sequences excluded",
            "sequences": seq_names,
            "has_proprio_head": has_prop,
            "sse": ps_sse.tolist() if has_prop else None, "sy": ps_sy.tolist(),
            "sy2": ps_sy2.tolist(), "n": ps_n.tolist(),
            "persistence_sse": qs_sse.tolist() if args.persistence else None,
        }
        sidecar.write_text(json.dumps(payload, indent=1) + "\n")
        npz = op.with_suffix(".perseq.npz")
        arrs = {}
        for h in range(H):
            if t_sid[h]:
                arrs[f"trav_sid_{h}"] = np.concatenate(t_sid[h])
                arrs[f"trav_score_{h}"] = np.concatenate(t_score[h])
                arrs[f"trav_label_{h}"] = np.concatenate(t_label[h])
            if u_sid[h]:
                arrs[f"pers_sid_{h}"] = np.concatenate(u_sid[h])
                arrs[f"pers_score_{h}"] = np.concatenate(u_score[h])
                arrs[f"pers_label_{h}"] = np.concatenate(u_label[h])
        if arrs:
            np.savez_compressed(npz, sequences=np.array(seq_names), **arrs)
        print(f"\n[wrote] {op}\n[wrote] {sidecar}" + (f"\n[wrote] {npz}" if arrs else ""))


if __name__ == "__main__":
    main()
