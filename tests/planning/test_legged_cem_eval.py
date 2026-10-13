"""Legged open-loop free-CEM eval — query construction, Δt schedule, prior, aggregation.

Everything here runs on synthesized legged missions (portable, no GrandTour on disk);
the CEM/model half is covered by the existing tests/planning/test_cem.py.

What is worth locking down, and why:
  - the goal frame sits at t + Σ(dt_schedule) anchors (3.4 s), NOT at the loader's own
    val_stride horizon — the whole reason the query is assembled by hand;
  - per-dataset norm ROUTING survives the hand assembly (a `tg` mission must not be
    normalized with `gt` stats — each corpus keeps its own normalization);
  - the movement gate / anchor stride / per-mission cap produce a deterministic, reportable
    query count;
  - the action prior is per-STRIDE (a 1.0 s step's rate band is narrower than a 0.1 s one).
"""
import json

import numpy as np
import pytest
import torch

from pathlib import Path

from feelwm.data.legged_dataset import LeggedSeqSpec
from feelwm.data.normalize import NormStats

from tools.eval.openloop_cem_legged import (
    ANCHOR_DT_S, build_context_dataset, build_queries, cum_offsets, cum_seconds,
    expert_actions, forward_speed_mps, legged_action_prior, n_supervised_steps,
    odom_waypoints, queries_from_manifest, realized_proprio, schedule_prior,
    _aggregate, _pearson,
)

SCHEDULE = [1, 1, 1, 1, 5, 5, 10, 10]
S4 = [4] * 8                     # the uniform stride-4 export grid
GOAL_OFF = 34
K, H = 3, 8
ACTIVE = ["roughness", "angular_instability", "slope_power", "cost_of_transport"]


# --------------------------------------------------------------------- fixtures


def _write_mission(seq_dir: Path, n: int, speed: float, n_patch=4, d_vis=8,
                   d_raw=53, t_win=5, seed=0) -> None:
    """A straight-line mission at constant `speed` (m/s) on a clean 10 Hz grid."""
    rng = np.random.default_rng(seed)
    seq_dir.mkdir(parents=True, exist_ok=True)
    pose = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    pose[:, 0, 3] = np.arange(n) * speed * ANCHOR_DT_S
    np.savez(
        seq_dir / "labels_legged.npz",
        anchor_idx=np.arange(n, dtype=np.int32),
        timestamp=np.arange(n, dtype=np.float64) * ANCHOR_DT_S,
        proprio_window=rng.standard_normal((n, t_win, d_raw)).astype(np.float32),
        proprio_feature=rng.standard_normal((n, 5)).astype(np.float32),
        pose=pose,
        action=rng.standard_normal((n, 3)).astype(np.float32) * 0.1,
        valid_mask=np.ones(n, np.uint8),
    )
    feat = seq_dir / "features" / "gt_dinov2reg_v1"
    feat.mkdir(parents=True, exist_ok=True)
    np.save(feat / "features.npy", np.zeros((n, n_patch, d_vis), np.float32))
    np.save(feat / "features_valid.npy", np.ones(n, bool))


def _write_stats(path: Path, action_mean=(0.0, 0.0, 0.0), action_std=(1.0, 1.0, 1.0),
                 d_raw=53) -> None:
    path.write_text(json.dumps({
        "feature_set": "legged",
        "proprio_feature": {"mean": [0] * 5, "std": [1] * 5},
        "proprio_window": {"mean": [0] * d_raw, "std": [1] * d_raw},
        "action": {"mean": list(action_mean), "std": list(action_std)},
    }))


@pytest.fixture
def legged_val(tmp_path):
    """Two held-out missions (one fast, one with a slow half) + a wm-config-shaped dict."""
    root = tmp_path / "grand_tour"
    _write_mission(root / "2024-12-fast", n=200, speed=0.8, seed=1)
    _write_mission(root / "2024-12-slow", n=200, speed=0.05, seed=2)
    stats = tmp_path / "gt.json"
    _write_stats(stats)
    specs = [LeggedSeqSpec("gt", "2024-12-fast", root / "2024-12-fast" / "labels_legged.npz",
                           root / "2024-12-fast" / "features" / "gt_dinov2reg_v1"),
             LeggedSeqSpec("gt", "2024-12-slow", root / "2024-12-slow" / "labels_legged.npz",
                           root / "2024-12-slow" / "features" / "gt_dinov2reg_v1")]
    wm_cfg = {
        "model": {"context_len": K, "rollout_horizon": H, "n_patch": 4, "d_vis": 8},
        "data": {"active_proprio_channels": ACTIVE, "proprio_horizon_s": 1.0},
    }
    return specs, {"gt": str(stats), "tg": str(stats)}, wm_cfg


# --------------------------------------------------------------------- schedule


def test_schedule_offsets_and_supervised_window():
    assert cum_offsets(SCHEDULE).tolist() == [1, 2, 3, 4, 9, 14, 24, 34]
    assert cum_seconds(SCHEDULE)[-1] == pytest.approx(3.4)
    # v2 was trained with proprio_horizon_s = 1.0 → steps 1..5 supervised, 6..8 extrapolation
    assert n_supervised_steps(SCHEDULE, 1.0) == 5
    assert n_supervised_steps(SCHEDULE, 4.0) == 8


# --------------------------------------------------------------------- queries


def test_goal_frame_sits_at_the_schedule_horizon_not_the_loader_stride(legged_val):
    """The reason the query is hand-assembled: the loader's stride-1 horizon is 8 anchors."""
    specs, norm, wm_cfg = legged_val
    ds = build_context_dataset(wm_cfg, specs, norm)
    assert ds.cfg.rollout_horizon == 1 and ds.cfg.val_stride == 1
    q = build_queries(ds, SCHEDULE, anchor_stride=4, speed_floor=0.3,
                      max_per_mission=None, seed=0)[0]
    goal_anchor = q["anchor"] + GOAL_OFF
    assert goal_anchor - q["anchor"] == 34            # 3.4 s, not the loader's 8
    assert goal_anchor in ds._records[q["rid"]].cached_set


def test_query_gate_stride_and_cap(legged_val):
    specs, norm, wm_cfg = legged_val
    ds = build_context_dataset(wm_cfg, specs, norm)
    q = build_queries(ds, SCHEDULE, anchor_stride=4, speed_floor=0.3,
                      max_per_mission=None, seed=0)
    missions = {r["mission"] for r in q}
    assert missions == {"2024-12-fast"}               # 0.05 m/s mission is gated out entirely
    assert all(r["anchor"] % 4 == 0 for r in q)
    assert all(r["anchor"] >= K and r["anchor"] + GOAL_OFF < 200 for r in q)
    assert all(r["window_speed_mps"] == pytest.approx(0.8, abs=1e-3) for r in q)
    assert [r["qid"] for r in q] == list(range(len(q)))

    capped = build_queries(ds, SCHEDULE, anchor_stride=4, speed_floor=0.3,
                           max_per_mission=5, seed=0)
    assert len(capped) == 5
    assert all(r["n_gated_in_mission"] == len(q) for r in capped)   # pool size stays reportable
    again = build_queries(ds, SCHEDULE, anchor_stride=4, speed_floor=0.3,
                          max_per_mission=5, seed=0)
    assert [r["anchor"] for r in again] == [r["anchor"] for r in capped]   # seeded subsample


def test_dropped_frame_windows_are_rejected(legged_val, tmp_path):
    """A camera gap inside [context start, goal] makes the emitted Δt a lie — the loader's own
    check only screens its stride-1 horizon, so the 34-anchor span is re-checked."""
    specs, norm, wm_cfg = legged_val
    gap_dir = tmp_path / "grand_tour" / "2024-12-gap"
    _write_mission(gap_dir, n=200, speed=0.8, seed=3)
    z = dict(np.load(gap_dir / "labels_legged.npz"))
    z["timestamp"] = z["timestamp"].copy()
    z["timestamp"][100:] += 0.5                        # 99 -> 100 spans 0.6 s
    np.savez(gap_dir / "labels_legged.npz", **z)
    gap_spec = LeggedSeqSpec("gt", "2024-12-gap", gap_dir / "labels_legged.npz",
                             gap_dir / "features" / "gt_dinov2reg_v1")
    ds = build_context_dataset(wm_cfg, [gap_spec], norm)
    anchors = {q["anchor"] for q in build_queries(ds, SCHEDULE, 1, 0.3, None, 0)}
    assert not any(a - K <= 99 < a + GOAL_OFF for a in anchors)
    assert 60 in anchors and 104 in anchors            # clear of the gap on both sides


# --------------------------------------------------------------------- pinned query set


def _manifest(path: Path, rows: list[tuple[int, str, int]]) -> Path:
    """`(qid, mission, our_anchor)` triples in the external manifest's shape."""
    path.write_text("\n".join(json.dumps(
        {"qid": q, "mission": m, "seq": m, "our_anchor": a, "export_ok": True})
        for q, m, a in rows) + "\n")
    return path


def test_query_manifest_keeps_the_manifest_qid_and_resolves_the_anchor(legged_val, tmp_path):
    """The pin's whole purpose: the same instances, under the SAME join key.

    A re-derived query set at a different schedule selects different anchors (the movement
    gate and the seeded cap both read the schedule), and a re-numbered qid mis-joins every
    paired row in `paired_ade.py`. So the manifest's qid must survive verbatim, out of order
    and with gaps.
    """
    specs, norm, wm_cfg = legged_val
    ds = build_context_dataset(wm_cfg, specs, norm)
    mf = _manifest(tmp_path / "m.jsonl",
                   [(70, "2024-12-fast", 60), (12, "2024-12-fast", 40)])
    q, skipped = queries_from_manifest(ds, S4, mf)

    assert skipped == []
    assert [r["qid"] for r in q] == [12, 70]                  # kept, sorted by qid
    assert [r["anchor"] for r in q] == [40, 60]
    assert all(ds._index[r["ds_index"]][1] == r["anchor"] for r in q)
    assert all(r["n_gated_in_mission"] == 2 for r in q)       # the pinned pool IS the manifest
    # the speed column describes THIS schedule's window, not the manifest's 3.4 s one
    assert all(r["window_speed_mps"] == pytest.approx(0.8, abs=1e-3) for r in q)


def test_query_manifest_reports_every_skip_with_a_reason(legged_val, tmp_path):
    """The manifest is a SELECTION, not a validity certificate — it was built for another
    horizon, so a row it names may not be plannable here, and n must not silently shrink."""
    specs, norm, wm_cfg = legged_val
    ds = build_context_dataset(wm_cfg, specs, norm)
    mf = _manifest(tmp_path / "m.jsonl", [
        (0, "2024-12-fast", 40),          # fine
        (1, "2024-12-fast", 190),         # goal at 190+32 is past the 200-anchor mission
        (2, "no-such-mission", 40),       # not in this split
        (3, "2024-12-fast", 1),           # too early for the k-frame context
    ])
    q, skipped = queries_from_manifest(ds, S4, mf)

    assert [r["qid"] for r in q] == [0]
    assert {s["qid"]: s["reason"] for s in skipped}.keys() == {1, 2, 3}
    reasons = {s["qid"]: s["reason"] for s in skipped}
    assert "past sequence end" in reasons[1]
    assert "not in the dataset split" in reasons[2]
    assert reasons[3]                      # a reason string, never a bare drop


def test_query_manifest_bypasses_the_gate_and_cap(legged_val, tmp_path):
    """The config's movement gate would re-introduce exactly the selection the pin removes:
    the slow mission is gated out of `build_queries` entirely, but a manifest naming it is
    honoured."""
    specs, norm, wm_cfg = legged_val
    ds = build_context_dataset(wm_cfg, specs, norm)
    assert not any(r["mission"] == "2024-12-slow"
                   for r in build_queries(ds, S4, 4, 0.3, None, 0))
    q, skipped = queries_from_manifest(ds, S4, _manifest(
        tmp_path / "m.jsonl", [(5, "2024-12-slow", 40)]))
    assert skipped == [] and [r["mission"] for r in q] == ["2024-12-slow"]
    assert q[0]["window_speed_mps"] < 0.3


def test_query_manifest_honours_the_missions_filter(legged_val, tmp_path):
    """--missions is a selection, so a filtered row is neither used nor counted as a skip."""
    specs, norm, wm_cfg = legged_val
    ds = build_context_dataset(wm_cfg, specs, norm)
    mf = _manifest(tmp_path / "m.jsonl",
                   [(0, "2024-12-fast", 40), (1, "2024-12-slow", 40)])
    q, skipped = queries_from_manifest(ds, S4, mf, missions={"2024-12-fast"})
    assert [r["qid"] for r in q] == [0] and skipped == []


# --------------------------------------------------------------------- action encoding


def test_expert_actions_carry_the_per_step_dt(legged_val):
    specs, norm, wm_cfg = legged_val
    ds = build_context_dataset(wm_cfg, specs, norm)
    a = expert_actions(ds, rid=0, anchor=40, schedule=SCHEDULE)
    assert a.shape == (H, 4)
    assert torch.allclose(a[:, 3], torch.tensor(SCHEDULE, dtype=torch.float32) * ANCHOR_DT_S)
    # the 3-dim part is a per-anchor RATE, so a constant-speed mission gives the SAME value
    # at every stride — the property that lets one set of norm-stats cover the schedule
    assert torch.allclose(a[:, 0], a[0, 0].expand(H), atol=1e-5)


def test_chained_expert_matches_the_odom_path_on_a_planar_mission(legged_val):
    """Planar SE(2) chaining and the raw pose path agree when there is no pitch/roll — so a
    nonzero `se2_chain_discrepancy_m` on real data is the 3-D geometry, not a bug here."""
    from feelwm.planning.trajectory import candidate_trajectories

    specs, norm, wm_cfg = legged_val
    ds = build_context_dataset(wm_cfg, specs, norm)
    stats = NormStats.from_json(norm["gt"])
    chained = candidate_trajectories(expert_actions(ds, 0, 40, SCHEDULE)[None], stats)[0]
    odom = odom_waypoints(ds._records[0].pose, 40, SCHEDULE)
    assert np.abs(chained.numpy() - odom).max() < 1e-3
    assert odom[-1, 0] == pytest.approx(0.8 * 3.4, abs=1e-3)   # 3.4 s at 0.8 m/s


def test_realized_proprio_reads_the_active_channels_at_the_schedule_anchors(legged_val):
    specs, norm, wm_cfg = legged_val
    ds = build_context_dataset(wm_cfg, specs, norm)
    real = realized_proprio(ds, 0, 40, SCHEDULE)
    assert real.shape == (H, len(ACTIVE))
    rec = ds._records[0]
    for h, off in enumerate(cum_offsets(SCHEDULE)):
        np.testing.assert_allclose(real[h], rec.feature[40 + int(off)][ds._chan_idx])


# --------------------------------------------------------------------- norm routing


def test_norm_routing_is_per_dataset(tmp_path):
    """A `tg` mission must be normalized with tg stats — never the gt ones (per-corpus stats).
    Same physical motion, different stats ⇒ different normalized action."""
    root = tmp_path / "d"
    _write_mission(root / "m_gt", n=120, speed=0.8, seed=4)
    _write_mission(root / "m_tg", n=120, speed=0.8, seed=5)
    gt_stats, tg_stats = tmp_path / "gt.json", tmp_path / "tg.json"
    _write_stats(gt_stats, action_mean=(0.0, 0.0, 0.0), action_std=(1.0, 1.0, 1.0))
    _write_stats(tg_stats, action_mean=(0.04, 0.0, 0.0), action_std=(2.0, 1.0, 1.0))
    specs = [LeggedSeqSpec("gt", "m_gt", root / "m_gt" / "labels_legged.npz",
                           root / "m_gt" / "features" / "gt_dinov2reg_v1"),
             LeggedSeqSpec("tg", "m_tg", root / "m_tg" / "labels_legged.npz",
                           root / "m_tg" / "features" / "gt_dinov2reg_v1")]
    wm_cfg = {"model": {"context_len": K, "rollout_horizon": H, "n_patch": 4, "d_vis": 8},
              "data": {"active_proprio_channels": ACTIVE, "proprio_horizon_s": 1.0}}
    norm = {"gt": str(gt_stats), "tg": str(tg_stats)}
    ds = build_context_dataset(wm_cfg, specs, norm)

    a_gt = expert_actions(ds, 0, 40, SCHEDULE)[0, 0]
    a_tg = expert_actions(ds, 1, 40, SCHEDULE)[0, 0]
    assert a_gt == pytest.approx(0.08, abs=1e-4)               # (0.08 - 0.0) / 1.0
    assert a_tg == pytest.approx((0.08 - 0.04) / 2.0, abs=1e-4)
    assert a_gt != pytest.approx(a_tg, abs=1e-3)


# --------------------------------------------------------------------- action prior


def test_action_prior_is_per_stride_and_bounded_by_percentiles(tmp_path):
    """Rate bands narrow as the step lengthens (a 1.0 s step time-averages the noise), which
    is exactly why CEM gets per-STEP init_std / clip bounds instead of one pooled band."""
    root = tmp_path / "train"
    rng = np.random.default_rng(0)
    for i in range(2):
        d = root / f"m{i}"
        _write_mission(d, n=600, speed=0.6, seed=10 + i)
        z = dict(np.load(d / "labels_legged.npz"))
        pose = z["pose"].copy()
        step = 0.06 + rng.standard_normal(600) * 0.02        # noisy per-anchor advance
        pose[:, 0, 3] = np.cumsum(step).astype(np.float32)
        z["pose"] = pose
        np.savez(d / "labels_legged.npz", **z)
    stats = tmp_path / "gt.json"
    _write_stats(stats)
    specs = [LeggedSeqSpec("gt", f"m{i}", root / f"m{i}" / "labels_legged.npz",
                           root / f"m{i}" / "features" / "gt_dinov2reg_v1") for i in range(2)]

    prior = legged_action_prior(specs, {"gt": str(stats)}, [1, 5, 10],
                                n_samples=400, seed=0)
    assert set(prior) == {1, 5, 10}
    assert prior[10]["std"][0] < prior[5]["std"][0] < prior[1]["std"][0]
    for s in (1, 5, 10):
        assert prior[s]["p1"][0] < prior[s]["mean"][0] < prior[s]["p99"][0]
        assert prior[s]["n"] == 400

    sp = schedule_prior(prior, SCHEDULE)
    for key in ("mean", "std", "p1", "p99"):
        assert sp[key].shape == (H, 3)
    torch.testing.assert_close(sp["std"][0], sp["std"][3])     # both stride-1 steps
    assert float(sp["std"][7, 0]) < float(sp["std"][0, 0])     # stride-10 step is tighter

    # the reported init placement: a ~0.6 m/s prior on a 0-mean/1-std stats file
    assert forward_speed_mps(sp["mean"], NormStats.from_json(str(stats)), SCHEDULE) == \
        pytest.approx(0.6, abs=0.05)


def test_action_prior_speed_floor_shifts_the_init(tmp_path):
    """The movement-gated prior ARM: standing anchors drag the raw prior below the (gated)
    query regime, and the floor is the knob that removes them."""
    root = tmp_path / "train"
    d = root / "m0"
    _write_mission(d, n=600, speed=0.8, seed=20)
    z = dict(np.load(d / "labels_legged.npz"))
    pose = z["pose"].copy()
    step = np.where(np.arange(600) < 300, 0.08, 0.0)          # half the mission is parked
    pose[:, 0, 3] = np.cumsum(step).astype(np.float32)
    z["pose"] = pose
    np.savez(d / "labels_legged.npz", **z)
    stats = tmp_path / "gt.json"
    _write_stats(stats)
    specs = [LeggedSeqSpec("gt", "m0", d / "labels_legged.npz",
                           d / "features" / "gt_dinov2reg_v1")]
    st = NormStats.from_json(str(stats))

    flat = [1] * H                                            # stride-1-only schedule
    raw = legged_action_prior(specs, {"gt": str(stats)}, flat, 400, 0, speed_floor=0.0)
    gated = legged_action_prior(specs, {"gt": str(stats)}, flat, 400, 0, speed_floor=0.3)
    assert raw[1]["mean"][0] < gated[1]["mean"][0]
    assert forward_speed_mps(schedule_prior(raw, flat)["mean"], st, flat) < 0.6
    assert forward_speed_mps(schedule_prior(gated, flat)["mean"], st, flat) == \
        pytest.approx(0.8, abs=0.05)


# --------------------------------------------------------------------- aggregation


def _row(qid, mission, ade, gs, gs_rand, prop_pred, prop_real):
    return {"qid": qid, "mission": mission, "anchor": qid * 4, "window_speed_mps": 0.8,
            "n_gated_in_mission": 10, "se2_chain_discrepancy_m": 0.01,
            "ade_goalonly": ade, "fde_goalonly": ade * 2, "ate_goalonly": ade,
            "rte_goalonly": 0.1, "ade_odom_goalonly": ade + 0.01, "goal_sim_goalonly": gs,
            "goal_sim_random_mean": gs_rand, "goal_sim_random_std": 0.01,
            "goal_sim_random_max": gs_rand + 0.02, "ade_random_mean": ade + 0.5,
            "ade_random_min": ade, "n_random": 8,
            "goal_sim_prior_mean": gs_rand, "ade_prior_mean": ade + 0.3,
            "prop_pred": prop_pred, "prop_real": prop_real}


def test_aggregate_splits_per_mission_and_separates_the_supervised_window():
    C = len(ACTIVE)
    pred = np.zeros((H, C)).tolist()
    real = np.concatenate([np.zeros((5, C)), np.ones((3, C))]).tolist()   # drift past 1.0 s
    rows = [_row(0, "12-03", 1.0, 0.9, 0.7, pred, real),
            _row(1, "12-03", 2.0, 0.8, 0.7, pred, real),
            _row(2, "12-09", 3.0, 0.7, 0.6, pred, real)]
    meta = {"arms": ["goalonly"], "primary_arm": "goalonly", "channel_names": ACTIVE,
            "n_supervised_steps": 5, "dt_schedule": SCHEDULE}
    res = _aggregate(rows, meta)

    assert res["label"] == "descriptive / non-comparison"
    assert set(res["per_mission"]) == {"12-03", "12-09"}
    assert res["per_mission"]["12-03"]["goalonly"]["ade_m"] == pytest.approx(1.5)
    assert res["pooled"]["goalonly"]["ade_m"] == pytest.approx(2.0)

    d = res["pooled"]["discrimination"]
    assert d["goal_sim_chosen_minus_random_mean"] == pytest.approx(0.1333, abs=1e-3)
    assert d["frac_chosen_beats_random_max"] == pytest.approx(1.0)
    assert d["ade_chosen_minus_prior_m"] == pytest.approx(-0.3)   # planning beat its own prior

    pa = res["pooled"]["proprio_alignment"]
    assert pa["n_queries"] == 3 and pa["n_supervised_steps"] == 5
    assert pa["within"]["roughness"]["mae_z"] == pytest.approx(0.0)
    assert pa["beyond"]["roughness"]["mae_z"] == pytest.approx(1.0)


def test_pearson_guards_degenerate_input():
    assert np.isnan(_pearson(np.zeros(2), np.zeros(2)))          # too few points
    assert np.isnan(_pearson(np.zeros(10), np.arange(10.0)))     # zero variance
    assert _pearson(np.arange(10.0), 2 * np.arange(10.0)) == pytest.approx(1.0)
