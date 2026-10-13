from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from feelwm.data.normalize import NormStats
from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel
from feelwm.training.curriculum import CurriculumConfig, CurriculumStage, stage_for_epoch
from feelwm.training.losses import LossConfig, compute_total_loss
from feelwm.training.scheduled_sampling import ScheduledSamplingConfig, teacher_probability
from feelwm.training import trainer as trainer_module
from feelwm.training.trainer import FullModelTrainer, TrainerConfig


B = 2
T = 2
H = 2
N = 4
D_VIS = 16
D_PROP = 8
D_RAW = 14
D_ACTION = 3
T_WIN = 6


class TinyFullModelDataset(Dataset):
    def __init__(self, n: int = 6) -> None:
        gen = torch.Generator().manual_seed(0)
        self.samples = []
        self._records = [SimpleNamespace(name="seq_a"), SimpleNamespace(name="seq_b")]
        self._index = []
        for i in range(n):
            self._index.append((i % 2, i))
            self.samples.append(
                {
                    "z_vis_context": torch.randn(T, N, D_VIS, generator=gen),
                    "proprio_window_context": torch.randn(T, T_WIN, D_RAW, generator=gen),
                    "action_context": torch.randn(T, D_ACTION, generator=gen),
                    "action_sequence": torch.randn(H, D_ACTION, generator=gen),
                    "current_feature": torch.randn(5, generator=gen),
                    "f_prop_target": torch.randn(H, 5, generator=gen),
                    "z_vis_target": torch.randn(H, N, D_VIS, generator=gen),
                    "binary_label_target": torch.randint(0, 2, (H,), generator=gen).float(),
                    "binary_valid_target_mask": torch.ones(H, dtype=torch.bool),
                    "valid_target_mask": torch.ones(H, dtype=torch.bool),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]

    def positive_rate(self) -> float:
        labels = torch.cat([s["binary_label_target"] for s in self.samples])
        return float(labels.mean())


class ForwardOnlyWrapper(nn.Module):
    """Minimal DDP-like wrapper that does not proxy custom module methods."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _stats() -> NormStats:
    return NormStats(
        feature_mean=np.zeros(5, dtype=np.float32),
        feature_std=np.ones(5, dtype=np.float32),
        window_mean=np.zeros(14, dtype=np.float32),
        window_std=np.ones(14, dtype=np.float32),
        action_mean=np.zeros(3, dtype=np.float32),
        action_std=np.ones(3, dtype=np.float32),
    )


def _model() -> TerrainWorldModel:
    return TerrainWorldModel(
        ModelConfig(
            context_len=T - 1,
            rollout_horizon=H,
            n_patch=N,
            d_vis=D_VIS,
            d_prop=D_PROP,
            d_raw=D_RAW,
            d_action=D_ACTION,
            n_layers=1,
            n_heads=4,
            ffn_mult=2,
            dropout=0.0,
            gradient_checkpointing=True,
        )
    )


def test_curriculum_stage_thresholds() -> None:
    cfg = CurriculumConfig(enabled=True, proprio_start_epoch=2, traversability_start_epoch=5)
    assert stage_for_epoch(0, cfg) is CurriculumStage.VISUAL_ONLY
    assert stage_for_epoch(1, cfg) is CurriculumStage.VISUAL_ONLY
    assert stage_for_epoch(2, cfg) is CurriculumStage.VISUAL_PROPRIO
    assert stage_for_epoch(4, cfg) is CurriculumStage.VISUAL_PROPRIO
    assert stage_for_epoch(5, cfg) is CurriculumStage.FULL


def test_scheduled_sampling_linear_floor() -> None:
    cfg = ScheduledSamplingConfig(start_epoch=5, end_epoch=15, p_start=1.0, p_end=0.5)
    assert teacher_probability(0, cfg) == 1.0
    assert teacher_probability(5, cfg) == 1.0
    assert teacher_probability(10, cfg) == 0.75
    assert teacher_probability(15, cfg) == 0.5
    assert teacher_probability(20, cfg) == 0.5


def test_loss_respects_curriculum_stages() -> None:
    model = _model()
    ds = TinyFullModelDataset(n=2)
    batch = {k: torch.stack([ds[0][k], ds[1][k]]) for k in ds[0].keys()}
    preds = model.forward_rollout(
        batch,
        batch["action_sequence"],
        use_ground_truth_visual=batch["z_vis_target"],
        use_ground_truth_proprio=batch["f_prop_target"],
        teacher_force_mask=torch.ones(B, H, dtype=torch.bool),
    )
    cfg = LossConfig()

    _, visual = compute_total_loss(preds, batch, cfg, CurriculumStage.VISUAL_ONLY)
    _, prop = compute_total_loss(preds, batch, cfg, CurriculumStage.VISUAL_PROPRIO)
    _, full = compute_total_loss(
        preds,
        batch,
        cfg,
        CurriculumStage.FULL,
        pos_weight=torch.tensor([1.0]),
    )
    assert "loss_proprio" not in visual
    assert "loss_traversability" not in prop
    assert "loss_proprio" in prop
    assert "loss_traversability" in full


def test_alignment_loss_added_only_when_enabled_and_present() -> None:
    preds = {
        "z_vis_preds": torch.zeros(1, 2, 1, 1),
        "f_prop_preds": torch.zeros(1, 2, 5),
        "z_prop_align_preds": torch.ones(1, 2, 4),
        "z_prop_align_target": torch.zeros(1, 2, 4),
    }
    targets = {
        "z_vis_target": torch.zeros(1, 2, 1, 1),
        "f_prop_target": torch.zeros(1, 2, 5),
        "valid_target_mask": torch.tensor([[True, True]]),
    }
    # disabled (lambda=0): no alignment term
    _, off = compute_total_loss(preds, targets, LossConfig(), CurriculumStage.VISUAL_PROPRIO)
    assert "loss_prop_align" not in off
    # enabled: term present, equals masked-mean of (1-0)^2 = 1.0, weighted into total
    total_on, on = compute_total_loss(
        preds, targets, LossConfig(lambda_prop_align=2.0), CurriculumStage.VISUAL_PROPRIO
    )
    assert "loss_prop_align" in on
    assert torch.isclose(on["loss_prop_align"], torch.tensor(1.0))
    assert torch.isclose(total_on, off["loss_total"] + 2.0 * 1.0)


def test_alignment_loss_ignores_proprio_horizon_cutoff() -> None:
    """Alignment is masked by valid_target_mask only, NOT proprio_horizon_mask:
    the conditioning must stay correct over the full horizon (aligned mode targets the >1 s gap)."""
    preds = {
        "z_vis_preds": torch.zeros(1, 2, 1, 1),
        "f_prop_preds": torch.zeros(1, 2, 5),
        "z_prop_align_preds": torch.ones(1, 2, 4),
        "z_prop_align_target": torch.zeros(1, 2, 4),
    }
    targets = {
        "z_vis_target": torch.zeros(1, 2, 1, 1),
        "f_prop_target": torch.zeros(1, 2, 5),
        "valid_target_mask": torch.tensor([[True, True]]),
        "proprio_horizon_mask": torch.tensor([[True, False]]),  # step 1 beyond 1s
    }
    _, out = compute_total_loss(
        preds, targets, LossConfig(lambda_prop_align=1.0), CurriculumStage.VISUAL_PROPRIO
    )
    # both steps counted (1.0), not just the in-horizon one
    assert torch.isclose(out["loss_prop_align"], torch.tensor(1.0))


def test_traversability_loss_uses_binary_valid_mask() -> None:
    preds = {
        "z_vis_preds": torch.zeros(1, 2, 1, 1),
        "f_prop_preds": torch.zeros(1, 2, 5),
        "c_pred_logits": torch.tensor([[[0.0], [10.0]]]),
    }
    targets = {
        "z_vis_target": torch.zeros(1, 2, 1, 1),
        "f_prop_target": torch.zeros(1, 2, 5),
        "binary_label_target": torch.tensor([[1.0, 0.0]]),
        "valid_target_mask": torch.tensor([[True, True]]),
        "binary_valid_target_mask": torch.tensor([[True, False]]),
    }
    _, full = compute_total_loss(
        preds,
        targets,
        LossConfig(),
        CurriculumStage.FULL,
        pos_weight=torch.tensor([1.0]),
    )
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
    )
    torch.testing.assert_close(full["loss_traversability"], expected)


def test_dummy_trainer_run_logs_and_checkpoints(tmp_path) -> None:
    train_ds = TinyFullModelDataset(n=4)
    val_ds = TinyFullModelDataset(n=4)
    trainer = FullModelTrainer(
        model=_model(),
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_stats=_stats(),
        trainer_cfg=TrainerConfig(
            epochs=3,
            batch_size=2,
            gradient_accumulation_steps=2,
            lr_warmup_epochs=1,
            num_workers=0,
            device="cpu",
            amp=False,
            progress=False,
            seed=123,
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
            per_sequence_dir=str(tmp_path / "per_sequence"),
        ),
        loss_cfg=LossConfig(),
        curriculum_cfg=CurriculumConfig(
            enabled=True,
            proprio_start_epoch=1,
            traversability_start_epoch=2,
        ),
        sampling_cfg=ScheduledSamplingConfig(
            enabled=True,
            start_epoch=0,
            end_epoch=2,
            p_start=1.0,
            p_end=0.5,
        ),
        pos_weight=1.0,
    )
    history = trainer.fit()
    assert [row["stage"] for row in history] == ["VISUAL_ONLY", "VISUAL_PROPRIO", "FULL"]
    assert history[0]["p_teacher"] == 1.0
    assert history[-1]["p_teacher"] == 0.5
    assert history[0]["lr"] == trainer.cfg.lr
    assert trainer.global_step == 3
    first_train_diag = history[0]["train"]["diagnostic_full_loss_components"]
    first_val_diag = history[0]["val"]["diagnostic_full_weighted_components"]
    assert {"loss_visual", "loss_proprio", "loss_traversability"} <= set(first_train_diag)
    assert {"visual", "proprio", "traversability", "total"} <= set(first_val_diag)
    assert set(history[-1]["train"]["binary_safety"]) >= {"auc", "f1", "threshold", "n", "pos_rate"}
    assert set(history[-1]["val"]["binary_safety"]) >= {"auc", "f1", "threshold", "n", "pos_rate"}
    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "per_sequence" / "val_epoch_002.md").exists()
    # default (no test_dataset / test_eval_every_n_epochs=0) logs no "test" block
    assert all("test" not in row for row in history)


def test_periodic_test_split_eval_logs_test_block(tmp_path) -> None:
    trainer = FullModelTrainer(
        model=_model(),
        train_dataset=TinyFullModelDataset(n=4),
        val_dataset=TinyFullModelDataset(n=4),
        test_dataset=TinyFullModelDataset(n=4),
        train_stats=_stats(),
        trainer_cfg=TrainerConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            amp=False,
            progress=False,
            seed=0,
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
            per_sequence_dir=str(tmp_path / "per_sequence"),
            test_eval_every_n_epochs=1,
        ),
        loss_cfg=LossConfig(),
        curriculum_cfg=CurriculumConfig(
            enabled=True, proprio_start_epoch=1, traversability_start_epoch=1
        ),
        sampling_cfg=ScheduledSamplingConfig(
            enabled=False, start_epoch=0, end_epoch=1, p_start=1.0, p_end=1.0
        ),
        pos_weight=1.0,
    )
    history = trainer.fit()
    # every epoch row carries a "test" block with r2 + binary_safety
    assert all("test" in row for row in history)
    assert "r2" in history[-1]["test"]
    assert set(history[-1]["test"]["binary_safety"]) >= {"auc", "f1"}
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all("test" in r for r in rows)


def test_save_epoch_checkpoints_writes_slim_per_epoch_files(tmp_path) -> None:
    trainer = FullModelTrainer(
        model=_model(),
        train_dataset=TinyFullModelDataset(n=4),
        val_dataset=TinyFullModelDataset(n=4),
        train_stats=_stats(),
        trainer_cfg=TrainerConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            amp=False,
            progress=False,
            seed=0,
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
            per_sequence_dir=str(tmp_path / "per_sequence"),
            save_epoch_checkpoints=True,
        ),
        loss_cfg=LossConfig(),
        curriculum_cfg=CurriculumConfig(enabled=False),
        sampling_cfg=ScheduledSamplingConfig(enabled=False),
        pos_weight=1.0,
    )
    trainer.fit()
    ck_dir = tmp_path / "epoch_ckpts"
    assert (ck_dir / "epoch_0.pt").exists() and (ck_dir / "epoch_1.pt").exists()
    # slim = model_state only (no optimizer), and loadable for recon
    slim = torch.load(ck_dir / "epoch_1.pt", map_location="cpu", weights_only=False)
    assert "model_state" in slim and "optimizer_state" not in slim


def test_trainer_rollout_unwraps_ddp_like_wrapper(tmp_path) -> None:
    trainer = FullModelTrainer(
        model=_model(),
        train_dataset=TinyFullModelDataset(n=2),
        val_dataset=TinyFullModelDataset(n=2),
        train_stats=_stats(),
        trainer_cfg=TrainerConfig(
            batch_size=1,
            num_workers=0,
            device="cpu",
            amp=False,
            progress=False,
            max_val_batches=1,
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
            per_sequence_dir=str(tmp_path / "per_sequence"),
        ),
        loss_cfg=LossConfig(),
        curriculum_cfg=CurriculumConfig(),
        sampling_cfg=ScheduledSamplingConfig(),
        pos_weight=1.0,
    )
    trainer.model = ForwardOnlyWrapper(trainer.model)

    summary = trainer.run_epoch(
        loader=DataLoader(trainer.val_dataset, batch_size=1, num_workers=0),
        epoch=0,
        stage=CurriculumStage.FULL,
        p_teacher=1.0,
        training=False,
    )

    assert "loss" in summary
    assert "r2" in summary


def test_seed_sets_local_cuda_device_before_torch_seed(monkeypatch) -> None:
    events = []
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        trainer_module.torch.cuda,
        "set_device",
        lambda rank: events.append(("set_device", rank)),
    )
    monkeypatch.setattr(
        trainer_module.torch,
        "manual_seed",
        lambda seed: events.append(("manual_seed", seed)),
    )
    monkeypatch.setattr(
        trainer_module.torch.cuda,
        "manual_seed_all",
        lambda seed: events.append(("manual_seed_all", seed)),
    )

    trainer_module._seed_everything(123)

    assert events == [
        ("set_device", 2),
        ("manual_seed", 123),
        ("manual_seed_all", 123),
    ]


def test_distributed_eval_sampler_shards_without_padding_or_drop() -> None:
    ds = TinyFullModelDataset(n=10)
    shards = [
        list(trainer_module._DistributedEvalSampler(ds, num_replicas=4, rank=rank))
        for rank in range(4)
    ]

    assert shards == [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]
    flat = [idx for shard in shards for idx in shard]
    assert sorted(flat) == list(range(len(ds)))


def test_r2_sufficient_stats_match_exact_summary(tmp_path) -> None:
    trainer = FullModelTrainer(
        model=_model(),
        train_dataset=TinyFullModelDataset(n=2),
        val_dataset=TinyFullModelDataset(n=2),
        train_stats=_stats(),
        trainer_cfg=TrainerConfig(
            batch_size=1,
            num_workers=0,
            device="cpu",
            amp=False,
            progress=False,
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
            per_sequence_dir=str(tmp_path / "per_sequence"),
        ),
        loss_cfg=LossConfig(),
        curriculum_cfg=CurriculumConfig(),
        sampling_cfg=ScheduledSamplingConfig(),
        pos_weight=1.0,
    )
    gen = torch.Generator().manual_seed(42)
    preds = [torch.randn(2, H, 5, generator=gen), torch.randn(1, H, 5, generator=gen)]
    targets = [torch.randn(2, H, 5, generator=gen), torch.randn(1, H, 5, generator=gen)]
    masks = [torch.tensor([[True, False], [True, True]]), torch.tensor([[False, True]])]

    exact = trainer._r2(torch.cat(preds), torch.cat(targets), torch.cat(masks))
    stats = trainer._r2_sufficient_stats(preds, targets, masks).cpu()
    reduced = trainer._r2_from_sufficient_stats(stats)

    np.testing.assert_allclose(reduced, exact, rtol=1e-6, atol=1e-6)

    exact_mae = trainer._mae(torch.cat(preds), torch.cat(targets), torch.cat(masks))
    reduced_mae = trainer._mae_from_sufficient_stats(stats)

    np.testing.assert_allclose(reduced_mae, exact_mae, rtol=1e-6, atol=1e-6)


def test_binary_sufficient_stats_survives_nonfinite_preds(tmp_path) -> None:
    # Regression: fp16 rollout overflow can emit NaN/inf c_preds. NaN survives
    # clamp(0,1) and casts to INT64_MIN, which bincount rejects. The stats pass must
    # drop non-finite preds instead of crashing, and match the finite-only result.
    trainer = FullModelTrainer(
        model=_model(),
        train_dataset=TinyFullModelDataset(n=2),
        val_dataset=TinyFullModelDataset(n=2),
        train_stats=_stats(),
        trainer_cfg=TrainerConfig(
            batch_size=1,
            num_workers=0,
            device="cpu",
            amp=False,
            progress=False,
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
            per_sequence_dir=str(tmp_path / "per_sequence"),
        ),
        loss_cfg=LossConfig(),
        curriculum_cfg=CurriculumConfig(),
        sampling_cfg=ScheduledSamplingConfig(),
        pos_weight=1.0,
    )
    clean_preds = [torch.tensor([[0.9, 0.2], [0.7, 0.1]])]
    clean_targets = [torch.tensor([[1, 0], [1, 0]])]
    clean_masks = [torch.tensor([[True, True], [True, True]])]
    clean = trainer._binary_sufficient_stats(clean_preds, clean_targets, clean_masks).cpu()

    # Same data but with a non-finite pred at a *masked-out* position, so dropping it
    # leaves the finite-only stats unchanged.
    dirty_preds = [torch.tensor([[0.9, float("nan")], [0.7, float("inf")]])]
    dirty_masks = [torch.tensor([[True, False], [True, False]])]
    dirty = trainer._binary_sufficient_stats(dirty_preds, clean_targets, dirty_masks).cpu()

    clean_kept = trainer._binary_sufficient_stats(
        [torch.tensor([[0.9], [0.7]])],
        [torch.tensor([[1], [1]])],
        [torch.tensor([[True], [True]])],
    ).cpu()
    assert torch.isfinite(dirty).all()
    torch.testing.assert_close(dirty, clean_kept)


def test_binary_sufficient_stats_drops_nonfinite_at_valid_position(tmp_path) -> None:
    # A non-finite pred at a *valid* position must be dropped (not crash, not poison).
    trainer = FullModelTrainer(
        model=_model(),
        train_dataset=TinyFullModelDataset(n=2),
        val_dataset=TinyFullModelDataset(n=2),
        train_stats=_stats(),
        trainer_cfg=TrainerConfig(
            batch_size=1,
            num_workers=0,
            device="cpu",
            amp=False,
            progress=False,
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
            per_sequence_dir=str(tmp_path / "per_sequence"),
        ),
        loss_cfg=LossConfig(),
        curriculum_cfg=CurriculumConfig(),
        sampling_cfg=ScheduledSamplingConfig(),
        pos_weight=1.0,
    )
    preds = [torch.tensor([[float("nan"), 0.8]])]
    targets = [torch.tensor([[1, 0]])]
    masks = [torch.tensor([[True, True]])]
    stats = trainer._binary_sufficient_stats(preds, targets, masks).cpu()
    assert torch.isfinite(stats).all()
    # Only the finite (pred=0.8, target=0) sample should be counted: 1 negative, 0 pos.
    tp, fp, fn, n_pos, n_neg = [float(x) for x in stats[:5]]
    assert n_pos == 0.0 and n_neg == 1.0


def test_loss_probe_returns_ranked_lambda_candidates(tmp_path) -> None:
    trainer = FullModelTrainer(
        model=_model(),
        train_dataset=TinyFullModelDataset(n=2),
        val_dataset=TinyFullModelDataset(n=2),
        train_stats=_stats(),
        trainer_cfg=TrainerConfig(
            batch_size=1,
            num_workers=0,
            device="cpu",
            amp=False,
            progress=False,
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
            per_sequence_dir=str(tmp_path / "per_sequence"),
        ),
        loss_cfg=LossConfig(),
        curriculum_cfg=CurriculumConfig(),
        sampling_cfg=ScheduledSamplingConfig(),
        pos_weight=1.0,
    )
    result = trainer.loss_probe(max_batches=1)
    assert set(result["raw_components"]) == {"visual", "proprio", "traversability"}
    assert len(result["candidates"]) == 9
    ratios = [row["max_min_ratio"] for row in result["candidates"]]
    assert ratios == sorted(ratios)
    assert result["recommended"] == result["candidates"][0]


def test_teacher_force_mask_granularity() -> None:
    import pytest

    from feelwm.training.scheduled_sampling import sample_teacher_force_mask

    torch.manual_seed(0)
    B, H = 512, 8
    # "sample": each row is constant (fully teacher-forced or fully free)
    m = sample_teacher_force_mask(B, H, 0.5, torch.device("cpu"), granularity="sample")
    assert m.shape == (B, H)
    assert torch.all(m.all(dim=1) | (~m).any(dim=1) == torch.ones(B, dtype=torch.bool))
    assert (m[:, :1] == m).all(), "sample granularity must broadcast one coin per row"
    frac = m[:, 0].float().mean().item()
    assert 0.4 < frac < 0.6  # expected teacher fraction preserved
    # "step" keeps legacy per-cell behavior: rows with mixed values must exist
    m2 = sample_teacher_force_mask(B, H, 0.5, torch.device("cpu"), granularity="step")
    mixed = ((m2.any(dim=1)) & (~m2.all(dim=1))).float().mean().item()
    assert mixed > 0.9
    # p extremes short-circuit for both granularities
    assert sample_teacher_force_mask(4, H, 1.0, torch.device("cpu"), granularity="sample").all()
    assert not sample_teacher_force_mask(4, H, 0.0, torch.device("cpu"), granularity="sample").any()
    with pytest.raises(ValueError):
        sample_teacher_force_mask(4, H, 0.5, torch.device("cpu"), granularity="block")
    with pytest.raises(ValueError):
        ScheduledSamplingConfig(mask_granularity="block")


def test_auto_resume_prefers_existing_checkpoint_over_resume_path(tmp_path) -> None:
    def make(resume_path=None, auto_resume=True, epochs=2):
        return FullModelTrainer(
            model=_model(),
            train_dataset=TinyFullModelDataset(n=4),
            val_dataset=TinyFullModelDataset(n=4),
            train_stats=_stats(),
            trainer_cfg=TrainerConfig(
                epochs=epochs,
                batch_size=2,
                num_workers=0,
                device="cpu",
                amp=False,
                progress=False,
                seed=0,
                checkpoint_path=str(tmp_path / "checkpoint.pt"),
                metrics_path=str(tmp_path / "metrics.jsonl"),
                per_sequence_dir=str(tmp_path / "per_sequence"),
                resume_path=resume_path,
                auto_resume=auto_resume,
            ),
            loss_cfg=LossConfig(),
            curriculum_cfg=CurriculumConfig(enabled=False),
            sampling_cfg=ScheduledSamplingConfig(
                enabled=False, start_epoch=0, end_epoch=1, p_start=1.0, p_end=1.0
            ),
            pos_weight=1.0,
        )

    # first launch: no checkpoint yet -> trains from scratch, leaves a full resume ckpt
    make().fit()

    # a warm-start init (model_state only, like a converted donor checkpoint)
    init = tmp_path / "init.pt"
    torch.save({"model_state": _model().state_dict()}, init)

    # relaunch with resume_path=init: auto-resume must pick checkpoint.pt instead
    resumed = make(resume_path=str(init), epochs=3)
    assert resumed.start_epoch == 2
    history = resumed.fit()
    assert [row["epoch"] for row in history] == [0, 1, 2]  # restored history + 1 new epoch

    # auto_resume=False honors resume_path: warm start at epoch 0
    fresh = make(resume_path=str(init), auto_resume=False)
    assert fresh.start_epoch == 0
    assert fresh.global_step == 0


def test_freeze_modules_excludes_params_from_training(tmp_path) -> None:
    import pytest

    def make(freeze):
        return FullModelTrainer(
            model=_model(),
            train_dataset=TinyFullModelDataset(n=4),
            val_dataset=TinyFullModelDataset(n=4),
            train_stats=_stats(),
            trainer_cfg=TrainerConfig(
                epochs=1,
                batch_size=2,
                num_workers=0,
                device="cpu",
                amp=False,
                progress=False,
                seed=0,
                checkpoint_path=str(tmp_path / "checkpoint.pt"),
                metrics_path=str(tmp_path / "metrics.jsonl"),
                per_sequence_dir=str(tmp_path / "per_sequence"),
                freeze_modules=freeze,
            ),
            loss_cfg=LossConfig(),
            curriculum_cfg=CurriculumConfig(enabled=False),
            sampling_cfg=ScheduledSamplingConfig(
                enabled=False, start_epoch=0, end_epoch=1, p_start=1.0, p_end=1.0
            ),
            pos_weight=1.0,
        )

    trainer = make(["dynamics", "film", "action_embedding", "visual_head"])
    model = trainer.model
    assert all(not p.requires_grad for p in model.dynamics.parameters())
    assert all(p.requires_grad for p in model.proprio_encoder.parameters())
    n_opt = sum(p.numel() for g in trainer.optimizer.param_groups for p in g["params"])
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_opt == n_trainable

    frozen_before = {k: v.clone() for k, v in model.dynamics.state_dict().items()}
    enc_before = {k: v.clone() for k, v in model.proprio_encoder.state_dict().items()}
    trainer.fit()
    assert all(torch.equal(frozen_before[k], v) for k, v in model.dynamics.state_dict().items())
    assert any(not torch.equal(enc_before[k], v) for k, v in model.proprio_encoder.state_dict().items())

    with pytest.raises(ValueError, match="freeze_modules"):
        make(["not_a_module"])


def test_noprop_freezes_film_proprio_proj(tmp_path) -> None:
    """use_proprio_input=False: the trainer freezes the orphaned FiLM proprio branch
    (the only gradless params — tests/models/test_query_trim.py) so DDP can run with
    find_unused_parameters=False; the params stay out of the optimizer but keep their
    state_dict keys."""

    def make(use_proprio_input):
        return FullModelTrainer(
            model=TerrainWorldModel(
                ModelConfig(
                    context_len=T - 1,
                    rollout_horizon=H,
                    n_patch=N,
                    d_vis=D_VIS,
                    d_prop=D_PROP,
                    d_raw=D_RAW,
                    d_action=D_ACTION,
                    n_layers=1,
                    n_heads=4,
                    ffn_mult=2,
                    dropout=0.0,
                    use_proprio_input=use_proprio_input,
                )
            ),
            train_dataset=TinyFullModelDataset(n=4),
            val_dataset=TinyFullModelDataset(n=4),
            train_stats=_stats(),
            trainer_cfg=TrainerConfig(
                epochs=1,
                batch_size=2,
                num_workers=0,
                device="cpu",
                amp=False,
                progress=False,
                seed=0,
                checkpoint_path=str(tmp_path / "checkpoint.pt"),
                metrics_path=str(tmp_path / "metrics.jsonl"),
                per_sequence_dir=str(tmp_path / "per_sequence"),
            ),
            loss_cfg=LossConfig(),
            curriculum_cfg=CurriculumConfig(enabled=False),
            sampling_cfg=ScheduledSamplingConfig(
                enabled=False, start_epoch=0, end_epoch=1, p_start=1.0, p_end=1.0
            ),
            pos_weight=1.0,
        )

    trainer = make(use_proprio_input=False)
    model = trainer.model
    assert all(not p.requires_grad for p in model.film.proprio_proj.parameters())
    opt_params = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    assert not any(id(p) in opt_params for p in model.film.proprio_proj.parameters())
    assert "film.proprio_proj.weight" in model.state_dict()  # ckpt keys unchanged

    trainer_on = make(use_proprio_input=True)
    assert all(p.requires_grad for p in trainer_on.model.film.proprio_proj.parameters())


def test_channel_indices_use_the_legged_registry_for_legged_stats(tmp_path) -> None:
    """Per-channel denorm maps active → canonical indices. With LEGGED stats the canonical
    order is LEGGED_FEATURE_CHANNELS, and a subset can be all names the WHEELED registry
    also has — name auto-detection would then resolve to the wrong columns (roughness /
    angular_instability / slope_power = legged [0,1,2] but wheeled [0,2,4])."""
    def make(feature_set: str, names: list[str]):
        stats = _stats()
        return FullModelTrainer(
            model=_model(),
            train_dataset=TinyFullModelDataset(n=2),
            val_dataset=TinyFullModelDataset(n=2),
            train_stats=NormStats(
                feature_mean=stats.feature_mean, feature_std=stats.feature_std,
                window_mean=stats.window_mean, window_std=stats.window_std,
                action_mean=stats.action_mean, action_std=stats.action_std,
                feature_set=feature_set,
            ),
            trainer_cfg=TrainerConfig(
                epochs=1, batch_size=2, num_workers=0, device="cpu", amp=False,
                progress=False, seed=0,
                checkpoint_path=str(tmp_path / "checkpoint.pt"),
                metrics_path=str(tmp_path / "metrics.jsonl"),
                per_sequence_dir=str(tmp_path / "per_sequence"),
            ),
            loss_cfg=LossConfig(channel_names=names, proprio_weights=[1.0] * len(names)),
            curriculum_cfg=CurriculumConfig(enabled=False),
            sampling_cfg=ScheduledSamplingConfig(
                enabled=False, start_epoch=0, end_epoch=1, p_start=1.0, p_end=1.0
            ),
            pos_weight=1.0,
        )

    shared = ["roughness", "angular_instability", "slope_power"]
    assert make("legged", shared).channel_indices == [0, 1, 2]
    assert make("wheeled", shared).channel_indices == [0, 2, 4]
    # legged v2 (with a legged-only name) is unchanged by the registry pin
    assert make("legged", shared + ["cost_of_transport"]).channel_indices == [0, 1, 2, 4]


def test_per_domain_val_datasets_get_their_own_metric_block(tmp_path) -> None:
    """Combined multi-corpus training (Gazebo + Isaac pooled): each
    corpus is validated on its own every epoch, through the SAME run_validation path as the
    pooled val, and lands under row["val_by_domain"][name]. A pooled val number cannot
    answer "does pooling beat each single-domain corpus on ITS OWN domain"."""
    trainer = FullModelTrainer(
        model=_model(),
        train_dataset=TinyFullModelDataset(n=4),
        val_dataset=TinyFullModelDataset(n=4),
        train_stats=_stats(),
        trainer_cfg=TrainerConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            amp=False,
            progress=False,
            seed=0,
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
            per_sequence_dir=str(tmp_path / "per_sequence"),
        ),
        loss_cfg=LossConfig(),
        curriculum_cfg=CurriculumConfig(enabled=False),
        sampling_cfg=ScheduledSamplingConfig(enabled=False),
        pos_weight=1.0,
        extra_val_datasets={"gazebo": TinyFullModelDataset(n=4),
                            "isaac": TinyFullModelDataset(n=2)},
    )
    history = trainer.fit()
    for row in history:
        assert set(row["val_by_domain"]) == {"gazebo", "isaac"}
        for block in row["val_by_domain"].values():
            # the endpoint metrics the readout needs: per-channel R2 and trav AUC
            assert set(block["r2"]) == set(row["val"]["r2"])
            assert set(block["binary_safety"]) >= {"auc", "f1"}
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all("val_by_domain" in r for r in rows)
    # ...and the pooled val / best-checkpoint logic is untouched by their presence
    assert "val" in history[-1] and (tmp_path / "checkpoint.pt").exists()


def test_no_extra_val_datasets_logs_no_domain_block(tmp_path) -> None:
    """Single-corpus runs (every existing sim/isaac/wheeled config) pay nothing: no extra
    loader, no extra pass, no extra key."""
    trainer = FullModelTrainer(
        model=_model(),
        train_dataset=TinyFullModelDataset(n=4),
        val_dataset=TinyFullModelDataset(n=4),
        train_stats=_stats(),
        trainer_cfg=TrainerConfig(
            epochs=1, batch_size=2, num_workers=0, device="cpu", amp=False, progress=False,
            seed=0,
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            metrics_path=str(tmp_path / "metrics.jsonl"),
            per_sequence_dir=str(tmp_path / "per_sequence"),
        ),
        loss_cfg=LossConfig(),
        curriculum_cfg=CurriculumConfig(enabled=False),
        sampling_cfg=ScheduledSamplingConfig(enabled=False),
        pos_weight=1.0,
    )
    history = trainer.fit()
    assert trainer.extra_val_datasets == {}
    assert all("val_by_domain" not in row for row in history)


def test_periodic_checkpoint_snapshots_mid_epoch_without_touching_canonical(tmp_path) -> None:
    """periodic_checkpoint_minutes: mid-epoch crash insurance for multi-hour epochs."""
    import os

    def make(minutes, epochs=2):
        return FullModelTrainer(
            model=_model(),
            train_dataset=TinyFullModelDataset(n=4),
            val_dataset=TinyFullModelDataset(n=4),
            train_stats=_stats(),
            trainer_cfg=TrainerConfig(
                epochs=epochs,
                batch_size=2,
                num_workers=0,
                device="cpu",
                amp=False,
                progress=False,
                seed=0,
                checkpoint_path=str(tmp_path / "checkpoint.pt"),
                metrics_path=str(tmp_path / "metrics.jsonl"),
                per_sequence_dir=str(tmp_path / "per_sequence"),
                periodic_checkpoint_minutes=minutes,
            ),
            loss_cfg=LossConfig(),
            curriculum_cfg=CurriculumConfig(enabled=False),
            sampling_cfg=ScheduledSamplingConfig(
                enabled=False, start_epoch=0, end_epoch=1, p_start=1.0, p_end=1.0
            ),
            pos_weight=1.0,
        )

    canonical = tmp_path / "checkpoint.pt"
    partial = tmp_path / "checkpoint_partial.pt"

    # off by default: no partial file, no behaviour change
    make(0.0).fit()
    assert not partial.exists()
    assert canonical.exists()
    canonical.unlink()
    (tmp_path / "metrics.jsonl").unlink()

    # tiny interval -> every optimizer step qualifies
    make(1e-9).fit()
    assert partial.exists()
    part = torch.load(partial, map_location="cpu", weights_only=False)
    assert part["partial_epoch"] == 1               # written during the last epoch
    assert part["next_epoch"] == 1                  # -> resume REPLAYS that epoch
    assert "optimizer_state" in part                # full resume payload, not the slim one
    # the canonical checkpoint stays the epoch-boundary artifact
    assert torch.load(canonical, map_location="cpu", weights_only=False)["next_epoch"] == 2
    assert not (tmp_path / "checkpoint.pt.tmp").exists()   # atomic write left no residue

    # auto-resume takes the newer of the two: canonical (written at epoch end) wins here
    assert make(0.0, epochs=3).start_epoch == 2
    # ...and the partial wins once it is the later state (i.e. a crash inside an epoch)
    os.utime(partial, (canonical.stat().st_mtime + 10, canonical.stat().st_mtime + 10))
    assert make(0.0, epochs=3).start_epoch == 1
