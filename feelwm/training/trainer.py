from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from feelwm.config import add_variant_arg, load_config
from feelwm.data.dataset import DatasetConfig, TartanDriveDataset
from feelwm.data.normalize import (
    FEATURE_CHANNELS,
    LEGGED_FEATURE_CHANNELS,
    NormStats,
    denormalize_feature,
)
from feelwm.models.terrain_wm import ModelConfig, TerrainWorldModel
from feelwm.training.curriculum import CurriculumConfig, CurriculumStage, stage_for_epoch
from feelwm.training.losses import (
    CHANNEL_NAMES,
    LossConfig,
    compute_total_loss,
    resolve_channel_indices,
)
from feelwm.training.scheduled_sampling import (
    ScheduledSamplingConfig,
    sample_teacher_force_mask,
    teacher_probability,
)


@dataclass
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    lr: float = 3e-4
    lr_warmup_epochs: int = 0
    # Post-warmup LR schedule. "constant" (default) holds base_lr after warmup — the
    # original behaviour. "cosine" decays base_lr → base_lr*lr_min_factor
    # over the post-warmup steps (cosine), to settle the late-epoch plateau/val-noise seen
    # with the constant schedule (best usually ep10–15 then bounces).
    lr_schedule: str = "constant"
    lr_min_factor: float = 0.0
    # Two-group LR. None (default) = ONE group at `lr`, unchanged behaviour. When set,
    # parameters whose qualified name starts with any `lr_fresh_prefixes` entry — a freshly
    # initialised module inserted into a pretrained stack — train at `lr_fresh`, and all
    # other trainable parameters (pretrained core, proprio encoder/head) at `lr`. Warmup
    # and the cosine decay scale BOTH groups multiplicatively, so the ratio holds for the
    # whole schedule. Motivation: inserting the token bottleneck into a pretrained stack
    # applied the lr tuned for 1.3M trainable params to 58.2M, 14.4M of them random.
    lr_fresh: float | None = None
    lr_fresh_prefixes: list[str] = field(default_factory=lambda: ["dynamics.bottleneck."])
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    num_workers: int = 4
    # Workers for the val/test loaders. None -> reuse num_workers. Eval loaders also run
    # with persistent_workers=False, so their workers are torn down after each eval instead
    # of sitting idle (holding process slots + RAM) through every training epoch. Set this
    # lower than num_workers when running several DDP jobs concurrently on one host: steady-
    # state CPU contention is driven by the TRAIN workers (sum over all jobs should stay
    # <= core count), and the epoch-boundary spike (train prefetch overlapping a just-
    # finished eval) is what tipped a contended box past the NCCL watchdog timeout.
    eval_num_workers: int | None = None
    # NCCL/gloo collective timeout (minutes). The default torch value is 10 min, which a
    # transient DataLoader stall on a CPU-oversubscribed host can exceed at an epoch
    # boundary, killing a multi-hour run. Raised so I/O contention degrades throughput
    # instead of aborting the job.
    dist_timeout_minutes: int = 30
    # Per-batch fetch timeout (seconds) for the train/val/test DataLoaders (0 = wait
    # forever, torch default). A worker wedged at spawn (fork-with-live-CUDA/NCCL-threads
    # deadlock) otherwise hangs its rank SILENTLY until the NCCL watchdog kills the job
    # 30 min later with an unrelated-looking collective timeout (observed with two DDP
    # jobs sharing one host: once at first val spawn, once at train spawn on 2/4
    # ranks — host RAM/CPU/disk/NCCL all verified healthy). With a timeout the stuck rank
    # raises a RuntimeError NAMING the loader instead. Set well above any real batch
    # latency (steady-state is seconds); only pathological stalls should trip it.
    dataloader_timeout_s: float = 0.0
    device: str = "cuda"
    amp: bool = True
    amp_dtype: str = "float16"
    compile: bool = False
    # torch.compile mode. "default" is the safe choice here: "reduce-overhead" wraps
    # steps in CUDA graphs, which interacts badly with DDP comm hooks and with the
    # eager (uncompiled) forward_rollout used in validation.
    compile_mode: str = "default"
    log_every: int = 20
    progress: bool = True
    seed: int = 0
    checkpoint_path: str = "runs/default/checkpoint.pt"
    resume_path: str | None = None
    # Channel subset (names) used for best-checkpoint selection; None = all
    # loss.channel_names. Use when one channel is structurally broken on the val split
    # (e.g. real-robot fwd/reverse split: slope_power sign-flips, its R² noise would
    # otherwise dominate the mean and corrupt checkpoint_best.pt).
    best_val_channels: list[str] | None = None
    # Top-level TerrainWorldModel submodule names whose parameters are frozen
    # (requires_grad=False, excluded from the optimizer). Low-data adaptation recipe
    # (real-robot fine-tuning): freeze the transplanted stack [dynamics, film,
    # action_embedding, visual_head] and train only the new proprio-side parts.
    # Losses are unchanged — e.g. the visual loss still flows THROUGH a frozen
    # dynamics into the trainable proprio encoder (conditioning signal).
    freeze_modules: list[str] = field(default_factory=list)
    # If checkpoint_path already exists at startup, resume from it — and IGNORE
    # resume_path (which then only seeds the FIRST launch, e.g. a warm-start init for
    # real-robot fine-tuning). Interrupted runs therefore continue with the unchanged
    # launch command. Set false (or delete the file) to intentionally restart a run.
    auto_resume: bool = True
    metrics_path: str = "runs/default/metrics.jsonl"
    per_sequence_dir: str = "runs/default/per_sequence"
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    # Linearly re-warm the lr over this many optimizer steps after each curriculum
    # stage transition (0 = off). Newly-activated losses (proprio/trav at FULL) hit
    # parameters whose Adam second moments were accumulated under the previous stage's
    # gradient scale; full-lr steps there caused a measured AdaLN training divergence
    # (~43 oversized steps -> AdaLN modulation blow-up -> permanent bf16 overflow).
    stage_rewarmup_steps: int = 0
    # ---- wandb (rank0-only; disabled when wandb_project is null) ----
    # Step-level: loss terms, grad_norm, lr, non-finite skip count every log_every
    # optimizer steps (grad_norm/skips are the early-warning signals for the divergence
    # class above). Epoch-level: the metrics.jsonl row, flattened.
    wandb_project: str | None = None
    wandb_run_name: str | None = None   # default: experiment dir name (checkpoint_path parent)
    wandb_mode: str = "online"          # "online" | "offline" | "disabled"
    # Run an OOD test-split eval every N epochs (0 = off); logged under the "test" key in
    # metrics.jsonl alongside "val". Needs a test_dataset passed to the trainer.
    test_eval_every_n_epochs: int = 0
    # Save a slim model-only checkpoint per epoch to <checkpoint_dir>/epoch_ckpts/epoch_<N>.pt
    # (for offline per-epoch reconstruction; optimizer/scaler omitted, NOT for resume).
    save_epoch_checkpoints: bool = False
    # Mid-epoch crash insurance (0 = off). Every N minutes rank0 writes a FULL resume
    # checkpoint to <checkpoint>_partial.pt at an optimizer-step boundary. The canonical
    # checkpoint_path is NEVER overwritten by it: the partial carries next_epoch=<current
    # epoch>, so resuming from it replays the current epoch from its start with the
    # weights reached so far. Needed when one epoch costs hours — without it a crash
    # (or a preempted GPU) loses everything since the last epoch boundary.
    periodic_checkpoint_minutes: float = 0.0


class _IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.dataset[idx])
        sample["_sample_idx"] = torch.tensor(idx, dtype=torch.long)
        return sample


class _DistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices without padding or dropping samples."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        if len(self.dataset) <= self.rank:
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP and torch.compile wrappers."""
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def build_param_groups(
    model: torch.nn.Module,
    lr: float,
    lr_fresh: float | None,
    fresh_prefixes: list[str],
) -> list[dict[str, Any]]:
    """AdamW param groups: fresh-module params at `lr_fresh`, every other trainable
    parameter at `lr`. Returns a single group when `lr_fresh` is None (the pre-run-2
    behaviour). The pretrained group comes FIRST so `_current_lr()` — and the `lr` field
    of metrics.jsonl — keep meaning the main lr. Pass an UNWRAPPED model: the prefixes
    are matched against qualified names, which DDP/compile would shift.
    """
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if lr_fresh is None:
        return [{"params": [p for _, p in trainable], "lr": lr, "name": "all"}]
    prefixes = tuple(fresh_prefixes)
    fresh = [p for n, p in trainable if n.startswith(prefixes)]
    base = [p for n, p in trainable if not n.startswith(prefixes)]
    if not fresh:
        raise ValueError(
            f"lr_fresh is set but no trainable parameter matches lr_fresh_prefixes "
            f"{list(fresh_prefixes)} — the second group would be silently empty"
        )
    groups = [{"params": base, "lr": lr, "name": "pretrained"}] if base else []
    groups.append({"params": fresh, "lr": lr_fresh, "name": "fresh"})
    return groups


def _flatten_numeric(d: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested metric dicts to 'train/loss', 'val/r2/roughness', ... keeping
    only numeric leaves (wandb-friendly)."""
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_numeric(v, prefix=f"{key}/"))
        elif isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def _model_for_custom_methods(model: torch.nn.Module) -> torch.nn.Module:
    """Return the module that exposes non-forward methods such as forward_rollout."""
    if hasattr(model, "forward_rollout"):
        return model
    wrapped = getattr(model, "module", None)
    if isinstance(wrapped, torch.nn.Module) and hasattr(wrapped, "forward_rollout"):
        return wrapped
    return model


def _set_ddp_cuda_device_from_env() -> int | None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or not torch.cuda.is_available():
        return None
    rank = int(local_rank)
    torch.cuda.set_device(rank)
    return rank


class FullModelTrainer:
    def __init__(
        self,
        model: TerrainWorldModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        train_stats: NormStats,
        trainer_cfg: TrainerConfig,
        loss_cfg: LossConfig,
        curriculum_cfg: CurriculumConfig,
        sampling_cfg: ScheduledSamplingConfig,
        pos_weight: float | None = None,
        test_dataset: Dataset | None = None,
        run_config: dict[str, Any] | None = None,
        extra_val_datasets: dict[str, Dataset] | None = None,
    ) -> None:
        # ---- distributed init ----
        self.is_ddp = "LOCAL_RANK" in os.environ
        if self.is_ddp:
            local_rank = int(os.environ["LOCAL_RANK"])
            _set_ddp_cuda_device_from_env()
            dist_timeout = timedelta(minutes=trainer_cfg.dist_timeout_minutes)
            dist.init_process_group(backend="nccl", timeout=dist_timeout)
            self.cpu_group = dist.new_group(backend="gloo", timeout=dist_timeout)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = local_rank
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.cpu_group = None
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            requested = trainer_cfg.device
            self.device = torch.device(
                requested if (torch.cuda.is_available() or requested == "cpu") else "cpu"
            )
        self.is_rank0 = self.rank == 0

        # ---- model: device → compile → DDP ----
        # compile-then-DDP (not DDP-then-compile): DDPOptimizer's bucket-split overlap
        # is irrelevant here — grads are ~190MB vs a multi-second step, so comm is noise
        # — and wrapping the compiled module keeps DDP semantics (find_unused_parameters,
        # no_sync) on the battle-tested eager path. Validation still runs eager:
        # _model_for_custom_methods unwraps to the original module's forward_rollout.
        self.model = model.to(self.device)
        if trainer_cfg.freeze_modules:
            frozen = trainable = 0
            for name in trainer_cfg.freeze_modules:
                mod = getattr(model, name, None)
                if not isinstance(mod, torch.nn.Module):
                    raise ValueError(
                        f"freeze_modules: model has no submodule {name!r} "
                        "(disabled by config, or a typo)"
                    )
                for p in mod.parameters():
                    p.requires_grad_(False)
            for p in model.parameters():
                (trainable, frozen) = (
                    (trainable + p.numel(), frozen)
                    if p.requires_grad
                    else (trainable, frozen + p.numel())
                )
            if self.is_rank0:
                print(
                    f"[freeze] {trainer_cfg.freeze_modules}: "
                    f"{frozen / 1e6:.1f}M params frozen, {trainable / 1e6:.1f}M trainable"
                )
        # No-proprio arm (use_proprio_input=False): the conditioning module's proprio
        # branch (film/cond .proprio_proj) is the ONLY parameter set that never receives
        # grad (verified: tests/models/test_query_trim.py grad-coverage test). Freeze it
        # so DDP can run in strict mode below — the same battle-tested regime as the
        # proprio-ON runs — instead of paying the per-iteration unused-parameter
        # traversal. Numerically identical: gradless params were never updated (AdamW
        # skips grad=None; no weight decay either) and state_dict keys are unchanged,
        # so old no-proprio checkpoints stay loadable. NOTE: this shrinks the optimizer
        # param list — do not flip this behaviour mid-run on an auto_resume.
        if not model.cfg.use_proprio_input:
            cond_module = getattr(model, "film", None) or getattr(model, "cond", None)
            if cond_module is not None and hasattr(cond_module, "proprio_proj"):
                for p in cond_module.proprio_proj.parameters():
                    p.requires_grad_(False)
        # Same treatment, same reason, for every rollout_proprio_mode except the default
        # "feedback": TerrainWorldModel._advance_proprio returns on the mode branch BEFORE
        # either rollout_prop_projection call site (the free-running one and the
        # teacher-forced GT one), so that Linear is unreachable and its weight/bias are
        # the only params without grad — DDP strict mode reported exactly these two
        # (indices 111/112) on the freeze arm. Verified by the grad-coverage test in
        # tests/models/test_query_trim.py, which asserts they are the ONLY orphans.
        # The modes are listed explicitly rather than as `!= "feedback"` so that a future
        # mode that DOES use the projection is not silently frozen.
        if (
            model.cfg.use_proprio_input
            and model.cfg.rollout_proprio_mode in {"freeze", "zero", "aligned"}
        ):
            proj = getattr(model, "rollout_prop_projection", None)
            if proj is not None:
                for p in proj.parameters():
                    p.requires_grad_(False)
        if trainer_cfg.compile:
            self.model = torch.compile(self.model, mode=trainer_cfg.compile_mode)
        if self.is_ddp:
            # find_unused_parameters is needed whenever some parameter can receive no grad:
            #   (a) the curriculum gates head losses (e.g. trav head gets no grad during
            #       VISUAL_ONLY — strict mode would crash now that rollout goes THROUGH
            #       DDP.forward, see TerrainWorldModel.forward); OR
            #   (b) proprio_injection="token" — proprio is routed to the dynamics token
            #       only, so the FiLM proprio_proj is never called (no grad).
            # use_proprio_input=False no longer needs the flag: its orphaned conditioning
            # params are frozen above (frozen params are excluded from the DDP reducer).
            # rollout_proprio_mode in {freeze, zero, aligned} likewise does NOT need it:
            # its one orphaned Linear (rollout_prop_projection) is frozen above too.
            # When neither holds, every parameter is used from step 0, so skip the extra
            # per-iteration autograd traversal the flag costs.
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=(
                    curriculum_cfg.enabled
                    or model.cfg.proprio_injection == "token"
                ),
            )

        # forward_rollout encodes proprio_window_target into alignment targets whenever
        # it is passed (encoder present) — but those targets are consumed only by the
        # aligned rollout mode (alignment loss) and the encode-oracle eval. Everywhere
        # else the encode is dead compute, so the loops below pass the windows only when
        # this is set (run_validation additionally passes them for encode_oracle_proprio).
        self._needs_prop_window_target = (
            model.cfg.rollout_proprio_mode == "aligned" and model.cfg.use_proprio_input
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        # Named extra val datasets, evaluated every epoch with the SAME code path as the
        # main val loop and reported under row["val_by_domain"][name]. Used by combined
        # multi-corpus training to keep the per-domain endpoints visible (a pooled val
        # number cannot answer "does pooling beat each corpus on its own domain").
        self.extra_val_datasets: dict[str, Dataset] = dict(extra_val_datasets or {})
        self.stats = train_stats
        self.cfg = trainer_cfg
        self.loss_cfg = loss_cfg
        self.channel_names = list(loss_cfg.channel_names)
        # Registry from the norm stats, not name auto-detection: a legged channel subset
        # can be all-shared-names (e.g. [roughness, angular_instability, slope_power]),
        # which auto-detect would resolve against the WHEELED registry → wrong canonical
        # indices → wrong denorm affine in per-channel R². feature_set is "legged" only
        # for legged stats ("wheeled"/"sim" both use FEATURE_CHANNELS).
        registry = (
            LEGGED_FEATURE_CHANNELS if train_stats.feature_set == "legged" else FEATURE_CHANNELS
        )
        self.channel_indices = resolve_channel_indices(self.channel_names, registry)
        self.best_val_channels = list(trainer_cfg.best_val_channels or self.channel_names)
        unknown = [c for c in self.best_val_channels if c not in self.channel_names]
        if unknown:
            raise ValueError(
                f"best_val_channels {unknown} not in loss.channel_names {self.channel_names}"
            )
        self.curriculum_cfg = curriculum_cfg
        self.sampling_cfg = sampling_cfg
        self.pos_weight = (
            torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
            if pos_weight is not None
            else None
        )
        param_groups = build_param_groups(
            _unwrap_model(self.model),
            lr=trainer_cfg.lr,
            lr_fresh=trainer_cfg.lr_fresh,
            fresh_prefixes=trainer_cfg.lr_fresh_prefixes,
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=trainer_cfg.lr,
            weight_decay=trainer_cfg.weight_decay,
        )
        if len(param_groups) > 1 and self.is_rank0:
            print(
                "[lr groups] "
                + "  ".join(
                    f"{g['name']}: {sum(p.numel() for p in g['params']) / 1e6:.1f}M @ {g['lr']:g}"
                    for g in param_groups
                )
            )
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.warmup_steps = 0
        self.global_step = 0
        self.use_amp = trainer_cfg.amp and self.device.type == "cuda"
        self.amp_dtype = torch.float16 if trainer_cfg.amp_dtype == "float16" else torch.bfloat16
        # Loss scaling exists to keep fp16 grads out of the subnormal range; bf16 has
        # fp32's exponent range, so the scaler is pure overhead there. Disabled it is a
        # no-op passthrough (scale/unscale_/update return early, step calls optimizer.step),
        # so the non-finite-grad guard path below works unchanged.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        self.start_epoch = 0
        self.history: list[dict[str, Any]] = []
        self.best_val_r2 = float("-inf")
        self._last_periodic_save = time.time()

        resume_from = trainer_cfg.resume_path
        if trainer_cfg.auto_resume and trainer_cfg.checkpoint_path:
            # Auto-resume from the LATEST of the epoch-boundary checkpoint and the
            # mid-epoch snapshot (periodic_checkpoint_minutes). Both are written by this
            # run in chronological order and atomically (never truncated), so mtime is
            # the correct ordering; the partial only ever wins when the process died
            # inside an epoch, and resuming from it replays that epoch from its start.
            candidates = [
                p for p in (Path(trainer_cfg.checkpoint_path),
                            Path(self._partial_checkpoint_path()))
                if p.exists()
            ]
            if candidates:
                newest = max(candidates, key=lambda p: p.stat().st_mtime)
                resume_from = str(newest)
                if self.is_rank0:
                    print(
                        f"[resume] existing {newest} -> auto-resuming "
                        "(overrides resume_path; set training.auto_resume: false or delete "
                        "the file to start fresh)"
                    )
        if resume_from:
            self.load_checkpoint(resume_from)
            if self.is_rank0:
                print(
                    f"[resume] loaded {resume_from}: start_epoch={self.start_epoch}, "
                    f"global_step={self.global_step}"
                )
                if self.start_epoch >= trainer_cfg.epochs:
                    print(
                        f"[resume] WARNING: start_epoch {self.start_epoch} >= epochs "
                        f"{trainer_cfg.epochs} — this run is already complete; fit() will exit"
                    )

        # wandb (rank0-only, lazy import so the dependency stays optional)
        self._wandb = None
        if self.is_rank0 and trainer_cfg.wandb_project:
            import wandb

            run_name = trainer_cfg.wandb_run_name or Path(trainer_cfg.checkpoint_path).parent.name
            self._wandb = wandb.init(
                project=trainer_cfg.wandb_project,
                name=run_name,
                mode=trainer_cfg.wandb_mode,
                config=run_config,
            )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self) -> list[dict[str, Any]]:
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)

        indexed_train = _IndexedDataset(self.train_dataset)
        indexed_val = _IndexedDataset(self.val_dataset)

        if self.is_ddp:
            train_sampler: Any = DistributedSampler(
                indexed_train,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.cfg.seed,
                drop_last=True,
            )
            val_sampler: Any = _DistributedEvalSampler(
                indexed_val,
                num_replicas=self.world_size,
                rank=self.rank,
            )
        else:
            train_sampler = None
            val_sampler = None

        persistent = self.cfg.num_workers > 0
        # Eval loaders: fewer workers (val/test are brief and infrequent vs train) and
        # NON-persistent, so their workers don't hold slots/RAM through every training
        # epoch. This caps the host-wide active-worker count to the train loaders.
        eval_workers = (
            self.cfg.eval_num_workers
            if self.cfg.eval_num_workers is not None
            else self.cfg.num_workers
        )
        train_loader = DataLoader(
            indexed_train,
            batch_size=self.cfg.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            generator=generator if train_sampler is None else None,
            drop_last=self.is_ddp,
            persistent_workers=persistent,
            timeout=self.cfg.dataloader_timeout_s if self.cfg.num_workers > 0 else 0,
        )
        val_loader = DataLoader(
            indexed_val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=eval_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            persistent_workers=False,
            timeout=self.cfg.dataloader_timeout_s if eval_workers > 0 else 0,
        )

        def eval_loader(dataset: Dataset) -> DataLoader:
            """Eval-only loader, mirroring the val construction above."""
            indexed = _IndexedDataset(dataset)
            return DataLoader(
                indexed,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                sampler=(
                    _DistributedEvalSampler(
                        indexed, num_replicas=self.world_size, rank=self.rank)
                    if self.is_ddp else None
                ),
                num_workers=eval_workers,
                pin_memory=self.device.type == "cuda",
                drop_last=False,
                persistent_workers=False,
                timeout=self.cfg.dataloader_timeout_s if eval_workers > 0 else 0,
            )

        # Optional OOD test-split loader (eval-only; mirrors val construction)
        test_loader = None
        if self.test_dataset is not None and self.cfg.test_eval_every_n_epochs > 0:
            test_loader = eval_loader(self.test_dataset)
        # Per-domain val loaders (multi-corpus runs; empty otherwise). Sorted so every rank
        # walks them in the same order — each one runs DDP collectives inside run_validation.
        domain_loaders = {
            name: eval_loader(ds) for name, ds in sorted(self.extra_val_datasets.items())
        }

        train_batches = len(train_loader)
        if self.cfg.max_train_batches is not None:
            train_batches = min(train_batches, self.cfg.max_train_batches)
        optim_steps_per_epoch = math.ceil(
            train_batches / max(1, self.cfg.gradient_accumulation_steps)
        )
        self.warmup_steps = max(0, int(self.cfg.lr_warmup_epochs) * optim_steps_per_epoch)
        self.total_steps = max(self.warmup_steps + 1, self.cfg.epochs * optim_steps_per_epoch)
        self._set_warmup_lr()

        last_stage: CurriculumStage | None = None
        for epoch in range(self.start_epoch, self.cfg.epochs):
            if self.is_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            stage = stage_for_epoch(epoch, self.curriculum_cfg)
            p_teacher = teacher_probability(epoch, self.sampling_cfg)
            if stage is not last_stage:
                if self.is_rank0:
                    print(f"[epoch {epoch}] curriculum -> {stage.name}")
                # Re-warm lr after a stage transition (not at the very start — the
                # initial warmup covers that): new losses just activated on params
                # with stale Adam state. See TrainerConfig.stage_rewarmup_steps.
                if last_stage is not None and self.cfg.stage_rewarmup_steps > 0:
                    self._rewarmup_start = self.global_step
                    self._rewarmup_end = self.global_step + self.cfg.stage_rewarmup_steps
                    if self.is_rank0:
                        print(
                            f"[epoch {epoch}] stage transition -> lr re-warmup over "
                            f"{self.cfg.stage_rewarmup_steps} steps"
                        )
                last_stage = stage
            if self.is_rank0:
                print(f"[epoch {epoch}] p_teacher={p_teacher:.4f}")

            t0 = time.time()
            skips_before = getattr(self, "_nonfinite_skips", 0)
            train_summary = self.run_epoch(train_loader, epoch, stage, p_teacher, training=True)
            val_summary, per_seq = self.run_validation(val_loader, epoch, stage)
            # Per-domain val slices (all ranks must enter for DDP collectives). Same metrics,
            # same code path — only the sequence set differs; per_seq is dropped because the
            # pooled per-sequence file already carries these rows under "<domain>/<name>".
            domain_summaries = {
                name: self.run_validation(
                    loader, epoch, stage, dataset=self.extra_val_datasets[name],
                    tag=f"val:{name}",
                )[0]
                for name, loader in domain_loaders.items()
            }
            # Optional OOD test-split eval (all ranks must enter for DDP collectives)
            test_summary = None
            if test_loader is not None and (
                (epoch + 1) % self.cfg.test_eval_every_n_epochs == 0
                or epoch + 1 == self.cfg.epochs
            ):
                test_summary, _ = self.run_validation(
                    test_loader, epoch, stage, dataset=self.test_dataset, tag="test"
                )
            elapsed = time.time() - t0

            if self.is_rank0:
                row = {
                    "epoch": epoch,
                    "stage": stage.name,
                    "p_teacher": p_teacher,
                    "lr": self._current_lr(),
                    "elapsed_s": elapsed,
                    "train": train_summary,
                    "val": val_summary,
                }
                # Divergence early-warning pair (gated bottleneck): how many optimizer steps
                # were dropped for non-finite grads, and how much of the gated bottleneck
                # path the model actually uses (mean |tanh(g)|, 0 at init).
                skips_total = getattr(self, "_nonfinite_skips", 0)
                row["nonfinite_skips_epoch"] = skips_total - skips_before
                row["nonfinite_skips_total"] = skips_total
                if len(self.optimizer.param_groups) > 1:
                    row["lr_fresh"] = float(self.optimizer.param_groups[-1]["lr"])
                gate = self._bottleneck_gate_abs_mean()
                if gate is not None:
                    row["gate_abs_mean"] = gate
                if domain_summaries:
                    row["val_by_domain"] = domain_summaries
                if test_summary is not None:
                    row["test"] = test_summary
                self.history.append(row)
                self._append_metrics(row)
                self._write_per_sequence(epoch, per_seq)
                self.save_checkpoint(self.cfg.checkpoint_path, epoch + 1)
                if self.cfg.save_epoch_checkpoints:
                    self._save_epoch_checkpoint(epoch)

                val_r2 = val_summary.get("r2", {})
                mean_r2 = (
                    sum(val_r2[name] for name in self.best_val_channels)
                    / len(self.best_val_channels)
                    if all(name in val_r2 for name in self.best_val_channels)
                    else float("-inf")
                )
                if mean_r2 > self.best_val_r2:
                    self.best_val_r2 = mean_r2
                    self.save_checkpoint(self._best_checkpoint_path(), epoch + 1)
                    print(f"[epoch {epoch}] new best mean val R2={mean_r2:.4f} -> {self._best_checkpoint_path()}")

        if self._wandb is not None:
            self._wandb.finish()
        if self.is_ddp:
            if self.cpu_group is not None:
                dist.destroy_process_group(self.cpu_group)
            dist.destroy_process_group()
        return self.history

    # ------------------------------------------------------------------
    # train epoch
    # ------------------------------------------------------------------

    def run_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        stage: CurriculumStage,
        p_teacher: float,
        training: bool,
        force_teacher_inputs: bool | None = None,
    ) -> dict[str, Any]:
        self.model.train(training)
        total_loss = 0.0
        n_batches = 0
        losses_acc: dict[str, float] = {}
        diagnostic_losses_acc: dict[str, float] = {}
        preds_all: list[Tensor] = []
        targets_all: list[Tensor] = []
        masks_all: list[Tensor] = []
        binary_preds_all: list[Tensor] = []
        binary_targets_all: list[Tensor] = []
        binary_masks_all: list[Tensor] = []
        use_teacher_inputs = training if force_teacher_inputs is None else force_teacher_inputs
        accumulation_steps = max(1, self.cfg.gradient_accumulation_steps)
        total_batches = len(loader)
        if training and self.cfg.max_train_batches is not None:
            total_batches = min(total_batches, self.cfg.max_train_batches)
        if not training and self.cfg.max_val_batches is not None:
            total_batches = min(total_batches, self.cfg.max_val_batches)
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        with grad_ctx:
            tag = "train" if training else "val"
            iterator = tqdm(
                enumerate(loader),
                total=total_batches,
                desc=f"{tag} epoch {epoch} {stage.name}",
                dynamic_ncols=True,
                leave=False,
                disable=not (self.cfg.progress and self.is_rank0),
            )
            for batch_idx, batch in iterator:
                if training and self.cfg.max_train_batches is not None and batch_idx >= self.cfg.max_train_batches:
                    break
                if not training and self.cfg.max_val_batches is not None and batch_idx >= self.cfg.max_val_batches:
                    break
                batch = self._to_device(batch)
                teacher_mask = (
                    sample_teacher_force_mask(
                        batch["action_sequence"].shape[0],
                        batch["action_sequence"].shape[1],
                        p_teacher,
                        self.device,
                        granularity=self.sampling_cfg.mask_granularity,
                    )
                    if training
                    else None
                )
                should_step = training and (
                    (batch_idx + 1) % accumulation_steps == 0
                    or (batch_idx + 1) >= total_batches
                )
                amp_ctx = (
                    torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
                    if self.use_amp
                    else nullcontext()
                )
                # Defer DDP grad all-reduce to the last micro-batch of each accumulation
                # group. no_sync() must wrap the FORWARD (DDP reads the flag there to arm
                # the reducer), not just backward — without it every micro-step pays a
                # full gradient sync, defeating accumulation.
                sync_ctx = (
                    self.model.no_sync()
                    if (self.is_ddp and training and not should_step)
                    else nullcontext()
                )
                with sync_ctx:
                    with amp_ctx:
                        # NOTE: must call self.model(...) (DDP.forward dispatch), NOT
                        # model.module.forward_rollout(...) — bypassing DDP.forward
                        # silently disables gradient synchronization across ranks.
                        preds = self.model(
                            batch,
                            batch["action_sequence"],
                            use_ground_truth_visual=batch["z_vis_target"] if use_teacher_inputs else None,
                            use_ground_truth_proprio=batch["f_prop_target"] if use_teacher_inputs else None,
                            teacher_force_mask=teacher_mask,
                            proprio_window_target=(
                                batch.get("proprio_window_target")
                                if self._needs_prop_window_target
                                else None
                            ),
                        )
                        loss, loss_dict = compute_total_loss(
                            preds=preds,
                            targets=batch,
                            cfg=self.loss_cfg,
                            stage=stage,
                            pos_weight=self.pos_weight,
                        )
                        # At FULL the diagnostic pass is identical to the training loss —
                        # reuse it instead of re-reducing the [B,H,P,D] visual terms.
                        if stage is CurriculumStage.FULL:
                            diagnostic_loss_dict = loss_dict
                        else:
                            with torch.no_grad():
                                _, diagnostic_loss_dict = compute_total_loss(
                                    preds=preds,
                                    targets=batch,
                                    cfg=self.loss_cfg,
                                    stage=CurriculumStage.FULL,
                                    pos_weight=self.pos_weight,
                                )

                    if training:
                        loss_for_backward = loss / accumulation_steps
                        self.scaler.scale(loss_for_backward).backward()

                if training:
                    if should_step:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                        self._set_warmup_lr()
                        # Skip the update on a non-finite grad so one transient spike
                        # (e.g. an AMP overflow) doesn't write NaN into every weight and
                        # poison the whole run. scaler.update() still runs to adapt scale.
                        if torch.isfinite(grad_norm):
                            self.scaler.step(self.optimizer)
                            self._consecutive_skips = 0
                        else:
                            self._nonfinite_skips = getattr(self, "_nonfinite_skips", 0) + 1
                            self._consecutive_skips = getattr(self, "_consecutive_skips", 0) + 1
                            if self.is_rank0 and (
                                self._nonfinite_skips <= 10 or self._nonfinite_skips % 100 == 0
                            ):
                                print(
                                    f"[warn] non-finite grad (norm={grad_norm}); skipping "
                                    f"optimizer step {self.global_step} "
                                    f"(total skips={self._nonfinite_skips})"
                                )
                            # The guard exists to absorb TRANSIENT spikes. If every step
                            # is non-finite the model has entered a divergent regime and
                            # training is frozen — fail loudly instead of burning GPU-days
                            # (observed: an AdaLN run sat frozen for 6 epochs at loss 3.6).
                            if self._consecutive_skips >= 200:
                                raise RuntimeError(
                                    f"{self._consecutive_skips} consecutive non-finite-grad "
                                    f"skips at step {self.global_step}: training has diverged "
                                    "(weights frozen). Lower lr / enable stage_rewarmup_steps "
                                    "and restart from the last good checkpoint."
                                )
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1

                        if self._wandb is not None and self.global_step % max(1, self.cfg.log_every) == 0:
                            step_log = {
                                "epoch": epoch,
                                "step/loss": float(loss.detach()),
                                "step/grad_norm": float(grad_norm),
                                "step/lr": self._current_lr(),
                                "step/nonfinite_skips": getattr(self, "_nonfinite_skips", 0),
                            }
                            step_log.update(
                                {f"step/{k}": float(v) for k, v in loss_dict.items()}
                            )
                            self._wandb.log(step_log, step=self.global_step)

                        # Mid-epoch crash insurance. rank0-only and collective-free, so
                        # the other ranks are not blocked and the wall-clock condition
                        # may differ between them without any divergence risk.
                        if (
                            self.is_rank0
                            and self.cfg.periodic_checkpoint_minutes > 0
                            and (time.time() - self._last_periodic_save)
                            >= self.cfg.periodic_checkpoint_minutes * 60.0
                        ):
                            self._save_periodic_checkpoint(epoch, batch_idx)
                            self._last_periodic_save = time.time()

                total_loss += float(loss.detach().cpu())
                n_batches += 1
                for key, val in loss_dict.items():
                    losses_acc[key] = losses_acc.get(key, 0.0) + float(val.detach().cpu())
                for key, val in diagnostic_loss_dict.items():
                    diagnostic_losses_acc[key] = diagnostic_losses_acc.get(key, 0.0) + float(val.detach().cpu())
                if self.cfg.progress and self.is_rank0 and (
                    batch_idx == 0
                    or (batch_idx + 1) % max(1, self.cfg.log_every) == 0
                    or (batch_idx + 1) >= total_batches
                ):
                    iterator.set_postfix(
                        loss=f"{total_loss / max(n_batches, 1):.4f}",
                        lr=f"{self._current_lr():.2e}",
                    )
                if preds.get("f_prop_preds") is not None:
                    preds_all.append(preds["f_prop_preds"].detach().cpu())
                    targets_all.append(batch["f_prop_target"].detach().cpu())
                    masks_all.append(batch["valid_target_mask"].detach().cpu())
                if preds.get("c_preds") is not None:
                    binary_preds_all.append(preds["c_preds"].detach().cpu())
                    binary_targets_all.append(batch["binary_label_target"].detach().cpu())
                    binary_masks_all.append(self._binary_metric_mask(batch).detach().cpu())

        if self.is_ddp:
            total_loss, n_batches, losses_acc = self._dist_reduce_scalars(
                total_loss, n_batches, losses_acc
            )
            _, _, diagnostic_losses_acc = self._dist_reduce_scalars(
                0.0, n_batches, diagnostic_losses_acc
            )
            summary = self._summarize_epoch_distributed(
                total_loss, n_batches, losses_acc,
                diagnostic_losses_acc,
                preds_all, targets_all, masks_all,
                binary_preds_all, binary_targets_all, binary_masks_all,
            )
        else:
            summary = self._summarize_epoch(
                total_loss, n_batches, losses_acc,
                diagnostic_losses_acc,
                preds_all, targets_all, masks_all,
                binary_preds_all, binary_targets_all, binary_masks_all,
            )

        tag = "train" if training else "val"
        self._print_summary(tag, epoch, stage, summary)
        return summary

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def run_validation(
        self,
        loader: DataLoader,
        epoch: int,
        stage: CurriculumStage,
        dataset: Dataset | None = None,
        tag: str = "val",
        oracle_proprio: bool = False,
        encode_oracle_proprio: bool = False,
    ) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
        # oracle_proprio: teacher-force the GT proprio (f_prop_target) into the rollout
        # conditioning -> upper bound on what any proprio-feedback scheme could reach
        # (diagnostic for choosing between freeze and aligned feedback).
        # encode_oracle_proprio: feed encode(true future raw window) directly as the
        # conditioning -> clean ceiling for the aligned scheme (embedding space, not
        # the projected-4-dim ceiling that oracle_proprio measures).
        # dataset is used only to map sample indices -> sequences for per_seq;
        # defaults to val_dataset (so the test loader maps against the test dataset).
        per_seq_dataset = dataset if dataset is not None else self.val_dataset
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        losses_acc: dict[str, float] = {}
        diagnostic_losses_acc: dict[str, float] = {}
        preds_all: list[Tensor] = []
        targets_all: list[Tensor] = []
        masks_all: list[Tensor] = []
        binary_preds_all: list[Tensor] = []
        binary_targets_all: list[Tensor] = []
        binary_masks_all: list[Tensor] = []
        seq_batches: list[dict[str, Any]] = []
        total_batches = len(loader)
        if self.cfg.max_val_batches is not None:
            total_batches = min(total_batches, self.cfg.max_val_batches)

        with torch.no_grad():
            iterator = tqdm(
                enumerate(loader),
                total=total_batches,
                desc=f"val epoch {epoch} {stage.name}",
                dynamic_ncols=True,
                leave=False,
                disable=not (self.cfg.progress and self.is_rank0),
            )
            for batch_idx, batch in iterator:
                if self.cfg.max_val_batches is not None and batch_idx >= self.cfg.max_val_batches:
                    break
                batch = self._to_device(batch)
                preds = _model_for_custom_methods(self.model).forward_rollout(
                    batch,
                    batch["action_sequence"],
                    use_ground_truth_proprio=batch["f_prop_target"] if oracle_proprio else None,
                    proprio_window_target=(
                        batch.get("proprio_window_target")
                        if (self._needs_prop_window_target or encode_oracle_proprio)
                        else None
                    ),
                    oracle_proprio_embedding=encode_oracle_proprio,
                )
                loss, loss_dict = compute_total_loss(
                    preds=preds,
                    targets=batch,
                    cfg=self.loss_cfg,
                    stage=stage,
                    pos_weight=self.pos_weight,
                )
                if stage is CurriculumStage.FULL:
                    diagnostic_loss_dict = loss_dict
                else:
                    _, diagnostic_loss_dict = compute_total_loss(
                        preds=preds,
                        targets=batch,
                        cfg=self.loss_cfg,
                        stage=CurriculumStage.FULL,
                        pos_weight=self.pos_weight,
                    )
                total_loss += float(loss.cpu())
                n_batches += 1
                for key, val in loss_dict.items():
                    losses_acc[key] = losses_acc.get(key, 0.0) + float(val.cpu())
                for key, val in diagnostic_loss_dict.items():
                    diagnostic_losses_acc[key] = diagnostic_losses_acc.get(key, 0.0) + float(val.cpu())
                if self.cfg.progress and self.is_rank0 and (
                    batch_idx == 0
                    or (batch_idx + 1) % max(1, self.cfg.log_every) == 0
                    or (batch_idx + 1) >= total_batches
                ):
                    iterator.set_postfix(loss=f"{total_loss / max(n_batches, 1):.4f}")

                if preds.get("f_prop_preds") is not None:
                    pred_cpu = preds["f_prop_preds"].detach().cpu()
                    target_cpu = batch["f_prop_target"].detach().cpu()
                    mask_cpu = batch["valid_target_mask"].detach().cpu()
                    preds_all.append(pred_cpu)
                    targets_all.append(target_cpu)
                    masks_all.append(mask_cpu)
                    seq_batches.append({
                        "sample_idxs": batch["_sample_idx"].detach().cpu().tolist(),
                        "pred": pred_cpu,
                        "target": target_cpu,
                        "current": batch.get("current_feature", torch.empty(0)).detach().cpu(),
                        "mask": mask_cpu,
                    })
                if preds.get("c_preds") is not None:
                    binary_preds_all.append(preds["c_preds"].detach().cpu())
                    binary_targets_all.append(batch["binary_label_target"].detach().cpu())
                    binary_masks_all.append(self._binary_metric_mask(batch).detach().cpu())

        if self.is_ddp:
            total_loss, n_batches, losses_acc = self._dist_reduce_scalars(
                total_loss, n_batches, losses_acc
            )
            _, _, diagnostic_losses_acc = self._dist_reduce_scalars(
                0.0, n_batches, diagnostic_losses_acc
            )
            gathered_seq: list[Any] = [None] * self.world_size
            dist.all_gather_object(gathered_seq, seq_batches, group=self.cpu_group)
            summary = self._summarize_epoch_distributed(
                total_loss, n_batches, losses_acc,
                diagnostic_losses_acc,
                preds_all, targets_all, masks_all,
                binary_preds_all, binary_targets_all, binary_masks_all,
            )
            if self.is_rank0:
                raw_per_seq: dict[str, dict[str, list[np.ndarray]]] = {}
                for rank_batches in gathered_seq:
                    for sb in rank_batches:
                        self._accumulate_per_sequence(
                            sample_idxs=sb["sample_idxs"],
                            pred=sb["pred"],
                            target=sb["target"],
                            current=sb["current"],
                            mask=sb["mask"],
                            out=raw_per_seq,
                            dataset=per_seq_dataset,
                        )
                per_seq = self._finalize_per_sequence(raw_per_seq)
            else:
                per_seq = {}
        else:
            raw_per_seq = {}
            for sb in seq_batches:
                self._accumulate_per_sequence(
                    sample_idxs=sb["sample_idxs"],
                    pred=sb["pred"],
                    target=sb["target"],
                    current=sb["current"],
                    mask=sb["mask"],
                    out=raw_per_seq,
                    dataset=per_seq_dataset,
                )
            per_seq = self._finalize_per_sequence(raw_per_seq)
            summary = self._summarize_epoch(
                total_loss, n_batches, losses_acc,
                diagnostic_losses_acc,
                preds_all, targets_all, masks_all,
                binary_preds_all, binary_targets_all, binary_masks_all,
            )
        self._print_summary(tag, epoch, stage, summary)
        return summary, per_seq

    # ------------------------------------------------------------------
    # DDP helpers
    # ------------------------------------------------------------------

    def _dist_reduce_scalars(
        self,
        total_loss: float,
        n_batches: int,
        losses_acc: dict[str, float],
    ) -> tuple[float, int, dict[str, float]]:
        keys = sorted(losses_acc.keys())
        values = [total_loss, float(n_batches)] + [losses_acc[k] for k in keys]
        t = torch.tensor(values, device=self.device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t_cpu = t.cpu().tolist()
        reduced_acc = {k: t_cpu[2 + i] for i, k in enumerate(keys)}
        return t_cpu[0], int(round(t_cpu[1])), reduced_acc

    # ------------------------------------------------------------------
    # loss probe
    # ------------------------------------------------------------------

    def loss_probe(self, max_batches: int = 2) -> dict[str, Any]:
        loader = DataLoader(
            _IndexedDataset(self.train_dataset),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        old_train_limit = self.cfg.max_train_batches
        old_val_limit = self.cfg.max_val_batches
        self.cfg.max_train_batches = max_batches
        self.cfg.max_val_batches = max_batches
        try:
            summary = self.run_epoch(
                loader=loader,
                epoch=0,
                stage=CurriculumStage.FULL,
                p_teacher=1.0,
                training=False,
                force_teacher_inputs=True,
            )
        finally:
            self.cfg.max_train_batches = old_train_limit
            self.cfg.max_val_batches = old_val_limit

        components = summary["loss_components"]
        raw = {
            "visual": float(components.get("loss_visual", 0.0)),
            "proprio": float(components.get("loss_proprio", 0.0)),
            "traversability": float(components.get("loss_traversability", 0.0)),
        }
        candidates = []
        for lambda_prop in (0.5, 1.0, 2.0):
            for lambda_trav in (0.25, 0.5, 1.0):
                weighted = {
                    "visual": raw["visual"],
                    "proprio": lambda_prop * raw["proprio"],
                    "traversability": lambda_trav * raw["traversability"],
                }
                positive = [v for v in weighted.values() if v > 0]
                ratio = max(positive) / max(min(positive), 1e-12) if positive else float("inf")
                candidates.append({
                    "lambda_prop": lambda_prop,
                    "lambda_trav": lambda_trav,
                    "weighted": weighted,
                    "max_min_ratio": ratio,
                })
        candidates.sort(key=lambda row: row["max_min_ratio"])
        return {
            "max_batches": max_batches,
            "raw_components": raw,
            "default_lambdas": {
                "lambda_prop": self.loss_cfg.lambda_prop,
                "lambda_trav": self.loss_cfg.lambda_trav,
                "lambda_cos": self.loss_cfg.lambda_cos,
            },
            "candidates": candidates,
            "recommended": candidates[0] if candidates else None,
        }

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------

    def _best_checkpoint_path(self) -> str:
        p = Path(self.cfg.checkpoint_path)
        return str(p.with_name(f"{p.stem}_best{p.suffix}"))

    def _partial_checkpoint_path(self) -> str:
        p = Path(self.cfg.checkpoint_path)
        return str(p.with_name(f"{p.stem}_partial{p.suffix}"))

    def _save_periodic_checkpoint(self, epoch: int, batch_idx: int) -> None:
        """Mid-epoch crash insurance (periodic_checkpoint_minutes), rank0-only.

        next_epoch=epoch (NOT epoch+1): the snapshot is taken mid-epoch, so a resume
        replays the whole current epoch with the weights reached so far. Nothing is
        lost, some batches are re-seen. checkpoint_path stays the epoch-boundary,
        post-validation artifact — this never writes to it."""
        path = self._partial_checkpoint_path()
        self.save_checkpoint(
            path, epoch,
            extra={"partial_epoch": epoch, "partial_batch_idx": batch_idx},
        )
        print(
            f"[epoch {epoch}] periodic checkpoint @ batch {batch_idx} "
            f"(step {self.global_step}) -> {path}"
        )

    def _save_epoch_checkpoint(self, epoch: int) -> None:
        """Slim model-only snapshot per epoch for offline per-epoch reconstruction.
        Optimizer/scaler/history omitted to save space — these are NOT resume checkpoints."""
        if not self.is_rank0:
            return
        d = Path(self.cfg.checkpoint_path).parent / "epoch_ckpts"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": _unwrap_model(self.model).state_dict(), "next_epoch": epoch + 1},
            d / f"epoch_{epoch}.pt",
        )

    def save_checkpoint(
        self, path: str, next_epoch: int, extra: dict[str, Any] | None = None
    ) -> None:
        if not self.is_rank0:
            return
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": _unwrap_model(self.model).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "global_step": self.global_step,
            "next_epoch": next_epoch,
            "best_val_r2": self.best_val_r2,
            "history": self.history,
            "trainer_config": asdict(self.cfg),
            "loss_config": asdict(self.loss_cfg),
            "curriculum_config": asdict(self.curriculum_cfg),
            "scheduled_sampling_config": asdict(self.sampling_cfg),
        }
        if extra:
            payload.update(extra)
        # Atomic: a kill DURING the write must not leave a truncated checkpoint behind —
        # auto_resume would then fail to load it and the run's only recovery point is
        # gone. Matters most for the periodic mid-epoch saves (frequent, unattended).
        tmp_path = ckpt_path.with_name(ckpt_path.name + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        _unwrap_model(self.model).load_state_dict(ckpt["model_state"])
        # Slim per-epoch snapshots (save_epoch_checkpoints) carry only model_state +
        # next_epoch — enough for eval/recon. Skip optimizer/scaler restore when absent
        # so such a checkpoint can be loaded for evaluation; a real resume checkpoint
        # always has these keys.
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        # Truthiness check: a disabled scaler (bf16 path) serializes {}, and an enabled
        # scaler raises on an empty state dict; loading enabled-state into a disabled
        # scaler is already a safe no-op.
        if ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step = int(ckpt.get("global_step", 0))
        self.start_epoch = int(ckpt.get("next_epoch", 0))
        self.best_val_r2 = float(ckpt.get("best_val_r2", float("-inf")))
        self.history = list(ckpt.get("history", []))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: val.to(self.device, non_blocking=True) if isinstance(val, Tensor) else val
            for key, val in batch.items()
        }

    @staticmethod
    def _binary_metric_mask(batch: dict[str, Tensor]) -> Tensor:
        mask = batch["valid_target_mask"].bool()
        if "binary_valid_target_mask" in batch:
            mask = mask & batch["binary_valid_target_mask"].bool()
        return mask

    def _summarize_epoch(
        self,
        total_loss: float,
        n_batches: int,
        losses_acc: dict[str, float],
        diagnostic_losses_acc: dict[str, float],
        preds: list[Tensor],
        targets: list[Tensor],
        masks: list[Tensor],
        binary_preds: list[Tensor] | None = None,
        binary_targets: list[Tensor] | None = None,
        binary_masks: list[Tensor] | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "loss": total_loss / max(n_batches, 1),
            "loss_components": {
                key: val / max(n_batches, 1)
                for key, val in sorted(losses_acc.items())
            },
        }
        self._attach_diagnostic_losses(summary, diagnostic_losses_acc, n_batches)
        if preds:
            pred, target, mask = torch.cat(preds), torch.cat(targets), torch.cat(masks)
            r2 = self._r2(pred, target, mask)
            summary["r2"] = {name: float(r2[i]) for i, name in enumerate(self.channel_names)}
            mae = self._mae(pred, target, mask)
            summary["mae"] = {name: float(mae[i]) for i, name in enumerate(self.channel_names)}
        if binary_preds:
            assert binary_targets is not None and binary_masks is not None
            summary["binary_safety"] = self._binary_metrics(
                torch.cat(binary_preds),
                torch.cat(binary_targets),
                torch.cat(binary_masks),
            )
        return summary

    def _summarize_epoch_distributed(
        self,
        total_loss: float,
        n_batches: int,
        losses_acc: dict[str, float],
        diagnostic_losses_acc: dict[str, float],
        preds: list[Tensor],
        targets: list[Tensor],
        masks: list[Tensor],
        binary_preds: list[Tensor] | None = None,
        binary_targets: list[Tensor] | None = None,
        binary_masks: list[Tensor] | None = None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "loss": total_loss / max(n_batches, 1),
            "loss_components": {
                key: val / max(n_batches, 1)
                for key, val in sorted(losses_acc.items())
            },
        }
        self._attach_diagnostic_losses(summary, diagnostic_losses_acc, n_batches)
        if preds:
            stats = self._r2_sufficient_stats(preds, targets, masks)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            r2 = self._r2_from_sufficient_stats(stats)
            summary["r2"] = {name: float(r2[i]) for i, name in enumerate(self.channel_names)}
            mae = self._mae_from_sufficient_stats(stats)
            summary["mae"] = {name: float(mae[i]) for i, name in enumerate(self.channel_names)}
        if binary_preds:
            assert binary_targets is not None and binary_masks is not None
            stats = self._binary_sufficient_stats(binary_preds, binary_targets, binary_masks)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            summary["binary_safety"] = self._binary_metrics_from_sufficient_stats(stats)
        return summary

    def _attach_diagnostic_losses(
        self,
        summary: dict[str, Any],
        diagnostic_losses_acc: dict[str, float],
        n_batches: int,
    ) -> None:
        if not diagnostic_losses_acc:
            return
        raw = {
            key: val / max(n_batches, 1)
            for key, val in sorted(diagnostic_losses_acc.items())
        }
        summary["diagnostic_full_loss_components"] = raw
        weighted: dict[str, float] = {}
        if "loss_visual" in raw:
            weighted["visual"] = raw["loss_visual"]
        if "loss_proprio" in raw:
            weighted["proprio"] = self.loss_cfg.lambda_prop * raw["loss_proprio"]
        if "loss_traversability" in raw:
            weighted["traversability"] = self.loss_cfg.lambda_trav * raw["loss_traversability"]
        if weighted:
            weighted["total"] = sum(weighted.values())
            summary["diagnostic_full_weighted_components"] = weighted

    def _r2_sufficient_stats(
        self,
        preds: list[Tensor],
        targets: list[Tensor],
        masks: list[Tensor],
    ) -> Tensor:
        n_chan = len(self.channel_names)
        pred = torch.cat(preds).reshape(-1, n_chan).to(torch.float64)
        target = torch.cat(targets).reshape(-1, n_chan).to(torch.float64)
        pred = self._denormalize_feature_tensor(pred)
        target = self._denormalize_feature_tensor(target)
        mask = torch.cat(masks).reshape(-1).to(torch.bool)
        valid = mask[:, None].to(torch.float64)
        ss_res = (((target - pred) ** 2) * valid).sum(dim=0)
        sum_y = (target * valid).sum(dim=0)
        sum_y2 = ((target ** 2) * valid).sum(dim=0)
        count = valid.sum().expand(n_chan)
        sum_abs = ((target - pred).abs() * valid).sum(dim=0)
        stats = torch.cat([ss_res, sum_y, sum_y2, count, sum_abs])
        return stats.to(device=self.device)

    def _r2_from_sufficient_stats(self, stats: Tensor) -> np.ndarray:
        c = len(self.channel_names)
        stats_cpu = stats.detach().cpu()
        ss_res = stats_cpu[:c]
        sum_y = stats_cpu[c:2 * c]
        sum_y2 = stats_cpu[2 * c:3 * c]
        count = torch.clamp(stats_cpu[3 * c:4 * c], min=1.0)
        ss_tot = sum_y2 - (sum_y ** 2) / count
        r2 = 1.0 - ss_res / torch.clamp(ss_tot, min=1e-12)
        return r2.numpy().astype(np.float64)

    def _mae_from_sufficient_stats(self, stats: Tensor) -> np.ndarray:
        """Per-channel MAE in denormalized (physical) units, from `_r2_sufficient_stats`."""
        c = len(self.channel_names)
        stats_cpu = stats.detach().cpu()
        count = torch.clamp(stats_cpu[3 * c:4 * c], min=1.0)
        sum_abs = stats_cpu[4 * c:5 * c]
        return (sum_abs / count).numpy().astype(np.float64)

    def _denormalize_feature_tensor(self, value: Tensor) -> Tensor:
        idx = self.channel_indices
        mean = torch.as_tensor(
            self.stats.feature_mean[idx], dtype=value.dtype, device=value.device
        )
        std = torch.as_tensor(
            self.stats.feature_std[idx], dtype=value.dtype, device=value.device
        )
        out = (value * std + mean).clone()
        if 3 in idx:  # ke_rate present: invert signed_log1p at its column
            pos = idx.index(3)
            ke_rate = out[..., pos]
            out[..., pos] = torch.sign(ke_rate) * torch.expm1(torch.abs(ke_rate))
        return out

    def _binary_sufficient_stats(
        self,
        preds: list[Tensor],
        targets: list[Tensor],
        masks: list[Tensor],
        bins: int = 1000,
    ) -> Tensor:
        pred = torch.cat(preds).reshape(-1).to(torch.float64)
        target = torch.cat(targets).reshape(-1).to(torch.int64)
        mask = torch.cat(masks).reshape(-1).to(torch.bool)
        # Drop non-finite preds: under fp16 autocast the autoregressive rollout can
        # overflow and emit NaN/inf c_preds. NaN survives clamp(0,1), and casting it to
        # long yields INT64_MIN, which bincount rejects. Exclude them so a metrics pass
        # never kills training; the proprio R2 path stays NaN-visible as the divergence
        # canary.
        mask = mask & torch.isfinite(pred)
        pred = pred[mask].clamp(0.0, 1.0)
        target = target[mask]
        pred_label = pred >= 0.5
        positive = target == 1
        negative = target == 0
        tp = (pred_label & positive).sum(dtype=torch.float64)
        fp = (pred_label & negative).sum(dtype=torch.float64)
        fn = ((~pred_label) & positive).sum(dtype=torch.float64)
        n_pos = positive.sum(dtype=torch.float64)
        n_neg = negative.sum(dtype=torch.float64)
        if pred.numel() == 0:
            hist_pos = torch.zeros(bins, dtype=torch.float64)
            hist_neg = torch.zeros(bins, dtype=torch.float64)
        else:
            idx = torch.clamp((pred * bins).to(torch.long), min=0, max=bins - 1)
            hist_pos = torch.bincount(idx[positive], minlength=bins).to(torch.float64)
            hist_neg = torch.bincount(idx[negative], minlength=bins).to(torch.float64)
        stats = torch.cat([
            torch.stack([tp, fp, fn, n_pos, n_neg]),
            hist_pos,
            hist_neg,
        ])
        return stats.to(device=self.device)

    @staticmethod
    def _binary_metrics_from_sufficient_stats(stats: Tensor, bins: int = 1000) -> dict[str, float]:
        stats_cpu = stats.detach().cpu()
        tp, fp, fn, n_pos, n_neg = [float(x) for x in stats_cpu[:5]]
        hist_pos = stats_cpu[5:5 + bins].numpy()
        hist_neg = stats_cpu[5 + bins:5 + 2 * bins].numpy()
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        auc = float("nan")
        if n_pos > 0 and n_neg > 0:
            neg_before = np.cumsum(np.concatenate([[0.0], hist_neg[:-1]]))
            pair_score = np.sum(hist_pos * neg_before + 0.5 * hist_pos * hist_neg)
            auc = float(pair_score / max(n_pos * n_neg, 1e-12))
        n = int(n_pos + n_neg)
        return {
            "auc": auc,
            "f1": float(f1),
            "threshold": 0.5,
            "n": n,
            "pos_rate": float(n_pos / max(n, 1)),
        }

    def _r2(self, pred: Tensor, target: Tensor, mask: Tensor) -> np.ndarray:
        n_chan = len(self.channel_names)
        idx = self.channel_indices
        # .float(): autocast(bf16) preds reach here un-upcast on the single-GPU path
        # (the DDP path reduces in fp32); numpy has no bfloat16.
        pred_np = denormalize_feature(pred.reshape(-1, n_chan).float().numpy(), self.stats, idx)
        target_np = denormalize_feature(target.reshape(-1, n_chan).float().numpy(), self.stats, idx)
        mask_np = mask.reshape(-1).numpy().astype(bool)
        out = []
        for c in range(n_chan):
            y = target_np[mask_np, c]
            p = pred_np[mask_np, c]
            ss_res = np.sum((y - p) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            out.append(1.0 - ss_res / max(ss_tot, 1e-12))
        return np.asarray(out, dtype=np.float64)

    def _mae(self, pred: Tensor, target: Tensor, mask: Tensor) -> np.ndarray:
        """Per-channel MAE in denormalized (physical) units."""
        n_chan = len(self.channel_names)
        idx = self.channel_indices
        pred_np = denormalize_feature(pred.reshape(-1, n_chan).float().numpy(), self.stats, idx)
        target_np = denormalize_feature(target.reshape(-1, n_chan).float().numpy(), self.stats, idx)
        mask_np = mask.reshape(-1).numpy().astype(bool)
        if not mask_np.any():
            return np.full(n_chan, np.nan, dtype=np.float64)
        return np.abs(target_np[mask_np] - pred_np[mask_np]).mean(axis=0).astype(np.float64)

    def _binary_metrics(self, pred: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
        p = pred.reshape(-1).float().numpy()
        y = target.reshape(-1).numpy().astype(np.int64)
        m = mask.reshape(-1).numpy().astype(bool)
        p, y = p[m], y[m]
        if len(y) == 0:
            return {"auc": float("nan"), "f1": float("nan"), "threshold": 0.5, "n": 0, "pos_rate": float("nan")}
        pred_label = p >= 0.5
        tp = float(np.sum(pred_label & (y == 1)))
        fp = float(np.sum(pred_label & (y == 0)))
        fn = float(np.sum((~pred_label) & (y == 1)))
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        return {
            "auc": float(self._auc_np(p, y)),
            "f1": float(f1),
            "threshold": 0.5,
            "n": int(len(y)),
            "pos_rate": float(np.mean(y)),
        }

    @staticmethod
    def _auc_np(score: np.ndarray, target: np.ndarray) -> float:
        pos = target == 1
        neg = target == 0
        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(score)
        ranks = np.empty_like(order, dtype=np.float64)
        sorted_score = score[order]
        i = 0
        while i < len(score):
            j = i + 1
            while j < len(score) and sorted_score[j] == sorted_score[i]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
            i = j
        return (float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    def _set_warmup_lr(self) -> None:
        scale = 1.0
        if self.warmup_steps > 0:
            scale = min(1.0, float(self.global_step + 1) / float(self.warmup_steps))
        # post-warmup cosine decay (config-gated; "constant" keeps the original behaviour).
        if self.cfg.lr_schedule == "cosine" and self.global_step >= self.warmup_steps:
            total = getattr(self, "total_steps", self.warmup_steps + 1)
            progress = float(self.global_step - self.warmup_steps) / float(
                max(1, total - self.warmup_steps)
            )
            progress = min(1.0, max(0.0, progress))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale *= self.cfg.lr_min_factor + (1.0 - self.cfg.lr_min_factor) * cos
        # stage-transition re-warmup (see TrainerConfig.stage_rewarmup_steps)
        rw_start = getattr(self, "_rewarmup_start", None)
        rw_end = getattr(self, "_rewarmup_end", None)
        if rw_start is not None and self.global_step < rw_end:
            scale *= min(1.0, float(self.global_step - rw_start + 1) / float(rw_end - rw_start))
        if scale >= 1.0 and self.warmup_steps <= 0 and rw_start is None and self.cfg.lr_schedule != "cosine":
            return
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _bottleneck_gate_abs_mean(self) -> float | None:
        """mean |tanh(g)| of the token bottleneck's residual gate; None unless it exists."""
        dynamics = getattr(_unwrap_model(self.model), "dynamics", None)
        bottleneck = getattr(dynamics, "bottleneck", None)
        return bottleneck.gate_abs_mean() if bottleneck is not None else None

    def _print_summary(self, tag: str, epoch: int, stage: CurriculumStage, summary: dict[str, Any]) -> None:
        if not self.is_rank0:
            return
        r2 = summary.get("r2", {})
        r2_text = "  ".join(f"{k}:R2={v:.3f}" for k, v in r2.items())
        print(f"[{tag} epoch {epoch} {stage.name}] loss={summary['loss']:.4f}  {r2_text}")

    def _append_metrics(self, row: dict[str, Any]) -> None:
        if not self.is_rank0:
            return
        path = Path(self.cfg.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if self._wandb is not None:
            self._wandb.log(_flatten_numeric(row), step=self.global_step)

    def _accumulate_per_sequence(
        self,
        sample_idxs: list[int],
        pred: Tensor,
        target: Tensor,
        current: Tensor,
        mask: Tensor,
        out: dict[str, dict[str, list[np.ndarray]]],
        dataset: Dataset | None = None,
    ) -> None:
        ds = dataset if dataset is not None else self.val_dataset
        if not hasattr(ds, "_index") or not hasattr(ds, "_records"):
            return
        for bi, sample_idx in enumerate(sample_idxs):
            # _index entries are (seq_id, anchor_t[, stride]); only seq_id is needed here.
            sid = ds._index[sample_idx][0]
            seq = ds._records[sid].name
            valid = mask[bi].numpy().astype(bool)
            if not valid.any():
                continue
            bucket = out.setdefault(seq, {"pred": [], "target": [], "current": []})
            bucket["pred"].append(pred[bi, valid].numpy())
            bucket["target"].append(target[bi, valid].numpy())
            cur = current[bi].numpy()[None, :].repeat(int(valid.sum()), axis=0)
            bucket["current"].append(cur)

    def _finalize_per_sequence(
        self,
        per_seq: dict[str, dict[str, list[np.ndarray]]],
    ) -> dict[str, dict[str, np.ndarray]]:
        return {
            seq: {
                key: denormalize_feature(
                    np.concatenate(parts, axis=0), self.stats, self.channel_indices
                )
                for key, parts in vals.items()
                if parts
            }
            for seq, vals in per_seq.items()
        }

    def _write_per_sequence(self, epoch: int, per_seq: dict[str, dict[str, np.ndarray]]) -> None:
        if not self.is_rank0:
            return
        out_dir = Path(self.cfg.per_sequence_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Task 8 Val Per-sequence Metrics — Epoch {epoch}",
            "",
            "| Sequence | N | " + " | ".join(self.channel_names) + " |",
            "|---|---:|" + "---:|" * len(self.channel_names),
        ]
        seq_rows: list[np.ndarray] = []
        for seq, vals in sorted(per_seq.items()):
            row = self._r2_np(vals["pred"], vals["target"])
            seq_rows.append(row)
            lines.append(f"| {seq} | {len(vals['target'])} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
        if seq_rows:
            med = np.nanmedian(np.stack(seq_rows), axis=0)
            lines += [
                "",
                "| Summary | " + " | ".join(self.channel_names) + " |",
                "|---|" + "---:|" * len(self.channel_names),
                "| median | " + " | ".join(f"{v:.4f}" for v in med) + " |",
            ]
        (out_dir / f"val_epoch_{epoch:03d}.md").write_text("\n".join(lines) + "\n")

    @staticmethod
    def _r2_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        vals = []
        for c in range(target.shape[1]):
            y, p = target[:, c], pred[:, c]
            ss_res = np.sum((y - p) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            vals.append(1.0 - ss_res / max(ss_tot, 1e-12))
        return np.asarray(vals, dtype=np.float64)


# ------------------------------------------------------------------
# config loading
# ------------------------------------------------------------------

def build_trainer_from_config(path: str, variant: str | None = None) -> FullModelTrainer:
    cfg_path = Path(path)
    cfg = load_config(cfg_path, variant)
    repo = cfg_path.resolve().parents[2] if cfg_path.name == "v1.yaml" else Path.cwd()

    trainer_cfg = _dataclass_from_dict(TrainerConfig, _resolve_paths(cfg["training"], repo))
    _seed_everything(trainer_cfg.seed)
    data_cfg = _dataclass_from_dict(DatasetConfig, _resolve_paths(cfg["data"], repo))
    train_data = TartanDriveDataset(data_cfg)
    val_data = TartanDriveDataset(_replace_dataclass(data_cfg, split="val"))
    test_data = (
        TartanDriveDataset(_replace_dataclass(data_cfg, split="test"))
        if trainer_cfg.test_eval_every_n_epochs > 0
        else None
    )
    stats = NormStats.from_json(data_cfg.norm_stats_path)
    pos_rate = train_data.positive_rate()
    pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-12)

    model_cfg = _dataclass_from_dict(ModelConfig, cfg["model"])
    loss_cfg = _dataclass_from_dict(LossConfig, cfg["loss"])
    _validate_proprio_channels(model_cfg, loss_cfg, data_cfg)
    if data_cfg.window_columns is not None and model_cfg.d_raw != len(data_cfg.window_columns):
        raise ValueError(
            f"model.d_raw ({model_cfg.d_raw}) != len(data.window_columns) "
            f"({len(data_cfg.window_columns)})"
        )

    model = TerrainWorldModel(model_cfg)
    return FullModelTrainer(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        train_stats=stats,
        trainer_cfg=trainer_cfg,
        loss_cfg=loss_cfg,
        curriculum_cfg=_dataclass_from_dict(CurriculumConfig, cfg["curriculum"]),
        sampling_cfg=_dataclass_from_dict(ScheduledSamplingConfig, cfg["scheduled_sampling"]),
        pos_weight=pos_weight,
        test_dataset=test_data,
        run_config=cfg,   # full yaml dump -> wandb.config (run-diff in the UI)
    )


def build_legged_trainer_from_config(path: str, variant: str | None = None) -> FullModelTrainer:
    """Additive legged builder. SEPARATE from the wheeled
    build_trainer_from_config (which is untouched).

    TG/GT mode (default): train = GrandTour-train + all TartanGround, each sample normed by
    ITS dataset stats; val = held-out GrandTour missions (real eval), GT-normed so
    R²=1-SS_res/SS_tot is exact (the per-channel affine denorm cancels). No binary trav
    (pos_weight unused, lambda_trav=0).

    Isaac ANYmal mode (data.isaac_root set): single corpus, single norm-stats
    (data.norm_stats_path, TRAIN-only), val = data.heldout_substrings path-groups (mirrors the
    wheeled Isaac Husky val design — same 9 groups → matched
    cross-embodiment val slices). This corpus HAS binary trav GT (rollover/stuck; no collision
    channel) → pos_weight computed from the train positive rate, trav head trainable.

    JOINT 3-corpus mode (isaac_root AND grandtour_root set): TG + GT-train + isaac-train
    pooled, per-TAG norm stats (never a shared sim+real norm), TAG-PURE per-domain vals
    (main val = heldout GrandTour missions; extra_val_datasets["isaac"] = the heldout isaac
    path-groups — R² is affine-invariant only within one tag's norm affine, so pooled
    mixed-tag val is deliberately not built). trav supervised by the isaac corpus only."""
    from feelwm.data.legged_dataset import (
        LeggedDataset, LeggedDatasetConfig, discover_legged_specs,
    )

    cfg_path = Path(path)
    cfg = load_config(cfg_path, variant)
    repo = Path(__file__).resolve().parents[2]

    def rp(x: str) -> str:
        return str(x if Path(x).is_absolute() else repo / x)

    trainer_cfg = _dataclass_from_dict(TrainerConfig, _resolve_paths(cfg["training"], repo))
    _seed_everything(trainer_cfg.seed)

    d, m = cfg["data"], cfg["model"]
    pos_weight = 1.0
    joint_val_isaac: list = []
    if d.get("isaac_root") and d.get("grandtour_root"):
        # --- JOINT 3-corpus mode (TG + GT + Isaac) ---
        # Per-TAG norm stats (never a shared sim+real norm; the TG+GT mode already trains
        # sim+real per-tag). Per-domain val is TAG-PURE — main val = the real GrandTour Dec
        # missions, extra val "isaac" = the 9 wheeled-mirror path-groups — because R²
        # is affine-invariant only within one norm affine: a mixed-tag pooled val
        # would be denormed with one tag's affine and be silently wrong.
        norm = {"gt": rp(d["norm_stats_path_gt"]), "tg": rp(d["norm_stats_path_tg"]),
                "isaac": rp(d["norm_stats_path_isaac"])}
        specs = discover_legged_specs(
            grandtour_root=d["grandtour_root"], tartanground_root=d["tartanground_root"],
            isaac_root=d["isaac_root"],
            feat_version_gt=d.get("feat_version_gt", "gt_dinov3_v1"),
            feat_version_tg=d.get("feat_version_tg", "tg_dinov3_v1"),
            feat_version_isaac=d.get("feat_version_isaac", "anymal_isaac_dinov3_v1"),
        )
        heldout_gt = set(d.get("heldout_grandtour_missions") or [])
        held_isaac = d.get("heldout_substrings") or []
        if not heldout_gt or not held_isaac:
            raise ValueError(
                "joint mode requires BOTH data.heldout_grandtour_missions and "
                "data.heldout_substrings (tag-pure per-domain vals)")
        val_specs = [s for s in specs if s.tag == "gt" and s.name in heldout_gt]
        joint_val_isaac = [s for s in specs if s.tag == "isaac"
                           and any(h in s.name for h in held_isaac)]
        if not val_specs or not joint_val_isaac:
            raise ValueError("joint mode: empty gt or isaac val slice")
        val_names = {s.name for s in val_specs} | {s.name for s in joint_val_isaac}
        train_specs = [s for s in specs if s.name not in val_names]
        # trav supervision exists ONLY on the isaac corpus (GT/TG binary masks are
        # all-zero) → pos_weight from the isaac train positive rate, the anchors the
        # BCE actually sees.
        n_pos = n_tot = 0
        for sp in train_specs:
            if sp.tag != "isaac":
                continue
            z = np.load(sp.label_npz)
            v = z["valid_mask"].astype(bool)
            n_pos += int(z["trav_gt"][v].sum())
            n_tot += int(v.sum())
        pos_rate = n_pos / max(n_tot, 1)
        pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-12)
        tags = {t: sum(1 for s in train_specs if s.tag == t) for t in ("gt", "tg", "isaac")}
        print(f"[legged-joint] train {tags} seqs; val gt {len(val_specs)} / isaac "
              f"{len(joint_val_isaac)}; isaac trav+ {n_pos}/{n_tot} ({100*pos_rate:.2f}%)"
              f" → pos_weight={pos_weight:.1f}")
        val_stats_tag = "gt"   # main val = real GrandTour; tag-pure → exact R²
    elif d.get("isaac_root"):
        # --- Isaac ANYmal single-corpus mode ---
        norm = {"isaac": rp(d["norm_stats_path"])}
        specs = discover_legged_specs(
            isaac_root=d["isaac_root"],
            feat_version_isaac=d.get("feat_version", "anymal_isaac_dinov3_v1"),
        )
        held = d.get("heldout_substrings")
        if not held:
            raise ValueError("isaac mode requires data.heldout_substrings (val path-groups)")
        val_specs = [s for s in specs if any(h in s.name for h in held)]
        train_specs = [s for s in specs if not any(h in s.name for h in held)]
        if not val_specs:
            raise ValueError("no sequences matched data.heldout_substrings")
        # pos_weight from the TRAIN positive rate over valid anchors (mirrors the sim builder)
        n_pos = n_tot = 0
        for sp in train_specs:
            z = np.load(sp.label_npz)
            v = z["valid_mask"].astype(bool)
            n_pos += int(z["trav_gt"][v].sum())
            n_tot += int(v.sum())
        pos_rate = n_pos / max(n_tot, 1)
        pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-12)
        print(f"[legged-isaac] train {len(train_specs)} seqs / val {len(val_specs)} seqs; "
              f"train trav+ {n_pos}/{n_tot} ({100*pos_rate:.2f}%) → pos_weight={pos_weight:.1f}")
        val_stats_tag = "isaac"   # val normed with the (train-only) isaac stats → exact R²
    else:
        norm = {"gt": rp(d["norm_stats_path_gt"]), "tg": rp(d["norm_stats_path_tg"])}
        specs = discover_legged_specs(
            grandtour_root=d["grandtour_root"], tartanground_root=d["tartanground_root"],
            feat_version_gt=d.get("feat_version_gt", "gt_dinov2reg_v1"),
            feat_version_tg=d.get("feat_version_tg", "tg_dinov2reg_v1"),
        )
        heldout = set(d.get("heldout_grandtour_missions", []))
        train_specs = [s for s in specs if not (s.tag == "gt" and s.name in heldout)]
        val_specs = [s for s in specs if s.tag == "gt" and s.name in heldout]
        if not val_specs:
            raise ValueError("no held-out GrandTour missions (set data.heldout_grandtour_missions)")
        # Optional TRAIN-side tag filter (val untouched). Consumer: the TG-only →
        # GrandTour zero-shot row — train on TG alone while keeping the TG+GT val
        # machinery verbatim, so the run is a corpus-only delta.
        train_tags = d.get("train_tags")
        if train_tags:
            train_specs = [s for s in train_specs if s.tag in set(train_tags)]
            if not train_specs:
                raise ValueError(f"data.train_tags {train_tags} left no train sequences")
            print(f"[legged] data.train_tags={train_tags} → train restricted to "
                  f"{len(train_specs)} seqs")
        val_stats_tag = "gt"      # val = held-out GT → GT stats give exact R²

    ld = LeggedDatasetConfig(
        context_len=m["context_len"], rollout_horizon=m["rollout_horizon"],
        n_patch=m["n_patch"], d_vis=m["d_vis"],
        require_vis_context=True, require_vis_target=True, norm_stats=norm,
        # multi-stride / Δt-conditioning (module docstring, feelwm/data/legged_dataset.py);
        # defaults reproduce single-stride behavior exactly when the config omits these keys.
        include_dt=d.get("include_dt", False),
        train_strides=d.get("train_strides", [1]),
        val_stride=d.get("val_stride", 1),
        proprio_horizon_s=d.get("proprio_horizon_s", 1.0),
        stride_sampling=d.get("stride_sampling", "all"),
        # active proprio OUTPUT channels (e.g. drop `slip`). None = all 5.
        active_proprio_channels=d.get("active_proprio_channels"),
    )

    model_cfg = _dataclass_from_dict(ModelConfig, m)
    loss_cfg = _dataclass_from_dict(LossConfig, cfg["loss"])
    if model_cfg.proprio_feature_dim != len(loss_cfg.channel_names):
        raise ValueError(
            f"proprio_feature_dim ({model_cfg.proprio_feature_dim}) != "
            f"len(channel_names) ({len(loss_cfg.channel_names)})"
        )
    active = ld.active_proprio_channels or list(LEGGED_FEATURE_CHANNELS)
    if list(loss_cfg.channel_names) != list(active):
        raise ValueError(
            f"loss.channel_names ({list(loss_cfg.channel_names)}) != "
            f"data.active_proprio_channels ({list(active)}) — the dataset emits exactly "
            "the active channels, in that order")

    train_data = LeggedDataset(train_specs, _replace_dataclass(ld, split="train"))
    val_data = LeggedDataset(val_specs, _replace_dataclass(ld, split="val"))
    # Per-tag WINDOW census + silent-partial-corpus guard: require_vis_context silently
    # drops every anchor whose visual feature cache is missing, so a corpus with no cache
    # trains as if it were absent — the spec census alone (which counts LABEL files)
    # cannot see it. Fail loudly.
    from collections import Counter
    tag_windows: Counter = Counter(
        train_data._records[idx[0]].spec.tag for idx in train_data._index)
    tag_anchors: Counter = Counter()
    for rec in train_data._records:
        tag_anchors[rec.spec.tag] += int(rec.valid_mask.sum())
    print(f"[legged] train WINDOWS by tag: {dict(tag_windows)} "
          f"(labeled anchors: {dict(tag_anchors)})")
    expected_tags = {s.tag for s in train_specs}
    missing = sorted(expected_tags - set(tag_windows))
    if missing:
        raise ValueError(
            f"corpus tag(s) {missing} contributed 0 train windows — their visual "
            "feature cache is missing/incomplete (require_vis_context drops uncached "
            "anchors silently). Refusing the silent partial-corpus run.")
    for t in sorted(expected_tags):
        if tag_anchors[t] and tag_windows[t] < 0.5 * tag_anchors[t]:
            print(f"[legged] ⚠ WARNING: tag {t!r} has only {tag_windows[t]} windows for "
                  f"{tag_anchors[t]} labeled anchors — feature cache likely PARTIAL.")
    extra_val: dict[str, Dataset] = {}
    if joint_val_isaac:
        extra_val["isaac"] = LeggedDataset(joint_val_isaac, _replace_dataclass(ld, split="val"))
    stats = NormStats.from_json(norm[val_stats_tag])
    # Probe one real sample: the config-level checks above cannot see a loader that
    # ignores the channel selection (an observed launch crash — a 5-dim
    # f_prop_target reached a 4-dim model and died inside DDP teacher forcing on the
    # first train batch). Fail here, at BUILD time, with a named error.
    # (restore the RNG: in random stride-sampling mode __getitem__ draws a stride, and the
    # probe must not shift the seeded stream — see _seed_everything above)
    _rng_state = random.getstate()
    try:
        _assert_sample_proprio_dim(train_data[0], model_cfg, "legged train")
        _assert_sample_proprio_dim(val_data[0], model_cfg, "legged val")
    finally:
        random.setstate(_rng_state)

    model = TerrainWorldModel(model_cfg)
    return FullModelTrainer(
        model=model, train_dataset=train_data, val_dataset=val_data, train_stats=stats,
        trainer_cfg=trainer_cfg, loss_cfg=loss_cfg,
        curriculum_cfg=_dataclass_from_dict(CurriculumConfig, cfg["curriculum"]),
        sampling_cfg=_dataclass_from_dict(ScheduledSamplingConfig, cfg["scheduled_sampling"]),
        pos_weight=pos_weight, test_dataset=None, run_config=cfg,
        extra_val_datasets=extra_val or None,
    )


def build_sim_datasets(
    cfg: dict[str, Any],
    splits: tuple[str, ...] = ("train", "val"),
    per_domain_val: bool = True,
    **overrides: Any,
):
    """Sim Husky datasets from a parsed config dict — the dataset half of
    build_sim_trainer_from_config (which calls it). Offline eval scripts get the exact same
    split + normalization without building a trainer, and without duplicating split logic.

    `splits` selects which datasets to build (building "train" scans ~900 episodes, so eval
    passes should ask for "val" only). `overrides` are applied to the shared SimDatasetConfig
    (e.g. rollout_horizon=16, val_stride=5). Returns ({split: SimHuskyDataset}, base_config).

    MULTI-CORPUS (see sim_corpora): "train" pools every corpus and "val" pools every
    corpus's held-out episodes, but with `per_domain_val` the returned dict ALSO carries a
    "val:<name>" dataset per corpus. A pooled val number hides exactly what combined
    training must answer ("does pooling beat each single-domain corpus on ITS OWN domain"), so the
    trainer validates each domain separately every epoch. Single-corpus configs are
    unaffected: no extra key, no extra pass.
    """
    from feelwm.data.sim_dataset import (
        SimHuskyDataset, SimDatasetConfig, discover_sim_specs, drop_excluded_specs,
        drop_specs, sim_corpora, split_sim_specs,
    )

    repo = Path(__file__).resolve().parents[2]

    def rp(x: str) -> str:
        return str(x if Path(x).is_absolute() else repo / x)

    d, m = cfg["data"], cfg["model"]
    corpora = sim_corpora(d)
    multi = len(corpora) > 1
    train_specs, val_specs = [], []
    domain_val: dict[str, list] = {}
    for c in corpora:
        found = discover_sim_specs(rp(c["seq_root"]), c["feat_version"], domain=c["name"],
                                   labels_name=c["labels_name"])
        if not found:
            raise ValueError(
                f"corpus {c['name'] or '<single>'}: no episodes under {c['seq_root']}")
        # Corpus composition first: dropped sequences are in NEITHER split (drop_specs).
        kept = drop_specs(found, c["drop_substrings"])
        if c["drop_substrings"] and len(kept) == len(found):
            raise ValueError(
                f"corpus {c['name'] or '<single>'}: drop_substrings "
                f"{c['drop_substrings']} matched no episode under {c['seq_root']} — a typo "
                "would silently leave near-duplicate-of-val sequences in train")
        tr, va = split_sim_specs(kept, c["heldout_substrings"])
        if not va:
            raise ValueError(
                f"no held-out sim episodes for corpus {c['name'] or '<single>'} "
                "(set heldout_substrings)")
        train_specs += tr
        val_specs += va
        domain_val[c["name"]] = va
    # The dataset-level list is the UNION: it is only used by the tail-holdout guard
    # (a sequence already held out whole is never tail-sliced), which is a name test.
    heldout = [s for c in corpora for s in c["heldout_substrings"]]

    base = SimDatasetConfig(
        # With multiple corpora these two are informational only (the loader reads each
        # spec's own absolute paths); they name the FIRST corpus so single-corpus callers
        # that read them back are unchanged.
        seq_root=rp(corpora[0]["seq_root"]), feat_version=corpora[0]["feat_version"],
        norm_stats_path=rp(d["norm_stats_path"]), heldout_substrings=heldout,
        # train-only sequence exclusion (SimDatasetConfig.exclude_sequences); [] = no-op
        exclude_sequences=list(d.get("exclude_sequences", [])),
        context_len=m["context_len"], rollout_horizon=m["rollout_horizon"],
        n_patch=m["n_patch"], d_vis=m["d_vis"],
        # encoder-input column subset (e.g. portable-11), sliced after normalization
        window_columns=d.get("window_columns"),
        # multi-stride / Δt-conditioning (module docstring, feelwm/data/sim_dataset.py);
        # defaults reproduce single-stride behavior exactly when the config omits these keys.
        include_dt=d.get("include_dt", False),
        train_strides=d.get("train_strides", [1]),
        val_stride=d.get("val_stride", 1),
        proprio_horizon_s=d.get("proprio_horizon_s", 1.0),
        stride_sampling=d.get("stride_sampling", "all"),
        # channel subset (e.g. drop ke_rate). None = all 5.
        active_proprio_channels=d.get("active_proprio_channels"),
        # real-robot localization-jump screen (None = off; sim/isaac indices unchanged)
        max_pose_step_m=d.get("max_pose_step_m"),
        # within-sequence tail holdout as a second val axis (off unless both are set)
        tail_val_substrings=d.get("tail_val_substrings", []),
        tail_val_fraction=d.get("tail_val_fraction", 0.0),
    )
    if overrides:
        base = _replace_dataclass(base, **overrides)
    # Tail-holdout sequences train on their head and validate on their tail, so they must
    # be handed to BOTH datasets; SimHuskyDataset keeps each side of the cut (tail_val_*).
    tail_specs = [s for s in train_specs
                  if base.tail_val_fraction > 0.0
                  and any(sub in s.name for sub in base.tail_val_substrings)]
    # Train-only exclusion, applied AFTER the tail pool is taken: an excluded sequence loses
    # its train windows but keeps any tail-val slice, so the val side of an arm that only
    # varies the training corpus is bit-identical to the empty-list case.
    train_specs = drop_excluded_specs(train_specs, base.exclude_sequences)
    if "train" in splits and not train_specs:
        raise ValueError(
            f"data.exclude_sequences {base.exclude_sequences} left no train sequences")
    spec_map = {"train": train_specs, "val": val_specs + tail_specs}
    unknown = set(splits) - set(spec_map)
    if unknown:
        raise ValueError(f"unknown sim split(s) {sorted(unknown)} (expected train/val)")
    # Per-domain val slices of the SAME pooled val set (same specs, same config) — not a
    # different split, so a domain's numbers are directly comparable with the single-corpus
    # arm that trained on that corpus alone.
    keys = list(splits)
    if multi and per_domain_val and "val" in splits:
        for name, va in domain_val.items():
            spec_map[f"val:{name}"] = va + [s for s in tail_specs if s.domain == name]
            keys.append(f"val:{name}")
    datasets = {
        s: SimHuskyDataset(spec_map[s], _replace_dataclass(base, split=_split_of(s)))
        for s in keys
    }
    return datasets, base


def _split_of(key: str) -> str:
    """Dataset key -> SimDatasetConfig.split ("val:isaac" is a val slice, not a new split)."""
    return key.split(":", 1)[0]


def build_sim_trainer_from_config(path: str, variant: str | None = None) -> FullModelTrainer:
    """Additive sim Husky builder. SEPARATE from the wheeled /
    legged builders. One sim domain (single norm, feature_set='sim'); train/val split by
    held-out episode substrings (in-distribution val until lake/forest land). Unlike legged,
    sim has TRUE binary trav GT → trav head trained, pos_weight from the train positive rate."""
    cfg_path = Path(path)
    cfg = load_config(cfg_path, variant)
    repo = Path(__file__).resolve().parents[2]

    trainer_cfg = _dataclass_from_dict(TrainerConfig, _resolve_paths(cfg["training"], repo))
    _seed_everything(trainer_cfg.seed)

    m = cfg["model"]
    datasets, base = build_sim_datasets(cfg)
    train_data, val_data = datasets["train"], datasets["val"]
    # Multi-corpus runs additionally validate each domain on its own (see build_sim_datasets)
    domain_val = {k.split(":", 1)[1]: v for k, v in datasets.items() if k.startswith("val:")}
    norm_path = base.norm_stats_path
    stats = NormStats.from_json(norm_path)

    model_cfg = _dataclass_from_dict(ModelConfig, m)
    loss_cfg = _dataclass_from_dict(LossConfig, cfg["loss"])
    if model_cfg.proprio_feature_dim != len(loss_cfg.channel_names):
        raise ValueError(
            f"proprio_feature_dim ({model_cfg.proprio_feature_dim}) != "
            f"len(channel_names) ({len(loss_cfg.channel_names)})")
    active = base.active_proprio_channels or list(FEATURE_CHANNELS)
    if list(loss_cfg.channel_names) != list(active):
        raise ValueError(
            f"loss.channel_names ({list(loss_cfg.channel_names)}) != "
            f"data.active_proprio_channels ({list(active)}) — the dataset emits exactly "
            "the active channels, in that order")

    pos_rate = train_data.positive_rate()
    pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-12)
    if domain_val:
        # Mixing is natural concatenation with NO reweighting (by design), so the
        # realized per-corpus sampling weight is an outcome of corpus size, not a knob —
        # print it once at build time instead of leaving it to be reconstructed later. Same
        # for pos_weight: one trav-BCE weight now covers two label conventions (Gazebo has a
        # collision channel, Isaac's bumper is off — rollover/stuck only).
        counts = train_data.domain_counts()
        total = sum(counts.values()) or 1
        share = ", ".join(f"{k}={v:,} ({v / total:.1%})" for k, v in sorted(counts.items()))
        print(f"[corpora] train windows: {share}; pooled pos_rate={pos_rate:.4f} "
              f"pos_weight={pos_weight:.1f}")
        for name, ds in sorted(domain_val.items()):
            print(f"[corpora] val:{name} windows: {len(ds):,}")

    model = TerrainWorldModel(model_cfg)
    return FullModelTrainer(
        model=model, train_dataset=train_data, val_dataset=val_data, train_stats=stats,
        trainer_cfg=trainer_cfg, loss_cfg=loss_cfg,
        curriculum_cfg=_dataclass_from_dict(CurriculumConfig, cfg["curriculum"]),
        sampling_cfg=_dataclass_from_dict(ScheduledSamplingConfig, cfg["scheduled_sampling"]),
        pos_weight=pos_weight, test_dataset=None, run_config=cfg,
        extra_val_datasets=domain_val or None,
    )


def _assert_sample_proprio_dim(sample: dict[str, Any], model_cfg: ModelConfig, what: str) -> None:
    """Build-time guard: the loader's emitted proprio width == the model's head width.

    Config-level checks (proprio_feature_dim vs channel_names vs active_proprio_channels)
    only prove the CONFIG is self-consistent; they cannot catch a data path that ignores
    the channel selection. That gap let a 5-dim `f_prop_target` reach a 4-dim legged model
    and crash inside DDP teacher forcing on the first train batch. Cheap to
    check once per build, and the error names the mismatch.
    """
    for key in ("f_prop_target", "current_feature"):
        if key not in sample:
            continue
        got = int(sample[key].shape[-1])
        if got != model_cfg.proprio_feature_dim:
            raise ValueError(
                f"{what} dataset emits {key} with last dim {got}, but "
                f"model.proprio_feature_dim is {model_cfg.proprio_feature_dim} — "
                "check data.active_proprio_channels / loss.channel_names"
            )


def _validate_proprio_channels(
    model_cfg: ModelConfig, loss_cfg: LossConfig, data_cfg: DatasetConfig
) -> None:
    """Ensure active-proprio-channel config is consistent across model/loss/data."""
    names = loss_cfg.channel_names
    n = len(names)
    if model_cfg.proprio_feature_dim != n:
        raise ValueError(
            f"model.proprio_feature_dim ({model_cfg.proprio_feature_dim}) != "
            f"len(loss.channel_names) ({n})"
        )
    if len(loss_cfg.proprio_weights) != n:
        raise ValueError(
            f"len(loss.proprio_weights) ({len(loss_cfg.proprio_weights)}) != "
            f"len(loss.channel_names) ({n})"
        )
    data_chans = data_cfg.active_proprio_channels or CHANNEL_NAMES
    if list(data_chans) != list(names):
        raise ValueError(
            f"data.active_proprio_channels ({list(data_chans)}) != "
            f"loss.channel_names ({list(names)})"
        )


def _seed_everything(seed: int) -> None:
    _set_ddp_cuda_device_from_env()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataclass_from_dict(cls, data: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def _replace_dataclass(obj, **updates):
    data = asdict(obj)
    data.update(updates)
    return type(obj)(**data)


def _resolve_paths(data: dict[str, Any], repo: Path) -> dict[str, Any]:
    out = dict(data)
    for key in ("split_path", "binary_label_root", "norm_stats_path", "checkpoint_path", "resume_path", "metrics_path", "per_sequence_dir"):
        if key in out and out[key] is not None:
            p = Path(out[key])
            out[key] = str(p if p.is_absolute() else repo / p)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--legged", action="store_true",
                        help="use the legged builder (TartanGround+GrandTour)")
    parser.add_argument("--sim", action="store_true",
                        help="use the sim Husky builder (NEGS-UGV Gazebo)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loss-probe", action="store_true")
    parser.add_argument("--probe-batches", type=int, default=2)
    parser.add_argument("--probe-out", default=None)
    add_variant_arg(parser)
    args = parser.parse_args(argv)
    try:
        load_config(args.config, args.variant)          # fail fast on a bad variant name
    except ValueError as e:
        parser.error(str(e))
    if args.legged:
        trainer = build_legged_trainer_from_config(args.config, args.variant)
    elif args.sim:
        trainer = build_sim_trainer_from_config(args.config, args.variant)
    else:
        trainer = build_trainer_from_config(args.config, args.variant)
    if args.dry_run:
        if trainer.is_rank0:
            print(f"train={len(trainer.train_dataset):,} val={len(trainer.val_dataset):,}")
            print(f"device={trainer.device} world_size={trainer.world_size} pos_weight={trainer.pos_weight}")
        return
    if args.loss_probe:
        result = trainer.loss_probe(max_batches=args.probe_batches)
        if trainer.is_rank0:
            text = json.dumps(result, indent=2)
            print(text)
            if args.probe_out:
                out = Path(args.probe_out)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text + "\n")
        return
    trainer.fit()


if __name__ == "__main__":
    main(sys.argv[1:])
