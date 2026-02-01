"""Main training orchestrator for SEM V5.5.

Integrates: data streaming, curriculum learning, self-distillation,
health monitoring, checkpointing, and logging into a single training loop.
"""

import os
import torch
import torch.nn as nn
import math
import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import asdict

from ..config import SEMConfig
from ..model import SEMModel
from ..utils.complex_adamw import ComplexAdamW
from ..quantizer.has_vq import HASVQ
from ..data.streaming import PackedStreamingDataset
from ..data.tokenizer import SEMTokenizer
from .scheduler import WSDScheduler
from .curriculum import CurriculumManager
from .distillation import EMATeacher, DistillationLoss
from .checkpoint import CheckpointManager
from .health import HealthMonitor
from .callbacks import WandbCallback, ConsoleCallback

logger = logging.getLogger(__name__)


class SEMTrainer:
    """Main training orchestrator for SEM V5.5.

    Args:
        config: Full SEMConfig
        device: Training device
        resume_from: Optional checkpoint path to resume from
    """

    def __init__(
        self,
        config: SEMConfig,
        device: str = "cpu",
        resume_from: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.config = config
        self.device = torch.device(device)
        self.global_step = 0
        self.dry_run = dry_run

        # Build model
        logger.info("Building SEM V5.5 model...")
        self.model = SEMModel(config).to(self.device)
        param_counts = self.model.count_parameters()
        logger.info(
            f"Parameters: {param_counts['total']['effective_real']:,} effective real"
        )

        # torch.compile disabled for XPU - requires C++ compiler for Triton kernels
        # CPU still uses torch.compile for oneDNN optimization
        if self.device.type == "cpu" and not self.dry_run:
            try:
                self.model = torch.compile(
                    self.model,
                    backend="inductor",
                    mode="reduce-overhead",
                )
                logger.info(
                    f"torch.compile enabled for CPU (Inductor, reduce-overhead)"
                )
            except Exception as e:
                logger.warning(
                    f"torch.compile failed for CPU, falling back to eager: {e}"
                )
        elif self.device.type == "xpu":
            logger.info("torch.compile disabled for XPU (eager mode)")

        # Optimizer
        self.optimizer = ComplexAdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler
        self.scheduler = WSDScheduler(
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            decay_steps=config.training.decay_steps,
            min_lr_ratio=config.training.lr_min_ratio,
        )

        # Curriculum
        self.curriculum = None
        if config.curriculum.enabled:
            self.curriculum = CurriculumManager(
                stages=config.curriculum.stages,
                transition_check_interval=config.curriculum.transition_check_interval,
                loss_plateau_threshold=config.curriculum.loss_plateau_threshold,
                loss_plateau_window=config.curriculum.loss_plateau_window,
                unitary_stability_threshold=config.curriculum.unitary_stability_threshold,
                lr_decay_per_stage=config.curriculum.lr_decay_per_stage,
                stage_warmup_steps=config.curriculum.stage_warmup_steps,
            )

        # Self-distillation (initialized later, at stage transition)
        self.ema_teacher = None
        self.distillation_loss = None
        if config.distillation.enabled:
            self.distillation_loss = DistillationLoss(
                alpha=config.distillation.alpha,
                temperature=config.distillation.temperature,
            )

        # Quantizer (Fisher tracking)
        self.quantizer = HASVQ(
            codebook_size=config.quantizer.codebook_size,
            group_size=config.quantizer.group_size,
            fisher_ema_decay=config.quantizer.fisher_ema_decay,
            outlier_percentile=config.quantizer.outlier_percentile,
        )

        # Health monitor
        self.health = HealthMonitor()

        # Checkpoint manager
        hub_repo_id = os.environ.get("HF_HUB_REPO_ID")
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_dir="runs",
            keep_checkpoints=config.training.keep_checkpoints,
            hub_repo_id=hub_repo_id,
        )

        # Callbacks
        self.wandb_cb = WandbCallback(
            project=config.training.wandb_project,
            config=asdict(config) if config.training.wandb_enabled else None,
            enabled=config.training.wandb_enabled,
        )
        self.console_cb = ConsoleCallback(log_interval=config.training.log_interval)

        # Tokenizer (skip in dry-run mode — use synthetic data instead)
        self.tokenizer = None
        if not self.dry_run:
            self.tokenizer = SEMTokenizer(config.training.tokenizer_path)
            assert self.tokenizer.vocab_size == config.model.vocab_size, (
                f"Tokenizer vocab {self.tokenizer.vocab_size} != "
                f"model vocab {config.model.vocab_size}"
            )

        # Resume from checkpoint
        if resume_from:
            self._resume(resume_from)

        # Gradient checkpointing
        if config.training.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on Mamba and Cayley layers.

        Wraps each Mamba layer's forward with torch.utils.checkpoint
        to trade compute for memory. The propagator layers are also wrapped.
        """
        from torch.utils.checkpoint import checkpoint

        for i, layer in enumerate(self.model.mamba_layers):
            original_forward = layer.forward

            def make_ckpt_forward(fwd):
                def ckpt_forward(x):
                    return checkpoint(fwd, x, use_reentrant=False)

                return ckpt_forward

            layer.forward = make_ckpt_forward(original_forward)

        logger.info(
            f"Gradient checkpointing enabled on {len(self.model.mamba_layers)} Mamba layers"
        )

    def _resume(self, path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from {path}")
        meta = self.checkpoint_mgr.load(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema_teacher=self.ema_teacher,
            curriculum_manager=self.curriculum,
            health_monitor=self.health,
            quantizer=self.quantizer,
            device=str(self.device),
        )
        self.global_step = meta["global_step"]
        if meta.get("wandb_run_id"):
            self.wandb_cb = WandbCallback(
                project=self.config.training.wandb_project,
                resume_id=meta["wandb_run_id"],
                enabled=self.config.training.wandb_enabled,
            )
        logger.info(f"Resumed at step {self.global_step}")

    def _build_dataloader(self):
        """Build dataloader for current curriculum stage."""
        seq_len = (
            self.curriculum.seq_len
            if self.curriculum
            else self.config.model.max_seq_length
        )

        if self.dry_run:
            return self._build_synthetic_dataloader(seq_len)

        min_score = self.curriculum.min_score if self.curriculum else 2

        dataset = PackedStreamingDataset(
            tokenizer=self.tokenizer,
            seq_len=seq_len,
            min_score=min_score,
            shuffle_buffer=self.config.training.shuffle_buffer_size,
            dataset_name=self.config.training.dataset_name,
        )

        return dataset.create_dataloader(
            batch_size=self.config.training.micro_batch_size,
            num_workers=self.config.training.num_workers,
        )

    def _build_synthetic_dataloader(self, seq_len: int):
        """Build a dataloader with random token IDs for dry-run validation."""
        num_batches = (
            self.config.training.max_steps
            * (self.config.training.batch_size // self.config.training.micro_batch_size)
            + 10
        )
        data = torch.randint(
            0,
            self.config.model.vocab_size,
            (num_batches, self.config.training.micro_batch_size, seq_len),
        )
        freqs = torch.ones(self.config.model.vocab_size) / self.config.model.vocab_size
        return [(data[i], freqs) for i in range(num_batches)]

    def _maybe_init_distillation(self):
        """Initialize EMA teacher when distillation stage is reached."""
        if self.ema_teacher is not None:
            return  # Already initialized
        if not self.config.distillation.enabled:
            return
        if self.curriculum is None:
            return
        if self.curriculum.current_stage < self.config.distillation.enable_at_stage:
            return

        logger.info("Initializing EMA teacher for self-distillation")
        self.ema_teacher = EMATeacher(
            self.model,
            decay_start=self.config.distillation.ema_decay_start,
            decay_end=self.config.distillation.ema_decay_end,
            decay_ramp_steps=self.config.distillation.ema_decay_ramp_steps,
        ).to(self.device)

    def _handle_stage_transition(self):
        """Handle curriculum stage transition."""
        old_stage = self.curriculum.current_stage
        new_config = self.curriculum.advance_stage(self.global_step)

        self.console_cb.on_stage_transition(
            old_stage, self.curriculum.current_stage, self.global_step
        )

        # Save stage transition checkpoint
        self.checkpoint_mgr.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            ema_teacher=self.ema_teacher,
            curriculum_manager=self.curriculum,
            health_monitor=self.health,
            quantizer=self.quantizer,
            wandb_run_id=self.wandb_cb.run_id,
            is_stage_transition=True,
        )

        # Reset scheduler with decayed LR
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.curriculum.lr_decay_per_stage
        self.scheduler.reset(new_warmup_steps=self.curriculum.stage_warmup_steps)

        # Maybe start distillation
        self._maybe_init_distillation()

        # Clean cache after stage transition
        if self.device.type == "xpu":
            torch.xpu.empty_cache()
            logger.info("XPU cache cleared after stage transition")

        # Log
        self.wandb_cb.log(
            {
                "curriculum/stage": self.curriculum.current_stage,
                "curriculum/seq_len": self.curriculum.seq_len,
                "curriculum/min_score": self.curriculum.min_score,
            },
            step=self.global_step,
        )

    def train(self):
        """Main training loop."""
        c = self.config.training
        accum_steps = c.batch_size // c.micro_batch_size

        logger.info(f"SEM V5.5 'Lean Crystal' Training")
        logger.info(f"Device: {self.device}")
        logger.info(
            f"Effective batch size: {c.batch_size} "
            f"(micro={c.micro_batch_size} x accum={accum_steps})"
        )
        if self.curriculum:
            logger.info(f"Curriculum: {len(self.curriculum.stages)} stages")
            logger.info(
                f"Starting stage {self.curriculum.current_stage}: "
                f"seq_len={self.curriculum.seq_len}, "
                f"min_score={self.curriculum.min_score}"
            )

        self.model.train()
        dataloader = self._build_dataloader()
        data_iter = iter(dataloader)
        micro_step = 0

        self.optimizer.zero_grad()

        while self.global_step < c.max_steps:
            self.console_cb.on_step_start()
            step_loss = 0.0
            step_unitary_divergence = 0.0
            step_metrics = {}

            # Gradient accumulation loop
            for _ in range(accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Rebuild dataloader (streaming reset)
                    dataloader = self._build_dataloader()
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                token_ids, token_freqs = batch
                token_ids = token_ids.to(self.device)
                token_freqs = token_freqs.to(self.device)

                # Forward pass
                output = self.model(
                    token_ids, targets=token_ids, token_freqs=token_freqs
                )

                # Loss computation
                if self.ema_teacher is not None and self.distillation_loss is not None:
                    loss, dist_metrics = self.distillation_loss.compute(
                        output,
                        self.ema_teacher.teacher,
                        token_ids,
                        token_ids,
                        token_freqs=token_freqs,
                    )
                    step_metrics.update(dist_metrics)
                else:
                    loss = output["loss"]

                # Scale loss for gradient accumulation
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

                step_loss += loss.item() / accum_steps
                if "unitary_divergence" in output:
                    step_unitary_divergence += (
                        output["unitary_divergence"].item() / accum_steps
                    )
                micro_step += 1

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), c.gradient_clip
            ).item()

            # Fisher tracker update (after clipping to avoid corrupted estimates from gradient spikes)
            self.quantizer.update_fisher(self.model)

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            # EMA teacher update
            if self.ema_teacher is not None:
                self.ema_teacher.update(self.model)

            self.global_step += 1

            # Record loss for curriculum
            if self.curriculum:
                self.curriculum.record_metrics(step_loss, step_unitary_divergence)

            # Metrics
            seq_len = (
                self.curriculum.seq_len
                if self.curriculum
                else self.config.model.max_seq_length
            )
            tokens_in_step = c.batch_size * seq_len

            step_metrics.update(
                {
                    "train/loss": step_loss,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm,
                }
            )

            # Console logging
            self.console_cb.on_step_end(self.global_step, step_metrics, tokens_in_step)

            # wandb logging
            if self.global_step % c.log_interval == 0:
                self.wandb_cb.log(step_metrics, step=self.global_step)

            # Health check
            if self.global_step % c.health_check_interval == 0:
                if self.device.type == "xpu":
                    torch.xpu.empty_cache()

                sample = token_ids[:1]  # Use first sample from last batch
                report = self.health.check(
                    self.model, sample, self.global_step, grad_norm
                )
                self.console_cb.on_health_report(report)
                self.wandb_cb.log(self.health.get_metrics_dict(), step=self.global_step)

                if report.has_error:
                    self.wandb_cb.alert("SEM Health Error", "\n".join(report.messages))
                    # Save emergency checkpoint
                    self.checkpoint_mgr.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        global_step=self.global_step,
                        ema_teacher=self.ema_teacher,
                        curriculum_manager=self.curriculum,
                        health_monitor=self.health,
                        quantizer=self.quantizer,
                        wandb_run_id=self.wandb_cb.run_id,
                    )
                    # Reduce LR
                    for pg in self.optimizer.param_groups:
                        pg["lr"] *= 0.5
                    logger.warning(f"Reduced LR by 50% due to health error")

            # Curriculum transition check
            if self.curriculum and self.curriculum.should_check_transition(
                self.global_step
            ):
                health_ok = (
                    not self.health.history[-1].has_error
                    if self.health.history
                    else True
                )
                if self.curriculum.check_transition(self.global_step, health_ok):
                    self._handle_stage_transition()
                    # Rebuild dataloader for new stage
                    dataloader = self._build_dataloader()
                    data_iter = iter(dataloader)

            # Regular checkpoint
            if self.global_step % c.checkpoint_interval == 0:
                self.checkpoint_mgr.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    global_step=self.global_step,
                    ema_teacher=self.ema_teacher,
                    curriculum_manager=self.curriculum,
                    health_monitor=self.health,
                    quantizer=self.quantizer,
                    wandb_run_id=self.wandb_cb.run_id,
                )

        # Final save
        logger.info("Training complete!")
        self.checkpoint_mgr.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            ema_teacher=self.ema_teacher,
            curriculum_manager=self.curriculum,
            health_monitor=self.health,
            quantizer=self.quantizer,
            wandb_run_id=self.wandb_cb.run_id,
        )
        self.wandb_cb.finish()
