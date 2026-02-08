"""Main training orchestrator for SEM V5.5.

Integrates: data streaming, curriculum learning, self-distillation,
health monitoring, checkpointing, and logging into a single training loop.
"""

import os
import torch
import torch.nn as nn
import math
import logging
import time
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


def _sync_and_time(device_type: str) -> float:
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "xpu":
        torch.xpu.synchronize()
    return time.perf_counter()


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
        v8_features = []
        v8 = getattr(config, "v8", None)
        if v8:
            if getattr(v8, "use_lindblad", False):
                v8_features.append("lindblad")
            if getattr(v8, "use_hybrid_automata", False):
                v8_features.append("hybrid_automata")
            if getattr(v8, "use_quaternionic", False):
                v8_features.append("quaternionic")
        if v8_features:
            logger.info(f"Building SEM model with V8 features: {v8_features}")
        else:
            logger.info("Building SEM model...")
        self.model = SEMModel(config).to(self.device)
        param_counts = self.model.count_parameters()
        logger.info(
            f"Parameters: {param_counts['total']['effective_real']:,} effective real"
        )

        self._use_amp = False
        self._amp_dtype = torch.float32
        self._grad_scaler = None
        no_compile = getattr(config.training, "no_compile", False)
        compile_mode = getattr(config.training, "compile_mode", "default")

        if self.device.type == "cuda":
            if not no_compile:
                try:
                    self.model = torch.compile(
                        self.model,
                        backend="inductor",
                        mode=compile_mode,
                    )
                    logger.info(f"torch.compile enabled for CUDA ({compile_mode})")
                except Exception as e:
                    logger.warning(f"torch.compile failed for CUDA: {e}")
            else:
                logger.info("torch.compile disabled (--no-compile)")
            no_amp = getattr(config.training, "no_amp", False)
            if no_amp:
                logger.info("AMP disabled (--no-amp)")
            elif torch.cuda.is_bf16_supported():
                self._use_amp = True
                self._amp_dtype = torch.bfloat16
                logger.info("AMP enabled with bf16 (native tensor core support)")
            else:
                self._use_amp = True
                self._amp_dtype = torch.float16
                logger.info("AMP enabled with fp16")
            self._grad_scaler = (
                torch.cuda.amp.GradScaler()
                if self._amp_dtype == torch.float16
                else None
            )
        elif self.device.type == "cpu" and not self.dry_run:
            if not no_compile:
                try:
                    self.model = torch.compile(
                        self.model,
                        backend="inductor",
                        mode="reduce-overhead",
                    )
                    logger.info("torch.compile enabled for CPU (reduce-overhead)")
                except Exception as e:
                    logger.warning(f"torch.compile failed for CPU: {e}")
        elif self.device.type == "xpu":
            logger.info("torch.compile disabled for XPU (eager mode)")

        # Optimizer with per-layer LR scaling (SEOP Fix 48)
        base_lr = config.training.learning_rate
        encoder_lr_scale = getattr(config.training, "encoder_lr_scale", 0.01)
        encoder_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name.startswith("encoder"):
                    encoder_params.append(param)
                else:
                    other_params.append(param)
        param_groups = [
            {
                "params": encoder_params,
                "lr": base_lr * encoder_lr_scale,
                "name": "encoder",
            },
            {"params": other_params, "lr": base_lr, "name": "other"},
        ]
        logger.info(
            f"Encoder LR={base_lr * encoder_lr_scale:.2e}, Other LR={base_lr:.2e}"
        )
        logger.info(
            f"  Encoder params: {len(encoder_params)}, Other params: {len(other_params)}"
        )
        self.optimizer = ComplexAdamW(
            param_groups,
            lr=base_lr,
            weight_decay=config.training.weight_decay,
            temperature=0,  # SEOP Fix 49: Disable TNGD noise — amplified by 1/denom during warmup
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
        self.console_cb = ConsoleCallback(
            log_interval=config.training.log_interval,
            timing_log_interval=config.training.timing_log_interval,
        )

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
        else:
            logger.info(
                "Gradient checkpointing DISABLED (config.training.gradient_checkpointing=False)"
            )

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on Mamba and Cayley layers.

        Wraps each layer's forward with torch.utils.checkpoint
        to trade compute for memory and prevent gradient graph explosion.
        """
        from torch.utils.checkpoint import checkpoint

        # Checkpoint Mamba layers
        for i, layer in enumerate(self.model.mamba_layers):
            original_forward = layer.forward

            def make_ckpt_forward(fwd):
                def ckpt_forward(x):
                    return checkpoint(fwd, x, use_reentrant=False)

                return ckpt_forward

            layer.forward = make_ckpt_forward(original_forward)

        # SEOP Fix 25: Also checkpoint propagator layers to prevent CG gradient explosion
        if hasattr(self.model.propagator, "layers"):
            for i, layer in enumerate(self.model.propagator.layers):
                original_forward = layer.forward

                def make_ckpt_forward(fwd):
                    def ckpt_forward(x):
                        return checkpoint(fwd, x, use_reentrant=False)

                    return ckpt_forward

                layer.forward = make_ckpt_forward(original_forward)

            logger.info(
                f"Gradient checkpointing enabled on {len(self.model.mamba_layers)} Mamba layers "
                f"and {len(self.model.propagator.layers)} propagator layers"
            )
        else:
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

        min_score = self.curriculum.min_score if self.curriculum else 2

        dataset = PackedStreamingDataset(
            tokenizer=self.tokenizer,
            seq_len=seq_len,
            min_score=min_score,
            shuffle_buffer=self.config.training.shuffle_buffer_size,
            dataset_name=self.config.training.dataset_name,
            timing_enabled=self.config.training.timing_enabled,
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

    def _auto_scale_micro_batch(self, new_seq_len: int):
        """Scale micro_batch inversely with seq_len to stay within VRAM budget.

        Uses the VRAM/token-slot ratio from the first stage to predict memory
        at new seq_len, then picks the largest micro_batch that fits.
        """
        if not hasattr(self, "_vram_per_token_slot"):
            return
        c = self.config.training
        if self.device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(0).total_memory
        else:
            return
        target_vram = int(total_vram * 0.85)
        new_micro = max(1, target_vram // (self._vram_per_token_slot * new_seq_len))
        new_micro = min(new_micro, c.batch_size)
        while c.batch_size % new_micro != 0 and new_micro > 1:
            new_micro -= 1
        old_micro = c.micro_batch_size
        c.micro_batch_size = new_micro
        logger.info(
            f"[AUTO-SCALE] seq_len {new_seq_len}: micro_batch {old_micro} -> {new_micro} "
            f"(accum={c.batch_size // new_micro})"
        )

    def _handle_stage_transition(self):
        """Handle curriculum stage transition."""
        old_stage = self.curriculum.current_stage
        new_config = self.curriculum.advance_stage(self.global_step)

        self.console_cb.on_stage_transition(
            old_stage, self.curriculum.current_stage, self.global_step
        )

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

        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.curriculum.lr_decay_per_stage
        self.scheduler.reset(new_warmup_steps=self.curriculum.stage_warmup_steps)

        self._auto_scale_micro_batch(self.curriculum.seq_len)

        self._maybe_init_distillation()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared after stage transition")
        elif self.device.type == "xpu":
            torch.xpu.empty_cache()
            logger.info("XPU cache cleared after stage transition")

        self.wandb_cb.log(
            {
                "curriculum/stage": self.curriculum.current_stage,
                "curriculum/seq_len": self.curriculum.seq_len,
                "curriculum/min_score": self.curriculum.min_score,
                "curriculum/micro_batch": self.config.training.micro_batch_size,
            },
            step=self.global_step,
        )

    def _check_nan_grads(self) -> bool:
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    logger.error(f"NaN/Inf gradient detected in: {name}")
                    return True
        return False

    def _check_nan_params(self) -> bool:
        for name, param in self.model.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                logger.error(f"NaN/Inf parameter value detected in: {name}")
                return True
        return False

    def _reduce_lr_on_nan(self, reason: str):
        old_lr = self.optimizer.param_groups[0]["lr"]
        for pg in self.optimizer.param_groups:
            pg["lr"] *= 0.5
        new_lr = self.optimizer.param_groups[0]["lr"]
        self._consecutive_nan_steps = getattr(self, "_consecutive_nan_steps", 0) + 1
        logger.warning(
            f"NaN detected ({reason}) - reducing LR by 50%: {old_lr:.2e} -> {new_lr:.2e} "
            f"(consecutive NaN: {self._consecutive_nan_steps})"
        )
        if self._consecutive_nan_steps >= 10:
            logger.error(
                f"FATAL: {self._consecutive_nan_steps} consecutive NaN steps. "
                f"Model is irrecoverable. Stopping training."
            )
            raise RuntimeError(
                f"Training diverged: {self._consecutive_nan_steps} consecutive NaN steps"
            )

    def train(self):
        """Main training loop."""
        c = self.config.training

        if c.batch_size % c.micro_batch_size != 0:
            raise ValueError(
                f"batch_size ({c.batch_size}) must be divisible by "
                f"micro_batch_size ({c.micro_batch_size})"
            )

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
        if (
            c.timing_enabled
            and hasattr(self.model, "propagator")
            and self.device.type != "cuda"
        ):
            self.model.propagator.set_timing(True)
        logger.info("[TRAIN] Building dataloader...")
        dataloader = self._build_dataloader()
        logger.info("[TRAIN] Creating data iterator...")
        data_iter = iter(dataloader)
        micro_step = 0

        self.optimizer.zero_grad()

        logger.info(f"[TRAIN] Starting training loop (max_steps={c.max_steps})...")
        logger.info("=" * 60)

        compile_warmup_done = False

        while self.global_step < c.max_steps:
            accum_steps = c.batch_size // c.micro_batch_size
            self.console_cb.on_step_start()
            step_loss = 0.0
            step_unitary_divergence = 0.0
            step_metrics = {}
            timings = {}
            current_phase = "init"
            nan_detected = False

            try:
                t_step_start = _sync_and_time(self.device.type)

                # Gradient accumulation loop
                for accum_idx in range(accum_steps):
                    current_phase = "batch_load"
                    if self.global_step == 0 and accum_idx == 0:
                        logger.info("[TRAIN] Loading first batch...")

                    t_batch_start = time.perf_counter()
                    try:
                        batch = next(data_iter)
                        if self.global_step == 0 and accum_idx == 0:
                            logger.info("[TRAIN] First batch loaded successfully!")
                    except StopIteration:
                        dataloader = self._build_dataloader()
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    t_batch_end = time.perf_counter()

                    current_phase = "device_transfer"
                    t_transfer_start = time.perf_counter()
                    token_ids, token_freqs = batch
                    token_ids = token_ids.to(self.device)
                    token_freqs = token_freqs.to(self.device)
                    t_transfer_end = _sync_and_time(self.device.type)

                    current_phase = "forward"
                    t_forward_start = _sync_and_time(self.device.type)
                    amp_ctx = (
                        torch.amp.autocast(self.device.type, dtype=self._amp_dtype)
                        if self._use_amp
                        else torch.amp.autocast(self.device.type, enabled=False)
                    )
                    with amp_ctx:
                        output = self.model(
                            token_ids, targets=token_ids, token_freqs=token_freqs
                        )
                    t_forward_end = _sync_and_time(self.device.type)

                    current_phase = "loss_compute"
                    t_loss_start = time.perf_counter()
                    if (
                        self.ema_teacher is not None
                        and self.distillation_loss is not None
                    ):
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
                    t_loss_end = _sync_and_time(self.device.type)

                    current_phase = "backward"
                    t_backward_start = time.perf_counter()
                    scaled_loss = loss / accum_steps
                    if self._grad_scaler is not None:
                        self._grad_scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    t_backward_end = _sync_and_time(self.device.type)

                    # NaN/Inf detection after backward
                    loss_val = loss.item()
                    if math.isnan(loss_val) or math.isinf(loss_val):
                        logger.error(
                            f"NaN/Inf loss={loss_val} at step {self.global_step} "
                            f"micro={accum_idx}"
                        )
                        self._reduce_lr_on_nan("loss is NaN/Inf")
                        self.optimizer.zero_grad()
                        nan_detected = True
                        break
                    if self._check_nan_grads():
                        self._reduce_lr_on_nan("gradient NaN/Inf")
                        self.optimizer.zero_grad()
                        nan_detected = True
                        break

                    step_loss += loss_val / accum_steps
                    if "unitary_divergence" in output:
                        step_unitary_divergence += (
                            output["unitary_divergence"].item() / accum_steps
                        )
                    micro_step += 1

                    if c.timing_enabled:
                        timings.setdefault("batch_load", []).append(
                            t_batch_end - t_batch_start
                        )
                        timings.setdefault("device_transfer", []).append(
                            t_transfer_end - t_transfer_start
                        )
                        timings.setdefault("forward", []).append(
                            t_forward_end - t_forward_start
                        )
                        timings.setdefault("loss_compute", []).append(
                            t_loss_end - t_loss_start
                        )
                        timings.setdefault("backward", []).append(
                            t_backward_end - t_backward_start
                        )
                        if hasattr(self.model, "propagator"):
                            prop_t = self.model.propagator.collect_and_clear_timing()
                            prop_agg = timings.setdefault("propagator", {})
                            for k, v in prop_t.items():
                                if isinstance(v, list):
                                    prop_agg.setdefault(k, []).extend(v)
                                else:
                                    prop_agg[k] = prop_agg.get(k, 0) + v

                if nan_detected:
                    self.global_step += 1
                    continue

                current_phase = "grad_clip"
                t_clip_start = _sync_and_time(self.device.type)
                if self._grad_scaler is not None:
                    self._grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), c.gradient_clip
                ).item()
                t_clip_end = _sync_and_time(self.device.type)

                current_phase = "fisher_update"
                t_fisher_start = time.perf_counter()
                self.quantizer.update_fisher(self.model)
                t_fisher_end = _sync_and_time(self.device.type)

                if self._check_nan_params():
                    self._reduce_lr_on_nan("parameter NaN/Inf before optimizer step")
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    continue

                current_phase = "optimizer_step"
                t_optim_start = time.perf_counter()
                if self._grad_scaler is not None:
                    self._grad_scaler.step(self.optimizer)
                    self._grad_scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                # Trigger cosine decay when warmup completes
                if self.global_step == self.config.training.warmup_steps:
                    self.scheduler.begin_decay()
                    logger.info(
                        f"[SCHEDULER] Triggered cosine decay at step {self.global_step}"
                    )
                t_optim_end = _sync_and_time(self.device.type)
                # Reset NaN counter on successful step
                self._consecutive_nan_steps = 0

                current_phase = "ema_update"
                t_ema_start = time.perf_counter()
                if self.ema_teacher is not None:
                    self.ema_teacher.update(self.model)
                t_ema_end = time.perf_counter()

                if c.timing_enabled:
                    timings["grad_clip"] = t_clip_end - t_clip_start
                    timings["fisher_update"] = t_fisher_end - t_fisher_start
                    timings["optimizer_step"] = t_optim_end - t_optim_start
                    timings["ema_update"] = (
                        t_ema_end - t_ema_start if self.ema_teacher else 0.0
                    )
                    timings["step_total"] = (
                        _sync_and_time(self.device.type) - t_step_start
                    )

                if c.timing_enabled and self.device.type == "cuda":
                    timings["vram_allocated_mb"] = (
                        torch.cuda.memory_allocated() / 1024**2
                    )
                    timings["vram_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
                    timings["vram_peak_mb"] = (
                        torch.cuda.max_memory_allocated() / 1024**2
                    )

                if (
                    not hasattr(self, "_vram_per_token_slot")
                    and self.device.type == "cuda"
                ):
                    peak = torch.cuda.max_memory_allocated()
                    seq_len = (
                        self.curriculum.seq_len
                        if self.curriculum
                        else self.config.model.max_seq_length
                    )
                    token_slots = c.micro_batch_size * seq_len
                    if token_slots > 0:
                        self._vram_per_token_slot = int(peak / token_slots)
                        logger.info(
                            f"[VRAM] Measured {peak / 1024**2:.0f}MB peak for "
                            f"{c.micro_batch_size}x{seq_len} = "
                            f"{self._vram_per_token_slot} bytes/token-slot"
                        )

                self.global_step += 1
                # Wire adaptive CG tolerance schedule to propagator
                if hasattr(self.model, "propagator"):
                    self.model.propagator.set_training_step(self.global_step)
                current_phase = "metrics"

                if not compile_warmup_done and self.global_step == 1:
                    compile_warmup_done = True
                    if c.timing_enabled:
                        logger.info(
                            f"[COMPILE] First step (includes compile warmup): {timings.get('step_total', 0) * 1000:.0f}ms"
                        )

            except Exception as e:
                logger.error(
                    f"[TRAIN] Exception during phase '{current_phase}' at step {self.global_step}"
                )
                if c.timing_enabled and timings:
                    logger.error(f"[TRAIN] Timings before failure: {timings}")
                raise

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

            # V8 diagnostic logging
            if hasattr(self.model, "get_diagnostics"):
                diag = self.model.get_diagnostics()
                for k, v in diag.items():
                    step_metrics[f"train/{k}"] = v

            # Console logging
            self.console_cb.on_step_end(
                self.global_step,
                step_metrics,
                tokens_in_step,
                timings if c.timing_enabled else None,
            )

            # wandb logging
            if self.global_step % c.log_interval == 0:
                self.wandb_cb.log(step_metrics, step=self.global_step)

            # Health check
            current_phase = "health_check"
            if self.global_step % c.health_check_interval == 0:
                t_health_start = time.perf_counter()
                if self.device.type == "xpu":
                    torch.xpu.empty_cache()

                sample = token_ids[:1]
                report = self.health.check(
                    self.model, sample, self.global_step, grad_norm
                )
                self.console_cb.on_health_report(report)
                self.wandb_cb.log(self.health.get_metrics_dict(), step=self.global_step)

                if c.timing_enabled:
                    timings["health_check"] = time.perf_counter() - t_health_start

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
            current_phase = "curriculum_transition"
            if self.curriculum and self.curriculum.should_check_transition(
                self.global_step
            ):
                health_ok = (
                    not self.health.history[-1].has_error
                    if self.health.history
                    else True
                )
                if self.curriculum.check_transition(self.global_step, health_ok):
                    t_transition_start = time.perf_counter()
                    self._handle_stage_transition()
                    dataloader = self._build_dataloader()
                    data_iter = iter(dataloader)
                    if c.timing_enabled:
                        timings["curriculum_transition"] = (
                            time.perf_counter() - t_transition_start
                        )

            # Regular checkpoint
            current_phase = "checkpoint_save"
            if self.global_step % c.checkpoint_interval == 0:
                t_ckpt_start = time.perf_counter()
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
                if c.timing_enabled:
                    timings["checkpoint_save"] = time.perf_counter() - t_ckpt_start

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
