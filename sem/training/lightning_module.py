import torch
import pytorch_lightning as L
import logging
from typing import Any, Dict, Optional
import math

from ..model import SEMModel
from ..utils.complex_adamw import ComplexAdamW
from .scheduler import WSDScheduler
from .distillation import EMATeacher, DistillationLoss

logger = logging.getLogger(__name__)


class SEMLightningModule(L.LightningModule):
    def __init__(self, config: Any):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.model = SEMModel(config)

        # SEOP Fix 27: Explicitly enable gradient checkpointing if configured
        if (
            hasattr(config.training, "gradient_checkpointing")
            and config.training.gradient_checkpointing
        ):
            logger.info(
                "Enabling gradient checkpointing on Mamba and Propagator layers"
            )
            from torch.utils.checkpoint import checkpoint

            # Monkey-patch forward methods for checkpointing
            def make_checkpointed(module):
                original_forward = module.forward

                def checkpointed_forward(*args, **kwargs):
                    # CRITICAL: Disable caching during checkpointed forward to ensure
                    # deterministic behavior (same tensors created in forward & recomputation)
                    if hasattr(module, "_psi_cache"):
                        old_cache = module._psi_cache
                        module._psi_cache = None
                        try:
                            # use_reentrant=False is required for complex control flow
                            # preserve_rng_state=False avoids device initialization errors
                            result = checkpoint(
                                original_forward,
                                *args,
                                use_reentrant=False,
                                preserve_rng_state=False,
                                **kwargs,
                            )
                        finally:
                            module._psi_cache = old_cache
                        return result
                    else:
                        # For modules without cache (Mamba layers)
                        return checkpoint(
                            original_forward,
                            *args,
                            use_reentrant=False,
                            preserve_rng_state=False,
                            **kwargs,
                        )

                module.forward = checkpointed_forward

            # Only checkpoint the propagator — Chebyshev KPM (16 iterations)
            # holds many intermediate tensors in the autograd graph.
            # Mamba layers are O(N) and relatively cheap, so we avoid the
            # recomputation overhead for them (SEOP Fix: remove Mamba checkpointing).
            if hasattr(self.model, "propagator"):
                make_checkpointed(self.model.propagator)
        else:
            logger.info(
                "Gradient checkpointing DISABLED (config.training.gradient_checkpointing=False)"
            )

        self.ema_teacher = None
        self.distillation_loss = None
        if config.distillation.enabled:
            self.distillation_loss = DistillationLoss(
                alpha=config.distillation.alpha,
                temperature=config.distillation.temperature,
            )

    def forward(self, x, targets=None, token_freqs=None):
        return self.model(x, targets=targets, token_freqs=token_freqs)

    def training_step(self, batch, batch_idx):
        # Propagator warmup: smooth alpha ramp 0→1 over warmup_steps.
        # Uses Chebyshev KPM (O(D)) instead of CG solver, avoiding prior convergence issues (SEOP Fix 48).

        # Wire adaptive CG tolerance schedule to propagator
        if hasattr(self.model, "propagator"):
            self.model.propagator.set_training_step(self.trainer.global_step)

        # Smooth propagator warmup: ramp alpha from 0→1 over warmup_steps
        warmup_steps = self.config.training.warmup_steps
        step = self.trainer.global_step
        if warmup_steps > 0 and step <= warmup_steps:
            raw_alpha = step / warmup_steps
            # SEOP Perf: Skip propagator entirely during early warmup (<10%)
            # to save 100% of propagator compute when contribution is negligible.
            if raw_alpha < 0.1:
                alpha = 0.0
            else:
                alpha = raw_alpha
            self.model.set_propagator_alpha(alpha)
        elif not self.model.propagator_enabled:
            self.model.set_propagator_alpha(1.0)

        token_ids, token_freqs = batch
        if getattr(self.config.training, "low_vram_mode", False):
            token_freqs = None

        if self.config.distillation.enabled and self.ema_teacher is None:
            # Check if we should enable at this stage
            # For simplicity, we just check if current stage >= enable_at_stage
            # We can get the stage from the curriculum callback if present
            current_stage = 0
            for cb in self.trainer.callbacks:
                if hasattr(cb, "curriculum") and cb.curriculum is not None:
                    current_stage = cb.curriculum.current_stage
                    break

            if current_stage >= self.config.distillation.enable_at_stage:
                self._init_ema_teacher()

        if self.ema_teacher is not None:
            output = self.model(token_ids, targets=token_ids, token_freqs=token_freqs)
            loss, dist_metrics = self.distillation_loss.compute(
                output,
                self.ema_teacher.teacher,
                token_ids,
                token_ids,
                token_freqs=token_freqs,
            )
            self.log_dict({f"distill/{k}": v for k, v in dist_metrics.items()})
        else:
            output = self.model(token_ids, targets=token_ids, token_freqs=token_freqs)
            loss = output["loss"]

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        if "unitary_divergence" in output:
            self.log("train/unitary_div", output["unitary_divergence"], prog_bar=True)

        # V8 diagnostic logging
        if hasattr(self.model, "get_diagnostics"):
            diag = self.model.get_diagnostics()
            for k, v in diag.items():
                if isinstance(v, (list, tuple)):
                    if len(v) > 0:
                        self.log(f"train/{k}_mean", sum(v) / len(v), prog_bar=False)
                        self.log(f"train/{k}_len", float(len(v)), prog_bar=False)
                elif isinstance(v, (int, float)):
                    self.log(f"train/{k}", float(v), prog_bar=False)
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    self.log(f"train/{k}", v.item(), prog_bar=False)

        # Return dict so callbacks can access outputs["loss"]
        return {"loss": loss}

    def on_after_backward(self):
        if self.trainer.global_step % 50 != 0:
            return
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    self._handle_nan(f"NaN/Inf gradient in {name}")
                    return

    def _handle_nan(self, reason: str):
        logger.error(f"STABILITY ALERT: {reason}")
        self.zero_grad()
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] *= 0.5

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_teacher is not None:
            self.ema_teacher.update(self.model)
        if self.trainer.global_step % 50 == 0:
            for name, param in self.named_parameters():
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    logger.error(f"CRITICAL ERROR: NaN/Inf parameter {name}")

    def configure_optimizers(self):
        base_lr = self.config.training.learning_rate
        encoder_lr_scale = getattr(self.config.training, "encoder_lr_scale", 0.01)
        wd = self.config.training.weight_decay

        # 3-group optimizer (matches trainer.py — SEOP Fix 48 + Mamba best practice)
        encoder_params = []
        ssm_nodecay_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder"):
                encoder_params.append(param)
            elif "A_log" in name or "log_A_mag" in name or ".D" in name:
                # SSM discrete parameters: no weight decay (Mamba paper, Appendix E)
                ssm_nodecay_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {
                "params": encoder_params,
                "lr": base_lr * encoder_lr_scale,
                "name": "encoder",
            },
            {
                "params": ssm_nodecay_params,
                "lr": base_lr,
                "weight_decay": 0.0,
                "name": "ssm_nodecay",
            },
            {"params": other_params, "lr": base_lr, "name": "other"},
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        logger.info(
            f"Optimizer groups: encoder={len(encoder_params)}, "
            f"ssm_nodecay={len(ssm_nodecay_params)}, other={len(other_params)}"
        )

        optimizer = ComplexAdamW(
            param_groups,
            lr=base_lr,
            weight_decay=wd,
            temperature=0,
        )

        # IPEX optimizer wrapping for XPU (if available)
        if self.device.type == "xpu":
            try:
                import intel_extension_for_pytorch as ipex

                logger.info("Wrapping optimizer with IPEX optimizations...")
                optimizer = ipex.optimize(optimizer)
            except ImportError:
                logger.debug("IPEX not available; using standard optimizer on XPU")

        scheduler = WSDScheduler(
            optimizer,
            warmup_steps=self.config.training.warmup_steps,
            decay_steps=self.config.training.decay_steps,
            min_lr_ratio=self.config.training.lr_min_ratio,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _init_ema_teacher(self):
        logger.info("Initializing EMA teacher for self-distillation")
        self.ema_teacher = EMATeacher(
            self.model,
            decay_start=self.config.distillation.ema_decay_start,
            decay_end=self.config.distillation.ema_decay_end,
            decay_ramp_steps=self.config.distillation.ema_decay_ramp_steps,
        ).to(self.device)

    def on_save_checkpoint(self, checkpoint):
        if self.ema_teacher is not None:
            checkpoint["ema_teacher_state_dict"] = self.ema_teacher.teacher.state_dict()
            checkpoint["ema_teacher_update_count"] = self.ema_teacher.update_count

    def on_load_checkpoint(self, checkpoint):
        if "ema_teacher_state_dict" in checkpoint:
            self._init_ema_teacher()
            self.ema_teacher.teacher.load_state_dict(
                checkpoint["ema_teacher_state_dict"]
            )
            self.ema_teacher.update_count = checkpoint["ema_teacher_update_count"]
