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
        self.save_hyperparameters()
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
            # We apply it to mamba_layers and propagator layers manually if needed,
            # or just rely on the model's internal support if we add it.
            # For now, let's use the old trainer's logic:
            from torch.utils.checkpoint import checkpoint

            # Monkey-patch forward methods for checkpointing
            def make_checkpointed(module):
                original_forward = module.forward

                def checkpointed_forward(*args, **kwargs):
                    return checkpoint(
                        original_forward, *args, use_reentrant=False, **kwargs
                    )

                module.forward = checkpointed_forward

            for m in self.model.mamba_layers:
                make_checkpointed(m)
            for m in self.model.propagator.layers:
                make_checkpointed(m)

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

        # Return dict so callbacks can access outputs["loss"]
        return {"loss": loss}

    def on_after_backward(self):
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
        for name, param in self.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                logger.error(f"CRITICAL ERROR: NaN/Inf parameter {name}")

    def configure_optimizers(self):
        optimizer = ComplexAdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            temperature=1e-5,
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
