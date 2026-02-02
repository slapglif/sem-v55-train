import time
import logging
from typing import Dict, Optional, Any
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback

from .curriculum import CurriculumManager
from .health import HealthMonitor

logger = logging.getLogger(__name__)


class SEMConsoleLogger(Callback):
    """Explicit console logging for HF Job visibility."""

    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = log_interval
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.perf_counter()
        logger.info("🚀 TRAINING START")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.log_interval != 0 or step == 0:
            return

        loss = outputs["loss"].item()
        # Handle multiple optimizers if present, otherwise take first
        opts = trainer.optimizers
        lr = opts[0].param_groups[0]["lr"] if opts else 0.0

        elapsed = time.perf_counter() - self.start_time
        avg_step_time = elapsed / max(1, step)

        # Calculate tokens/sec
        batch_size = pl_module.config.training.batch_size
        seq_len = 1024
        if hasattr(trainer, "datamodule") and hasattr(trainer.datamodule, "dataset"):
            seq_len = trainer.datamodule.dataset.seq_len
        tok_per_sec = (batch_size * seq_len) / max(1e-6, avg_step_time)

        logger.info(
            f"Step {step}: loss={loss:.4f} | lr={lr:.2e} | "
            f"step_time={avg_step_time * 1000:.1f}ms | tok/s={tok_per_sec:.0f}"
        )


class SEMCurriculumCallback(Callback):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.curriculum:
            return
        loss = outputs["loss"].item()
        unitary_div = trainer.callback_metrics.get(
            "train/unitary_div", torch.tensor(0.0)
        ).item()
        self.curriculum.record_metrics(loss, unitary_div)
        if self.curriculum.should_check_transition(trainer.global_step):
            if self.curriculum.check_transition(trainer.global_step, True):
                self._handle_transition(trainer, pl_module)

    def _handle_transition(self, trainer, pl_module):
        old_stage = self.curriculum.current_stage
        self.curriculum.advance_stage(trainer.global_step)
        logger.info(
            f"✨ Curriculum Transition: Stage {old_stage} -> {self.curriculum.current_stage}"
        )
        if hasattr(trainer, "datamodule") and hasattr(
            trainer.datamodule, "update_seq_len"
        ):
            trainer.datamodule.update_seq_len(
                self.curriculum.seq_len, self.curriculum.min_score
            )
        for opt in trainer.optimizers:
            for pg in opt.param_groups:
                pg["lr"] *= self.curriculum.lr_decay_per_stage
        pl_module.log_dict(
            {
                "curriculum/stage": float(self.curriculum.current_stage),
                "curriculum/seq_len": float(self.curriculum.seq_len),
                "curriculum/min_score": float(self.curriculum.min_score),
            }
        )


class SEMHealthCallback(Callback):
    def __init__(self, check_interval: int = 500):
        super().__init__()
        self.monitor = HealthMonitor()
        self.check_interval = check_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.check_interval != 0:
            return
        token_ids, _ = batch
        sample = token_ids[:1]
        grad_norm = trainer.callback_metrics.get(
            "train/grad_norm", torch.tensor(0.0)
        ).item()
        report = self.monitor.check(
            pl_module.model, sample, trainer.global_step, grad_norm
        )
        pl_module.log_dict(self.monitor.get_metrics_dict())
        if report.has_error:
            logger.error(
                f"🚨 Health Error at step {trainer.global_step}: {'; '.join(report.messages)}"
            )
            for opt in trainer.optimizers:
                for pg in opt.param_groups:
                    pg["lr"] *= 0.5
