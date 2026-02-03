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
        logger.info("TRAINING START")

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
            f"Curriculum Transition: Stage {old_stage} -> {self.curriculum.current_stage}"
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
        if trainer.global_step == 0 or trainer.global_step % self.check_interval != 0:
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
                f"Health Error at step {trainer.global_step}: {'; '.join(report.messages)}"
            )
            for opt in trainer.optimizers:
                for pg in opt.param_groups:
                    pg["lr"] *= 0.5


# Legacy non-Lightning callbacks for hf_train.py


class WandbCallback:
    """Legacy wandb callback for non-Lightning trainer."""

    def __init__(
        self,
        project: str,
        config: Any = None,
        enabled: bool = True,
        resume_id: str = None,
    ):
        self.project = project
        self.config = config
        self.enabled = enabled
        self.run_id = None
        self._run = None
        if enabled:
            try:
                import wandb

                if resume_id:
                    self._run = wandb.init(
                        project=project, config=config, resume="must", id=resume_id
                    )
                else:
                    self._run = wandb.init(project=project, config=config)
                self.run_id = self._run.id if self._run else None
            except Exception as e:
                logger.warning(f"wandb init failed: {e}")
                self.enabled = False

    def log(self, metrics: Dict[str, float], step: int):
        if self.enabled and self._run:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"wandb log failed: {e}")

    def alert(self, title: str, text: str):
        if self.enabled and self._run:
            try:
                import wandb

                wandb.alert(title=title, text=text)
            except Exception:
                pass

    def finish(self):
        if self.enabled and self._run:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass


class ConsoleCallback:
    """Legacy console logging callback for non-Lightning trainer."""

    def __init__(self, log_interval: int = 10, timing_log_interval: int = 10):
        self.log_interval = log_interval
        self.timing_log_interval = timing_log_interval
        self._step_times = []

    def on_step_start(self):
        pass

    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, Any],
        tokens_in_step: int,
        timings: Optional[Dict] = None,
    ):
        if step % self.log_interval != 0:
            return
        loss = metrics.get("train/loss", 0.0)
        lr = metrics.get("train/lr", 0.0)
        grad_norm = metrics.get("train/grad_norm", 0.0)

        msg = f"Step {step}: loss={loss:.4f} | lr={lr:.2e} | grad_norm={grad_norm:.4f}"

        if timings and "step_total" in timings:
            step_time_ms = timings["step_total"] * 1000
            tok_per_sec = tokens_in_step / max(timings["step_total"], 1e-6)
            msg += f" | step_time={step_time_ms:.1f}ms | tok/s={tok_per_sec:.0f}"

        logger.info(msg)

    def on_health_report(self, report):
        if report.has_error:
            logger.error(f"Health Error: {'; '.join(report.messages)}")
        elif report.has_warning:
            logger.warning(f"Health Warning: {'; '.join(report.messages)}")

    def on_stage_transition(self, old_stage: int, new_stage: int, step: int):
        logger.info(
            f"Curriculum Transition: Stage {old_stage} -> {new_stage} at step {step}"
        )
