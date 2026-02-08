import time
import logging
from typing import Dict, Optional, Any, cast
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
        self._last_step_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.perf_counter()
        logger.info("TRAINING START")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.log_interval != 0 or step == 0:
            return
        start_time = self.start_time
        if start_time is None:
            return

        if hasattr(self, "_last_logged_step") and self._last_logged_step == step:
            return
        self._last_logged_step = step

        # Read loss from callback_metrics (real un-divided loss)
        loss_metric = trainer.callback_metrics.get("train/loss")
        if loss_metric is None:
            return
        loss = (
            loss_metric.item()
            if isinstance(loss_metric, torch.Tensor)
            else float(loss_metric)
        )

        # Handle multiple optimizers if present, otherwise take first
        trainer_obj = cast(Any, trainer)
        opts = trainer_obj.optimizers
        lr = opts[0].param_groups[0]["lr"] if opts else 0.0

        # Per-step timing instead of running average
        now = time.perf_counter()
        if self._last_step_time is None:
            step_time = now - start_time
        else:
            step_time = now - self._last_step_time
        self._last_step_time = now

        # Calculate tokens/sec with global batch accounting
        module = cast(Any, pl_module)
        config = getattr(module, "config", None)
        if config is None:
            return
        micro_batch_size = config.training.micro_batch_size
        seq_len = 1024
        datamodule = getattr(trainer_obj, "datamodule", None)
        dataset = getattr(datamodule, "dataset", None)
        if dataset is not None and hasattr(dataset, "seq_len"):
            seq_len = dataset.seq_len

        # Global tokens = micro_batch_size * seq_len * num_devices * accumulate_grad_batches
        num_devices = max(1, trainer_obj.num_devices)
        accumulate_grad_batches = max(1, trainer_obj.accumulate_grad_batches)
        global_tokens = (
            micro_batch_size * seq_len * num_devices * accumulate_grad_batches
        )
        tok_per_sec = global_tokens / max(1e-6, step_time)

        logger.info(
            f"Step {step}: loss={loss:.4f} | lr={lr:.2e} | "
            f"step_time={step_time * 1000:.1f}ms | tok/s={tok_per_sec:.0f}"
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
        if not isinstance(outputs, dict) or "loss" not in outputs:
            return
        curriculum = cast(CurriculumManager, self.curriculum)
        outputs_dict = cast(Dict[str, Any], outputs)
        loss_val = outputs_dict["loss"]
        loss = (
            loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)
        )
        unitary_div = trainer.callback_metrics.get(
            "train/unitary_div", torch.tensor(0.0)
        ).item()
        curriculum.record_metrics(loss, unitary_div)
        if curriculum.should_check_transition(trainer.global_step):
            if curriculum.check_transition(trainer.global_step, True):
                self._handle_transition(trainer, pl_module, curriculum)

    def _handle_transition(self, trainer, pl_module, curriculum: CurriculumManager):
        old_stage = curriculum.current_stage
        curriculum.advance_stage(trainer.global_step)
        logger.info(
            f"Curriculum Transition: Stage {old_stage} -> {curriculum.current_stage}"
        )
        if hasattr(trainer, "datamodule") and hasattr(
            trainer.datamodule, "update_seq_len"
        ):
            trainer.datamodule.update_seq_len(curriculum.seq_len, curriculum.min_score)
        for opt in trainer.optimizers:
            for pg in opt.param_groups:
                pg["lr"] *= curriculum.lr_decay_per_stage
        pl_module.log_dict(
            {
                "curriculum/stage": float(curriculum.current_stage),
                "curriculum/seq_len": float(curriculum.seq_len),
                "curriculum/min_score": float(curriculum.min_score),
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
        if (
            hasattr(self, "_last_health_step")
            and self._last_health_step == trainer.global_step
        ):
            return
        self._last_health_step = trainer.global_step
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
        resume_id: Optional[str] = None,
    ):
        self.project = project
        self.config = config
        self.enabled = enabled
        self.run_id = None
        self._run = None
        if enabled:
            try:
                import importlib

                wandb = importlib.import_module("wandb")
                resume = resume_id
                if resume is not None:
                    self._run = wandb.init(
                        project=project, config=config, resume="must", id=resume
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
                import importlib

                wandb = importlib.import_module("wandb")
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"wandb log failed: {e}")

    def alert(self, title: str, text: str):
        if self.enabled and self._run:
            try:
                import importlib

                wandb = importlib.import_module("wandb")
                wandb.alert(title=title, text=text)
            except Exception:
                pass

    def finish(self):
        if self.enabled and self._run:
            try:
                import importlib

                wandb = importlib.import_module("wandb")
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
        self._step_start_time = time.perf_counter()

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
