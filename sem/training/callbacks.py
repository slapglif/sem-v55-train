"""Training callbacks for logging and monitoring.

Provides wandb integration and console progress display.
"""
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class WandbCallback:
    """Wandb logging callback.

    Args:
        project: wandb project name
        config: Config dict to log
        resume_id: wandb run ID for resumption
        enabled: Whether wandb is enabled
    """

    def __init__(self, project: str, config: dict = None,
                 resume_id: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        self.run = None
        if enabled:
            try:
                import wandb
                self.run = wandb.init(
                    project=project,
                    config=config,
                    resume="allow",
                    id=resume_id,
                )
            except ImportError:
                logger.warning("wandb not installed, disabling logging")
                self.enabled = False
            except Exception as e:
                logger.warning(f"wandb init failed: {e}, disabling")
                self.enabled = False

    def log(self, metrics: Dict[str, float], step: int):
        if self.enabled and self.run:
            import wandb
            wandb.log(metrics, step=step)

    def alert(self, title: str, text: str):
        if self.enabled and self.run:
            import wandb
            wandb.alert(title=title, text=text)

    @property
    def run_id(self) -> Optional[str]:
        if self.enabled and self.run:
            return self.run.id
        return None

    def finish(self):
        if self.enabled and self.run:
            import wandb
            wandb.finish()


class ConsoleCallback:
    """Console logging with progress tracking.

    Args:
        log_interval: Steps between log messages
    """

    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self._step_start_time = None
        self._tokens_processed = 0

    def on_step_start(self):
        self._step_start_time = time.time()

    def on_step_end(self, step: int, metrics: Dict[str, float],
                    tokens_in_step: int = 0):
        self._tokens_processed += tokens_in_step

        if step % self.log_interval != 0:
            return

        elapsed = time.time() - self._step_start_time if self._step_start_time else 0

        loss = metrics.get("train/loss", 0)
        lr = metrics.get("train/lr", 0)
        grad_norm = metrics.get("train/grad_norm", 0)

        parts = [f"Step {step}:"]
        parts.append(f"loss={loss:.4f}")
        parts.append(f"lr={lr:.6f}")
        if grad_norm > 0:
            parts.append(f"grad_norm={grad_norm:.4f}")
        if elapsed > 0 and tokens_in_step > 0:
            tok_per_sec = tokens_in_step / elapsed
            parts.append(f"tok/s={tok_per_sec:.0f}")

        logger.info(" | ".join(parts))

    def on_stage_transition(self, old_stage: int, new_stage: int, step: int):
        logger.info(
            f"{'='*60}\n"
            f"CURRICULUM: Stage {old_stage} -> {new_stage} at step {step}\n"
            f"{'='*60}"
        )

    def on_health_report(self, report):
        if report.has_error:
            for msg in report.messages:
                logger.error(f"HEALTH ERROR: {msg}")
        elif report.has_warning:
            for msg in report.messages:
                logger.warning(f"HEALTH WARNING: {msg}")
