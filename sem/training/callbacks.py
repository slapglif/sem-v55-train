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

    def __init__(
        self,
        project: str,
        config: dict = None,
        resume_id: Optional[str] = None,
        enabled: bool = True,
    ):
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

    def __init__(self, log_interval: int = 10, timing_log_interval: int = 10):
        self.log_interval = log_interval
        self.timing_log_interval = timing_log_interval
        self._step_start_time = None
        self._tokens_processed = 0

    def on_step_start(self):
        self._step_start_time = time.time()

    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
        tokens_in_step: int = 0,
        timings: Optional[Dict] = None,
    ):
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

        if timings and step % self.timing_log_interval == 0:
            timing_parts = []
            if "step_total" in timings:
                timing_parts.append(f"total={timings['step_total'] * 1000:.1f}ms")

            for key, label in [
                ("batch_load", "batch"),
                ("device_transfer", "xfer"),
                ("forward", "fwd"),
                ("loss_compute", "loss"),
                ("backward", "bwd"),
            ]:
                vals = timings.get(key, [])
                if vals:
                    avg_ms = sum(vals) / len(vals) * 1000
                    timing_parts.append(f"{label}={avg_ms:.1f}ms")

            for key, label in [
                ("grad_clip", "clip"),
                ("fisher_update", "fisher"),
                ("optimizer_step", "opt"),
                ("ema_update", "ema"),
                ("health_check", "health"),
                ("curriculum_transition", "curric"),
                ("checkpoint_save", "ckpt"),
            ]:
                if key in timings and timings[key]:
                    timing_parts.append(f"{label}={timings[key] * 1000:.1f}ms")

            if timing_parts:
                logger.info(f"  Timing: {' | '.join(timing_parts)}")

            prop_timings = timings.get("propagator", {})
            if prop_timings:
                prop_parts = []
                for key, label in [
                    ("cache_weights", "cache"),
                    ("rhs_matvec", "rhs"),
                    ("lazy_gate", "gate"),
                    ("cg_solve", "cg"),
                    ("total", "prop_total"),
                ]:
                    vals = prop_timings.get(key, [])
                    if isinstance(vals, list) and vals:
                        avg_ms = sum(vals) / len(vals) * 1000
                        prop_parts.append(f"{label}={avg_ms:.1f}ms")
                    elif isinstance(vals, (int, float)) and vals:
                        prop_parts.append(f"{label}={vals * 1000:.1f}ms")
                cg_iters = prop_timings.get("cg_iterations", [])
                if cg_iters:
                    avg_iters = sum(cg_iters) / len(cg_iters)
                    prop_parts.append(f"cg_iters={avg_iters:.1f}")
                skips = prop_timings.get("cg_skips", 0)
                total = prop_timings.get("cg_calls", 0)
                if total:
                    prop_parts.append(f"cg_skip={skips}/{total}")
                cache_hits = prop_timings.get("cache_hits", 0)
                if cache_hits:
                    prop_parts.append(f"cache_hit={cache_hits}")
                if prop_parts:
                    logger.info(f"  Propagator: {' | '.join(prop_parts)}")

            vram_alloc = timings.get("vram_allocated_mb")
            if vram_alloc is not None:
                vram_peak = timings.get("vram_peak_mb", 0)
                vram_reserved = timings.get("vram_reserved_mb", 0)
                logger.info(
                    f"  VRAM: alloc={vram_alloc:.0f}MB | peak={vram_peak:.0f}MB | reserved={vram_reserved:.0f}MB"
                )

    def on_stage_transition(self, old_stage: int, new_stage: int, step: int):
        logger.info(
            f"{'=' * 60}\n"
            f"CURRICULUM: Stage {old_stage} -> {new_stage} at step {step}\n"
            f"{'=' * 60}"
        )

    def on_health_report(self, report):
        if report.has_error:
            for msg in report.messages:
                logger.error(f"HEALTH ERROR: {msg}")
        elif report.has_warning:
            for msg in report.messages:
                logger.warning(f"HEALTH WARNING: {msg}")
