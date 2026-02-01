"""Curriculum learning for SEM V5.5 training.

Manages progressive training stages that increase data quality
and sequence length, allowing the model's convergence-sensitive
components (Sinkhorn encoder, CG solver) to stabilize before
scaling up.

Stages are defined by:
- min_score: FineWeb-Edu education quality threshold
- seq_len: Sequence length for packing
- min_steps: Minimum steps before transition is allowed
"""

import logging
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)


class CurriculumManager:
    """Manages curriculum learning stage transitions.

    Args:
        stages: List of stage configs [{"min_score": int, "seq_len": int, "min_steps": int}]
        transition_check_interval: Steps between transition checks
        loss_plateau_threshold: Max relative change in rolling loss to consider plateau
        loss_plateau_window: Number of steps for rolling loss average
        unitary_stability_threshold: Max average unitary divergence to allow transition
        lr_decay_per_stage: LR multiplier when advancing stage
        stage_warmup_steps: Warmup steps for new stage's WSD cycle
    """

    def __init__(
        self,
        stages: List[Dict],
        transition_check_interval: int = 500,
        loss_plateau_threshold: float = 0.01,
        loss_plateau_window: int = 1000,
        unitary_stability_threshold: float = 0.05,
        lr_decay_per_stage: float = 0.7,
        stage_warmup_steps: int = 500,
    ):
        self.stages = stages
        self.transition_check_interval = transition_check_interval
        self.loss_plateau_threshold = loss_plateau_threshold
        self.loss_plateau_window = loss_plateau_window
        self.unitary_stability_threshold = unitary_stability_threshold
        self.lr_decay_per_stage = lr_decay_per_stage
        self.stage_warmup_steps = stage_warmup_steps

        self.current_stage = 0
        self.stage_start_step = 0
        self.loss_history = deque(maxlen=loss_plateau_window)
        self.unitary_history = deque(maxlen=loss_plateau_window)
        self.transition_history = []  # List of (stage, step) tuples

    @property
    def current_config(self) -> Dict:
        """Get the current stage configuration."""
        return self.stages[self.current_stage]

    @property
    def min_score(self) -> int:
        return self.current_config["min_score"]

    @property
    def seq_len(self) -> int:
        return self.current_config["seq_len"]

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage >= len(self.stages) - 1

    def record_metrics(self, loss: float, unitary_divergence: float):
        """Record training metrics for transition checks."""
        self.loss_history.append(loss)
        self.unitary_history.append(unitary_divergence)

    def should_check_transition(self, global_step: int) -> bool:
        """Whether it's time to check for stage transition."""
        if self.is_final_stage:
            return False
        return global_step % self.transition_check_interval == 0

    def check_transition(self, global_step: int, health_ok: bool = True) -> bool:
        """Check if conditions are met to advance to next stage.

        Args:
            global_step: Current training step
            health_ok: Whether health metrics are within bounds
        Returns:
            True if transition should occur
        """
        if self.is_final_stage:
            return False

        # Check minimum steps
        steps_in_stage = global_step - self.stage_start_step
        min_steps = self.current_config.get("min_steps", 0)
        if steps_in_stage < min_steps:
            return False

        # Check health
        if not health_ok:
            return False

        # Check loss plateau
        if len(self.loss_history) < self.loss_plateau_window // 2:
            return False  # Not enough data

        losses = list(self.loss_history)
        first_half = sum(losses[: len(losses) // 2]) / (len(losses) // 2)
        second_half = sum(losses[len(losses) // 2 :]) / (len(losses) - len(losses) // 2)

        if first_half == 0:
            return False

        relative_change = abs(second_half - first_half) / abs(first_half)
        plateau_met = relative_change < self.loss_plateau_threshold

        if plateau_met:
            logger.info(
                f"Loss plateau detected: relative change {relative_change:.4f} "
                f"< threshold {self.loss_plateau_threshold}"
            )

        # Check unitary stability
        unitary_avg = sum(self.unitary_history) / len(self.unitary_history)
        unitary_met = unitary_avg < self.unitary_stability_threshold

        if unitary_met:
            logger.info(
                f"Unitary stability detected: avg divergence {unitary_avg:.4f} "
                f"< threshold {self.unitary_stability_threshold}"
            )

        return plateau_met and unitary_met

    def advance_stage(self, global_step: int) -> Dict:
        """Advance to the next curriculum stage.

        Args:
            global_step: Current training step
        Returns:
            New stage config dict
        """
        old_stage = self.current_stage
        self.current_stage += 1
        self.stage_start_step = global_step
        self.loss_history.clear()
        self.unitary_history.clear()
        self.transition_history.append((self.current_stage, global_step))

        logger.info(
            f"Curriculum: Stage {old_stage} -> {self.current_stage} at step {global_step}. "
            f"New config: min_score={self.min_score}, seq_len={self.seq_len}"
        )

        return self.current_config

    def state_dict(self) -> dict:
        return {
            "current_stage": self.current_stage,
            "stage_start_step": self.stage_start_step,
            "loss_history": list(self.loss_history),
            "unitary_history": list(self.unitary_history),
            "transition_history": self.transition_history,
        }

    def load_state_dict(self, state: dict):
        self.current_stage = state["current_stage"]
        self.stage_start_step = state["stage_start_step"]
        self.loss_history = deque(
            state.get("loss_history", []), maxlen=self.loss_plateau_window
        )
        self.unitary_history = deque(
            state.get("unitary_history", []), maxlen=self.loss_plateau_window
        )
        self.transition_history = state.get("transition_history", [])
