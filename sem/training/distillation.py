"""EMA self-distillation for SEM V5.5.

The Born Collapse sampler is inherently a "measurement" that destroys
quantum information (phase). Self-distillation via an EMA teacher
provides smoothed Born probabilities as soft targets, averaging over
multiple "measurement outcomes" — a direct quantum mechanical analog.

The EMA operates in Cartesian (real+imag) space on complex parameters.
At high decay rates (0.999+), teacher and student phases diverge by
at most (1-decay) * grad_step per update, which is << pi, so
Cartesian interpolation is safe (no phase wrapping issues).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EMATeacher:
    """Exponential Moving Average teacher model for self-distillation.

    Args:
        student_model: The student SEMModel to copy
        decay_start: Initial EMA decay rate
        decay_end: Final EMA decay rate (after ramp)
        decay_ramp_steps: Steps over which to linearly ramp decay
    """

    def __init__(
        self,
        student_model: nn.Module,
        decay_start: float = 0.999,
        decay_end: float = 0.9999,
        decay_ramp_steps: int = 10000,
    ):
        self.teacher = copy.deepcopy(student_model)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.decay_start = decay_start
        self.decay_end = decay_end
        self.decay_ramp_steps = decay_ramp_steps
        self.update_count = 0

    @property
    def decay(self) -> float:
        """Current EMA decay rate (linearly ramped)."""
        if self.update_count >= self.decay_ramp_steps:
            return self.decay_end
        progress = self.update_count / max(1, self.decay_ramp_steps)
        return self.decay_start + (self.decay_end - self.decay_start) * progress

    @decay.setter
    def decay(self, value: float):
        """Allow setting decay directly (for checkpoint restore)."""
        self.decay_start = value

    @torch.no_grad()
    def update(self, student_model: nn.Module):
        """Update teacher parameters via EMA.

        Uses Phase-Magnitude EMA for complex parameters to avoid destructive
        interference (magnitude collapse) during phase shifts.
        """
        d = self.decay
        alpha = 1 - d
        for t_param, s_param in zip(
            self.teacher.parameters(), student_model.parameters()
        ):
            if t_param.is_complex():
                m_s = s_param.data.abs()
                m_t = t_param.data.abs()

                m_new = torch.lerp(m_t, m_s, alpha)

                p_s = s_param.data / (m_s + 1e-12)
                p_t = t_param.data / (m_t + 1e-12)
                p_new = torch.lerp(p_t, p_s, alpha)
                p_new = p_new / (p_new.abs() + 1e-12)

                t_param.data.copy_(m_new * p_new)
            else:
                t_param.data.mul_(d).add_(s_param.data, alpha=alpha)

        self.update_count += 1

    def to(self, device):
        """Move teacher to device."""
        self.teacher = self.teacher.to(device)
        return self


class DistillationLoss:
    """Combined hard + soft loss for self-distillation.

    Loss = alpha * hard_loss + (1-alpha) * KL(teacher || student)

    Where hard_loss is the standard NLL from Born collapse,
    and soft_loss is KL divergence between teacher and student
    Born probabilities.

    Args:
        alpha: Weight on hard loss (1-alpha on soft loss)
        temperature: Temperature for soft target smoothing
    """

    def __init__(self, alpha: float = 0.7, temperature: float = 2.0):
        self.alpha = alpha
        self.temperature = temperature

    def compute(
        self,
        student_output: dict,
        teacher_model: nn.Module,
        token_ids: Tensor,
        targets: Tensor,
        token_freqs: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        """Compute combined distillation loss.

        Args:
            student_output: Dict from SEMModel.forward() with 'loss' and 'log_probs'
            teacher_model: EMA teacher SEMModel
            token_ids: [B, S] input tokens
            targets: [B, S] target tokens
            token_freqs: [B, V] or [V] EMA of token frequencies
        Returns:
            (total_loss, metrics_dict) tuple
        """
        hard_loss = student_output["loss"]

        # Get teacher predictions
        with torch.no_grad():
            teacher_output = teacher_model(token_ids, token_freqs=token_freqs)
            teacher_logits = teacher_output["logits"][:, :-1, :].contiguous()

        # Student logits (already computed)
        student_logits = student_output["logits"][:, :-1, :].contiguous()

        # Apply temperature scaling
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        # KL divergence (scaled by T^2 per Hinton et al.)
        soft_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (self.temperature**2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        metrics = {
            "distillation/hard_loss": hard_loss.item(),
            "distillation/soft_loss": soft_loss.item(),
            "distillation/total_loss": total_loss.item(),
            "distillation/alpha": self.alpha,
        }

        return total_loss, metrics
