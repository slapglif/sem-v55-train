"""Empirical Fisher Information tracker for HAS-VQ.

Tracks the diagonal of the Fisher Information Matrix using an
exponential moving average of squared gradients:
    F_ema = decay * F_ema + (1 - decay) * grad^2

For complex parameters, tracks real and imaginary parts separately.
High-Fisher parameters contain critical knowledge and should be
stored at high precision (FP16/INT8).
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict


class FisherTracker:
    """Tracks empirical Fisher information for all model parameters.

    Used by HAS-VQ to identify which parameters are "outliers"
    (high Fisher info = critical knowledge) vs "bulk" (low Fisher
    info = compressible).
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.fisher_ema: Dict[str, Tensor] = {}
        self.step_count = 0

    def update(self, named_parameters):
        """Update Fisher EMA with current gradients.

        Call after loss.backward() but before optimizer.step().

        Args:
            named_parameters: Iterator of (name, param) pairs
        """
        self.step_count += 1

        for name, param in named_parameters:
            if param.grad is None:
                continue

            grad = param.grad

            # For complex params, compute Fisher on real/imag separately
            if param.is_complex():
                grad_real = torch.view_as_real(grad)
                grad_sq = grad_real ** 2
            else:
                grad_sq = grad ** 2

            if name not in self.fisher_ema:
                self.fisher_ema[name] = grad_sq.detach().clone()
            else:
                self.fisher_ema[name].mul_(self.decay).add_(
                    grad_sq.detach(), alpha=1 - self.decay
                )

    def get_fisher(self, name: str) -> Tensor:
        """Get Fisher information for a named parameter."""
        return self.fisher_ema.get(name, None)

    def get_outlier_mask(self, name: str, percentile: float = 0.01) -> Tensor:
        """Get mask of high-Fisher "outlier" parameters.

        Args:
            name: Parameter name
            percentile: Top fraction to mark as outliers (e.g., 0.01 = top 1%)
        Returns:
            Boolean mask (True = outlier)
        """
        fisher = self.fisher_ema.get(name)
        if fisher is None:
            return None

        threshold = torch.quantile(fisher.flatten().float(), 1.0 - percentile)
        return fisher >= threshold

    def state_dict(self) -> dict:
        return {
            'fisher_ema': {k: v.cpu() for k, v in self.fisher_ema.items()},
            'step_count': self.step_count,
            'decay': self.decay,
        }

    def load_state_dict(self, state: dict):
        self.fisher_ema = {k: v for k, v in state['fisher_ema'].items()}
        self.step_count = state['step_count']
        self.decay = state['decay']
