"""Complex-aware AdamW optimizer.

PyTorch's built-in AdamW works with complex parameters via Wirtinger
calculus, but we provide explicit handling for better numerical behavior
with phase-encoded representations.
"""

import torch
from torch.optim import Optimizer
from torch import Tensor
import math


class ComplexAdamW(Optimizer):
    """AdamW optimizer with explicit complex parameter support.

    Handles complex parameters by operating on real and imaginary
    parts with coupled momentum, ensuring phase-coherent updates.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        processed = set()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p in processed or p.grad is None:
                    continue

                # Handle Real-Block Isomorphism pairs (weight_real/weight_imag)
                if hasattr(p, "_complex_partner"):
                    partner = p._complex_partner
                    if partner.grad is not None:
                        p_real = p if getattr(p, "_is_complex_real", False) else partner
                        p_imag = (
                            partner
                            if getattr(partner, "_is_complex_imag", False)
                            else p
                        )

                        self._coupled_adam_step(
                            p_real, p_imag, lr, beta1, beta2, eps, weight_decay
                        )
                        processed.add(p_real)
                        processed.add(p_imag)
                        continue

                # Existing complex64 support
                if p.is_complex():
                    # For complex params, work with the view_as_real representation
                    # This ensures coupled real/imag momentum
                    p_view = torch.view_as_real(p)
                    g_view = torch.view_as_real(p.grad)
                    self._adam_step(
                        p_view,
                        g_view,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        p,
                        coupled=True,
                    )
                else:
                    self._adam_step(p, p.grad, lr, beta1, beta2, eps, weight_decay, p)

                processed.add(p)

        return loss

    def _coupled_adam_step(
        self,
        p_real: Tensor,
        p_imag: Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ):
        """AdamW step for a pair of real/imag parameters with coupled momentum."""
        grad_real, grad_imag = p_real.grad, p_imag.grad
        state_real = self.state[p_real]
        state_imag = self.state[p_imag]

        # Initialize states if needed
        if len(state_real) == 0:
            state_real["step"] = 0
            state_real["exp_avg"] = torch.zeros_like(p_real)
            state_real["exp_avg_sq"] = torch.zeros_like(p_real)

        if len(state_imag) == 0:
            state_imag["step"] = 0
            state_imag["exp_avg"] = torch.zeros_like(p_imag)
            # Share the exp_avg_sq tensor to ensure coupled behavior
            state_imag["exp_avg_sq"] = state_real["exp_avg_sq"]

        state_real["step"] += 1
        state_imag["step"] = state_real["step"]
        step = state_real["step"]

        # Decoupled weight decay
        if weight_decay != 0:
            p_real.mul_(1 - lr * weight_decay)
            p_imag.mul_(1 - lr * weight_decay)

        # Momentum updates (m_t)
        state_real["exp_avg"].mul_(beta1).add_(grad_real, alpha=1 - beta1)
        state_imag["exp_avg"].mul_(beta1).add_(grad_imag, alpha=1 - beta1)

        # Coupled Second Momentum update (v_t = beta2 * v_{t-1} + (1-beta2) * |g|^2)
        # Note: |g|^2 = grad_real^2 + grad_imag^2
        # Since state_imag['exp_avg_sq'] is the same tensor as state_real['exp_avg_sq'],
        # we only update it once.
        grad_sq = grad_real.pow(2) + grad_imag.pow(2)
        state_real["exp_avg_sq"].mul_(beta2).add_(grad_sq, alpha=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1

        # Denominator uses shared v_t
        denom = (state_real["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(
            eps
        )

        # Apply updates
        p_real.addcdiv_(state_real["exp_avg"], denom, value=-step_size)
        p_imag.addcdiv_(state_imag["exp_avg"], denom, value=-step_size)

    def _adam_step(
        self,
        param_view: Tensor,
        grad_view: Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        original_param: Tensor,
        coupled: bool = False,
    ):
        """Core AdamW step on a (possibly real view of complex) parameter."""
        state = self.state[original_param]

        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad_view)
            state["exp_avg_sq"] = torch.zeros_like(grad_view)
            if coupled:
                # If coupled, exp_avg_sq will store a single value per complex pair
                # Shape (..., 2) -> (..., 1) for broadcast or just keep (..., 2) and fill both
                pass

        state["step"] += 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step = state["step"]

        # Decoupled weight decay
        if weight_decay != 0:
            param_view.mul_(1 - lr * weight_decay)

        # Momentum updates
        exp_avg.mul_(beta1).add_(grad_view, alpha=1 - beta1)

        if coupled:
            # For complex params, we want phase-coherent updates: share v_t between re/im
            # |g|^2 = sum(g_i^2) across the last dimension (re/im)
            grad_sq = grad_view.pow(2).sum(dim=-1, keepdim=True)
            exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)
            # exp_avg_sq is (..., 1), will broadcast to (..., 2) in addcdiv
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad_view, grad_view, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        param_view.addcdiv_(exp_avg, denom, value=-step_size)
