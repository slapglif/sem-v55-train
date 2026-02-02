"""Thermodynamic Complex-aware AdamW optimizer.

Integrates principles from Thermodynamic Natural Gradient Descent (TNGD)
by treating the optimization as an Ornstein-Uhlenbeck process at
equilibrium.

Ref: https://www.nature.com/articles/s44335-025-00049-x
"""

import torch
from torch.optim import Optimizer
from torch import Tensor
import math
from typing import Optional


class ComplexAdamW(Optimizer):
    """Thermodynamic AdamW with explicit complex parameter support.

    Implements a hybrid digital-analog inspired optimization loop:
    1. Preconditions gradients with the diagonal Fisher Information Matrix (Adam v_t).
    2. Inject thermodynamic noise (Langevin dynamics) to sample the
       equilibrium distribution of the natural gradient.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        temperature: float = 1e-4,  # Thermodynamic noise scale
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            temperature=temperature,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = float(closure())

        processed = set()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            temperature = group["temperature"]

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

                        self._coupled_thermodynamic_step(
                            p_real,
                            p_imag,
                            lr,
                            beta1,
                            beta2,
                            eps,
                            weight_decay,
                            temperature,
                        )
                        processed.add(p_real)
                        processed.add(p_imag)
                        continue

                # Existing complex64 support
                if p.is_complex():
                    p_view = torch.view_as_real(p)
                    g_view = torch.view_as_real(p.grad)
                    self._thermodynamic_step(
                        p_view,
                        g_view,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        temperature,
                        p,
                        coupled=True,
                    )
                else:
                    self._thermodynamic_step(
                        p, p.grad, lr, beta1, beta2, eps, weight_decay, temperature, p
                    )

                processed.add(p)

        return loss

    def _coupled_thermodynamic_step(
        self,
        p_real: Tensor,
        p_imag: Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        temperature: float,
    ):
        """AdamW step with coupled momentum and thermodynamic noise."""
        grad_real, grad_imag = p_real.grad, p_imag.grad
        if grad_real is None or grad_imag is None:
            return

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

        # Coupled Second Momentum update (diagonal Fisher estimate)
        grad_sq = grad_real.pow(2) + grad_imag.pow(2)
        state_real["exp_avg_sq"].mul_(beta2).add_(grad_sq, alpha=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr / bias_correction1

        # Denominator (Preconditioner)
        denom = (state_real["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(
            eps
        )

        # Apply updates (Natural Gradient Step)
        p_real.addcdiv_(state_real["exp_avg"], denom, value=-step_size)
        p_imag.addcdiv_(state_imag["exp_avg"], denom, value=-step_size)

        # TNGD Factoring: Inject Thermodynamic Noise (Langevin Dynamics)
        if temperature > 0:
            noise_scale = math.sqrt(2 * temperature * lr)
            noise_r = torch.randn_like(p_real) * noise_scale
            noise_i = torch.randn_like(p_imag) * noise_scale

            # Stay in natural gradient geometry
            p_real.add_(noise_r / (denom.sqrt() + 1e-8))
            p_imag.add_(noise_i / (denom.sqrt() + 1e-8))

    def _thermodynamic_step(
        self,
        param_view: Tensor,
        grad_view: Tensor,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
        temperature: float,
        original_param: Tensor,
        coupled: bool = False,
    ):
        """Core AdamW step with thermodynamic noise."""
        state = self.state[original_param]

        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(grad_view)
            state["exp_avg_sq"] = torch.zeros_like(grad_view)

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
            grad_sq = grad_view.pow(2).sum(dim=-1, keepdim=True)
            exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad_view, grad_view, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        param_view.addcdiv_(exp_avg, denom, value=-step_size)

        # TNGD Noise injection
        if temperature > 0:
            noise_scale = math.sqrt(2 * temperature * lr)
            noise = torch.randn_like(param_view) * noise_scale
            param_view.add_(noise / (denom.sqrt() + 1e-8))
