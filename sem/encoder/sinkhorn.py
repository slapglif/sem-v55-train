"""Log-domain Sinkhorn optimal transport for MESH-SDR encoder.

Implements differentiable Sinkhorn iterations in the log domain
for numerical stability. At epsilon=0.05 with S=2048, raw-domain
Sinkhorn would overflow -- log-domain is mandatory.

The Sinkhorn algorithm solves the entropy-regularized OT problem:
    min_T <C, T> - epsilon * H(T)
    s.t. T @ 1 = r, T^T @ 1 = c
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import math


class LogSinkhorn(nn.Module):
    """Log-domain Sinkhorn algorithm for optimal transport.

    Solves entropy-regularized OT: min_T <C,T> - eps*H(T)
    subject to marginal constraints.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        max_iter: int = 50,
        tol: float = 1e-3,
        auto_epsilon: bool = False,
        auto_epsilon_scale: float = 0.05,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.auto_epsilon = auto_epsilon
        self.auto_epsilon_scale = auto_epsilon_scale

    def forward(
        self,
        cost: Tensor,
        source_marginal: Optional[Tensor] = None,
        target_marginal: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute optimal transport plan via log-domain Sinkhorn.

        Args:
            cost: Cost matrix [B, N, M] (non-negative)
            source_marginal: Source distribution [B, N] (default: uniform)
            target_marginal: Target distribution [B, M] (default: uniform)

        Returns:
            Transport plan T: [B, N, M] (doubly stochastic up to marginals)
        """
        B, N, M = cost.shape
        device = cost.device
        dtype = cost.dtype

        # Default uniform marginals
        if source_marginal is None:
            log_r = torch.full((B, N), -math.log(N), device=device, dtype=dtype)
        else:
            log_r = (source_marginal + 1e-12).log()

        if target_marginal is None:
            log_c = torch.full((B, M), -math.log(M), device=device, dtype=dtype)
        else:
            log_c = (target_marginal + 1e-12).log()

        # Cost-aware epsilon: scale to median of cost matrix per batch element
        if self.auto_epsilon:
            cost_flat = cost.reshape(B, -1)
            median_cost = cost_flat.median(dim=-1).values.clamp(min=1e-6)  # [B]
            eps = (
                (self.auto_epsilon_scale * median_cost).unsqueeze(-1).unsqueeze(-1)
            )  # [B,1,1]
            log_K = -cost / eps
        else:
            log_K = -cost / self.epsilon  # [B, N, M]

        # Initialize dual variables
        log_u = torch.zeros(B, N, device=device, dtype=dtype)  # [B, N]
        log_v = torch.zeros(B, M, device=device, dtype=dtype)  # [B, M]

        # PERF: Run fixed iteration count — no early-exit convergence check.
        # The old code called `(log_u - log_u_prev).abs().max()` which triggers
        # an implicit .item() GPU-CPU sync every iteration (up to max_iter syncs
        # per micro-batch). For 16 micro-batches that's 16*max_iter pipeline stalls.
        for iteration in range(self.max_iter):
            # Update u: log_u = log_r - logsumexp(log_K + log_v[None,:], dim=-1)
            log_u = log_r - torch.logsumexp(log_K + log_v.unsqueeze(-2), dim=-1)

            # Update v: log_v = log_c - logsumexp(log_K + log_u[:, None], dim=-2)
            log_v = log_c - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=-2)

        # Recover transport plan: T = diag(u) @ K @ diag(v)
        # log T = log_u[:, :, None] + log_K + log_v[:, None, :]
        log_T = log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(-2)
        T = log_T.exp()

        return T
