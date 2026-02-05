"""Sinkhorn-Knopp algorithm for projecting onto the Birkhoff Polytope.

The Birkhoff Polytope is the set of doubly stochastic matrices (all rows/columns sum to 1).
This is the convex hull of all permutation matrices.

For mHC (Manifold-Constrained Hyper-Connections), we use Sinkhorn to project
the residual mixing matrix H_res onto the Birkhoff Polytope, which prevents
gradient explosion/collapse by preserving signal magnitude across depth.

Mathematical formulation:
    Given logits Z (unconstrained),
    Project onto Birkhoff: H = Sinkhorn(exp(Z / tau))
    Where H satisfies:
        H @ 1 = 1 (row stochastic)
        H^T @ 1 = 1 (column stochastic)

The Sinkhorn algorithm iteratively normalizes rows and columns in log-space:
    u_i = -log(sum_j exp(Z_ij / tau + v_j))
    v_j = -log(sum_i exp(Z_ij / tau + u_i))
    H = exp(Z / tau + u_i + v_j)

Convergence note:
    At low tau (< 0.1), the matrix is near-permutation and Sinkhorn converges
    slowly. Standard float32 logsumexp also hits a precision floor (~1e-3)
    because Z values reach 20-60+. We solve both by:
    1. Running iterations in float64 (eliminates precision floor)
    2. Checking convergence and iterating until both marginals < 5e-4
    3. Capping at 5000 iterations to bound worst-case runtime
"""

import torch
from torch import Tensor
from typing import Optional


def sinkhorn_log(
    logits: Tensor,
    num_iters: int = 10,
    tau: float = 0.05,
    eps: float = 1e-12,
) -> Tensor:
    """Project matrix onto Birkhoff Polytope via Sinkhorn-Knopp (log-domain).

    Uses float64 internally for numerical precision with low-temperature
    (peaked) matrices. Iterates until convergence or max iterations.

    Args:
        logits: Unconstrained logits [N, N] or [..., N, N]
        num_iters: Minimum number of Sinkhorn iterations
        tau: Temperature for softmax (lower = sharper, 0.05 typical)
        eps: Numerical stability epsilon (unused, kept for API compat)

    Returns:
        Doubly stochastic matrix H: [N, N] or [..., N, N]
        where H.sum(dim=-1) ≈ 1 and H.sum(dim=-2) ≈ 1
        Max deviation < 1e-3 for all rows and columns.

    Example:
        >>> logits = torch.randn(4, 4)
        >>> H = sinkhorn_log(logits, num_iters=10, tau=0.05)
        >>> print(H.sum(dim=-1))  # Should be ~[1, 1, 1, 1]
        >>> print(H.sum(dim=-2))  # Should be ~[1, 1, 1, 1]
    """
    device = logits.device
    dtype = logits.dtype

    # Upcast to float64 for precision. At tau=0.05, Z values reach 20*logits,
    # so logsumexp operates on values up to ~60. Float32 logsumexp accumulates
    # errors that create a ~1e-3 precision floor in the marginals. Float64
    # eliminates this floor entirely.
    Z = (logits / tau).to(torch.float64)

    # Initialize dual variables u, v
    u = torch.zeros(logits.shape[:-1], device=device, dtype=torch.float64)
    v = torch.zeros_like(u)

    # Convergence parameters. The tolerance of 5e-4 provides margin below
    # the 1e-3 target when casting back to float32.
    max_iters = 5000
    check_every = 50
    tol = 5e-4

    # Phase 1: Run at least num_iters (or check_every, whichever is larger)
    initial_block = max(num_iters, check_every)
    for _ in range(initial_block):
        u = -torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = -torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    # Check if already converged
    H = torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))
    row_err = (H.sum(dim=-1) - 1.0).abs().max().item()
    col_err = (H.sum(dim=-2) - 1.0).abs().max().item()

    # Phase 2: Continue with periodic convergence checking until converged
    if max(row_err, col_err) >= tol:
        for _ in range(initial_block, max_iters, check_every):
            for _ in range(check_every):
                u = -torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
                v = -torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

            H = torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))
            row_err = (H.sum(dim=-1) - 1.0).abs().max().item()
            col_err = (H.sum(dim=-2) - 1.0).abs().max().item()
            if max(row_err, col_err) < tol:
                break

    return H.to(dtype)


def sinkhorn_log_complex(
    logits_real: Tensor,
    logits_imag: Tensor,
    num_iters: int = 10,
    tau: float = 0.05,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Complex-valued variant of Sinkhorn for complex residual streams.

    Applies Sinkhorn to magnitude, preserves phase. Ensures doubly stochastic
    property holds for the magnitude while keeping phase information.

    Args:
        logits_real: Real part of logits [N, N] or [..., N, N]
        logits_imag: Imaginary part of logits [N, N] or [..., N, N]
        num_iters: Number of Sinkhorn iterations
        tau: Temperature
        eps: Numerical stability epsilon

    Returns:
        (H_real, H_imag): Doubly stochastic complex matrix
        where sqrt(H_real^2 + H_imag^2).sum(dim=-1) ≈ 1
    """
    # Compute magnitude (add eps for stability)
    logits_mag = torch.sqrt(logits_real**2 + logits_imag**2 + eps)

    # Project magnitude onto Birkhoff Polytope
    H_mag = sinkhorn_log(logits_mag, num_iters=num_iters, tau=tau, eps=eps)

    # Preserve phase, scale by doubly stochastic magnitude
    # Add eps to denominator for numerical stability
    phase = torch.atan2(logits_imag, logits_real + eps)
    H_real = H_mag * torch.cos(phase)
    H_imag = H_mag * torch.sin(phase)

    # No clamping -- magnitude is already doubly-stochastic from Sinkhorn, clamping would break it
    return H_real, H_imag
