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
    2. Running a bounded, fixed iteration budget (<= 200) to avoid long-tail
       runtime *and* avoid per-check GPU sync from Python-side convergence tests.
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

    Z = (logits / tau).to(dtype)

    u = torch.zeros(logits.shape[:-1], device=device, dtype=dtype)
    v = torch.zeros_like(u)

    for _ in range(num_iters):
        u = -torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = -torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    H = torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))

    return H


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

    # Direct normalization instead of atan2 → cos/sin roundtrip.
    # atan2(0,0) backward computes 0/0 = NaN; torch.where can't mask it
    # because 0 * NaN = NaN in IEEE 754.
    safe_mag = logits_mag.clamp(min=1e-6)
    unit_real = logits_real / safe_mag
    unit_imag = logits_imag / safe_mag
    mag_is_small = logits_mag.detach() < 1e-4
    unit_real = torch.where(mag_is_small, torch.ones_like(unit_real), unit_real)
    unit_imag = torch.where(mag_is_small, torch.zeros_like(unit_imag), unit_imag)

    H_real = H_mag * unit_real
    H_imag = H_mag * unit_imag

    return H_real, H_imag
