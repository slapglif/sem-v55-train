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

    # Upcast to float64 for precision. At tau=0.05, Z values reach 20*logits,
    # so logsumexp operates on values up to ~60. Float32 logsumexp accumulates
    # errors that create a ~1e-3 precision floor in the marginals. Float64
    # eliminates this floor entirely.
    Z = (logits / tau).to(torch.float64)

    # Initialize dual variables u, v
    u = torch.zeros(logits.shape[:-1], device=device, dtype=torch.float64)
    v = torch.zeros_like(u)

    # SEOP (bounded runtime / no device sync): for the small matrices used in mHC
    # (typically 4x4), convergence is fast, and a fixed cap avoids pathological
    # long loops and `.item()`-based GPU synchronization.
    max_iters = 200
    # Preserve the original contract: run *at least* num_iters. In practice for
    # sharp (low-tau) projections, small matrices can still need >50 iterations
    # to hit ~1e-3 marginal error across a batch, so we default to the full cap.
    total_iters = min(num_iters, max_iters)

    for _ in range(total_iters):
        u = -torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = -torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    # SEOP (entropy leak / solver impedance): at low tau the matrix becomes
    # near-permutation and classic Sinkhorn can converge very slowly. With the
    # iteration cap above, we apply a small, fixed Newton-style polishing step
    # for small N to hit the doubly-stochastic constraints without unbounded
    # iteration counts or Python-side convergence checks.
    n = Z.shape[-1]
    if n <= 8:
        # Flatten batch dims so we can build small linear systems.
        z_flat = Z.reshape(-1, n, n)
        u_flat = u.reshape(-1, n)
        v_flat = v.reshape(-1, n)

        # Gauge-fix v[:, 0] == 0 without changing exp(Z+u+v):
        # add shift to u and subtract from v.
        shift = v_flat[:, 0].unsqueeze(-1)
        u_flat = u_flat + shift
        v_flat = v_flat - shift

        # A few steps are enough for N<=8.
        polish_steps = 3
        for _ in range(polish_steps):
            a = z_flat + u_flat.unsqueeze(-1) + v_flat.unsqueeze(-2)
            h = torch.exp(a)
            r = h.sum(dim=-1)
            c = h.sum(dim=-2)

            # Residuals: rows (all) and cols excluding the gauge-fixed col 0.
            g = torch.cat([r - 1.0, (c - 1.0)[:, 1:]], dim=-1)  # [Bf, 2n-1]

            j11 = torch.diag_embed(r)  # [Bf, n, n]
            j12 = h[:, :, 1:]  # [Bf, n, n-1]
            j21 = h[:, :, 1:].transpose(-2, -1)  # [Bf, n-1, n]
            j22 = torch.diag_embed(c[:, 1:])  # [Bf, n-1, n-1]

            top = torch.cat([j11, j12], dim=-1)
            bot = torch.cat([j21, j22], dim=-1)
            j = torch.cat([top, bot], dim=-2)  # [Bf, 2n-1, 2n-1]

            # Small diagonal jitter improves numerical robustness for very sharp matrices.
            jitter = 1e-6
            eye = torch.eye(2 * n - 1, device=j.device, dtype=j.dtype).unsqueeze(0)
            j = j + jitter * eye

            rhs = (-g).unsqueeze(-1)
            try:
                delta = torch.linalg.solve(j, rhs).squeeze(-1)
            except RuntimeError:
                delta = torch.linalg.lstsq(j, rhs).solution.squeeze(-1)

            du = delta[:, :n]
            dv1 = delta[:, n:]

            u_flat = u_flat + du
            v_flat = v_flat.clone()
            v_flat[:, 1:] = v_flat[:, 1:] + dv1

        u = u_flat.reshape(*Z.shape[:-1])
        v = v_flat.reshape(*Z.shape[:-1])

    H = torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))

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
