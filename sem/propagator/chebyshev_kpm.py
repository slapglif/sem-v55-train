"""Kernel Polynomial Method (KPM) for Chebyshev expansion of the Cayley resolvent.

Implements the Chebyshev polynomial expansion of (I + iαH)^{-1} as an alternative
to the CG iterative solver. The KPM uses Jackson kernel damping to suppress Gibbs
oscillations, following Rev. Mod. Phys. 78, 275 (2006).

This approach is inspired by ChebyKAN (Chebyshev Kolmogorov–Arnold Networks) and
classical KPM from condensed matter physics. The Chebyshev recurrence is O(k·N)
where k is the polynomial degree and N is the system size, with no inner products
or convergence checks — making it fully deterministic and GPU-friendly.

All operations use real-block float32 (no complex dtypes) to maintain XPU
compatibility with the rest of the SEM propagator stack.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

import torch
from torch import Tensor


class _HasDiagonal(Protocol):
    def get_diagonal(self) -> Tensor: ...


def jackson_kernel(N: int, device: torch.device | None = None) -> Tensor:
    """Jackson kernel damping coefficients to suppress Gibbs oscillations.

    From Rev. Mod. Phys. 78, 275 (2006), Eq. (71)::

        g_m = ((N+1-m)cos(πm/(N+1)) + sin(πm/(N+1))/tan(π/(N+1))) / (N+1)

    The Jackson kernel provides optimal damping in the sense of minimising the
    integrated squared deviation while maintaining non-negativity of the
    resulting spectral density.

    Args:
        N: Number of Chebyshev moments (degree + 1).
        device: Target device for the output tensor.

    Returns:
        Tensor of shape ``[N]``, float32, Jackson kernel coefficients.
    """
    m = torch.arange(N, dtype=torch.float32, device=device)
    Np1 = float(N + 1)
    theta = math.pi / Np1
    cos_term = (Np1 - m) * torch.cos(m * theta)
    sin_term = torch.sin(m * theta) / math.tan(theta)
    g = (cos_term + sin_term) / Np1
    return g


def resolvent_chebyshev_coeffs(
    alpha: float,
    lambda_max: float,
    degree: int,
    device: torch.device | None = None,
) -> Tensor:
    """Compute Chebyshev expansion coefficients for the resolvent f(λ) = 1/(1 + iαλ).

    The function *f* is evaluated on the rescaled spectrum Ĥ = 2H/λ_max − I,
    which maps eigenvalues from [0, λ_max] to [−1, 1].  The original eigenvalue
    corresponding to rescaled *x* is::

        λ = λ_max · (x + 1) / 2

    so the target function on [−1, 1] is::

        f(x) = 1 / (1 + i · α · λ_max · (x + 1) / 2)

    Chebyshev coefficients are computed via the discrete cosine transform on
    Chebyshev nodes::

        x_k = cos(π(k + 0.5) / M)    for k = 0 … M−1
        c_n = (2/M) Σ_k f(x_k) cos(n·π(k + 0.5) / M)

    with *c_0* halved (standard DCT-II convention).  Jackson kernel damping is
    applied to suppress Gibbs oscillations.

    The result is returned as a real-block tensor ``[degree, 2]`` where the last
    dimension holds ``(real, imag)`` parts of each complex coefficient.

    Args:
        alpha: dt/2 — the half-timestep coefficient.
        lambda_max: Spectral radius of H (Gershgorin bound).
        degree: Number of Chebyshev terms.
        device: Target device for the output tensor.

    Returns:
        Tensor of shape ``[degree, 2]``, float32, Chebyshev coefficients in
        real-block form.
    """
    M = max(4 * degree, 128)

    k = torch.arange(M, dtype=torch.float32, device=device)
    x_k = torch.cos(math.pi * (k + 0.5) / M)

    # Resolvent in real-block: 1/(1+iαλ) = (1-iαλ)/(1+α²λ²), λ = λ_max·(x+1)/2
    lam = lambda_max * (x_k + 1.0) / 2.0
    a_lam = alpha * lam
    denom = 1.0 + a_lam * a_lam
    f_real = 1.0 / denom
    f_imag = -a_lam / denom

    # DCT-II on Chebyshev nodes: c_n = (2/M) Σ_k f(x_k) cos(nπ(k+0.5)/M)
    n = torch.arange(degree, dtype=torch.float32, device=device)
    angles = n.unsqueeze(1) * (math.pi * (k.unsqueeze(0) + 0.5) / M)
    cos_basis = torch.cos(angles)

    c_real = (2.0 / M) * (cos_basis @ f_real)
    c_imag = (2.0 / M) * (cos_basis @ f_imag)

    # Halve c_0 (standard DCT-II convention)
    c_real[0] = c_real[0] * 0.5
    c_imag[0] = c_imag[0] * 0.5

    g = jackson_kernel(degree, device=device)
    c_real = c_real * g
    c_imag = c_imag * g

    return torch.stack([c_real, c_imag], dim=-1)


def estimate_spectral_radius(hamiltonian: _HasDiagonal) -> float:
    """Estimate spectral radius of multi-scale Hamiltonian via Gershgorin circle theorem.

    For a Laplacian H = D − A, the Gershgorin bound gives λ_max ≤ 2 · max(degree).
    This is exact for the largest eigenvalue of graph Laplacians.

    Args:
        hamiltonian: ``MultiScaleHamiltonian`` or ``GraphLaplacianHamiltonian``
            instance (must expose ``.get_diagonal()`` returning a complex tensor
            whose real part is the degree vector).

    Returns:
        float: Upper bound on spectral radius.
    """
    diag = hamiltonian.get_diagonal().real  # type: ignore[union-attr]  # degree vector, shape [D]
    return float(2.0 * diag.max().item())


def chebyshev_kpm_solve(
    matvec_fn: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    rhs_r: Tensor,
    rhs_i: Tensor,
    coeffs: Tensor,
    lambda_max: float | Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Apply KPM Chebyshev expansion to approximate (I + iαH)^{−1} · rhs.

    Uses the 3-term Chebyshev recurrence in real-block form:

    .. math::

        T_0(\hat{H}) v &= v \\
        T_1(\hat{H}) v &= \hat{H} v \\
        T_{n+1}(\hat{H}) v &= 2 \hat{H} T_n(\hat{H}) v - T_{n-1}(\hat{H}) v

    where Ĥ rescales eigenvalues from [0, λ_max] to [−1, 1]::

        Ĥ · v = (2/λ_max) · H · v − v

    The result is accumulated as::

        out = Σ_{n=0}^{k−1} c_n · T_n(Ĥ) · rhs

    where *c_n* are complex coefficients in ``[k, 2]`` real-block form.
    Complex multiplication follows ``(a + ib)(v_r + iv_i) = (av_r − bv_i, av_i + bv_r)``.

    Args:
        matvec_fn: Callable ``(vr, vi) -> (Hvr, Hvi)``, applies the **unscaled**
            Hamiltonian H in real-block form.
        rhs_r: ``[B, S, D]`` float32, real part of the right-hand side.
        rhs_i: ``[B, S, D]`` float32, imaginary part of the right-hand side.
        coeffs: ``[k, 2]`` float32, Chebyshev coefficients ``(real, imag)``.
        lambda_max: Spectral radius for rescaling (Gershgorin bound).

    Returns:
        ``(out_r, out_i)``: tuple of ``[B, S, D]`` float32 tensors, the
        approximate resolvent applied to the RHS.
    """
    k = coeffs.shape[0]
    if isinstance(lambda_max, torch.Tensor):
        # SEOP Fix: Handle tensor lambda_max to avoid GPU-CPU sync
        lmax = torch.maximum(
            lambda_max,
            torch.tensor(1e-8, device=lambda_max.device, dtype=lambda_max.dtype),
        )
    else:
        lmax = max(lambda_max, 1e-8)
    scale = 2.0 / lmax

    # Ĥ·v = (2/λ_max)·H·v − v  (rescale eigenvalues to [-1, 1])
    def rescaled_matvec(vr: Tensor, vi: Tensor) -> tuple[Tensor, Tensor]:
        Hvr, Hvi = matvec_fn(vr, vi)
        return scale * Hvr - vr, scale * Hvi - vi

    # T_0 = rhs
    T_prev_r, T_prev_i = rhs_r, rhs_i

    c0_re, c0_im = coeffs[0, 0], coeffs[0, 1]
    out_r = c0_re * T_prev_r - c0_im * T_prev_i
    out_i = c0_re * T_prev_i + c0_im * T_prev_r

    if k == 1:
        return out_r, out_i

    # T_1 = Ĥ · rhs
    T_curr_r, T_curr_i = rescaled_matvec(T_prev_r, T_prev_i)

    c1_re, c1_im = coeffs[1, 0], coeffs[1, 1]
    out_r = out_r + c1_re * T_curr_r - c1_im * T_curr_i
    out_i = out_i + c1_re * T_curr_i + c1_im * T_curr_r

    # 3-term recurrence: T_{n+1} = 2·Ĥ·T_n − T_{n-1}
    for n in range(2, k):
        Ht_r, Ht_i = rescaled_matvec(T_curr_r, T_curr_i)
        T_next_r = 2.0 * Ht_r - T_prev_r
        T_next_i = 2.0 * Ht_i - T_prev_i

        cn_re, cn_im = coeffs[n, 0], coeffs[n, 1]
        out_r = out_r + cn_re * T_next_r - cn_im * T_next_i
        out_i = out_i + cn_re * T_next_i + cn_im * T_next_r

        T_prev_r, T_prev_i = T_curr_r, T_curr_i
        T_curr_r, T_curr_i = T_next_r, T_next_i

    return out_r, out_i
