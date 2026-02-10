"""Tests for KPM Chebyshev propagator module."""

import pytest
import time
import torch
from sem.utils.complex_ops import safe_complex


def test_jackson_kernel_properties():
    """Jackson kernel should have expected mathematical properties."""
    from sem.propagator.chebyshev_kpm import jackson_kernel

    g = jackson_kernel(16)
    assert g.shape == (16,)
    assert g.dtype == torch.float32
    # g[0] should be close to 1.0 (first coefficient)
    assert abs(g[0].item() - 1.0) < 0.05
    # All coefficients should be finite
    assert torch.all(torch.isfinite(g))
    # Coefficients should generally decrease in magnitude
    assert g[0].item() > g[-1].item()


def test_resolvent_coeffs_shape_and_dtype():
    """Resolvent coefficients should have correct shape and dtype."""
    from sem.propagator.chebyshev_kpm import resolvent_chebyshev_coeffs

    coeffs = resolvent_chebyshev_coeffs(alpha=0.05, lambda_max=10.0, degree=12)
    assert coeffs.shape == (12, 2)  # [degree, (real, imag)]
    assert coeffs.dtype == torch.float32
    assert torch.all(torch.isfinite(coeffs))
    # c_0 real part should be positive (resolvent has positive DC component)
    assert coeffs[0, 0].item() > 0


def test_kpm_solve_matches_direct():
    """KPM solve should approximate direct solve within reasonable tolerance."""
    from sem.propagator.chebyshev_kpm import (
        chebyshev_kpm_solve,
        resolvent_chebyshev_coeffs,
    )
    from sem.propagator.hamiltonian import MultiScaleHamiltonian

    torch.manual_seed(42)
    D = 64
    B, S = 2, 16
    dt = 0.1
    alpha = dt / 2.0

    ham = MultiScaleHamiltonian(D, num_scales=2, base_sparsity=3)
    ham.cache_weights()

    # Get dense H for direct solve reference
    H = ham.get_hamiltonian_dense()
    I_mat = torch.eye(D, dtype=H.dtype, device=H.device)
    A_plus = I_mat + 1j * alpha * H

    # Random RHS
    psi = torch.randn(B, S, D, dtype=torch.complex64)
    rhs = safe_complex(torch.randn(B, S, D), torch.randn(B, S, D))
    rhs_r, rhs_i = rhs.real, rhs.imag

    # Direct solve reference
    rhs_flat = rhs.reshape(-1, D)
    x_direct = torch.linalg.solve(
        A_plus.to(torch.complex128), rhs_flat.to(torch.complex128).T
    ).T.to(torch.complex64)
    x_direct = x_direct.reshape(B, S, D)

    # Gershgorin bound
    diag = ham.get_diagonal().real
    lmax = float(2.0 * diag.max().item())
    lmax = max(lmax, 1e-8)

    # KPM solve
    coeffs = resolvent_chebyshev_coeffs(alpha=alpha, lambda_max=lmax, degree=16)
    out_r, out_i = chebyshev_kpm_solve(
        matvec_fn=ham.matvec_real_fused,
        rhs_r=rhs_r,
        rhs_i=rhs_i,
        coeffs=coeffs,
        lambda_max=lmax,
    )
    x_kpm = safe_complex(out_r, out_i)

    ham.clear_cache()

    # Relative error should be < 5% (KPM is approximate)
    rel_error = (x_kpm - x_direct).norm() / (x_direct.norm() + 1e-12)
    assert rel_error.item() < 0.05, (
        f"KPM relative error {rel_error.item():.4f} exceeds 5%"
    )


def test_kpm_unitarity():
    """KPM propagation should approximately preserve norm."""
    from sem.propagator.chebyshev_kpm import (
        chebyshev_kpm_solve,
        resolvent_chebyshev_coeffs,
    )
    from sem.propagator.hamiltonian import MultiScaleHamiltonian

    torch.manual_seed(42)
    D = 64
    B, S = 2, 32
    dt = 0.1
    alpha = dt / 2.0

    ham = MultiScaleHamiltonian(D, num_scales=2, base_sparsity=3)
    ham.cache_weights()

    # Build RHS from the Cayley Crank-Nicolson: rhs = A_minus @ psi
    # where A_minus = I - i*alpha*H
    psi = torch.randn(B, S, D, dtype=torch.complex64)
    psi_r, psi_i = psi.real, psi.imag
    H_r, H_i = ham.matvec_real_fused(psi_r, psi_i)
    # A_minus = I - i*alpha*H, in real-block: (vr + alpha*Hvi, vi - alpha*Hvr)
    rhs_r = psi_r + alpha * H_i
    rhs_i = psi_i - alpha * H_r

    diag = ham.get_diagonal().real
    lmax = float(2.0 * diag.max().item())
    lmax = max(lmax, 1e-8)

    coeffs = resolvent_chebyshev_coeffs(alpha=alpha, lambda_max=lmax, degree=16)
    out_r, out_i = chebyshev_kpm_solve(
        matvec_fn=ham.matvec_real_fused,
        rhs_r=rhs_r,
        rhs_i=rhs_i,
        coeffs=coeffs,
        lambda_max=lmax,
    )

    ham.clear_cache()

    # Compare norms: input vs output (should be close for unitary transform)
    norm_in = (psi_r**2 + psi_i**2).sum(dim=-1)
    norm_out = (out_r**2 + out_i**2).sum(dim=-1)
    rel_norm_change = ((norm_out - norm_in).abs() / (norm_in + 1e-12)).mean()

    assert rel_norm_change.item() < 0.10, (
        f"Norm change {rel_norm_change.item():.4f} exceeds 10%"
    )


@pytest.mark.slow
def test_kpm_speed_scaling():
    """KPM should scale linearly with D (not cubically like direct solve)."""
    if not torch.cuda.is_available():
        pytest.skip("Speed test requires CUDA")

    from sem.propagator.chebyshev_kpm import (
        chebyshev_kpm_solve,
        resolvent_chebyshev_coeffs,
    )
    from sem.propagator.hamiltonian import MultiScaleHamiltonian

    device = torch.device("cuda")
    times = {}

    for D in [128, 256]:
        torch.manual_seed(42)
        ham = MultiScaleHamiltonian(D, num_scales=2, base_sparsity=3).to(device)
        ham.cache_weights()

        diag = ham.get_diagonal().real
        lmax = float(2.0 * diag.max().item())
        lmax = max(lmax, 1e-8)

        coeffs = resolvent_chebyshev_coeffs(
            alpha=0.05, lambda_max=lmax, degree=12, device=device
        )
        rhs_r = torch.randn(2, 64, D, device=device)
        rhs_i = torch.randn(2, 64, D, device=device)

        # Warmup
        for _ in range(3):
            chebyshev_kpm_solve(ham.matvec_real_fused, rhs_r, rhs_i, coeffs, lmax)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            chebyshev_kpm_solve(ham.matvec_real_fused, rhs_r, rhs_i, coeffs, lmax)
        torch.cuda.synchronize()
        times[D] = (time.perf_counter() - t0) / 10

        ham.clear_cache()

    # D=256 should be less than 5x of D=128 (linear scaling allows ~2x, generous bound)
    ratio = times[256] / times[128]
    assert ratio < 5.0, f"Speed ratio {ratio:.2f} suggests non-linear scaling"


def test_learnable_coeffs_gradient():
    """Chebyshev coefficients should receive gradients during backward pass."""
    from sem.propagator.chebyshev_kpm import (
        chebyshev_kpm_solve,
        resolvent_chebyshev_coeffs,
    )
    from sem.propagator.hamiltonian import MultiScaleHamiltonian

    torch.manual_seed(42)
    D = 32
    B, S = 1, 8

    ham = MultiScaleHamiltonian(D, num_scales=2, base_sparsity=3)
    ham.cache_weights()

    diag = ham.get_diagonal().real
    lmax = float(2.0 * diag.max().item())
    lmax = max(lmax, 1e-8)

    coeffs = resolvent_chebyshev_coeffs(alpha=0.05, lambda_max=lmax, degree=8)
    coeffs = torch.nn.Parameter(coeffs)  # Make learnable

    rhs_r = torch.randn(B, S, D)
    rhs_i = torch.randn(B, S, D)

    out_r, out_i = chebyshev_kpm_solve(
        matvec_fn=ham.matvec_real_fused,
        rhs_r=rhs_r,
        rhs_i=rhs_i,
        coeffs=coeffs,
        lambda_max=lmax,
    )

    loss = (out_r**2 + out_i**2).sum()
    loss.backward()

    ham.clear_cache()

    assert coeffs.grad is not None, "Coefficients did not receive gradients"
    assert torch.all(torch.isfinite(coeffs.grad)), "Gradients contain NaN/Inf"
    assert coeffs.grad.norm().item() > 0, "Gradients are all zero"
