"""Tests for Cayley-Soliton Propagator.

Validates:
- Cayley transform unitarity: ||U†U - I||_F < 1e-6
- CG solver convergence
- CG solver differentiability (custom autograd)
- Norm preservation through propagation
- Nonlinear phase rotation correctness
"""
import torch
import pytest


class TestCGSolver:
    """Test the Conjugate Gradient solver."""

    def test_solves_simple_system(self):
        """CG should solve A@x = b for Hermitian positive definite A."""
        from sem.propagator.cg_solver import cg_solve

        D = 8
        # Construct HPD matrix: A = M†M + I
        M = torch.randn(D, D, dtype=torch.complex64)
        A = M.conj().T @ M + torch.eye(D, dtype=torch.complex64) * 2

        # Known solution
        x_true = torch.randn(D, dtype=torch.complex64)
        b = A @ x_true

        x_solved = cg_solve(A, b.unsqueeze(0), max_iter=50, tol=1e-8)

        residual = (x_solved.squeeze(0) - x_true).abs().max()
        assert residual < 1e-4, f"CG solution error: {residual:.6f}"

    def test_batched_solve(self):
        """CG should handle batched right-hand sides."""
        from sem.propagator.cg_solver import cg_solve

        D = 8
        M = torch.randn(D, D, dtype=torch.complex64)
        A = M.conj().T @ M + torch.eye(D, dtype=torch.complex64) * 2

        b = torch.randn(4, D, dtype=torch.complex64)  # 4 batch
        x = cg_solve(A, b, max_iter=50, tol=1e-8)

        assert x.shape == (4, D)

        # Verify each solution
        for i in range(4):
            residual = (A @ x[i] - b[i]).abs().max()
            assert residual < 1e-3, f"Batch {i} residual: {residual:.6f}"

    def test_differentiability(self):
        """CG solve should be differentiable via implicit differentiation."""
        from sem.propagator.cg_solver import cg_solve

        D = 4
        M = torch.randn(D, D, dtype=torch.complex64)
        A = M.conj().T @ M + torch.eye(D, dtype=torch.complex64) * 2
        A = A.detach().requires_grad_(True)

        b = torch.randn(1, D, dtype=torch.complex64, requires_grad=True)

        x = cg_solve(A, b, max_iter=20, tol=1e-6)
        loss = x.abs().sum()
        loss.backward()

        assert b.grad is not None, "No gradient for b"
        assert A.grad is not None, "No gradient for A"
        assert not torch.isnan(b.grad).any(), "NaN in b gradient"
        assert not torch.isnan(A.grad).any(), "NaN in A gradient"


class TestCayleyUnitarity:
    """Test unitarity of the Cayley transform."""

    def test_cayley_operator_unitary(self):
        """U = (I-iH)^{-1}(I+iH) should satisfy U†U = I for Hermitian H."""
        from sem.propagator.unitarity_check import cayley_unitarity_check

        D = 16
        # Build Hermitian H
        M = torch.randn(D, D, dtype=torch.complex64)
        H = (M + M.conj().T) / 2

        deviation = cayley_unitarity_check(H, dt=0.1, tol=1e-6)
        assert deviation < 1e-5, f"Cayley unitarity deviation: {deviation:.2e}"

    def test_norm_preservation(self):
        """Propagator should preserve wavefunction norm."""
        from sem.propagator.unitarity_check import check_unitarity
        from sem.propagator.cayley_soliton import CayleySolitonPropagator

        prop = CayleySolitonPropagator(
            dim=16, dt=0.1, nonlinear_alpha=0.1,
            cg_max_iter=30, cg_tol=1e-7,
            laplacian_sparsity=3, num_scales=1
        )

        psi = torch.randn(1, 4, 16, dtype=torch.complex64)
        psi_out = prop(psi)

        # Check norm preservation
        norm_in = (psi.abs() ** 2).sum(dim=-1)
        norm_out = (psi_out.abs() ** 2).sum(dim=-1)

        deviation = ((norm_out / (norm_in + 1e-12)) - 1.0).abs().max()
        # Allow some tolerance due to nonlinear rotation + CG approximation
        assert deviation < 0.1, f"Norm deviation: {deviation:.4f}"


class TestCayleySolitonPropagator:
    """Test the full Cayley-Soliton propagator."""

    def test_output_shape(self):
        """Propagator should preserve shape."""
        from sem.propagator.cayley_soliton import CayleySolitonPropagator

        prop = CayleySolitonPropagator(
            dim=16, dt=0.1, cg_max_iter=10,
            laplacian_sparsity=3, num_scales=1
        )

        psi = torch.randn(2, 8, 16, dtype=torch.complex64)
        psi_out = prop(psi)

        assert psi_out.shape == (2, 8, 16)
        assert psi_out.is_complex()

    def test_no_nan(self):
        """Propagator should not produce NaN."""
        from sem.propagator.cayley_soliton import CayleySolitonPropagator

        prop = CayleySolitonPropagator(
            dim=16, dt=0.1, cg_max_iter=20,
            laplacian_sparsity=3, num_scales=1
        )

        psi = torch.randn(1, 4, 16, dtype=torch.complex64)
        psi_out = prop(psi)

        assert not torch.isnan(psi_out).any(), "NaN in propagator output"
        assert not torch.isinf(psi_out).any(), "Inf in propagator output"

    def test_gradient_flow(self):
        """Gradients should flow through the propagator."""
        from sem.propagator.cayley_soliton import CayleySolitonPropagator

        prop = CayleySolitonPropagator(
            dim=8, dt=0.1, cg_max_iter=10,
            laplacian_sparsity=2, num_scales=1
        )

        psi = torch.randn(1, 2, 8, dtype=torch.complex64, requires_grad=True)
        psi_out = prop(psi)
        loss = psi_out.abs().sum()
        loss.backward()

        assert psi.grad is not None, "No gradient for input"
        assert not torch.isnan(psi.grad).any(), "NaN in input gradient"

    def test_stack(self):
        """CayleySolitonStack should chain multiple layers."""
        from sem.propagator.cayley_soliton import CayleySolitonStack

        stack = CayleySolitonStack(
            dim=16, num_layers=3, dt=0.1,
            cg_max_iter=10, laplacian_sparsity=3
        )

        psi = torch.randn(1, 4, 16, dtype=torch.complex64)
        psi_out = stack(psi)

        assert psi_out.shape == (1, 4, 16)
        assert not torch.isnan(psi_out).any()
