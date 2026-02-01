"""Tests for end-to-end unitarity and information conservation.

Validates the core invariant of SEM V5.5: the propagation pipeline
conserves the total information content (L2 norm) of the wavefunction.
"""
import torch
import pytest


class TestEndToEndUnitarity:
    """Test norm preservation across the full pipeline."""

    def test_single_layer_norm(self):
        """Single propagator layer should approximately preserve norm."""
        from sem.propagator.cayley_soliton import CayleySolitonPropagator

        prop = CayleySolitonPropagator(
            dim=32, dt=0.05, nonlinear_alpha=0.05,
            cg_max_iter=30, cg_tol=1e-7,
            laplacian_sparsity=3, num_scales=1
        )

        # Normalized input
        psi = torch.randn(2, 8, 32, dtype=torch.complex64)
        norm_in = (psi.abs() ** 2).sum(dim=-1)

        psi_out = prop(psi)
        norm_out = (psi_out.abs() ** 2).sum(dim=-1)

        ratio = norm_out / (norm_in + 1e-12)
        # Should be close to 1.0
        assert (ratio - 1.0).abs().max() < 0.2, \
            f"Norm ratio max deviation: {(ratio-1.0).abs().max():.4f}"

    def test_multi_layer_norm(self):
        """Multiple propagation layers should not amplify norm drift."""
        from sem.propagator.cayley_soliton import CayleySolitonStack

        stack = CayleySolitonStack(
            dim=16, num_layers=4, dt=0.05,
            nonlinear_alpha=0.05, cg_max_iter=30,
            cg_tol=1e-7, laplacian_sparsity=3
        )

        psi = torch.randn(1, 4, 16, dtype=torch.complex64)
        norm_in = (psi.abs() ** 2).sum(dim=-1)

        psi_out = stack(psi)
        norm_out = (psi_out.abs() ** 2).sum(dim=-1)

        ratio = norm_out / (norm_in + 1e-12)
        # Allow more tolerance for stacked layers
        assert (ratio - 1.0).abs().max() < 0.5, \
            f"Stacked norm ratio deviation: {(ratio-1.0).abs().max():.4f}"

    def test_information_density_metric(self):
        """Information density should be computable and in [0, 1]."""
        from sem.utils.metrics import information_density

        z = torch.randn(2, 8, 32, dtype=torch.complex64)
        density = information_density(z)

        assert 0 <= density <= 1, f"Information density out of range: {density}"

    def test_phase_coherence_metric(self):
        """Phase coherence should be computable and in [0, 1]."""
        from sem.utils.metrics import phase_coherence

        # Random phases: low coherence
        z_random = torch.randn(2, 8, 32, dtype=torch.complex64)
        coherence_random = phase_coherence(z_random)

        # Aligned phases: high coherence
        mag = torch.rand(2, 8, 32)
        z_aligned = torch.polar(mag, torch.zeros_like(mag))
        coherence_aligned = phase_coherence(z_aligned)

        assert coherence_aligned > coherence_random, \
            "Aligned phases should have higher coherence"
