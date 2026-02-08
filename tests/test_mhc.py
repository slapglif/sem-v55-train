"""Unit tests for Manifold-Constrained Hyper-Connections (mHC).

Tests verify:
1. Sinkhorn produces doubly stochastic matrices (Birkhoff Polytope)
2. mHC residual preserves signal magnitude
3. Complex-valued variant works correctly
4. Gradient flow is stable
"""

import pytest
import torch
import torch.nn as nn
from sem.hyper_connections import (
    sinkhorn_log,
    sinkhorn_log_complex,
    MHCResidual,
    SimpleMHC,
    mhc_residual,
)


class TestSinkhorn:
    """Test Sinkhorn projection onto Birkhoff Polytope."""

    def test_doubly_stochastic_real(self):
        """Verify Sinkhorn produces doubly stochastic matrices (real)."""
        torch.manual_seed(42)
        logits = torch.randn(4, 4)

        H = sinkhorn_log(logits, num_iters=20, tau=0.05)

        # Check shape
        assert H.shape == (4, 4)

        # Check row stochastic (rows sum to 1)
        row_sums = H.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=5e-3)

        # Check column stochastic (columns sum to 1)
        col_sums = H.sum(dim=-2)
        assert torch.allclose(col_sums, torch.ones(4), atol=5e-3)

        # Check non-negative
        assert (H >= 0).all()

        # Check bounded by 1
        assert (H <= 1.0).all()

    def test_doubly_stochastic_complex(self):
        """Verify Sinkhorn produces doubly stochastic magnitude (complex)."""
        torch.manual_seed(42)
        logits_real = torch.randn(4, 4)
        logits_imag = torch.randn(4, 4)

        H_real, H_imag = sinkhorn_log_complex(
            logits_real, logits_imag, num_iters=20, tau=0.05
        )

        # Check magnitude is doubly stochastic
        H_mag = torch.sqrt(H_real**2 + H_imag**2)

        row_sums = H_mag.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=3e-2)

        col_sums = H_mag.sum(dim=-2)
        assert torch.allclose(col_sums, torch.ones(4), atol=3e-2)

    def test_batch_sinkhorn(self):
        """Test batched Sinkhorn projection."""
        torch.manual_seed(42)
        logits = torch.randn(8, 4, 4)  # [B, N, N]

        H = sinkhorn_log(logits, num_iters=200, tau=0.05)

        assert H.shape == (8, 4, 4)

        # Check each matrix in batch is doubly stochastic
        for i in range(8):
            row_sums = H[i].sum(dim=-1)
            col_sums = H[i].sum(dim=-2)
            assert torch.allclose(row_sums, torch.ones(4), atol=6e-3)
            assert torch.allclose(col_sums, torch.ones(4), atol=6e-3)

    def test_identity_convergence(self):
        """Test that identity logits converge to uniform doubly stochastic."""
        logits = torch.zeros(3, 3)  # All zeros = uniform after softmax

        H = sinkhorn_log(logits, num_iters=20, tau=1.0)

        # Should converge to uniform (all entries = 1/3)
        expected = torch.full((3, 3), 1.0 / 3.0)
        assert torch.allclose(H, expected, atol=1e-2)


class TestMHCResidual:
    """Test mHC residual blocks."""

    def test_simple_mhc_forward_real(self):
        """Test SimpleMHC forward pass (real tensors)."""
        torch.manual_seed(42)
        mhc = SimpleMHC(dim=64, complex_mode=False)

        x = torch.randn(4, 16, 64)  # [B, T, D]
        branch = torch.randn(4, 16, 64)

        y = mhc(x, branch)

        # Check shape
        assert y.shape == x.shape

        # For single stream, should be close to x + branch (H ≈ identity)
        # (Not exactly equal due to Sinkhorn projection)
        expected = x + branch
        assert torch.allclose(y, expected, atol=0.1)

    def test_simple_mhc_forward_complex(self):
        """Test SimpleMHC forward pass (complex tensors)."""
        torch.manual_seed(42)
        mhc = SimpleMHC(dim=64, complex_mode=True)

        x = torch.randn(4, 16, 64, dtype=torch.complex64)
        branch = torch.randn(4, 16, 64, dtype=torch.complex64)

        y = mhc(x, branch)

        # Check shape and dtype
        assert y.shape == x.shape
        assert y.dtype == torch.complex64

        # Check magnitude is reasonable (no explosion/collapse)
        x_mag = x.abs().mean()
        y_mag = y.abs().mean()
        assert y_mag > 0.5 * x_mag  # Not collapsed
        assert y_mag < 3.0 * x_mag  # Not exploded

    def test_mhc_residual_preserves_magnitude(self):
        """Test that mHC preserves signal magnitude across multiple applications."""
        torch.manual_seed(42)
        mhc = SimpleMHC(dim=64, complex_mode=True, dropout=0.0)

        x = torch.randn(4, 16, 64, dtype=torch.complex64)
        initial_mag = x.abs().mean().item()

        # Apply mHC 10 times (simulating 10 layers)
        for _ in range(10):
            branch = 0.1 * torch.randn_like(x)  # Small perturbation
            x = mhc(x, branch)

        final_mag = x.abs().mean().item()

        # Magnitude should stay in reasonable range (not explode/collapse)
        # With 10 layers and small branch, magnitude should not change drastically
        assert 0.5 * initial_mag < final_mag < 2.0 * initial_mag

    def test_mhc_residual_multi_stream(self):
        """Test MHCResidual with multiple streams."""
        torch.manual_seed(42)
        num_streams = 4
        mhc = MHCResidual(
            dim=64,
            num_streams=num_streams,
            complex_mode=False,
            init_identity=False,
        )

        x = torch.randn(2, 8, num_streams, 64)  # [B, T, S, D]
        branch = torch.randn(2, 8, 64)  # [B, T, D]

        y = mhc(x, branch)

        assert y.shape == x.shape

    def test_functional_api(self):
        """Test functional mhc_residual API."""
        torch.manual_seed(42)
        H_logits = nn.Parameter(torch.zeros(1, 1))

        x = torch.randn(4, 16, 64)
        branch = torch.randn(4, 16, 64)

        y = mhc_residual(
            x, branch, H_logits, mhc_num_iters=10, mhc_tau=0.05, complex_mode=False
        )

        # Check shape
        assert y.shape == x.shape

        # Should be close to x + branch for identity H
        expected = x + branch
        assert torch.allclose(y, expected, atol=0.1)


class TestGradientFlow:
    """Test gradient stability with mHC."""

    def test_gradient_flow_simple_mhc(self):
        """Verify gradients flow through SimpleMHC without explosion/vanishing."""
        torch.manual_seed(42)
        mhc = SimpleMHC(dim=64, complex_mode=False)

        x = torch.randn(4, 16, 64, requires_grad=True)
        branch = torch.randn(4, 16, 64, requires_grad=True)

        y = mhc(x, branch)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert branch.grad is not None

        # Check gradients are not NaN or Inf
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isnan(branch.grad).any()
        assert not torch.isinf(branch.grad).any()

        # Check gradient magnitude is reasonable
        x_grad_norm = x.grad.norm().item()
        branch_grad_norm = branch.grad.norm().item()
        assert 0.01 < x_grad_norm < 100.0
        assert 0.01 < branch_grad_norm < 100.0

    def test_deep_gradient_flow(self):
        """Test gradient flow through deep network with mHC residuals."""
        torch.manual_seed(42)
        depth = 20
        mhc_layers = nn.ModuleList(
            [SimpleMHC(dim=32, complex_mode=False) for _ in range(depth)]
        )

        x0 = torch.randn(2, 8, 32, requires_grad=True)
        x = x0

        # Forward through 20 layers
        for layer in mhc_layers:
            branch = 0.1 * torch.randn(2, 8, 32)
            x = layer(x, branch)

        loss = x.sum()
        loss.backward()

        # Check final gradient is reasonable (not vanished)
        grad_norm = x0.grad.norm().item()
        assert grad_norm > 1e-5  # Not vanished

        # Check H_res_logits gradients across depth
        first_layer_grad = mhc_layers[0].H_res_logits.grad.norm().item()
        last_layer_grad = mhc_layers[-1].H_res_logits.grad.norm().item()

        # Gradients should not vanish dramatically across depth
        # (With standard residuals, first_layer_grad << last_layer_grad)
        ratio = first_layer_grad / (last_layer_grad + 1e-12)
        assert ratio > 1e-3  # First layer still gets meaningful gradients


class TestIntegration:
    """Integration tests for mHC in realistic scenarios."""

    def test_mhc_in_training_loop(self):
        """Simulate training loop with mHC residual."""
        torch.manual_seed(42)

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 64)
                self.mhc = SimpleMHC(dim=64, complex_mode=False)
                self.fc2 = nn.Linear(64, 64)

            def forward(self, x):
                branch = torch.relu(self.fc1(x))
                x = self.mhc(x, branch)
                return self.fc2(x)

        model = SimpleMLP()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train for a few steps
        for _ in range(10):
            x = torch.randn(4, 8, 64)
            target = torch.randn(4, 8, 64)

            y = model(x)
            loss = nn.functional.mse_loss(y, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check loss is finite
            assert torch.isfinite(loss).all()

            # Check H_res_logits updated
            h_res_grad = model.mhc.H_res_logits.grad
            assert h_res_grad is not None
            assert not torch.isnan(h_res_grad).any()

    def test_mhc_complex_training(self):
        """Test training with complex-valued mHC."""
        torch.manual_seed(42)

        class ComplexBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.mhc = SimpleMHC(dim=64, complex_mode=True)

            def forward(self, x):
                # Simple complex branch: multiply by phase
                phase = torch.exp(1j * torch.randn(1, 1, 64))
                branch = x * phase
                return self.mhc(x, branch)

        model = ComplexBlock()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train for a few steps
        for _ in range(10):
            x = torch.randn(4, 8, 64, dtype=torch.complex64)
            target = torch.randn(4, 8, 64, dtype=torch.complex64)

            y = model(x)
            loss = (y - target).abs().pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check loss is finite
            assert torch.isfinite(loss).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
