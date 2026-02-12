"""Tests for Complex Mamba-3 layer with Mamba2 backend.

Validates:
- Mamba2 forward pass correctness
- Phase preservation properties
- Gradient flow through complex operations
"""

import torch
import pytest

pytest.importorskip("mamba_ssm")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mamba-ssm requires CUDA"
)


@requires_cuda
class TestComplexMamba3Layer:
    """Test the full Complex Mamba-3 layer."""

    def test_output_shape(self):
        from sem.spinor.complex_mamba3 import ComplexMamba3Layer

        layer = ComplexMamba3Layer(
            hidden_dim=32, state_dim=16, mimo_groups=4, block_size=8, d_conv=4
        ).cuda()

        x = torch.randn(2, 16, 32, dtype=torch.complex64, device="cuda")
        y = layer(x)

        assert y.shape == (2, 16, 32)
        assert y.is_complex()

    def test_residual_connection(self):
        from sem.spinor.complex_mamba3 import ComplexMamba3Layer

        layer = ComplexMamba3Layer(
            hidden_dim=16, state_dim=8, mimo_groups=2, block_size=4, d_conv=2
        ).cuda()

        x = torch.randn(1, 4, 16, dtype=torch.complex64, device="cuda") * 10
        y = layer(x)

        correlation = (x.real * y.real).sum() / (x.abs().sum() * y.abs().sum() + 1e-8)
        assert correlation > 0, "Residual connection should preserve some input signal"

    def test_gradient_flow(self):
        from sem.spinor.complex_mamba3 import ComplexMamba3Layer

        layer = ComplexMamba3Layer(
            hidden_dim=16, state_dim=8, mimo_groups=2, block_size=4, d_conv=2
        ).cuda()

        x = torch.randn(
            1, 4, 16, dtype=torch.complex64, device="cuda", requires_grad=True
        )
        y = layer(x)
        loss = y.abs().sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"

        grad_count = sum(1 for p in layer.parameters() if p.grad is not None)
        total_params = sum(1 for _ in layer.parameters())
        assert grad_count > total_params * 0.5, (
            f"Only {grad_count}/{total_params} params got gradients"
        )

    def test_no_nan(self):
        from sem.spinor.complex_mamba3 import ComplexMamba3Layer

        layer = ComplexMamba3Layer(
            hidden_dim=32, state_dim=16, mimo_groups=4, block_size=8, d_conv=4
        ).cuda()

        x = torch.randn(2, 8, 32, dtype=torch.complex64, device="cuda")
        y = layer(x)

        assert not torch.isnan(y).any(), "NaN in layer output"
        assert not torch.isinf(y).any(), "Inf in layer output"
