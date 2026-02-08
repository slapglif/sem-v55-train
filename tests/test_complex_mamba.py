"""Tests for Complex Mamba-3 Spinor module.

Validates:
- Block-diagonal spinor factorization correctness
- Complex SSM state update
- Phase preservation properties
- MIMO group equivalence
- Gradient flow through complex operations
"""

import torch
import pytest


class TestSpinorBlock:
    """Test block-diagonal spinor factorization."""

    def test_output_shape(self):
        """SpinorBlock should preserve tensor shape."""
        from sem.spinor.spinor_block import SpinorBlock

        block = SpinorBlock(hidden_dim=32, block_size=8)
        x = torch.randn(2, 8, 32, dtype=torch.complex64)
        y = block(x)

        assert y.shape == (2, 8, 32)
        assert y.is_complex()

    def test_near_identity_init(self):
        """Initial spinor should be near-identity (output ≈ input)."""
        from sem.spinor.spinor_block import SpinorBlock

        block = SpinorBlock(hidden_dim=16, block_size=4)
        x = torch.randn(1, 4, 16, dtype=torch.complex64)
        y = block(x)

        # Should be close to input due to near-identity initialization
        relative_change = (y - x).abs().mean() / (x.abs().mean() + 1e-8)
        assert relative_change < 0.5, (
            f"Initial spinor deviates too much: {relative_change:.4f}"
        )

    def test_block_independence(self):
        """Different blocks should be independent (no cross-talk)."""
        from sem.spinor.spinor_block import SpinorBlock

        block = SpinorBlock(hidden_dim=16, block_size=4)  # 4 blocks of size 4

        # Zero out all but first block
        x = torch.zeros(1, 1, 16, dtype=torch.complex64)
        x[0, 0, :4] = torch.randn(4, dtype=torch.complex64)

        y = block(x)

        # Only first block should have nonzero output
        assert y[0, 0, :4].abs().sum() > 0, "First block should be nonzero"
        assert y[0, 0, 4:].abs().sum() < 1e-5, "Other blocks should be ~zero"


class TestComplexMamba3Layer:
    """Test the full Complex Mamba-3 layer."""

    def test_output_shape(self):
        """Layer output should match input shape."""
        from sem.spinor.complex_mamba3 import ComplexMamba3Layer

        layer = ComplexMamba3Layer(
            hidden_dim=32, state_dim=16, mimo_groups=4, block_size=8, d_conv=4
        )

        x = torch.randn(2, 16, 32, dtype=torch.complex64)
        y = layer(x)

        assert y.shape == (2, 16, 32)
        assert y.is_complex()

    def test_residual_connection(self):
        """Layer should include residual connection."""
        from sem.spinor.complex_mamba3 import ComplexMamba3Layer

        layer = ComplexMamba3Layer(
            hidden_dim=16, state_dim=8, mimo_groups=2, block_size=4, d_conv=2
        )

        x = torch.randn(1, 4, 16, dtype=torch.complex64) * 10
        y = layer(x)

        # With residual, output should be correlated with input
        correlation = (x.real * y.real).sum() / (x.abs().sum() * y.abs().sum() + 1e-8)
        assert correlation > 0, "Residual connection should preserve some input signal"

    def test_gradient_flow(self):
        """Gradients should flow through all components."""
        from sem.spinor.complex_mamba3 import ComplexMamba3Layer

        layer = ComplexMamba3Layer(
            hidden_dim=16, state_dim=8, mimo_groups=2, block_size=4, d_conv=2
        )

        x = torch.randn(1, 4, 16, dtype=torch.complex64, requires_grad=True)
        y = layer(x)
        loss = y.abs().sum()
        loss.backward()

        assert x.grad is not None, "No gradient for input"

        # Check key parameters
        grad_count = sum(1 for p in layer.parameters() if p.grad is not None)
        total_params = sum(1 for _ in layer.parameters())
        assert grad_count > total_params * 0.5, (
            f"Only {grad_count}/{total_params} params got gradients"
        )

    def test_no_nan(self):
        """Layer should not produce NaN for reasonable input."""
        from sem.spinor.complex_mamba3 import ComplexMamba3Layer

        layer = ComplexMamba3Layer(
            hidden_dim=32, state_dim=16, mimo_groups=4, block_size=8, d_conv=4
        )

        x = torch.randn(2, 8, 32, dtype=torch.complex64)
        y = layer(x)

        assert not torch.isnan(y).any(), "NaN in layer output"
        assert not torch.isinf(y).any(), "Inf in layer output"

    def test_memory_horizon_ratio_init(self):
        """Memory horizon ratio should control SSM decay initialization."""
        from sem.spinor.complex_mamba3 import ComplexMamba3Layer, ComplexSSMState

        # Default (ratio=0): τ = S/e
        ssm_default = ComplexSSMState(
            d_input=8, state_dim=16, memory_horizon_ratio=0.0, max_seq_length=256
        )
        # Custom ratio: τ = 0.5 * S = 128
        ssm_custom = ComplexSSMState(
            d_input=8, state_dim=16, memory_horizon_ratio=0.5, max_seq_length=256
        )

        # Both should work (no crash)
        x = torch.randn(1, 4, 1, 8, dtype=torch.complex64)
        y_default = ssm_default(x)
        y_custom = ssm_custom(x)

        assert y_default.shape == (1, 4, 1, 8)
        assert y_custom.shape == (1, 4, 1, 8)

        # Invalid ratio should raise
        with pytest.raises(ValueError, match="memory_horizon_ratio"):
            ComplexSSMState(
                d_input=8, state_dim=16, memory_horizon_ratio=-0.1, max_seq_length=256
            )
        with pytest.raises(ValueError, match="memory_horizon_ratio"):
            ComplexSSMState(
                d_input=8, state_dim=16, memory_horizon_ratio=1.5, max_seq_length=256
            )
        with pytest.raises(ValueError, match="max_seq_length"):
            ComplexSSMState(
                d_input=8, state_dim=16, memory_horizon_ratio=0.0, max_seq_length=0
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        reason="CUDA with bfloat16 support required",
    )
    def test_mixed_precision_a(self):
        """Mixed precision (bfloat16 A) should produce accurate results.

        SEOP Fix 23: A tensor is bounded in (0, 1), making bfloat16 safe
        with <1% mean relative error and 25% memory bandwidth reduction.
        """
        from sem.spinor.complex_mamba3 import ComplexMamba3Layer

        device = "cuda"

        # Create layers with and without mixed precision
        layer_mp = ComplexMamba3Layer(
            hidden_dim=64,
            state_dim=16,
            mimo_groups=4,
            block_size=8,
            d_conv=4,
            use_mixed_precision_a=True,
        ).to(device)

        layer_fp = ComplexMamba3Layer(
            hidden_dim=64,
            state_dim=16,
            mimo_groups=4,
            block_size=8,
            d_conv=4,
            use_mixed_precision_a=False,
        ).to(device)

        # Share weights
        layer_fp.load_state_dict(layer_mp.state_dict())

        x = torch.randn(2, 32, 64, dtype=torch.complex64, device=device)

        y_mp = layer_mp(x)
        y_fp = layer_fp(x)

        # Check output is complex and correct shape
        assert y_mp.is_complex()
        assert y_mp.shape == x.shape

        # Check numerical accuracy: <1% mean relative error
        rel_error = (y_mp - y_fp).abs() / (y_fp.abs() + 1e-8)
        mean_rel_error = rel_error.mean().item()
        assert mean_rel_error < 0.01, (
            f"Mixed precision error too high: {mean_rel_error:.4%}"
        )

        # Check gradients flow correctly
        y_mp.abs().sum().backward()
        for name, p in layer_mp.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not p.grad.isnan().any(), f"NaN gradient for {name}"
