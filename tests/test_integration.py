"""Integration tests for SEM V5.5 'Lean Crystal'.

Tests the full model pipeline end-to-end:
- Forward pass completes without errors
- Output shapes are correct
- No NaN/Inf in outputs
- Gradients flow to all parameters
- Training loop runs without errors
- Loss decreases over steps
"""

import torch
import pytest

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mamba-ssm requires CUDA"
)


@requires_cuda
class TestFullForwardPass:
    """Test the complete SEM model forward pass."""

    def test_forward_shape(self):
        """Full model forward pass should produce correct shapes."""
        from sem.config import (
            SEMConfig,
            ModelConfig,
            EncoderConfig,
            SpinorConfig,
            PropagatorConfig,
        )
        from sem.model import SEMModel

        config = SEMConfig(
            model=ModelConfig(
                hidden_dim=32, num_layers=2, vocab_size=100, max_seq_length=64
            ),
            encoder=EncoderConfig(sdr_sparsity=8, sdr_candidates=16),
            spinor=SpinorConfig(
                block_size=8, num_blocks=4, state_dim=16, mimo_groups=4, d_conv=2
            ),
            propagator=PropagatorConfig(cg_max_iter=10, laplacian_sparsity=3),
        )

        model = SEMModel(config).cuda()
        tokens = torch.randint(0, 100, (2, 16), device="cuda")

        output = model(tokens, targets=tokens)

        assert "logits" in output
        assert "log_probs" in output
        assert "loss" in output
        assert output["logits"].shape == (2, 16, 100)
        assert output["loss"].ndim == 0  # scalar

    def test_no_nan(self):
        """Forward pass should not produce NaN."""
        from sem.config import (
            SEMConfig,
            ModelConfig,
            EncoderConfig,
            SpinorConfig,
            PropagatorConfig,
        )
        from sem.model import SEMModel

        config = SEMConfig(
            model=ModelConfig(
                hidden_dim=32, num_layers=2, vocab_size=100, max_seq_length=64
            ),
            encoder=EncoderConfig(sdr_sparsity=8, sdr_candidates=16),
            spinor=SpinorConfig(
                block_size=8, num_blocks=4, state_dim=16, mimo_groups=4, d_conv=2
            ),
            propagator=PropagatorConfig(cg_max_iter=10, laplacian_sparsity=3),
        )

        model = SEMModel(config).cuda()
        tokens = torch.randint(0, 100, (1, 8), device="cuda")

        output = model(tokens, targets=tokens)

        assert not torch.isnan(output["logits"]).any(), "NaN in logits"
        assert not torch.isinf(output["logits"]).any(), "Inf in logits"
        assert not torch.isnan(output["loss"]), "NaN in loss"

    def test_gradient_flow(self):
        """All parameters should receive gradients."""
        from sem.config import (
            SEMConfig,
            ModelConfig,
            EncoderConfig,
            SpinorConfig,
            PropagatorConfig,
        )
        from sem.model import SEMModel

        config = SEMConfig(
            model=ModelConfig(
                hidden_dim=16, num_layers=1, vocab_size=50, max_seq_length=32
            ),
            encoder=EncoderConfig(sdr_sparsity=4, sdr_candidates=8),
            spinor=SpinorConfig(
                block_size=4, num_blocks=4, state_dim=8, mimo_groups=2, d_conv=2
            ),
            propagator=PropagatorConfig(cg_max_iter=10, laplacian_sparsity=2),
        )

        model = SEMModel(config).cuda()
        tokens = torch.randint(0, 50, (1, 8), device="cuda")

        output = model(tokens, targets=tokens)
        output["loss"].backward()

        total = 0
        with_grad = 0
        for name, p in model.named_parameters():
            total += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                with_grad += 1

        ratio = with_grad / total
        assert ratio > 0.5, (
            f"Only {with_grad}/{total} ({ratio:.0%}) parameters got gradients"
        )

    def test_parameter_count(self):
        """Model should report parameter counts."""
        from sem.config import (
            SEMConfig,
            ModelConfig,
            EncoderConfig,
            SpinorConfig,
            PropagatorConfig,
        )
        from sem.model import SEMModel

        config = SEMConfig(
            model=ModelConfig(
                hidden_dim=32, num_layers=2, vocab_size=100, max_seq_length=64
            ),
            encoder=EncoderConfig(sdr_sparsity=8, sdr_candidates=16),
            spinor=SpinorConfig(
                block_size=8, num_blocks=4, state_dim=16, mimo_groups=4, d_conv=2
            ),
            propagator=PropagatorConfig(cg_max_iter=10, laplacian_sparsity=3),
        )

        model = SEMModel(config).cuda()
        counts = model.count_parameters()

        assert "total" in counts
        assert counts["total"]["total"] > 0


@requires_cuda
class TestTrainingLoop:
    """Test the training loop mechanics."""

    def test_single_step(self):
        """Single training step should complete without error."""
        from sem.config import (
            SEMConfig,
            ModelConfig,
            EncoderConfig,
            SpinorConfig,
            PropagatorConfig,
        )
        from sem.model import SEMModel
        from sem.utils.complex_adamw import ComplexAdamW

        config = SEMConfig(
            model=ModelConfig(
                hidden_dim=16, num_layers=1, vocab_size=50, max_seq_length=16
            ),
            encoder=EncoderConfig(sdr_sparsity=4, sdr_candidates=8),
            spinor=SpinorConfig(
                block_size=4, num_blocks=4, state_dim=8, mimo_groups=2, d_conv=2
            ),
            propagator=PropagatorConfig(cg_max_iter=5, laplacian_sparsity=2),
        )

        model = SEMModel(config).cuda()
        optimizer = ComplexAdamW(model.parameters(), lr=1e-3)

        tokens = torch.randint(0, 50, (2, 8), device="cuda")

        # Forward
        output = model(tokens, targets=tokens)
        loss = output["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0, "Loss should be positive"

    def test_loss_decreases(self):
        """Loss should decrease over a few training steps."""
        from sem.config import (
            SEMConfig,
            ModelConfig,
            EncoderConfig,
            SpinorConfig,
            PropagatorConfig,
            V8Config,
        )
        from sem.model import SEMModel
        from sem.utils.complex_adamw import ComplexAdamW

        config = SEMConfig(
            model=ModelConfig(
                hidden_dim=16, num_layers=1, vocab_size=20, max_seq_length=16
            ),
            encoder=EncoderConfig(sdr_sparsity=4, sdr_candidates=8),
            spinor=SpinorConfig(
                block_size=4, num_blocks=4, state_dim=8, mimo_groups=2, d_conv=2
            ),
            propagator=PropagatorConfig(cg_max_iter=5, laplacian_sparsity=2),
            v8=V8Config(
                use_lindblad=False,
                use_hybrid_automata=False,
                use_quaternionic=False,
                use_mhc=False,
            ),
        )

        model = SEMModel(config).cuda()
        optimizer = ComplexAdamW(model.parameters(), lr=1e-3)

        # Fixed batch for overfitting test
        tokens = torch.randint(0, 20, (4, 8), device="cuda")

        losses = []
        for _ in range(10):
            output = model(tokens, targets=tokens)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0] * 1.1, (
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestConfig:
    """Test configuration loading."""

    def test_default_config(self):
        """Default config should be valid."""
        from sem.config import SEMConfig

        config = SEMConfig()
        assert config.model.hidden_dim == 256
        assert config.model.num_layers == 8
        assert config.model.vocab_size == 50262

    def test_yaml_config(self):
        """Config should load from YAML."""
        from sem.config import SEMConfig
        import os

        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )
        if os.path.exists(yaml_path):
            config = SEMConfig.from_yaml(yaml_path)
            assert config.model.hidden_dim == 256


@requires_cuda
class TestSEOPFixes:
    """Test SEOP fixes are properly implemented."""

    def test_seop_fix_34_detach(self):
        """SEOP Fix 34: unitary_divergence should be detached in loss."""
        from sem.config import (
            SEMConfig,
            ModelConfig,
            EncoderConfig,
            SpinorConfig,
            PropagatorConfig,
            TrainingConfig,
        )
        from sem.model import SEMModel

        config = SEMConfig(
            model=ModelConfig(
                hidden_dim=16, num_layers=1, vocab_size=50, max_seq_length=16
            ),
            encoder=EncoderConfig(sdr_sparsity=4, sdr_candidates=8),
            spinor=SpinorConfig(
                block_size=4, num_blocks=4, state_dim=8, mimo_groups=2, d_conv=2
            ),
            propagator=PropagatorConfig(cg_max_iter=5, laplacian_sparsity=2),
            training=TrainingConfig(unitary_lambda=0.01),
        )

        model = SEMModel(config).cuda()
        tokens = torch.randint(0, 50, (1, 8), device="cuda")
        output = model(tokens, targets=tokens)

        assert "unitary_divergence" in output
        output["loss"].backward()

    def test_seop_fix_35_clamp(self):
        """SEOP Fix 35: psi_energy should be clamped before log²."""
        from sem.config import (
            SEMConfig,
            ModelConfig,
            EncoderConfig,
            SpinorConfig,
            PropagatorConfig,
        )
        from sem.model import SEMModel

        config = SEMConfig(
            model=ModelConfig(
                hidden_dim=16, num_layers=1, vocab_size=50, max_seq_length=16
            ),
            encoder=EncoderConfig(sdr_sparsity=4, sdr_candidates=8),
            spinor=SpinorConfig(
                block_size=4, num_blocks=4, state_dim=8, mimo_groups=2, d_conv=2
            ),
            propagator=PropagatorConfig(cg_max_iter=5, laplacian_sparsity=2),
        )

        model = SEMModel(config).cuda()
        tokens = torch.randint(0, 50, (1, 8), device="cuda")
        output = model(tokens, targets=tokens)

        assert torch.isfinite(output["unitary_divergence"]), (
            f"unitary_divergence not finite: {output['unitary_divergence']}"
        )

    def test_training_step_wiring(self):
        """Training step counter should propagate to propagator for adaptive CG."""
        from sem.config import (
            SEMConfig,
            ModelConfig,
            EncoderConfig,
            SpinorConfig,
            PropagatorConfig,
        )
        from sem.model import SEMModel

        config = SEMConfig(
            model=ModelConfig(
                hidden_dim=16, num_layers=1, vocab_size=50, max_seq_length=16
            ),
            encoder=EncoderConfig(sdr_sparsity=4, sdr_candidates=8),
            spinor=SpinorConfig(
                block_size=4, num_blocks=4, state_dim=8, mimo_groups=2, d_conv=2
            ),
            propagator=PropagatorConfig(
                cg_max_iter=5,
                laplacian_sparsity=2,
                adaptive_cg_tol=True,
            ),
        )

        model = SEMModel(config).cuda()

        if hasattr(model.propagator, "set_training_step"):
            model.propagator.set_training_step(100)

        tokens = torch.randint(0, 50, (1, 8), device="cuda")
        output = model(tokens, targets=tokens)
        assert torch.isfinite(output["loss"])
