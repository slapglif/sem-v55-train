"""Quick convergence test - verify model can overfit on a tiny dataset.

This test creates a minimal synthetic dataset and trains for 100-200 steps
to verify the model CAN learn when conditions are right.

Expected behavior:
- Loss should DECREASE by at least 50% from initial value
- No NaN in loss or gradients
- Gradients should flow (non-zero)
"""

import torch
import pytest

from sem.model import SEMModel
from sem.config import (
    SEMConfig,
    ModelConfig,
    EncoderConfig,
    SpinorConfig,
    PropagatorConfig,
    QuantizerConfig,
    SamplerConfig,
    TrainingConfig,
    DistillationConfig,
    V8Config,
)
from sem.utils.complex_adamw import ComplexAdamW


@pytest.fixture
def minimal_config():
    """Ultra-minimal config for fast convergence test."""
    return SEMConfig(
        model=ModelConfig(
            hidden_dim=128,  # Very small for speed
            num_layers=2,  # Minimal depth
            vocab_size=1000,  # Small vocab
            max_seq_length=128,  # Short sequences
        ),
        encoder=EncoderConfig(
            sdr_sparsity=16,
            sdr_candidates=64,
            sinkhorn_epsilon=0.05,
            sinkhorn_max_iter=50,
            sinkhorn_tol=1e-3,
        ),
        spinor=SpinorConfig(
            block_size=8,
            num_blocks=16,
            state_dim=32,
            mimo_groups=4,
            d_conv=4,
        ),
        propagator=PropagatorConfig(
            cayley_dt=0.1,
            cg_max_iter=10,  # Fewer iterations for speed
            cg_tol=1e-4,  # Looser tolerance
            nonlinear_alpha=0.1,
            laplacian_sparsity=5,
            lazy_cg=False,
            lazy_cg_tol=1e-3,
            direct_solve=True,  # Use direct solve for small models
            pit_gamma=0.1,
        ),
        quantizer=QuantizerConfig(
            codebook_size=256,
            group_size=2,
            fisher_ema_decay=0.99,
            outlier_percentile=0.01,
            dead_code_threshold=100,
        ),
        sampler=SamplerConfig(
            temperature=1.0,
            top_k=50,
            top_p=0.95,
        ),
        training=TrainingConfig(
            batch_size=8,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_steps=20,
            max_steps=200,
            gradient_clip=10.0,
            dtype="complex64",
            low_vram_mode=False,
            unitary_lambda=0.01,
            decay_steps=180,
            lr_min_ratio=0.1,
        ),
        distillation=DistillationConfig(
            enabled=False,
        ),
        v8=V8Config(
            use_lindblad=True,
            use_hybrid_automata=True,
            use_quaternionic=True,
            use_mhc=True,
        ),
    )


def test_quick_convergence(minimal_config):
    """Verify model can overfit on a tiny synthetic dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Create synthetic data: simple repetitive pattern
    # Use repeating sequence [1, 2, 3, 4, 5, ..., 20] to make it easy to learn
    vocab_size = minimal_config.model.vocab_size
    seq_length = 32
    batch_size = 8
    num_batches = 10  # Only 10 batches total, repeat them

    # Create a simple repeating pattern
    pattern_length = 20
    pattern = torch.arange(1, pattern_length + 1, dtype=torch.long)  # [1, 2, ..., 20]

    # Repeat pattern to fill sequence length
    num_repeats = (seq_length // pattern_length) + 1
    base_seq = pattern.repeat(num_repeats)[:seq_length]

    # Create batch by adding small variations
    synthetic_data = []
    for i in range(num_batches):
        # Same pattern for all samples in batch (maximize overfitting potential)
        batch = base_seq.unsqueeze(0).repeat(batch_size, 1)
        synthetic_data.append(batch)

    # 2. Initialize model
    model = SEMModel(minimal_config).to(device)
    model.train()

    # 3. Initialize optimizer with aggressive settings
    optimizer = ComplexAdamW(
        model.parameters(),
        lr=minimal_config.training.learning_rate,
        weight_decay=0.0,  # No regularization
        temperature=1e-5,
    )

    # 4. Training loop
    num_steps = 200
    losses = []
    grad_norms = []

    print("\n" + "=" * 60)
    print("QUICK CONVERGENCE TEST - Model Overfitting Check")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Batch size: {batch_size}")
    print(f"Number of training steps: {num_steps}")
    print(f"Learning rate: {minimal_config.training.learning_rate}")
    print("=" * 60 + "\n")

    for step in range(num_steps):
        # Cycle through synthetic data
        batch_idx = step % num_batches
        token_ids = synthetic_data[batch_idx].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(token_ids, targets=token_ids)
        loss = output["loss"]

        # Check for NaN
        if torch.isnan(loss):
            pytest.fail(f"NaN loss at step {step}")

        # Backward pass
        loss.backward()

        # Check gradients
        total_grad_norm = 0.0
        num_params_with_grad = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm**2
                num_params_with_grad += 1

                # Check for NaN/Inf gradients
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    pytest.fail(f"NaN/Inf gradient in {name} at step {step}")

        total_grad_norm = total_grad_norm**0.5
        grad_norms.append(total_grad_norm)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), minimal_config.training.gradient_clip
        )

        # Optimizer step
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Print progress every 20 steps
        if (step + 1) % 20 == 0 or step == 0:
            print(
                f"Step {step + 1:3d}/{num_steps} | Loss: {loss.item():.4f} | "
                f"Grad norm: {total_grad_norm:.4f}"
            )

    # 5. Verify convergence
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss

    print("\n" + "=" * 60)
    print("CONVERGENCE RESULTS")
    print("=" * 60)
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Loss reduction: {loss_reduction * 100:.1f}%")
    print(f"Average grad norm: {sum(grad_norms) / len(grad_norms):.4f}")
    print(f"Min loss: {min(losses):.4f}")
    print(f"Max loss: {max(losses):.4f}")
    print("=" * 60 + "\n")

    # Assertions
    assert loss_reduction >= 0.5, (
        f"Loss should decrease by at least 50%, got {loss_reduction * 100:.1f}%"
    )
    assert final_loss < initial_loss, f"Final loss should be lower than initial loss"
    assert all(grad_norm > 0 for grad_norm in grad_norms), (
        "All gradient norms should be positive (gradients flowing)"
    )
    assert not any(torch.isnan(torch.tensor(loss)) for loss in losses), (
        "No NaN in losses"
    )

    print("✅ CONVERGENCE TEST PASSED")
    print(f"   Model successfully learned the synthetic pattern")
    print(f"   Loss reduced by {loss_reduction * 100:.1f}%")
    print(
        f"   Gradients flowing correctly (avg norm: {sum(grad_norms) / len(grad_norms):.4f})"
    )


if __name__ == "__main__":
    # Allow running directly for debugging
    config = SEMConfig(
        model=ModelConfig(
            hidden_dim=128,
            num_layers=2,
            vocab_size=1000,
            max_seq_length=128,
        ),
        encoder=EncoderConfig(
            sdr_sparsity=16,
            sdr_candidates=64,
            sinkhorn_epsilon=0.05,
            sinkhorn_max_iter=50,
            sinkhorn_tol=1e-3,
        ),
        spinor=SpinorConfig(
            block_size=8,
            num_blocks=16,
            state_dim=32,
            mimo_groups=4,
            d_conv=4,
        ),
        propagator=PropagatorConfig(
            cayley_dt=0.1,
            cg_max_iter=10,
            cg_tol=1e-4,
            nonlinear_alpha=0.1,
            laplacian_sparsity=5,
            lazy_cg=False,
            lazy_cg_tol=1e-3,
            direct_solve=True,
            pit_gamma=0.1,
        ),
        quantizer=QuantizerConfig(
            codebook_size=256,
            group_size=2,
            fisher_ema_decay=0.99,
            outlier_percentile=0.01,
            dead_code_threshold=100,
        ),
        sampler=SamplerConfig(temperature=1.0, top_k=50, top_p=0.95),
        training=TrainingConfig(
            batch_size=8,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_steps=20,
            max_steps=200,
            gradient_clip=1.0,
            dtype="complex64",
            low_vram_mode=False,
            unitary_lambda=0.01,
            decay_steps=180,
            lr_min_ratio=0.1,
        ),
        distillation=DistillationConfig(enabled=False),
    )
    test_quick_convergence(config)
