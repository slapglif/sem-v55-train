"""Test gradient checkpointing with complex Cayley propagator."""

import torch
import pytest
from pathlib import Path

from sem.config import SEMConfig, V8Config
from sem.training.lightning_module import SEMLightningModule


def test_gradient_checkpointing_no_tensor_mismatch():
    """Verify gradient checkpointing works without tensor count mismatch.

    Bug: torch.utils.checkpoint.CheckpointError - different number of tensors
    saved during forward (152) vs recomputation (131).

    Root cause: Cayley propagator uses conditional caching that creates different
    code paths in forward vs recomputation.

    Fix: Temporarily disable cache during checkpointed forward passes.
    """
    # Load a real config and override for a small test model.
    # Use SEMConfig to ensure defaults exist for fields omitted in YAML.
    config_path = Path(__file__).parent.parent / "configs" / "max_aggression.yaml"
    config = SEMConfig.from_yaml(config_path)

    config.model.hidden_dim = 64
    config.model.num_layers = 2
    config.model.vocab_size = 256
    config.model.max_seq_length = 128
    config.spinor.num_blocks = 8
    config.propagator.lazy_cg_tol = 1e-6
    config.training.gradient_checkpointing = True
    config.training.batch_size = 2
    config.training.low_vram_mode = False
    config.distillation.enabled = False
    config.v8 = V8Config(
        use_lindblad=False,
        use_hybrid_automata=False,
        use_quaternionic=False,
        use_mhc=False,
    )

    # Create model with gradient checkpointing enabled
    module = SEMLightningModule(config)
    module.eval()  # Disable dropout for deterministic behavior

    # Create dummy input
    B, S = 2, 32
    token_ids = torch.randint(0, config.model.vocab_size, (B, S))

    # Run forward pass (this will use checkpointing)
    with torch.enable_grad():
        output = module(token_ids, targets=token_ids)
        loss = output["loss"]

        # CRITICAL: Backward pass triggers recomputation with checkpoint
        # If cache interferes, this will raise:
        # CheckpointError: different number of tensors saved (152 vs 131)
        loss.backward()

    # Verify gradients exist (checkpointing worked)
    has_grads = any(p.grad is not None for p in module.parameters() if p.requires_grad)
    assert has_grads, "Gradient checkpointing failed - no gradients computed"

    # Verify no NaN gradients
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    print("✓ Gradient checkpointing working correctly - no tensor mismatch")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
