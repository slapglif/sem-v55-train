"""Quick XPU test to verify no NaN loss before HF deployment."""

import torch
import sys

sys.path.insert(0, ".")

from sem.model import SEMModel
from sem.config import SEMConfig


def test_xpu():
    print("=" * 60)
    print("XPU LOCAL TEST - Verify No NaN Loss")
    print("=" * 60)

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("XPU not available, testing on CPU...")
        device = torch.device("cpu")
    else:
        device = torch.device("xpu")
        print(f"XPU: {torch.xpu.get_device_name(0)}")
        print(
            f"VRAM: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )

    # Minimal config for fast testing
    config = SEMConfig()
    config.model.hidden_dim = 64
    config.model.num_layers = 1
    config.model.vocab_size = 512
    config.model.max_seq_length = 64
    config.propagator.cg_max_iter = 4
    config.training.micro_batch_size = 1
    config.training.batch_size = 1
    config.encoder.sdr_sparsity = 8
    config.encoder.sdr_candidates = 32
    config.encoder.sinkhorn_max_iter = 10
    config.encoder.sinkhorn_epsilon = 0.1
    config.encoder.sinkhorn_tol = 1.0e-2

    print(f"\nConfig: D={config.model.hidden_dim}, layers={config.model.num_layers}")
    print(f"CG max_iter: {config.propagator.cg_max_iter}")
    print(f"Device: {device}")

    # Build model
    print("\nBuilding model...")
    model = SEMModel(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test data
    B, S = 1, 64
    token_ids = torch.randint(0, config.model.vocab_size, (B, S), device=device)
    token_freqs = torch.rand(config.model.vocab_size, device=device)

    # Forward + backward test
    print("\nTesting forward + backward...")
    model.train()

    for step in range(3):
        output = model(token_ids, targets=token_ids, token_freqs=token_freqs)
        loss = output["loss"]

        if torch.isnan(loss):
            print(f"[FAIL] Step {step}: NaN loss detected!")
            return False

        print(f"[PASS] Step {step}: loss={loss.item():.4f}")

        loss.backward()

        # Check gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                if torch.isnan(p.grad).any():
                    print(f"[FAIL] NaN gradient detected!")
                    return False

        if not has_grad:
            print("[FAIL] No gradients computed!")
            return False

        # Zero grads for next step
        model.zero_grad()

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED - No NaN, gradients flowing correctly")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_xpu()
    sys.exit(0 if success else 1)
