"""Profile a single training step to verify backward pass optimization."""

import torch
import time
import sys

# Add the project root to path
sys.path.insert(0, r"C:\Users\freeb\work\ml")

from sem.model import SEMModel
from sem.config import SEMConfig
from sem.training.trainer import SEMTrainer


def profile_training_step():
    """Profile a single training step with detailed timing."""
    print("=" * 70)
    print("SEM Training Step Profiling - Backward Pass Optimization")
    print("=" * 70)

    # Create minimal config for testing
    config = SEMConfig()
    config.model.hidden_dim = 256
    config.model.num_layers = 4  # Reduced for faster testing
    config.model.vocab_size = 1000  # Small vocab for testing
    config.model.max_seq_length = 512  # Shorter sequences
    config.training.micro_batch_size = 2
    config.training.batch_size = 2
    config.training.gradient_checkpointing = True
    config.propagator.cg_max_iter = 5

    # Create model
    print("\n1. Building model...")
    model = SEMModel(config)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Move to device
    device = torch.device("cpu")  # Test on CPU first
    model = model.to(device)
    print(f"   Device: {device}")

    # Create synthetic batch
    batch_size = 2
    seq_len = 512
    token_ids = torch.randint(
        0, config.model.vocab_size, (batch_size, seq_len), device=device
    )
    targets = torch.randint(
        0, config.model.vocab_size, (batch_size, seq_len), device=device
    )
    token_freqs = torch.rand(batch_size, config.model.vocab_size, device=device)

    # Warm up
    print("\n2. Warm-up passes...")
    for _ in range(2):
        model.zero_grad()
        outputs = model(token_ids, targets, token_freqs)
        loss = outputs["loss"]
        loss.backward()

    # Clear gradients
    model.zero_grad(set_to_none=True)

    # Profile one step
    print("\n3. Profiling one training step...")
    print("-" * 70)

    torch.set_num_threads(8)

    # Forward pass timing
    t0 = time.perf_counter()
    outputs = model(token_ids, targets, token_freqs)
    t_forward = time.perf_counter() - t0
    loss = outputs["loss"]

    # Backward pass timing
    t0 = time.perf_counter()
    loss.backward()
    t_backward = time.perf_counter() - t0

    # Optimizer step timing (simulated)
    t0 = time.perf_counter()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data -= 0.001 * p.grad
    t_opt = time.perf_counter() - t0

    total = t_forward + t_backward + t_opt

    print(
        f"Forward pass:   {t_forward * 1000:8.2f}ms ({t_forward / total * 100:5.1f}%)"
    )
    print(
        f"Backward pass:  {t_backward * 1000:8.2f}ms ({t_backward / total * 100:5.1f}%)"
    )
    print(f"Optimizer step: {t_opt * 1000:8.2f}ms ({t_opt / total * 100:5.1f}%)")
    print("-" * 70)
    print(f"Total step:     {total * 1000:8.2f}ms")
    print(f"Throughput:     {(batch_size * seq_len) / total:,.0f} tokens/sec")

    # Analyze backward/forward ratio
    ratio = t_backward / t_forward
    print(f"\nBwd/Fwd ratio:  {ratio:.2f}x")

    if ratio < 3.0:
        print("[OK] Backward pass is well-optimized (< 3x forward)")
    elif ratio < 5.0:
        print("[WARN] Backward pass is slower than expected (3-5x forward)")
    else:
        print("[CRITICAL] Backward pass is severely bottlenecked (> 5x forward)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    profile_training_step()
