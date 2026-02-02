import sys

sys.path.insert(0, r"C:\Users\freeb\work\ml")

import time
import torch
from sem.config import SEMConfig
from sem.model import SEMModel
from sem.utils.complex_adamw import ComplexAdamW

print("=" * 70)
print("SEM BOTTLENECK FIX VERIFICATION")
print("=" * 70)

# Full-size model config
config = SEMConfig()
print(f"\nModel config:")
print(f"  hidden_dim: {config.model.hidden_dim}")
print(f"  num_layers: {config.model.num_layers}")
print(f"  vocab_size: {config.model.vocab_size}")

# Build model
print("\nBuilding model...")
model = SEMModel(config)
counts = model.count_parameters()
print(f"Parameters: {counts['total']['effective_real']:,} effective real")

# Optimizer
optimizer = ComplexAdamW(model.parameters(), lr=3e-4)
model.train()

# Test with micro_batch_size from config
B = config.training.micro_batch_size  # 8
S = 1024  # sequence length


def get_batch():
    return torch.randint(0, config.model.vocab_size, (B, S))


print(f"\nBatch size: {B}, Sequence length: {S}")
print(f"Tokens per batch: {B * S:,}")

# Warmup
print("\nWarming up (3 steps)...")
for _ in range(3):
    batch = get_batch()
    output = model(batch, targets=batch)
    loss = output["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Profile 10 steps
print("\n" + "=" * 70)
print("PROFILING 10 TRAINING STEPS")
print("=" * 70)

times_forward = []
times_backward = []
times_optim = []
times_total = []

for step in range(10):
    t0 = time.time()

    # Data
    batch = get_batch()

    # Forward
    t1 = time.time()
    output = model(batch, targets=batch)
    loss = output["loss"]
    t2 = time.time()

    # Backward
    loss.backward()
    t3 = time.time()

    # Optimizer
    optimizer.step()
    optimizer.zero_grad()
    t4 = time.time()

    dt_forward = t2 - t1
    dt_backward = t3 - t2
    dt_optim = t4 - t3
    dt_total = t4 - t0

    times_forward.append(dt_forward)
    times_backward.append(dt_backward)
    times_optim.append(dt_optim)
    times_total.append(dt_total)

    print(
        f"Step {step:2d}: loss={loss.item():.4f} | "
        f"fwd={dt_forward:.2f}s bwd={dt_backward:.2f}s opt={dt_optim:.2f}s | "
        f"total={dt_total:.2f}s | {B * S / dt_total:.0f} tok/s"
    )

# Summary
print("\n" + "=" * 70)
print("AVERAGES (last 5 steps, excluding warmup)")
print("=" * 70)

avg_fwd = sum(times_forward[5:]) / 5
avg_bwd = sum(times_backward[5:]) / 5
avg_opt = sum(times_optim[5:]) / 5
avg_total = sum(times_total[5:]) / 5

print(f"Forward pass:   {avg_fwd:.2f}s ({avg_fwd / avg_total * 100:.0f}%)")
print(f"Backward pass:  {avg_bwd:.2f}s ({avg_bwd / avg_total * 100:.0f}%)")
print(f"Optimizer step: {avg_opt:.2f}s ({avg_opt / avg_total * 100:.0f}%)")
print(f"Total per step: {avg_total:.2f}s")
print(f"Throughput:     {B * S / avg_total:.0f} tokens/second")

# Bottleneck identification
print("\n" + "=" * 70)
print("BOTTLENECK ANALYSIS")
print("=" * 70)

components = [
    ("Forward", avg_fwd),
    ("Backward", avg_bwd),
    ("Optimizer", avg_opt),
]
components.sort(key=lambda x: x[1], reverse=True)

print("\nRanked by time (slowest first):")
for i, (name, t) in enumerate(components, 1):
    print(f"  {i}. {name:<15} {t:.2f}s ({t / avg_total * 100:.0f}%)")

print(f"\nBottleneck: {components[0][0]} ({components[0][1]:.2f}s)")

print("\n" + "=" * 70)
print("EXPECTED TRAINING TIME FOR 100K STEPS")
print("=" * 70)
steps_remaining = 100000
hours = (steps_remaining * avg_total) / 3600
print(f"At {avg_total:.2f}s per step: {hours:.1f} hours")
print(f"Old estimate (11s per step): {(steps_remaining * 11) / 3600:.1f} hours")
print(f"Speedup: {11 / avg_total:.1f}x")

print("\n" + "=" * 70)
