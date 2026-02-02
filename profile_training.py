"""Profiling script to identify CPU training bottlenecks in SEM V5.5.

Measures timing for:
- Data loading (synthetic batch creation)
- Forward pass
- Backward pass
- Optimizer step
- EMA teacher update (if enabled)

Run with: python profile_training.py
"""

import time
import torch
import torch.nn as nn
from pathlib import Path

from sem.config import SEMConfig
from sem.model import SEMModel
from sem.utils.complex_adamw import ComplexAdamW
from sem.training.distillation import EMATeacher

# Suppress logging noise
import logging

logging.basicConfig(level=logging.WARNING)
for name in ["sem", "transformers", "datasets"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def get_synthetic_batch(batch_size: int, seq_length: int, vocab_size: int, device: str):
    """Create synthetic batch of token IDs."""
    return torch.randint(0, vocab_size, (batch_size, seq_length), device=device)


def profile_training(
    num_steps: int = 10,
    batch_size: int = 4,
    seq_length: int = 512,
    device: str = "cpu",
    use_ema: bool = False,
):
    """Profile training loop with detailed timing breakdown."""

    print(f"\n{'=' * 70}")
    print(f"SEM V5.5 Training Profiler")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Seq length: {seq_length}")
    print(f"Num steps: {num_steps}")
    print(f"Use EMA teacher: {use_ema}")
    print(f"{'=' * 70}\n")

    # Setup
    config = SEMConfig()
    config.model.hidden_dim = 128  # Smaller for faster profiling
    config.model.num_layers = 4
    config.training.batch_size = batch_size
    config.training.seq_length = seq_length

    model = SEMModel(config).to(device)
    optimizer = ComplexAdamW(model.parameters(), lr=3e-4)

    # Optional EMA teacher
    ema_teacher = None
    if use_ema:
        ema_teacher = EMATeacher(model, decay=0.999)

    # Warmup: compile/JIT if available
    print("Warming up model...")
    with torch.no_grad():
        _ = model(
            get_synthetic_batch(batch_size, seq_length, config.model.vocab_size, device)
        )

    # Timing accumulators
    times = {
        "data_load": [],
        "forward": [],
        "backward": [],
        "optimizer": [],
        "ema_update": [],
        "total": [],
    }

    losses = []

    print(
        f"{'Step':<6} {'Data':<8} {'Forward':<8} {'Backward':<8} {'Optim':<8} {'EMA':<8} {'Total':<8} {'Loss':<10}"
    )
    print("-" * 80)

    # Profile loop
    for step in range(num_steps):
        step_start = time.perf_counter()

        # Data loading
        t0 = time.perf_counter()
        batch = get_synthetic_batch(
            batch_size, seq_length, config.model.vocab_size, device
        )
        t_data = time.perf_counter() - t0
        times["data_load"].append(t_data)

        # Forward pass
        t0 = time.perf_counter()
        output = model(batch, targets=batch)
        loss = output["loss"]
        t_forward = time.perf_counter() - t0
        times["forward"].append(t_forward)
        losses.append(loss.item())

        # Backward pass
        t0 = time.perf_counter()
        loss.backward()
        t_backward = time.perf_counter() - t0
        times["backward"].append(t_backward)

        # Optimizer step
        t0 = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        t_optim = time.perf_counter() - t0
        times["optimizer"].append(t_optim)

        # EMA update
        t_ema = 0.0
        if ema_teacher is not None:
            t0 = time.perf_counter()
            ema_teacher.update(model)
            t_ema = time.perf_counter() - t0
            times["ema_update"].append(t_ema)

        t_total = time.perf_counter() - step_start
        times["total"].append(t_total)

        # Print per-step breakdown
        print(
            f"{step:<6} "
            f"{t_data * 1000:<8.2f} "
            f"{t_forward * 1000:<8.2f} "
            f"{t_backward * 1000:<8.2f} "
            f"{t_optim * 1000:<8.2f} "
            f"{t_ema * 1000:<8.2f} "
            f"{t_total * 1000:<8.2f} "
            f"{loss.item():<10.4f}"
        )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (milliseconds)")
    print("=" * 80)

    def print_stats(name, values):
        if not values:
            return
        avg = sum(values) / len(values)
        min_v = min(values)
        max_v = max(values)
        print(
            f"{name:<20} avg={avg * 1000:>8.2f}ms  min={min_v * 1000:>8.2f}ms  max={max_v * 1000:>8.2f}ms"
        )

    print_stats("Data Loading", times["data_load"])
    print_stats("Forward Pass", times["forward"])
    print_stats("Backward Pass", times["backward"])
    print_stats("Optimizer Step", times["optimizer"])
    if times["ema_update"]:
        print_stats("EMA Update", times["ema_update"])
    print_stats("Total Step", times["total"])

    # Breakdown percentages
    print("\n" + "=" * 80)
    print("TIME BREAKDOWN (% of total step time)")
    print("=" * 80)

    avg_total = sum(times["total"]) / len(times["total"])
    avg_data = sum(times["data_load"]) / len(times["data_load"])
    avg_forward = sum(times["forward"]) / len(times["forward"])
    avg_backward = sum(times["backward"]) / len(times["backward"])
    avg_optim = sum(times["optimizer"]) / len(times["optimizer"])
    avg_ema = (
        sum(times["ema_update"]) / len(times["ema_update"])
        if times["ema_update"]
        else 0.0
    )

    print(f"Data Loading:  {avg_data / avg_total * 100:>6.1f}%")
    print(f"Forward Pass:  {avg_forward / avg_total * 100:>6.1f}%")
    print(f"Backward Pass: {avg_backward / avg_total * 100:>6.1f}%")
    print(f"Optimizer:     {avg_optim / avg_total * 100:>6.1f}%")
    if avg_ema > 0:
        print(f"EMA Update:    {avg_ema / avg_total * 100:>6.1f}%")

    # Identify bottleneck
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)

    components = [
        ("Data Loading", avg_data),
        ("Forward Pass", avg_forward),
        ("Backward Pass", avg_backward),
        ("Optimizer", avg_optim),
    ]
    if avg_ema > 0:
        components.append(("EMA Update", avg_ema))

    components.sort(key=lambda x: x[1], reverse=True)

    print(f"\nRanked by time (slowest first):")
    for i, (name, time_val) in enumerate(components, 1):
        pct = time_val / avg_total * 100
        print(f"  {i}. {name:<20} {time_val * 1000:>8.2f}ms ({pct:>5.1f}%)")

    bottleneck_name, bottleneck_time = components[0]
    print(
        f"\n⚠️  PRIMARY BOTTLENECK: {bottleneck_name} ({bottleneck_time * 1000:.2f}ms per step)"
    )

    # Loss trend
    print("\n" + "=" * 80)
    print("LOSS TREND")
    print("=" * 80)
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss:   {losses[-1]:.4f}")
    print(f"Change:       {losses[-1] - losses[0]:+.4f}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Profile on CPU
    profile_training(
        num_steps=10,
        batch_size=4,
        seq_length=512,
        device="cpu",
        use_ema=False,
    )

    # Optional: Profile with EMA
    print("\n\nProfiling WITH EMA teacher enabled...\n")
    profile_training(
        num_steps=10,
        batch_size=4,
        seq_length=512,
        device="cpu",
        use_ema=True,
    )
