"""Minimal profiler to find SEM bottleneck on XPU."""

import torch
import time
import sys

sys.path.insert(0, ".")

from sem.model import SEMModel
from sem.config import SEMConfig


def profile_forward():
    print("=" * 60)
    print("SEM COMPONENT PROFILER - XPU")
    print("=" * 60)

    # Minimal config for fast testing
    config = SEMConfig()
    config.model.hidden_dim = 256
    config.model.num_layers = 2  # Reduced for testing
    config.model.vocab_size = 1000
    config.model.max_seq_length = 512  # Shorter
    config.propagator.cg_max_iter = 3  # SEOP FIX: Use 3 instead of 40
    config.propagator.lazy_cg = True

    device = torch.device(
        "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"CG max_iter: {config.propagator.cg_max_iter}")
    print(f"Num layers: {config.model.num_layers}")
    print()

    # Build model
    print("Building model...")
    t0 = time.perf_counter()
    model = SEMModel(config).to(device)
    t_build = time.perf_counter() - t0
    print(f"Model build: {t_build:.2f}s")
    print()

    # Test input
    B, S = 2, 512
    token_ids = torch.randint(0, config.model.vocab_size, (B, S), device=device)
    token_freqs = torch.rand(config.model.vocab_size, device=device)

    # Warm up
    print("Warming up...")
    for _ in range(2):
        with torch.no_grad():
            _ = model(token_ids, token_freqs=token_freqs)
    print()

    # Profile components
    print("PROFILING COMPONENTS:")
    print("-" * 60)

    # 1. Encoder only
    psi = model.encoder(token_ids, token_freqs=token_freqs)
    torch.xpu.synchronize() if device.type == "xpu" else None

    # 2. Mamba layers (the sequential scan)
    print("Testing Mamba layers (sequential scan)...")
    psi_test = psi.clone()
    for i, mamba_layer in enumerate(model.mamba_layers):
        t0 = time.perf_counter()
        psi_test = mamba_layer(psi_test)
        torch.xpu.synchronize() if device.type == "xpu" else None
        t_layer = time.perf_counter() - t0
        print(f"  Mamba layer {i}: {t_layer * 1000:.1f}ms")

    total_mamba = sum(1 for _ in model.mamba_layers)  # Placeholder for actual timing
    print()

    # 3. Propagator (CG solver) - NO GRAD TEST ONLY
    print("Testing Propagator (CG solver)...")
    with torch.no_grad():
        t0 = time.perf_counter()
        psi_test = model.propagator(psi)
        torch.xpu.synchronize() if device.type == "xpu" else None
        t_prop = time.perf_counter() - t0
    print(f"  Propagator forward: {t_prop * 1000:.1f}ms")

    # Check CG skip rate
    for i, layer in enumerate(model.propagator.layers):
        print(f"  Layer {i} CG skip rate: {layer.cg_skip_rate * 100:.1f}%")
    print()

    # 4. Full forward with backward
    print("Testing FULL forward + backward...")
    model.train()

    t0 = time.perf_counter()
    output = model(token_ids, targets=token_ids, token_freqs=token_freqs)
    loss = output["loss"]
    torch.xpu.synchronize() if device.type == "xpu" else None
    t_forward = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss.backward()
    torch.xpu.synchronize() if device.type == "xpu" else None
    t_backward = time.perf_counter() - t0

    print(f"  Forward:  {t_forward * 1000:.1f}ms")
    print(f"  Backward: {t_backward * 1000:.1f}ms")
    print(f"  Total:    {(t_forward + t_backward) * 1000:.1f}ms")
    print(f"  Loss:     {loss.item():.4f}")

    print()
    print("=" * 60)
    print("EXPECTED STEP TIME (with accum_steps=1):")
    print(f"  {t_forward + t_backward:.2f}s per step")
    print(f"  ~{int((B * S) / (t_forward + t_backward))} tok/s")
    print("=" * 60)


if __name__ == "__main__":
    profile_forward()
