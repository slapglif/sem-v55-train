import sys
sys.path.insert(0, r'C:\Users\freeb\work\ml')

import time
import torch
from sem.propagator.hamiltonian import GraphLaplacianHamiltonian

# Test with D=256 (typical SEM config)
D = 256
H = GraphLaplacianHamiltonian(D, sparsity=5)
H.cache_weights()

# Test data
B, S = 8, 1024
vr = torch.randn(B, S, D)
vi = torch.randn(B, S, D)

print(f"Testing matvec with D={D}, sparsity={H.sparsity}")
print(f"Input shape: [{B}, {S}, {D}]")

# Warmup (this builds the dense cache)
print("\nWarming up (building dense cache)...")
_ = H.matvec_real(vr)
if hasattr(H, '_cached_A_dense'):
    print(f"Dense cache built: {H._cached_A_dense.shape}")

# Time matvec
n = 10
times = []
for _ in range(n):
    t0 = time.time()
    Hvr = H.matvec_real(vr)
    t1 = time.time()
    times.append(t1-t0)

avg_time = sum(times)/len(times)
print(f"\nAverage matvec time: {avg_time*1000:.2f}ms")
print(f"Speedup vs old: {113.72/avg_time:.1f}x")

# Estimate full training step
print(f"\nCG solver cost estimate:")
print(f"  - 5 iterations per layer")
print(f"  - 8 layers")
print(f"  - ~80 matvecs per step")
print(f"  - Estimated matvec time: {80 * avg_time * 1000:.0f}ms ({80 * avg_time:.1f}s)")
print(f"  - Old time: ~11000ms (11s)")
