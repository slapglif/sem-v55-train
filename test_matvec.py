import sys
sys.path.insert(0, r'C:\Users\freeb\work\ml')

import time
import torch
from sem.propagator.hamiltonian import GraphLaplacianHamiltonian

# Test sparse matvec performance
D = 256
H = GraphLaplacianHamiltonian(D, sparsity=5)
H.cache_weights()

# Test data
B, S = 8, 1024
vr = torch.randn(B, S, D)
vi = torch.randn(B, S, D)

print(f"Testing matvec with D={D}, sparsity={H.sparsity}")
print(f"Input shape: [{B}, {S}, {D}]")

# Warmup
_ = H.matvec_real(vr)

# Time separate matvecs
n = 10
t0 = time.time()
for _ in range(n):
    Hvr = H.matvec_real(vr)
    Hvi = H.matvec_real(vi)
t1 = time.time()
print(f"\nSeparate matvecs: {(t1-t0)/n*1000:.2f}ms per call")

# Time fused matvec
t0 = time.time()
for _ in range(n):
    Hvr, Hvi = H.matvec_real_fused(vr, vi)
t1 = time.time()
print(f"Fused matvec:     {(t1-t0)/n*1000:.2f}ms per call")

# Cost per matvec
print(f"\nSparse matrix: {H.edge_indices.shape[1]} edges")
print(f"Cost per matvec: O({H.edge_indices.shape[1]} * {D}) = {H.edge_indices.shape[1] * D:,} ops")

# Estimate CG cost
print(f"\nCG solver cost estimate:")
print(f"  - 5 iterations per layer")
print(f"  - 8 layers")
print(f"  - 2 matvecs per iteration (forward + backward)")
print(f"  - Total matvecs per step: ~80")
print(f"  - Estimated matvec time: {80 * (t1-t0)/n * 1000:.0f}ms")
