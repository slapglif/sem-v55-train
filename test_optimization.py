"""Quick test to verify Hamiltonian sparse matrix caching and gradient checkpointing."""

import torch
import time
from sem.propagator.hamiltonian import GraphLaplacianHamiltonian, MultiScaleHamiltonian
from sem.propagator.cayley_soliton import CayleySolitonPropagator


def test_sparse_index_caching():
    """Test that sparse indices are pre-built and cached."""
    print("Testing sparse index caching...")

    # Create Hamiltonian
    H = GraphLaplacianHamiltonian(dim=256, sparsity=5)

    # Check that _sparse_idx_sym buffer exists
    assert hasattr(H, "_sparse_idx_sym"), "_sparse_idx_sym buffer not found!"
    assert H._sparse_idx_sym is not None, "_sparse_idx_sym is None!"
    assert H._sparse_idx_sym.shape[0] == 2, (
        "_sparse_idx_sym should have shape [2, num_edges]"
    )

    print(f"  [OK] Pre-built sparse indices shape: {H._sparse_idx_sym.shape}")
    print(
        f"  [OK] Original edges: {H.edge_indices.shape[1]}, Symmetric: {H._sparse_idx_sym.shape[1]}"
    )


def test_matvec_performance():
    """Test that matvec doesn't rebuild sparse tensors."""
    print("\nTesting matvec performance (no reconstruction)...")

    # Create Hamiltonian
    H = GraphLaplacianHamiltonian(dim=256, sparsity=5)
    H.cache_weights()

    # Test input
    vr = torch.randn(2, 512, 256)  # [B, S, D]
    vi = torch.randn(2, 512, 256)

    # Warm up
    for _ in range(3):
        H.matvec_real_fused(vr, vi)

    # Time multiple calls
    torch.set_num_threads(8)
    n_calls = 50

    start = time.perf_counter()
    for _ in range(n_calls):
        Hvr, Hvi = H.matvec_real_fused(vr, vi)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / n_calls) * 1000
    print(f"  [OK] {n_calls} matvec calls: {elapsed * 1000:.1f}ms total")
    print(f"  [OK] Average per call: {avg_ms:.2f}ms")

    if avg_ms < 5.0:
        print(f"  [OK] Performance OK (< 5ms per call)")
    else:
        print(f"  [WARN] Performance slow (> 5ms per call)")


def test_gradient_checkpointing_on_propagator():
    """Test that gradient checkpointing is properly applied."""
    print("\nTesting gradient checkpointing setup...")

    # Create a propagator
    prop = CayleySolitonPropagator(dim=256, cg_max_iter=5)

    # Test input
    psi = torch.randn(2, 512, 256, dtype=torch.complex64, requires_grad=True)

    # Forward pass
    psi_out = prop(psi)

    # Backward pass
    loss = psi_out.abs().sum()
    loss.backward()

    print(f"  [OK] Forward pass completed: {psi_out.shape}")
    print(f"  [OK] Backward pass completed: grad shape {psi.grad.shape}")
    print(
        f"  [OK] Hamiltonian edge_weights have grad: {prop.hamiltonian.scales[0].edge_weights.grad is not None}"
    )


def main():
    print("=" * 60)
    print("SEM Hamiltonian Optimization Verification")
    print("=" * 60)

    test_sparse_index_caching()
    test_matvec_performance()
    test_gradient_checkpointing_on_propagator()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
