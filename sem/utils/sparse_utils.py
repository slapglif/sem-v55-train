"""Sparse tensor utilities for SEM V5.5.

Provides helpers for constructing sparse matrices (Graph Laplacian,
transport plans) and sparse-dense operations used in the Cayley
propagator and MESH encoder.
"""

import torch
from torch import Tensor
from sem.utils.complex_ops import safe_complex


def build_graph_laplacian(
    dim: int,
    sparsity: int = 5,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Build a sparse Graph Laplacian matrix for the Cayley propagator.

    Constructs a symmetric, positive semi-definite Laplacian L = D - A
    where A is a sparse adjacency with ~sparsity connections per node.

    Args:
        dim: Matrix dimension (D x D)
        sparsity: Average number of neighbors per node
        dtype: Tensor dtype (complex64 for wave propagation)
        device: Target device

    Returns:
        Sparse COO tensor [dim, dim]
    """
    # Build adjacency: connect each node to ~sparsity nearest neighbors
    # Use a ring + random long-range connections (small-world topology)
    rows, cols, vals = [], [], []

    for i in range(dim):
        # Local ring connections (nearest neighbors)
        for offset in range(1, max(2, sparsity // 2) + 1):
            j = (i + offset) % dim
            rows.extend([i, j])
            cols.extend([j, i])
            weight = 1.0 / offset  # Distance-decaying weight
            vals.extend([weight, weight])

        # Random long-range connections
        num_random = max(0, sparsity - sparsity // 2 * 2)
        if num_random > 0:
            targets = torch.randint(0, dim, (num_random,)).tolist()
            for j in targets:
                if j != i:
                    rows.extend([i, j])
                    cols.extend([j, i])
                    vals.extend([0.5, 0.5])

    # Build adjacency
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)

    # Remove duplicates by coalescing
    A = torch.sparse_coo_tensor(indices, values, (dim, dim)).coalesce()

    # Degree matrix
    degree = torch.sparse.sum(A, dim=1).to_dense()
    D_indices = torch.arange(dim).unsqueeze(0).expand(2, -1)
    D = torch.sparse_coo_tensor(D_indices, degree, (dim, dim))

    # Laplacian L = D - A
    L = (D - A).coalesce()

    # Convert to target dtype
    if dtype.is_complex:
        L = safe_complex(L.to_dense().float(), torch.zeros(dim, dim)).to_sparse()

    return L.to(device)


def sparse_complex_mv(sparse_mat: Tensor, vec: Tensor) -> Tensor:
    """Sparse matrix-vector multiply for complex tensors.

    Args:
        sparse_mat: Sparse [D, D] complex64
        vec: Dense [*, D] complex64
    Returns:
        Dense [*, D] complex64
    """
    if sparse_mat.is_sparse:
        # Convert to dense for complex matmul (PyTorch sparse complex is limited)
        dense_mat = sparse_mat.to_dense()
        return torch.matmul(vec, dense_mat.T)
    return torch.matmul(vec, sparse_mat.T)


def top_k_sparsify(x: Tensor, k: int, dim: int = -1) -> Tensor:
    """Keep only top-k values along dimension, zero out the rest.

    Args:
        x: Input tensor
        k: Number of values to keep
        dim: Dimension to sparsify along
    Returns:
        Sparsified tensor (same shape, most values zeroed)
    """
    topk_vals, topk_idx = torch.topk(x.abs(), k, dim=dim)
    mask = torch.zeros_like(x)
    mask.scatter_(dim, topk_idx, 1.0)
    return x * mask
