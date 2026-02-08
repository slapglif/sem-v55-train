"""Sparse Hamiltonian (Graph Laplacian) for wave propagation.

The Hamiltonian H defines the connectivity of the Crystal Manifold.
It is a sparse, Hermitian, positive semi-definite matrix that acts
as the Graph Laplacian of the propagation topology.

The sparsity pattern is a small-world graph: local ring connections
plus random long-range connections, giving O(sparsity * D) nonzeros
instead of O(D^2) for dense.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from sem.utils.complex_ops import safe_complex


class GraphLaplacianHamiltonian(nn.Module):
    edge_indices: Tensor
    edge_weights: nn.Parameter
    _cached_w: Tensor | None
    _cached_degree: Tensor | None
    _cached_sparse_A: Tensor | None

    """Learnable sparse Graph Laplacian Hamiltonian.

    H = D - A where A is a learnable sparse adjacency matrix
    and D is the degree matrix. H is Hermitian by construction.
    """

    def __init__(self, dim: int, sparsity: int = 5):
        """
        Args:
            dim: Matrix dimension D
            sparsity: Average nonzeros per row
        """
        super().__init__()
        self.dim = dim
        self.sparsity = sparsity

        # Build fixed sparsity pattern (small-world topology)
        edges = self._build_sparsity_pattern(dim, sparsity)
        self.register_buffer("edge_indices", edges)  # [2, num_edges]

        num_edges = edges.shape[1]
        # Learnable edge weights (real, will be symmetrized)
        self.edge_weights = nn.Parameter(torch.randn(num_edges) * 0.1)
        self._cached_w = None
        self._cached_degree = None
        self._cached_dense_A = None
        self._cached_sparse_A = None
        # SEOP Fix 24: Pre-build symmetric sparse indices to avoid reconstruction
        self._cached_sparse_idx: Tensor | None = None
        self.register_buffer("_sparse_idx_sym", self._build_symmetric_indices(edges))

    def _build_sparsity_pattern(self, dim: int, sparsity: int) -> Tensor:
        """Build small-world sparsity pattern.

        Returns edge indices [2, num_edges] for upper triangle only
        (will be symmetrized when building H).
        """
        rows, cols = [], []

        # Ring connections
        ring_k = max(1, sparsity // 2)
        for i in range(dim):
            for offset in range(1, ring_k + 1):
                j = (i + offset) % dim
                if i < j:  # Upper triangle only
                    rows.append(i)
                    cols.append(j)
                else:
                    rows.append(j)
                    cols.append(i)

        # Random long-range (deterministic seed for reproducibility)
        gen = torch.Generator().manual_seed(42)
        num_random = max(0, dim * sparsity // 2 - len(rows))
        for _ in range(num_random):
            i = torch.randint(0, dim, (1,), generator=gen).item()
            j = torch.randint(0, dim, (1,), generator=gen).item()
            if i != j:
                a, b = min(i, j), max(i, j)
                rows.append(a)
                cols.append(b)

        edges = torch.tensor([rows, cols], dtype=torch.long)
        # Remove duplicate edges
        edges = torch.unique(edges, dim=1)
        return edges

    def _build_symmetric_indices(self, edges: Tensor) -> Tensor:
        """Pre-build symmetric COO indices for sparse matvec.

        Returns [2, 2*num_edges] tensor with both (row,col) and (col,row) indices.
        This avoids rebuilding the indices on every matvec call.
        """
        rows, cols = edges[0], edges[1]
        # Concatenate both directions for symmetric matrix
        idx = torch.cat([torch.stack([rows, cols]), torch.stack([cols, rows])], dim=1)
        return idx

    def get_hamiltonian_dense(self) -> Tensor:
        """Build dense Hermitian Hamiltonian matrix.

        Returns:
            H: [D, D] complex64, Hermitian positive semi-definite
        """
        D = self.dim
        # SEOP Fix 7: Smooth positive constraint (softplus, not abs)
        # abs() has non-differentiable cusp at zero → gradient oscillation
        # softplus is smooth everywhere: ℝ→ℝ⁺, ≈identity for x>>0
        w = F.softplus(self.edge_weights)

        # Build adjacency A
        A = torch.zeros(D, D, device=w.device)
        rows, cols = self.edge_indices[0], self.edge_indices[1]
        A[rows, cols] = w
        A[cols, rows] = w  # Symmetric

        # Degree matrix
        degree = A.sum(dim=1)
        D_mat = torch.diag(degree)

        # Laplacian L = D - A (real, symmetric, PSD)
        L = D_mat - A

        # Convert to complex Hermitian (cast avoids allocating D×D zeros for imag)
        return L.to(torch.complex64)

    def cache_weights(self):
        """Cache softplus weights + degree for reuse across CG iterations.

        SEOP Fix 19: softplus + degree scatter_add called ~6× per CG solve.
        With 8 propagator layers, that's 48 redundant calls per step.
        Cache once, reuse across all matvec calls within one forward pass.
        """
        self._cached_w = F.softplus(self.edge_weights)
        rows, cols = self.edge_indices[0], self.edge_indices[1]
        degree = torch.zeros(self.dim, device=self._cached_w.device)
        degree.scatter_add_(0, rows, self._cached_w)
        degree.scatter_add_(0, cols, self._cached_w)
        self._cached_degree = degree
        D = self.dim
        device = self._cached_w.device
        if D <= 512 or device.type == "xpu":
            A_dense = torch.zeros(D, D, device=device, dtype=self._cached_w.dtype)
            A_dense[rows, cols] = self._cached_w
            A_dense[cols, rows] = self._cached_w
            self._cached_dense_A = A_dense
            self._cached_sparse_A = None
        else:
            self._cached_dense_A = None
            # Issue #1 fix: Cache the sparse COO tensor to avoid rebuilding
            # + coalesce() on every matvec call. The indices are fixed (topology
            # doesn't change), only weights change per forward pass.
            w_sym = torch.cat([self._cached_w, self._cached_w])
            idx = self._sparse_idx_sym
            self._cached_sparse_A = torch.sparse_coo_tensor(
                idx, w_sym, (D, D)
            ).coalesce()

    def clear_cache(self):
        """Clear cached weights."""
        self._cached_w = None
        self._cached_degree = None
        self._cached_dense_A = None
        self._cached_sparse_A = None

    def get_diagonal(self) -> Tensor:
        """Get diagonal of Laplacian H: diag(H) = degree vector.

        For Laplacian H = D - A, the diagonal is just the degree matrix D,
        which is the sum of edge weights for each node.

        Returns:
            diag_H: [D] complex64, diagonal entries of H
        """
        w = (
            self._cached_w
            if self._cached_w is not None
            else F.softplus(self.edge_weights)
        )
        rows, cols = self.edge_indices[0], self.edge_indices[1]
        degree = torch.zeros(self.dim, device=w.device)
        degree.scatter_add_(0, rows, w)
        degree.scatter_add_(0, cols, w)
        return degree.to(torch.complex64)

    def matvec_real(self, v: Tensor) -> Tensor:
        """Compute H @ v where H and v are both real.

        H = D - A where A is sparse adjacency.
        (H @ v)_i = degree_i * v_i - sum_{j in neighbors(i)} w_ij * v_j

        O(sparsity * D) instead of O(D^2).

        Args:
            v: [..., D] real tensor
        Returns:
            Hv: [..., D] real tensor
        """
        # SEOP Fix 19: Use cached weights to avoid recomputing softplus per CG iteration
        w = (
            self._cached_w
            if self._cached_w is not None
            else F.softplus(self.edge_weights)
        )
        rows, cols = self.edge_indices[0], self.edge_indices[1]

        # Build sparse adjacency matrix A (symmetric)
        # Use torch.sparse for memory-efficient matvec
        D = self.dim
        device = w.device

        # Sparse matvec: A @ v where A is [D, D] sparse
        # v is [..., D], we reshape to [prod(...), D], multiply, then reshape back
        batch_shape = v.shape[:-1]
        v_flat = v.reshape(-1, D)  # [B*S, D]
        w_sym = torch.cat([w, w])  # Duplicate weights for symmetry

        # Dense matmul for small D or XPU (sparse.mm not supported on XPU)
        if D <= 512 or device.type == "xpu":
            if self._cached_dense_A is not None:
                A_dense = self._cached_dense_A
            else:
                A_dense = torch.zeros(D, D, device=device, dtype=w.dtype)
                A_dense[rows, cols] = w
                A_dense[cols, rows] = w
            Av_flat = v_flat @ A_dense.t()
        else:
            if self._cached_sparse_A is not None:
                A_sparse = self._cached_sparse_A
            else:
                idx = self._sparse_idx_sym
                A_sparse = torch.sparse_coo_tensor(idx, w_sym, (D, D)).coalesce()
            Av_flat = torch.sparse.mm(A_sparse, v_flat.t()).t()

        res = Av_flat.reshape(*batch_shape, D)

        if self._cached_degree is not None:
            degree = self._cached_degree
        else:
            degree = torch.zeros(self.dim, device=w.device)
            degree.scatter_add_(0, rows, w)
            degree.scatter_add_(0, cols, w)

        return degree * v - res

    def matvec_real_fused(self, vr: Tensor, vi: Tensor) -> tuple[Tensor, Tensor]:
        """Compute H @ v for real tensors with a fused sparse matvec.

        H = D - A where A is sparse adjacency.
        (H @ v)_i = degree_i * v_i - sum_{j in neighbors(i)} w_ij * v_j

        O(sparsity * D) instead of O(D^2).

        Args:
            vr: [..., D] real tensor
            vi: [..., D] real tensor
        Returns:
            Hvr: [..., D] real tensor
            Hvi: [..., D] real tensor
        """
        # SEOP Fix 23: Fuse real/imag matvecs to halve sparse.mm launches and reuse A.
        w = (
            self._cached_w
            if self._cached_w is not None
            else F.softplus(self.edge_weights)
        )
        rows, cols = self.edge_indices[0], self.edge_indices[1]

        D = self.dim
        device = w.device

        batch_shape = vr.shape[:-1]
        vr_flat = vr.reshape(-1, D)
        vi_flat = vi.reshape(-1, D)
        v_combined = torch.cat([vr_flat, vi_flat], dim=0)

        if D <= 512 or device.type == "xpu":
            if self._cached_dense_A is not None:
                A_dense = self._cached_dense_A
            else:
                A_dense = torch.zeros(D, D, device=device, dtype=w.dtype)
                A_dense[rows, cols] = w
                A_dense[cols, rows] = w
            Av_combined = v_combined @ A_dense.t()
        else:
            if self._cached_sparse_A is not None:
                A_sparse = self._cached_sparse_A
            else:
                idx = self._sparse_idx_sym
                w_sym = torch.cat([w, w])
                A_sparse = torch.sparse_coo_tensor(idx, w_sym, (D, D)).coalesce()
            Av_combined = torch.sparse.mm(A_sparse, v_combined.t()).t()

        Avr_flat, Avi_flat = Av_combined.split(vr_flat.shape[0], dim=0)

        Avr = Avr_flat.reshape(*batch_shape, D)
        Avi = Avi_flat.reshape(*batch_shape, D)

        if self._cached_degree is not None:
            degree = self._cached_degree
        else:
            degree = torch.zeros(self.dim, device=w.device)
            degree.scatter_add_(0, rows, w)
            degree.scatter_add_(0, cols, w)

        return degree * vr - Avr, degree * vi - Avi

    def matvec(self, v: Tensor) -> Tensor:
        """Compute H @ v using sparse Laplacian structure.

        H = D - A where A is sparse adjacency.
        (H @ v)_i = degree_i * v_i - sum_{j in neighbors(i)} w_ij * v_j

        O(sparsity * D) instead of O(D^2).

        Args:
            v: [..., D] complex64
        Returns:
            Hv: [..., D] complex64
        """
        # XPU Refactor: Use explicit real matvecs
        return safe_complex(self.matvec_real(v.real), self.matvec_real(v.imag))


class MultiScaleHamiltonian(nn.Module):
    scales: nn.ModuleList
    scale_weights: nn.Parameter
    _cached_scale_weights: Tensor | None

    """Graph Laplacian Pyramid: multi-scale Hamiltonian.

    Combines Hamiltonians at different connectivity scales,
    allowing both local and global information flow.
    """

    def __init__(self, dim: int, num_scales: int = 3, base_sparsity: int = 5):
        super().__init__()
        self.scales = nn.ModuleList(
            [
                GraphLaplacianHamiltonian(dim, sparsity=base_sparsity * (2**s))
                for s in range(num_scales)
            ]
        )
        # Learnable scale mixing weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        self._cached_scale_weights = None

    def cache_weights(self):
        """Cache scale mixing weights and child Hamiltonian weights."""
        self._cached_scale_weights = torch.softmax(self.scale_weights, dim=0)
        for scale in self.scales:
            if hasattr(scale, "cache_weights"):
                scale.cache_weights()

    def clear_cache(self):
        """Clear all caches."""
        self._cached_scale_weights = None
        for scale in self.scales:
            if hasattr(scale, "clear_cache"):
                scale.clear_cache()

    def get_hamiltonian_dense(self) -> Tensor:
        """Get combined multi-scale Hamiltonian."""
        weights = (
            self._cached_scale_weights
            if self._cached_scale_weights is not None
            else torch.softmax(self.scale_weights, dim=0)
        )
        # Use torch.stack for cleaner type checking than sum() on generator
        terms = [
            w * scale.get_hamiltonian_dense() for w, scale in zip(weights, self.scales)
        ]
        return torch.stack(terms).sum(dim=0)

    def get_diagonal(self) -> Tensor:
        """Get diagonal of combined multi-scale Hamiltonian.

        Returns:
            diag_H: [D] complex64, diagonal entries of multi-scale H
        """
        weights = (
            self._cached_scale_weights
            if self._cached_scale_weights is not None
            else torch.softmax(self.scale_weights, dim=0)
        )
        terms = [w * scale.get_diagonal() for w, scale in zip(weights, self.scales)]
        return torch.stack(terms).sum(dim=0)

    def matvec_real(self, v: Tensor) -> Tensor:
        """Multi-scale H @ v for real tensors."""
        weights = (
            self._cached_scale_weights
            if self._cached_scale_weights is not None
            else torch.softmax(self.scale_weights, dim=0)
        )
        terms = [w * scale.matvec_real(v) for w, scale in zip(weights, self.scales)]
        return torch.stack(terms).sum(dim=0)

    def matvec_real_fused(self, vr: Tensor, vi: Tensor) -> tuple[Tensor, Tensor]:
        """Multi-scale H @ v for fused real tensors."""
        # SEOP Fix 23: Fuse real/imag passes per scale to avoid duplicate sparse.mm.
        weights = (
            self._cached_scale_weights
            if self._cached_scale_weights is not None
            else torch.softmax(self.scale_weights, dim=0)
        )
        terms_r, terms_i = [], []
        for w, scale in zip(weights, self.scales):
            Hvr, Hvi = scale.matvec_real_fused(vr, vi)
            terms_r.append(w * Hvr)
            terms_i.append(w * Hvi)
        return torch.stack(terms_r).sum(dim=0), torch.stack(terms_i).sum(dim=0)

    def matvec(self, v: Tensor) -> Tensor:
        """Multi-scale H @ v for complex tensors."""
        return safe_complex(self.matvec_real(v.real), self.matvec_real(v.imag))
