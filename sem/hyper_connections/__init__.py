"""Manifold-Constrained Hyper-Connections (mHC) for gradient-stable residual learning.

This module provides residual connection mechanisms that prevent gradient explosion/collapse
by constraining the residual mixing matrix to the Birkhoff Polytope (doubly stochastic matrices).

Key Components:
    - sinkhorn_log: Projects matrices onto Birkhoff Polytope via Sinkhorn-Knopp
    - MHCResidual: Full mHC residual block with multi-stream support
    - SimpleMHC: Minimal single-stream version for SEM V8.0
    - mhc_residual: Functional API for stateless residual connection

Why mHC prevents gradient death:
    1. Standard residual: x' = x + f(x) can cause explosion/collapse across depth
    2. mHC residual: x' = H_res @ x + beta * f(x) with H_res doubly stochastic
    3. Doubly stochastic = all rows/cols sum to 1 = information preserving
    4. Birkhoff Polytope = convex hull of permutations = structured mixing
    5. Sinkhorn = differentiable projection onto Birkhoff = trainable constraint

Original paper: tokenbender/mHC-manifold-constrained-hyper-connections
Adapted for: SEM V8.0 complex-valued architecture

Example:
    >>> from sem.hyper_connections import SimpleMHC
    >>> mhc = SimpleMHC(dim=256, complex_mode=True)
    >>> x = torch.randn(8, 128, 256, dtype=torch.complex64)
    >>> branch_out = torch.randn_like(x)
    >>> y = mhc(x, branch_out)  # Gradient-stable residual connection
"""

from .sinkhorn import (
    sinkhorn_log,
    sinkhorn_log_complex,
)

from .mhc import (
    MHCResidual,
    SimpleMHC,
    mhc_residual,
)

__all__ = [
    # Sinkhorn projection
    'sinkhorn_log',
    'sinkhorn_log_complex',

    # mHC residual blocks
    'MHCResidual',
    'SimpleMHC',
    'mhc_residual',
]
