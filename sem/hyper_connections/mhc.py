"""Manifold-Constrained Hyper-Connections (mHC) for gradient-stable residual learning.

The core insight: standard residual connections (x + f(x)) can cause gradient explosion
or collapse across depth. mHC solves this by constraining the residual mixing matrix
to the Birkhoff Polytope (doubly stochastic matrices), which preserves signal magnitude.

Mathematical formulation:
    Standard residual: x_{l+1} = x_l + f(x_l)
    mHC residual: x_{l+1} = H_res @ x_l + beta * f(x_l)

    Where H_res is doubly stochastic (all rows/cols sum to 1), enforced via Sinkhorn.

Why this prevents gradient death:
    1. Doubly stochastic = information-preserving (no magnitude change)
    2. Birkhoff Polytope = convex hull of permutations (structured mixing)
    3. Sinkhorn projection = differentiable constraint enforcement

Original paper: tokenbender/mHC-manifold-constrained-hyper-connections
Adapted for: SEM V8.0 complex-valued residual streams
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import math

from .sinkhorn import sinkhorn_log, sinkhorn_log_complex


class MHCResidual(nn.Module):
    """Manifold-Constrained Hyper-Connection residual block.

    Replaces standard residual (x + f(x)) with Birkhoff-constrained mixing:
        x' = H_res @ x + beta * f(x)

    Where H_res is projected onto the Birkhoff Polytope via Sinkhorn.

    Args:
        dim: Hidden dimension
        num_streams: Number of residual streams (1 for SEM V8.0 single-stream)
        mhc_num_iters: Sinkhorn iterations (10-20 typical)
        mhc_tau: Temperature for Sinkhorn softmax (0.01-0.1)
        complex_mode: If True, operates on complex-valued tensors
        init_identity: If True, initialize H_res to identity (default: diagonal-dominant)
        dropout: Dropout probability for output

    Example:
        >>> mhc = MHCResidual(dim=256, num_streams=1)
        >>> x = torch.randn(8, 128, 256)  # [batch, seq, dim]
        >>> branch_out = torch.randn_like(x)
        >>> y = mhc(x, branch_out)
        >>> print(y.shape)  # [8, 128, 256]
    """

    def __init__(
        self,
        dim: int,
        num_streams: int = 1,
        mhc_num_iters: int = 10,
        mhc_tau: float = 0.05,
        complex_mode: bool = True,
        init_identity: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.mhc_num_iters = mhc_num_iters
        self.mhc_tau = mhc_tau
        self.complex_mode = complex_mode
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # H_res: Learnable residual mixing matrix (will be projected to Birkhoff)
        # Initialize as diagonal-dominant for stable training start
        if init_identity:
            # Start at identity (no mixing)
            init_h_res = torch.zeros(num_streams, num_streams)
            init_h_res.fill_diagonal_(1.0)
        else:
            # Start diagonal-dominant: diag = 0, off-diag = -8 (softmax ≈ 0)
            init_h_res = torch.full((num_streams, num_streams), -8.0)
            init_h_res.fill_diagonal_(0.0)

        self.H_res_logits = nn.Parameter(init_h_res)

        # beta: Output routing weight (learnable per-stream)
        self.beta_logits = nn.Parameter(torch.zeros(num_streams))

        if complex_mode:
            # Complex variant: separate logits for real/imag
            self.H_res_logits_imag = nn.Parameter(init_h_res.clone())
            self.beta_logits_imag = nn.Parameter(torch.zeros(num_streams))

    def get_h_res(self) -> Tensor:
        """Compute doubly stochastic H_res via Sinkhorn projection."""
        if self.complex_mode:
            H_real, H_imag = sinkhorn_log_complex(
                self.H_res_logits,
                self.H_res_logits_imag,
                num_iters=self.mhc_num_iters,
                tau=self.mhc_tau,
            )
            # Return complex tensor
            H = torch.complex(H_real, H_imag)
        else:
            H = sinkhorn_log(
                self.H_res_logits,
                num_iters=self.mhc_num_iters,
                tau=self.mhc_tau,
            )
        return H

    def get_beta(self) -> Tensor:
        """Compute output routing weights (softmax for normalization)."""
        if self.complex_mode:
            beta_real = torch.softmax(self.beta_logits, dim=-1)
            beta_imag = torch.softmax(self.beta_logits_imag, dim=-1)
            beta = torch.complex(beta_real, beta_imag)
        else:
            beta = torch.softmax(self.beta_logits, dim=-1)
        return beta

    def forward(self, residual: Tensor, branch_output: Tensor) -> Tensor:
        """Apply mHC residual connection.

        Args:
            residual: Input residual stream [B, ..., S, D] or [B, ..., D]
                where S = num_streams (if present)
            branch_output: Branch function output [B, ..., D]

        Returns:
            Updated residual stream [B, ..., S, D] or [B, ..., D]
        """
        # Get doubly stochastic mixing matrix
        H_res = self.get_h_res()  # [S, S]

        # Get output routing weights
        beta = self.get_beta()  # [S]

        # If single stream, squeeze/unsqueeze for consistency
        if self.num_streams == 1:
            # residual: [B, T, D] -> [B, T, 1, D]
            # branch_output: [B, T, D] stays as-is
            residual_expanded = residual.unsqueeze(-2)  # [B, ..., 1, D]

            # Apply residual mixing: H_res @ residual
            # [1, 1] @ [B, ..., 1, D] -> [B, ..., 1, D]
            residual_mixed = torch.einsum("ss,...sd->...sd", H_res, residual_expanded)

            # Apply output routing: beta * branch_output
            # [1] * [B, ..., D] -> [B, ..., 1, D]
            branch_weighted = beta.unsqueeze(0) * branch_output.unsqueeze(-2)

            # Combine and squeeze back
            output = residual_mixed + branch_weighted  # [B, ..., 1, D]
            output = output.squeeze(-2)  # [B, ..., D]
        else:
            # Multi-stream case
            # residual: [B, ..., S, D]
            # Apply residual mixing
            residual_mixed = torch.einsum("st,...td->...sd", H_res, residual)

            # Apply output routing
            # branch_output: [B, ..., D] -> [B, ..., S, D]
            # beta: [S] -> [1, ..., 1, S, 1] (broadcast compatible)
            ndim_before_sd = len(branch_output.shape) - 1  # Number of dims before D
            beta_shape = [1] * ndim_before_sd + [self.num_streams, 1]
            beta_expanded = beta.view(*beta_shape)
            branch_weighted = beta_expanded * branch_output.unsqueeze(-2)

            output = residual_mixed + branch_weighted

        return self.dropout(output)


class SimpleMHC(nn.Module):
    """Simplified mHC for direct use in SEM V8.0 (no multi-stream complexity).

    This is the minimal version for single residual stream:
        x' = sinkhorn_project(H_res) @ x + branch_output

    Where H_res is constrained to Birkhoff Polytope (doubly stochastic).

    Args:
        dim: Hidden dimension
        mhc_num_iters: Sinkhorn iterations
        mhc_tau: Sinkhorn temperature
        complex_mode: If True, handles complex tensors
        dropout: Output dropout

    Example:
        >>> mhc = SimpleMHC(dim=256)
        >>> x = torch.randn(8, 128, 256, dtype=torch.complex64)
        >>> branch_out = torch.randn_like(x)
        >>> y = mhc(x, branch_out)
    """

    def __init__(
        self,
        dim: int,
        num_streams: int = 4,
        mhc_num_iters: int = 10,
        mhc_tau: float = 0.05,
        complex_mode: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.mhc_num_iters = mhc_num_iters
        self.mhc_tau = mhc_tau
        self.complex_mode = complex_mode
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # We keep a single-stream *interface* (residual, branch_output) -> updated residual,
        # but internally we partition channels into multiple streams so the Birkhoff/Sinkhorn
        # constraint is meaningful (a 1x1 Sinkhorn projection is constant and would yield
        # zero gradients for H_res_logits).
        if self.num_streams < 1:
            raise ValueError("num_streams must be >= 1")
        if self.dim % self.num_streams != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by num_streams ({self.num_streams})"
            )
        self._stream_dim = self.dim // self.num_streams

        # H_res: learnable stream-mixing matrix projected onto the Birkhoff polytope.
        # Initialize mildly diagonal-dominant for a stable start WITHOUT saturating Sinkhorn.
        # With tau=0.05, very negative off-diagonals (e.g. -8.0) become ~-160 after scaling
        # and can underflow exp(), yielding near-exact identity and zero gradients.
        # NOTE: we want H close to identity for a residual-like start, but not so sharp
        # that Sinkhorn saturates and gradients vanish.
        init_h = torch.full((self.num_streams, self.num_streams), -0.25)
        init_h.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h)

        if complex_mode:
            self.H_res_logits_imag = nn.Parameter(init_h.clone())

        # Per-stream branch injection gain. We avoid softmax normalization here because
        # streams are channel partitions (not duplicated parallel residual streams).
        # Using beta ~= 1 preserves the expected x + branch behavior.
        self.beta_logits = nn.Parameter(torch.zeros(self.num_streams))

    def forward(self, residual: Tensor, branch_output: Tensor) -> Tensor:
        """Apply simplified mHC residual.

        For single stream, this degenerates to:
            x' = identity_weight * x + branch_output

        But we keep the Sinkhorn machinery for multi-stream extension.

        Args:
            residual: [B, T, D] (complex or real)
            branch_output: [B, T, D]

        Returns:
            [B, T, D]
        """
        if self.complex_mode:
            H_real, H_imag = sinkhorn_log_complex(
                self.H_res_logits,
                self.H_res_logits_imag,
                num_iters=self.mhc_num_iters,
                tau=self.mhc_tau,
            )
            H = torch.complex(H_real, H_imag)
        else:
            H = sinkhorn_log(
                self.H_res_logits,
                num_iters=self.mhc_num_iters,
                tau=self.mhc_tau,
            )

        # Partition channels into streams: [B, T, D] -> [B, T, S, Ds]
        # Then apply stream mixing H:[S,S] to the stream axis.
        b, t, d = residual.shape
        if d != self.dim:
            raise ValueError(f"Expected residual last-dim={self.dim}, got {d}")
        residual_s = residual.view(b, t, self.num_streams, self._stream_dim)
        mixed_s = torch.einsum("ij,btjd->btid", H, residual_s)
        mixed = mixed_s.reshape(b, t, d)

        # Branch injection (per-channel-partition gain). beta in (0, 2) with beta=1 at init.
        beta = 2.0 * torch.sigmoid(self.beta_logits).to(dtype=branch_output.dtype)
        branch_s = branch_output.view(b, t, self.num_streams, self._stream_dim)
        branch_weighted = (beta.view(1, 1, self.num_streams, 1) * branch_s).reshape(
            b, t, d
        )

        output = mixed + branch_weighted
        return self.dropout(output)


def mhc_residual(
    x: Tensor,
    branch_output: Tensor,
    H_res_logits: Tensor,
    H_res_logits_imag: Optional[Tensor] = None,
    mhc_num_iters: int = 10,
    mhc_tau: float = 0.05,
    complex_mode: bool = True,
) -> Tensor:
    """Functional API for mHC residual connection (stateless).

    This is the minimal interface for SEM V8.0 integration.

    Args:
        x: Input residual [B, T, D]
        branch_output: Branch function output [B, T, D]
        H_res_logits: Learnable mixing matrix logits [1, 1] or [S, S]
        H_res_logits_imag: Optional imaginary logits for complex mode.
            If None, zeros are used (symmetric/real-only Sinkhorn).
        mhc_num_iters: Sinkhorn iterations
        mhc_tau: Sinkhorn temperature
        complex_mode: Handle complex tensors

    Returns:
        Updated residual [B, T, D]

    Example:
        >>> H_logits = torch.nn.Parameter(torch.zeros(1, 1))
        >>> x = torch.randn(8, 128, 256, dtype=torch.complex64)
        >>> branch = torch.randn_like(x)
        >>> y = mhc_residual(x, branch, H_logits)
    """
    if complex_mode:
        # Split complex tensor into real/imag
        x_real = x.real
        x_imag = x.imag
        branch_real = branch_output.real
        branch_imag = branch_output.imag

        # Use provided imaginary logits or fall back to zeros
        imag_logits = (
            H_res_logits_imag
            if H_res_logits_imag is not None
            else torch.zeros_like(H_res_logits)
        )

        H_real, H_imag = sinkhorn_log_complex(
            H_res_logits,
            imag_logits,
            num_iters=mhc_num_iters,
            tau=mhc_tau,
        )

        # Apply to real and imag separately
        # For [1,1] matrix: just scalar multiplication
        residual_real = H_real[0, 0] * x_real
        residual_imag = H_imag[0, 0] * x_imag

        output_real = residual_real + branch_real
        output_imag = residual_imag + branch_imag

        return torch.complex(output_real, output_imag)
    else:
        H = sinkhorn_log(H_res_logits, num_iters=mhc_num_iters, tau=mhc_tau)
        residual_weighted = H[0, 0] * x
        return residual_weighted + branch_output
