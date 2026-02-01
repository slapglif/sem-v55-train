"""Cayley-Soliton Propagator for unitary wave evolution.

Two-step propagator that preserves information (norm) exactly:

Step 1: Nonlinear Phase Rotation (pointwise, O(D))
    psi_rot = psi * exp(i * alpha * |psi|^2)
    This is a Kerr-like self-phase modulation. The phase of each
    component rotates proportionally to its own intensity.

Step 2: Sparse Cayley Diffusion (O(sparsity * D) via CG)
    Solve: (I + i*dt/2 * H) psi_out = (I - i*dt/2 * H) psi_rot
    This is the Crank-Nicolson scheme applied to the Schrodinger
    equation i*dpsi/dt = H*psi, using the Cayley transform to
    guarantee unitarity.

The "soliton" is a self-sustaining wave packet that maintains its
form through the balance of nonlinear self-focusing (Step 1) and
linear diffusion (Step 2).
"""

import torch
import torch.nn as nn
from torch import Tensor

from .hamiltonian import MultiScaleHamiltonian
from .cg_solver import cg_solve, cg_solve_sparse
from .unitarity_check import check_unitarity
from sem.spinor.complex_ops import complex_mul_real


class CayleySolitonPropagator(nn.Module):
    """Single Cayley-Soliton propagation step.

    Combines nonlinear phase rotation with unitary Cayley diffusion.
    """

    def __init__(
        self,
        dim: int,
        dt: float = 0.1,
        nonlinear_alpha: float = 0.1,
        cg_max_iter: int = 20,
        cg_tol: float = 1e-6,
        laplacian_sparsity: int = 5,
        num_scales: int = 3,
        check_unitarity_flag: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.check_unitarity_flag = check_unitarity_flag

        # SEOP Fix 9: Per-dimension Kerr coefficient
        # Scalar α applies uniform nonlinear coupling to all feature channels.
        # Different dimensions encode different features with different magnitude
        # distributions — each channel needs its own coupling strength.
        # Vector α ∈ ℝ^D lets each dimension learn its optimal rotation rate.
        self.alpha = nn.Parameter(torch.full((dim,), nonlinear_alpha))

        # Multi-scale Hamiltonian (Graph Laplacian Pyramid)
        self.hamiltonian = MultiScaleHamiltonian(
            dim, num_scales=num_scales, base_sparsity=laplacian_sparsity
        )

    def forward(self, psi: Tensor) -> Tensor:
        """Propagate wavefunction through one Cayley-Soliton step.

        Args:
            psi: [B, S, D] complex64 input wavefunction
        Returns:
            psi_out: [B, S, D] complex64 propagated wavefunction
        """
        # Boundary: Convert to real-block representation [B, S, D, 2]
        # This eliminates complex64 overhead on XPU/NPU architectures.
        psi_real_block = torch.view_as_real(psi)
        psi_r, psi_i = psi_real_block.unbind(-1)

        # Step 1: Nonlinear phase rotation (Kerr effect)
        # SEOP Fix 13: Normalize intensity for uniform phase encoding
        # Raw |ψ|² is χ²-distributed (heavy tail). Normalizing by mean ensures
        # average rotation = α regardless of magnitude, spreading information
        # uniformly across the phase circle.
        intensity = psi_r * psi_r + psi_i * psi_i  # |psi|², no sqrt
        intensity = intensity / (intensity.mean(dim=-1, keepdim=True) + 1e-8)
        phase_shift = self.alpha * intensity  # [B, S, D] real

        # Explicit real math rotation: psi * (cos(p) + i*sin(p))
        # Eliminates torch.exp(1j * ...) complex allocation.
        cos_p = torch.cos(phase_shift)
        sin_p = torch.sin(phase_shift)
        psi_rot_r, psi_rot_i = complex_mul_real(psi_r, psi_i, cos_p, sin_p)

        # Step 2: Cayley diffusion via sparse Hamiltonian matvec
        # Cayley scheme: (I + i*dt/2*H) psi_out = (I - i*dt/2*H) psi_rot
        # Using sparse H.matvec instead of dense [D,D] matmul: O(sparsity*D) vs O(D²)
        half_dt = self.dt / 2

        # SEOP Fix 19: Cache Hamiltonian weights for reuse across CG iterations
        self.hamiltonian.cache_weights()
        D = self.dim

        def a_plus_matvec(v_pair):
            """(I - i*dt/2*H) @ v_pair"""
            vr, vi = v_pair
            # (1 - i*k*H)(vr + i*vi) = (vr + k*H*vi) + i*(vi - k*H*vr)
            Hvr = self.hamiltonian.matvec_real(vr)
            Hvi = self.hamiltonian.matvec_real(vi)
            return (vr + half_dt * Hvi, vi - half_dt * Hvr)

        def a_minus_matvec(v_pair):
            """(I + i*dt/2*H) @ v_pair"""
            vr, vi = v_pair
            # (1 + i*k*H)(vr + i*vi) = (vr - k*H*vi) + i*(vi + k*H*vr)
            Hvr = self.hamiltonian.matvec_real(vr)
            Hvi = self.hamiltonian.matvec_real(vi)
            return (vr - half_dt * Hvi, vi + half_dt * Hvr)

        # Compute RHS: A_plus @ psi_rot
        rhs_r, rhs_i = a_plus_matvec((psi_rot_r, psi_rot_i))

        # Solve A_minus @ psi_out = rhs via CG with sparse matvec
        # Wrapped to operate on [..., D, 2] real tensors for cg_solve_sparse.
        def a_minus_matvec_wrapped(v_real_block):
            vr, vi = v_real_block.unbind(-1)
            out_r, out_i = a_minus_matvec((vr, vi))
            return torch.stack([out_r, out_i], dim=-1)

        B, S, _ = rhs_r.shape
        rhs_real_block = torch.stack([rhs_r, rhs_i], dim=-1).reshape(-1, D, 2)

        psi_out_real_block = cg_solve_sparse(
            a_minus_matvec_wrapped, rhs_real_block, self.cg_max_iter, self.cg_tol
        )

        # Boundary: Convert back to complex
        psi_out = torch.view_as_complex(psi_out_real_block.reshape(B, S, D, 2))

        # Clear cached weights (free memory, ensure fresh compute next forward)
        self.hamiltonian.clear_cache()

        # Debug: check unitarity
        if self.check_unitarity_flag and not self.training:
            # Re-wrap for utility which expects complex
            check_unitarity(psi, psi_out)

        return psi_out


class CayleySolitonStack(nn.Module):
    """Stack of Cayley-Soliton propagation layers.

    Each layer has its own learnable Hamiltonian and nonlinear strength,
    allowing the model to learn different propagation dynamics at each depth.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 8,
        dt: float = 0.1,
        nonlinear_alpha: float = 0.1,
        cg_max_iter: int = 20,
        cg_tol: float = 1e-6,
        laplacian_sparsity: int = 5,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CayleySolitonPropagator(
                    dim=dim,
                    dt=dt,
                    nonlinear_alpha=nonlinear_alpha,
                    cg_max_iter=cg_max_iter,
                    cg_tol=cg_tol,
                    laplacian_sparsity=laplacian_sparsity,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, psi: Tensor) -> Tensor:
        """Propagate through all layers.

        Args:
            psi: [B, S, D] complex64
        Returns:
            [B, S, D] complex64
        """
        for layer in self.layers:
            psi = layer(psi)
        return psi
