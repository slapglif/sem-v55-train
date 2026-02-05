"""Lindblad Dissipation for SEM V8.0.

Implements Maxwell's Demon selective forgetting via Lindblad master equation.
Instead of perfect unitarity (which causes catastrophic retention), we allow
controlled information loss through dissipative coupling to a "cold bath".

Theory:
    The Lindblad master equation for density matrix ρ:
    dρ/dt = -i[H, ρ] + γ Σ_k (L_k ρ L_k† - 0.5 {L_k† L_k, ρ})

    For pure state evolution (|ψ⟩ instead of ρ):
    d|ψ⟩/dt = -i H |ψ⟩ - 0.5 γ Σ_k L_k† L_k |ψ⟩ + √γ Σ_k L_k |ψ⟩ dW_k

    We implement the deterministic part (dissipation term) without the noise.
    The key: L operators act as "entropy sinks" that selectively evaporate
    high-entropy components while preserving low-entropy signal.

Architecture:
    - Learnable Lindblad operators L_k (K operators, each D×D complex)
    - Dissipation strength γ (learnable or fixed)
    - Applied after Remizov-Cayley propagation step

SEOP Analysis:
    Input: Complex spinor ψ ~ CN(0, σ²I) (Gaussian distribution)
    Output: ψ_dissipated = ψ - 0.5·γ·dt·(Σ L_k† L_k)·ψ

    Distributional Effect:
    - L_k† L_k is positive semidefinite → eigenvalues λ_j ≥ 0
    - Dissipation reduces magnitude preferentially along eigenvectors with large λ
    - Creates "dissipation basin" that drains entropy while preserving signal subspace

    Entropy Transfer per FLOP:
    - Forward: O(K·D²) FLOPs for computing Σ L_k† L_k
    - Entropy reduction: ΔS ≈ -γ·dt·Tr(Σ L_k† L_k) nats
    - Efficiency: ΔS / FLOP ≈ -γ·dt·Tr(Σ L_k† L_k) / (K·D²)
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional
from ..utils.complex_ops import safe_complex


class LindbladDissipation(nn.Module):
    """Lindblad dissipation layer for selective forgetting.

    Implements the dissipative term of the Lindblad master equation:
    dψ/dt = -0.5·γ·Σ_k (L_k† L_k)·ψ

    This acts as Maxwell's Demon: selectively evaporating high-entropy
    components (noise) while preserving low-entropy signal.

    Args:
        dim: Spinor dimension D
        num_lindblad_ops: Number of Lindblad operators K (default 4)
        gamma: Dissipation strength (default 0.01)
        learnable_gamma: If True, gamma is a learnable parameter
        init_scale: Initialization scale for L operators (default 0.01)
    """

    def __init__(
        self,
        dim: int,
        num_lindblad_ops: int = 4,
        gamma: float = 0.01,
        learnable_gamma: bool = True,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_lindblad_ops = num_lindblad_ops

        # Dissipation strength (log-parameterized for stability)
        if learnable_gamma:
            self.log_gamma = nn.Parameter(torch.tensor(math.log(gamma)))
        else:
            self.register_buffer('log_gamma', torch.tensor(math.log(gamma)))

        # Lindblad operators L_k (K operators, each D×D complex)
        # Represented as real/imag components for numerical stability
        self.L_real = nn.Parameter(
            torch.randn(num_lindblad_ops, dim, dim) * init_scale
        )
        self.L_imag = nn.Parameter(
            torch.randn(num_lindblad_ops, dim, dim) * init_scale
        )

        # Initialize as diagonal-dominant for stability
        with torch.no_grad():
            for k in range(num_lindblad_ops):
                # Add diagonal boost to prevent collapse
                self.L_real[k].diagonal().add_(0.1)

    @property
    def gamma(self) -> float:
        """Current dissipation strength."""
        return self.log_gamma.exp().item()

    def compute_lindblad_term(self) -> Tensor:
        """Compute Σ_k L_k† L_k (the dissipation operator).

        Returns:
            L_dag_L_sum: [D, D] complex64 - Sum of L_k† L_k operators
        """
        # Construct complex Lindblad operators
        L = safe_complex(self.L_real, self.L_imag)  # [K, D, D]

        # Compute L† for each operator
        L_dag = L.conj().transpose(-2, -1)  # [K, D, D]

        # Compute L_k† L_k for each k
        L_dag_L = torch.bmm(L_dag, L)  # [K, D, D]

        # Sum over all Lindblad operators
        L_dag_L_sum = L_dag_L.sum(dim=0)  # [D, D]

        return L_dag_L_sum

    def forward(self, psi: Tensor, dt: float = 1.0) -> Tensor:
        """Apply Lindblad dissipation to spinor state.

        Implements: ψ_out = ψ - 0.5·γ·dt·(Σ L_k† L_k)·ψ

        Args:
            psi: [B, S, D] complex64 - Spinor state
            dt: Time step (default 1.0)

        Returns:
            psi_dissipated: [B, S, D] complex64 - State after dissipation
        """
        # Compute dissipation operator Σ L_k† L_k
        L_dag_L_sum = self.compute_lindblad_term()  # [D, D]

        # Apply dissipation: -0.5·γ·dt·(Σ L_k† L_k)·ψ
        # psi: [B, S, D], L_dag_L_sum: [D, D]
        gamma = self.log_gamma.exp()
        dissipation_coeff = -0.5 * gamma * dt

        # Matrix-vector product: [D, D] @ [B, S, D]
        # Use einsum for clarity: (i,j), (b,s,j) -> (b,s,i)
        dissipation = dissipation_coeff * torch.einsum(
            'ij,bsj->bsi',
            L_dag_L_sum,
            psi
        )

        # Apply dissipation to state
        psi_out = psi + dissipation

        return psi_out

    def dissipation_rate(self, psi: Tensor) -> Tensor:
        """Compute instantaneous dissipation rate (entropy loss).

        Rate = -γ·⟨ψ|Σ L_k† L_k|ψ⟩

        Args:
            psi: [B, S, D] complex64

        Returns:
            rate: [B, S] float32 - Dissipation rate (negative = entropy loss)
        """
        # Compute Σ L_k† L_k
        L_dag_L_sum = self.compute_lindblad_term()  # [D, D]

        # Compute ⟨ψ|L†L|ψ⟩ = ψ† (L†L) ψ
        # ψ†: [B, S, D], L†L: [D, D], ψ: [B, S, D]
        L_psi = torch.einsum('ij,bsj->bsi', L_dag_L_sum, psi)  # [B, S, D]
        expectation = (psi.conj() * L_psi).sum(dim=-1).real  # [B, S]

        gamma = self.log_gamma.exp()
        rate = -gamma * expectation

        return rate

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, num_ops={self.num_lindblad_ops}, '
            f'gamma={self.gamma:.6f}'
        )


class AdaptiveLindbladDissipation(nn.Module):
    """Adaptive Lindblad dissipation with input-dependent γ.

    Instead of fixed dissipation strength, γ is computed from the input state.
    This allows the model to learn when to dissipate (high entropy) vs when
    to preserve (low entropy).

    Dissipation control: γ(ψ) = σ(W·|ψ|² + b)·γ_max

    Args:
        dim: Spinor dimension D
        num_lindblad_ops: Number of Lindblad operators K (default 4)
        gamma_max: Maximum dissipation strength (default 0.05)
        init_scale: Initialization scale for L operators (default 0.01)
    """

    def __init__(
        self,
        dim: int,
        num_lindblad_ops: int = 4,
        gamma_max: float = 0.05,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_lindblad_ops = num_lindblad_ops
        self.gamma_max = gamma_max

        # Lindblad operators (same as fixed-gamma version)
        self.L_real = nn.Parameter(
            torch.randn(num_lindblad_ops, dim, dim) * init_scale
        )
        self.L_imag = nn.Parameter(
            torch.randn(num_lindblad_ops, dim, dim) * init_scale
        )

        # Initialize as diagonal-dominant
        with torch.no_grad():
            for k in range(num_lindblad_ops):
                self.L_real[k].diagonal().add_(0.1)

        # Adaptive gamma network: |ψ|² -> γ
        # Input: magnitude squared [B, S, D]
        # Output: scalar gamma [B, S, 1]
        self.gamma_proj = nn.Linear(dim, 1, bias=True)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.constant_(self.gamma_proj.bias, 0.0)  # Start with γ ≈ γ_max/2

    def compute_lindblad_term(self) -> Tensor:
        """Compute Σ_k L_k† L_k (same as fixed-gamma version)."""
        L = safe_complex(self.L_real, self.L_imag)  # [K, D, D]
        L_dag = L.conj().transpose(-2, -1)  # [K, D, D]
        L_dag_L = torch.bmm(L_dag, L)  # [K, D, D]
        L_dag_L_sum = L_dag_L.sum(dim=0)  # [D, D]
        return L_dag_L_sum

    def compute_gamma(self, psi: Tensor) -> Tensor:
        """Compute adaptive dissipation strength from input state.

        Args:
            psi: [B, S, D] complex64

        Returns:
            gamma: [B, S, 1] float32
        """
        # Magnitude squared as entropy proxy
        mag_sq = psi.real**2 + psi.imag**2  # [B, S, D]

        # Project to scalar and apply sigmoid
        gamma_logit = self.gamma_proj(mag_sq)  # [B, S, 1]
        gamma = torch.sigmoid(gamma_logit) * self.gamma_max

        return gamma

    def forward(self, psi: Tensor, dt: float = 1.0) -> Tensor:
        """Apply adaptive Lindblad dissipation.

        Args:
            psi: [B, S, D] complex64
            dt: Time step

        Returns:
            psi_dissipated: [B, S, D] complex64
        """
        # Compute adaptive γ(ψ)
        gamma = self.compute_gamma(psi)  # [B, S, 1]

        # Compute dissipation operator
        L_dag_L_sum = self.compute_lindblad_term()  # [D, D]

        # Apply dissipation with adaptive strength
        dissipation_coeff = -0.5 * gamma * dt  # [B, S, 1]

        # Matrix-vector with broadcasting
        L_psi = torch.einsum('ij,bsj->bsi', L_dag_L_sum, psi)  # [B, S, D]
        dissipation = dissipation_coeff * L_psi

        psi_out = psi + dissipation

        return psi_out

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, num_ops={self.num_lindblad_ops}, '
            f'gamma_max={self.gamma_max:.4f}'
        )


# Gradient checkpointing support
def checkpoint_lindblad_dissipation(
    lindblad: LindbladDissipation,
    psi: Tensor,
    dt: float = 1.0
) -> Tensor:
    """Wrapper for gradient checkpointing Lindblad dissipation.

    Use with torch.utils.checkpoint.checkpoint() to trade compute for memory.

    Example:
        from torch.utils.checkpoint import checkpoint
        psi_out = checkpoint(
            checkpoint_lindblad_dissipation,
            lindblad_layer,
            psi,
            dt,
            use_reentrant=False
        )
    """
    return lindblad(psi, dt)
