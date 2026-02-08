"""Lindblad Dissipation for SEM V8.0.

Implements a deterministic dissipation/forgetting term inspired by Lindblad
dynamics.

Theory (context):
    The GKSL/Lindblad master equation for a density matrix ρ is:
        dρ/dt = -i[H, ρ] + γ Σ_k (L_k ρ L_k† - 0.5 {L_k† L_k, ρ})

    A common pure-state quantum-trajectory form has both a deterministic drift
    and stochastic jump/noise terms. This module implements ONLY the
    deterministic ("no-jump") drift term:
        d|ψ⟩/dt = -0.5·γ·(Σ_k L_k† L_k)·|ψ⟩

    The stochastic jump terms (e.g. √γ·L_k·dW_k or discrete jumps) are omitted.
    This is intentional for a deterministic neural network setting.

Notes:
    - Without jump terms, only A = Σ_k L_k† L_k affects the evolution.
    - Applying A to ψ is O(B·S·D²); forming A via dense matmuls is O(K·D³).
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional
from ..utils.complex_ops import safe_complex


class LindbladDissipation(nn.Module):
    """Lindblad dissipation layer for selective forgetting.

    Implements the deterministic (no-jump) dissipation drift:
        dψ/dt = -0.5·γ·Σ_k (L_k† L_k)·ψ

    This is equivalent to evolving with a non-Hermitian effective Hamiltonian
    drift term H_eff ∝ -i·0.5·γ·Σ_k L_k† L_k. The stochastic jump terms from the
    full Lindblad / quantum-trajectory formulation are intentionally omitted.

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

        # Forward Euler stability note:
        #   psi <- (I - 0.5*gamma*dt*A) psi, A = sum_k L_k^† L_k (PSD)
        # Explicit Euler can diverge if 0.5*gamma*dt*lambda_max(A) is too large.
        # Bound gamma to a small range to avoid runaway dissipation.
        self.gamma_max = 0.1

        # Backward-compatibility note:
        # Historically this module exposed `log_gamma` (log of an unconstrained
        # positive gamma). Some tests and callers still mutate/inspect it.
        # We keep `log_gamma` but compute a sigmoid-bounded effective gamma.
        if learnable_gamma:
            self.log_gamma = nn.Parameter(torch.tensor(math.log(gamma)))
        else:
            self.register_buffer("log_gamma", torch.tensor(math.log(gamma)))

        # Lindblad operators L_k (K operators, each D×D complex)
        # Represented as real/imag components for numerical stability
        self.L_real = nn.Parameter(torch.randn(num_lindblad_ops, dim, dim) * init_scale)
        self.L_imag = nn.Parameter(torch.randn(num_lindblad_ops, dim, dim) * init_scale)

        # Initialize as diagonal-dominant for stability
        with torch.no_grad():
            for k in range(num_lindblad_ops):
                # Add diagonal boost to prevent collapse
                self.L_real[k].diagonal().add_(0.1)

    @property
    def gamma(self) -> float:
        """Current dissipation strength."""
        gamma = (
            torch.sigmoid(self.log_gamma - math.log(self.gamma_max)) * self.gamma_max
        )
        return gamma.item()

    def compute_lindblad_term(self) -> Tensor:
        """Compute Σ_k L_k† L_k (the dissipation operator).

        Returns:
            L_dag_L_sum: [D, D] complex64 - Sum of L_k† L_k operators
        """
        # Construct complex Lindblad operators
        L = safe_complex(self.L_real, self.L_imag)  # [K, D, D]

        # NOTE: Without jump terms, only A = Σ_k L_k† L_k matters for the
        # deterministic drift. Individual L_k are retained for:
        #   (1) future jump-term implementations,
        #   (2) per-operator analysis/regularization,
        #   (3) gradient diversity (K small matrices can train differently than one A).

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
        gamma = (
            torch.sigmoid(self.log_gamma - math.log(self.gamma_max)) * self.gamma_max
        )
        dissipation_coeff = -0.5 * gamma * dt

        # Matrix-vector product: [D, D] @ [B, S, D]
        # Use einsum for clarity: (i,j), (b,s,j) -> (b,s,i)
        # Cost: O(D^2) per (batch, seq) position
        dissipation = dissipation_coeff * torch.einsum("ij,bsj->bsi", L_dag_L_sum, psi)

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
        L_psi = torch.einsum("ij,bsj->bsi", L_dag_L_sum, psi)  # [B, S, D]
        expectation = (psi.conj() * L_psi).sum(dim=-1).real  # [B, S]

        gamma = (
            torch.sigmoid(self.log_gamma - math.log(self.gamma_max)) * self.gamma_max
        )
        rate = -gamma * expectation

        return rate

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_ops={self.num_lindblad_ops}, gamma={self.gamma:.6f}"
        )


class AdaptiveLindbladDissipation(nn.Module):
    """Adaptive Lindblad dissipation with input-dependent γ.

    Instead of fixed dissipation strength, γ is computed from the input state.
    This allows the model to learn when to dissipate (high entropy) vs when
    to preserve (low entropy).

    Dissipation control: γ(ψ) = σ(W·|ψ|² + b)·γ_max

    This implements ONLY the deterministic (no-jump) drift term. Without jump
    terms, only A = Σ_k L_k† L_k affects the evolution, but we keep individual
    L_k for future extensibility and analysis.

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
        self.L_real = nn.Parameter(torch.randn(num_lindblad_ops, dim, dim) * init_scale)
        self.L_imag = nn.Parameter(torch.randn(num_lindblad_ops, dim, dim) * init_scale)

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
        # Cost: O(D^2) per (batch, seq) position
        L_psi = torch.einsum("ij,bsj->bsi", L_dag_L_sum, psi)  # [B, S, D]
        dissipation = dissipation_coeff * L_psi

        psi_out = psi + dissipation

        return psi_out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_ops={self.num_lindblad_ops}, "
            f"gamma_max={self.gamma_max:.4f}"
        )


# Gradient checkpointing support
def checkpoint_lindblad_dissipation(
    lindblad: LindbladDissipation, psi: Tensor, dt: float = 1.0
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
