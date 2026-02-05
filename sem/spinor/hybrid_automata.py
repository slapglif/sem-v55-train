"""Hybrid Automata for SEM V8.0.

Detects singularities via Lie bracket curvature and triggers quantum jumps.

Theory:
    The Hamiltonian H(t) evolves smoothly most of the time, but occasionally
    hits "spiky" attention patterns where smooth flow breaks down (eigenvalues
    approach imaginary axis → Cayley transform singular).

    Lie Bracket Curvature:
        K = ||[H, dH/dt]|| = ||H·(dH/dt) - (dH/dt)·H||_F

    Regime Detection:
        - K < ε: Smooth regime → use Remizov-Cayley propagation
        - K > ε: Spiky regime → trigger Landau-Zener transition

    Landau-Zener Transition:
        Instantaneous unitary teleportation through singularity:
        |ψ⟩ → U_LZ |ψ⟩ where U_LZ is a learned unitary operator

        Implemented as a complex-valued linear layer with orthogonality
        regularization (soft constraint via loss term).

Architecture:
    - Monitor: Compute Lie bracket curvature from H(t), H(t-1)
    - Decide: Compare K to threshold ε (learnable)
    - Act: Apply Landau-Zener jump if K > ε

SEOP Analysis:
    Input Distribution: H ~ CN(0, σ²I) (complex Gaussian Hamiltonian)
    Output: Discrete mixture of two processes

    Smooth Regime (P(K<ε)):
        No change to ψ (handled by Remizov-Cayley elsewhere)

    Jump Regime (P(K>ε)):
        ψ → U_LZ·ψ where U_LZ is nearly unitary
        Distributional effect: Rotation in complex state space
        Entropy preserved (unitary → reversible)

    Impedance Check:
        - Remizov-Cayley expects "smooth" H evolution
        - Spiky H would cause eigenvalue→i singularity (NaN/inf)
        - Hybrid automata resolves: jump → smooth → jump sequence
        - No information loss (both processes are unitary)

    Entropy Transfer per FLOP:
        - Monitor: O(D³) FLOPs (matrix multiply for Lie bracket)
        - Jump: O(D²) FLOPs (matrix-vector multiply)
        - Decision overhead: O(1) (scalar comparison)
        - Entropy change: 0 (unitary operations preserve entropy)
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Tuple, Optional, Union
from ..utils.complex_ops import safe_complex


class HybridAutomata(nn.Module):
    """Hybrid automata for singularity detection and quantum jumps.

    Monitors Lie bracket curvature K = ||[H, dH/dt]|| and triggers
    Landau-Zener transitions when K exceeds threshold.

    Args:
        dim: Spinor dimension D
        curvature_threshold: Initial threshold ε (default 0.1)
        learnable_threshold: If True, threshold is learnable (default True)
        ortho_reg_weight: Weight for orthogonality regularization (default 0.01)
        softness: Controls transition sharpness for soft sigmoid gate (default 0.01)
    """

    def __init__(
        self,
        dim: int,
        curvature_threshold: float = 0.1,
        learnable_threshold: bool = True,
        ortho_reg_weight: float = 0.01,
        softness: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.ortho_reg_weight = ortho_reg_weight
        self.softness = softness

        # Curvature threshold (log-parameterized for stability)
        if learnable_threshold:
            self.log_threshold = nn.Parameter(
                torch.tensor(math.log(curvature_threshold))
            )
        else:
            self.register_buffer(
                'log_threshold',
                torch.tensor(math.log(curvature_threshold))
            )

        # Landau-Zener transition operator (complex linear)
        # Represented as 2 real linear layers for numerical stability
        # U_LZ: D×D complex = 2D×2D real block matrix [[A, -B], [B, A]]
        self.lz_real = nn.Linear(dim, dim, bias=False)
        self.lz_imag = nn.Linear(dim, dim, bias=False)

        # Initialize as near-identity (small perturbation from I)
        # This ensures U_LZ starts close to unitary
        with torch.no_grad():
            nn.init.eye_(self.lz_real.weight)  # Real part = I
            nn.init.zeros_(self.lz_imag.weight)  # Imag part = 0
            # Add small noise for symmetry breaking
            self.lz_real.weight.add_(torch.randn_like(self.lz_real.weight) * 0.01)
            self.lz_imag.weight.add_(torch.randn_like(self.lz_imag.weight) * 0.01)

    @property
    def threshold(self) -> float:
        """Current curvature threshold."""
        return self.log_threshold.exp().item()

    def compute_curvature(
        self,
        H_t: Tensor,
        H_prev: Tensor,
        dt: float = 1.0
    ) -> Tensor:
        """Compute Lie bracket curvature K = ||[H, dH/dt]||.

        Args:
            H_t: [B, D, D] complex64 - Current Hamiltonian
            H_prev: [B, D, D] complex64 - Previous Hamiltonian
            dt: Time step (default 1.0)

        Returns:
            K: [B] float32 - Curvature values (Frobenius norm)
        """
        # Compute dH/dt
        dH_dt = (H_t - H_prev) / dt  # [B, D, D]

        # Lie bracket: [H, dH/dt] = H @ dH/dt - dH/dt @ H
        lie_bracket = torch.bmm(H_t, dH_dt) - torch.bmm(dH_dt, H_t)  # [B, D, D]

        # Frobenius norm: sqrt(Σ |element|²)
        K = torch.linalg.norm(lie_bracket.flatten(start_dim=-2), dim=-1)  # [B]

        return K

    def landau_zener_jump(self, psi: Tensor) -> Tensor:
        """Apply Landau-Zener transition (instantaneous teleportation).

        Implements: ψ → U_LZ·ψ where U_LZ is a learned complex matrix.

        Args:
            psi: [B, S, D] complex64

        Returns:
            psi_jumped: [B, S, D] complex64
        """
        # Apply complex linear: (A + iB)·ψ = A·ψ_r - B·ψ_i + i(B·ψ_r + A·ψ_i)
        psi_r = psi.real
        psi_i = psi.imag

        # Real part: A·ψ_r - B·ψ_i
        out_r = self.lz_real(psi_r) - self.lz_imag(psi_i)
        # Imag part: B·ψ_r + A·ψ_i
        out_i = self.lz_imag(psi_r) + self.lz_real(psi_i)

        psi_jumped = safe_complex(out_r, out_i)

        return psi_jumped

    def compute_unitarity_loss(self) -> Tensor:
        """Compute orthogonality regularization loss.

        Soft constraint: ||U†U - I||²_F to encourage unitarity.

        Returns:
            loss: scalar - Deviation from unitarity
        """
        # Construct complex matrix U = A + iB
        U = safe_complex(self.lz_real.weight, self.lz_imag.weight)  # [D, D]

        # Compute U†U
        U_dag = U.conj().t()  # [D, D]
        U_dag_U = torch.mm(U_dag, U)  # [D, D]

        # Target: Identity matrix
        I = torch.eye(self.dim, dtype=U.dtype, device=U.device)

        # Frobenius norm: ||U†U - I||²
        loss = torch.linalg.norm(U_dag_U - I, ord='fro')**2

        return loss * self.ortho_reg_weight

    def forward(
        self,
        psi: Tensor,
        H_t: Tensor,
        H_prev: Tensor,
        dt: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, dict]]:
        """Apply hybrid automata logic.

        Uses a differentiable soft sigmoid gate instead of a hard boolean
        threshold, ensuring gradients flow through the jump decision.

        Args:
            psi: [B, S, D] complex64 - Spinor state
            H_t: [B, D, D] complex64 - Current effective Hamiltonian
            H_prev: [B, D, D] complex64 - Previous Hamiltonian
            dt: Time step (default 1.0)
            return_diagnostics: If True, return curvature values

        Returns:
            psi_out: [B, S, D] complex64 - Output state
            jump_weight: [B] float32 - Soft jump weights (0=smooth, 1=jumped)
            (Optional) diagnostics: dict with 'curvature', 'threshold', 'jump_rate'
        """
        threshold = self.log_threshold.exp()
        K = self.compute_curvature(H_t, H_prev, dt)  # [B]

        # Differentiable soft sigmoid gate instead of hard boolean
        # jump_weight ≈ 0 when K << threshold, ≈ 1 when K >> threshold
        jump_weight = torch.sigmoid((K - threshold) / self.softness)  # [B]

        # Always compute jumped path for gradient flow
        psi_jumped = self.landau_zener_jump(psi)

        # Soft blend between smooth and jump paths
        # Broadcast jump_weight: [B] -> [B, 1, 1] for tensor blending
        jump_weight_3d = jump_weight.view(-1, 1, 1)
        psi_out = (1.0 - jump_weight_3d) * psi + jump_weight_3d * psi_jumped

        if return_diagnostics:
            diagnostics = {
                'curvature': K,
                'threshold': threshold,
                'jump_rate': jump_weight.mean().item(),
            }
            return psi_out, jump_weight, diagnostics
        else:
            return psi_out, jump_weight

    def extra_repr(self) -> str:
        return f'dim={self.dim}, threshold={self.threshold:.6f}'


class AdaptiveHybridAutomata(nn.Module):
    """Adaptive hybrid automata with input-dependent threshold.

    Instead of fixed threshold ε, compute ε(H) from the Hamiltonian itself.
    This allows the model to learn context-dependent singularity detection.

    Threshold control: ε(H) = σ(W·||H||²_F + b)·ε_max

    Args:
        dim: Spinor dimension D
        threshold_max: Maximum curvature threshold (default 0.5)
        ortho_reg_weight: Weight for orthogonality regularization (default 0.01)
        softness: Controls transition sharpness for soft sigmoid gate (default 0.01)
    """

    def __init__(
        self,
        dim: int,
        threshold_max: float = 0.5,
        ortho_reg_weight: float = 0.01,
        softness: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.threshold_max = threshold_max
        self.ortho_reg_weight = ortho_reg_weight
        self.softness = softness

        # Landau-Zener operator (same as fixed-threshold version)
        self.lz_real = nn.Linear(dim, dim, bias=False)
        self.lz_imag = nn.Linear(dim, dim, bias=False)

        # Initialize as near-identity
        with torch.no_grad():
            nn.init.eye_(self.lz_real.weight)
            nn.init.zeros_(self.lz_imag.weight)
            self.lz_real.weight.add_(torch.randn_like(self.lz_real.weight) * 0.01)
            self.lz_imag.weight.add_(torch.randn_like(self.lz_imag.weight) * 0.01)

        # Adaptive threshold network: ||H||² -> ε
        # Input: Frobenius norm squared (scalar per batch)
        # Output: threshold ε [B, 1]
        self.threshold_proj = nn.Linear(1, 1, bias=True)
        nn.init.zeros_(self.threshold_proj.weight)
        nn.init.constant_(self.threshold_proj.bias, 0.0)  # Start at ε_max/2

    def compute_curvature(
        self,
        H_t: Tensor,
        H_prev: Tensor,
        dt: float = 1.0
    ) -> Tensor:
        """Compute Lie bracket curvature (same as fixed-threshold version)."""
        dH_dt = (H_t - H_prev) / dt
        lie_bracket = torch.bmm(H_t, dH_dt) - torch.bmm(dH_dt, H_t)
        K = torch.linalg.norm(lie_bracket.flatten(start_dim=-2), dim=-1)
        return K

    def compute_threshold(self, H_t: Tensor) -> Tensor:
        """Compute adaptive threshold from Hamiltonian.

        Args:
            H_t: [B, D, D] complex64

        Returns:
            threshold: [B, 1] float32
        """
        # Frobenius norm squared as complexity proxy
        H_norm_sq = torch.linalg.norm(
            H_t.flatten(start_dim=-2), dim=-1
        )**2  # [B]

        # Project to threshold
        threshold_logit = self.threshold_proj(H_norm_sq.unsqueeze(-1))  # [B, 1]
        threshold = torch.sigmoid(threshold_logit) * self.threshold_max

        return threshold

    def landau_zener_jump(self, psi: Tensor) -> Tensor:
        """Apply Landau-Zener jump (same as fixed-threshold version)."""
        psi_r = psi.real
        psi_i = psi.imag
        out_r = self.lz_real(psi_r) - self.lz_imag(psi_i)
        out_i = self.lz_imag(psi_r) + self.lz_real(psi_i)
        psi_jumped = safe_complex(out_r, out_i)
        return psi_jumped

    def compute_unitarity_loss(self) -> Tensor:
        """Compute orthogonality regularization loss (same as fixed version)."""
        U = safe_complex(self.lz_real.weight, self.lz_imag.weight)
        U_dag = U.conj().t()
        U_dag_U = torch.mm(U_dag, U)
        I = torch.eye(self.dim, dtype=U.dtype, device=U.device)
        loss = torch.linalg.norm(U_dag_U - I, ord='fro')**2
        return loss * self.ortho_reg_weight

    def forward(
        self,
        psi: Tensor,
        H_t: Tensor,
        H_prev: Tensor,
        dt: float = 1.0,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, dict]]:
        """Apply adaptive hybrid automata logic.

        Uses a differentiable soft sigmoid gate for the jump decision.

        Args:
            psi: [B, S, D] complex64
            H_t: [B, D, D] complex64
            H_prev: [B, D, D] complex64
            dt: Time step
            return_diagnostics: If True, return diagnostics

        Returns:
            psi_out: [B, S, D] complex64
            jump_weight: [B] float32 - Soft jump weights (0=smooth, 1=jumped)
            (Optional) diagnostics: dict
        """
        # Compute adaptive threshold
        threshold = self.compute_threshold(H_t).squeeze(-1)  # [B]

        # Compute curvature
        K = self.compute_curvature(H_t, H_prev, dt)  # [B]

        # Differentiable soft sigmoid gate
        jump_weight = torch.sigmoid((K - threshold) / self.softness)  # [B]

        # Always compute jumped path for gradient flow
        psi_jumped = self.landau_zener_jump(psi)

        # Soft blend between smooth and jump paths
        jump_weight_3d = jump_weight.view(-1, 1, 1)
        psi_out = (1.0 - jump_weight_3d) * psi + jump_weight_3d * psi_jumped

        if return_diagnostics:
            diagnostics = {
                'curvature': K,
                'threshold': threshold,
                'jump_rate': jump_weight.mean().item(),
            }
            return psi_out, jump_weight, diagnostics
        else:
            return psi_out, jump_weight

    def extra_repr(self) -> str:
        return f'dim={self.dim}, threshold_max={self.threshold_max:.4f}'


# Gradient checkpointing support
def checkpoint_hybrid_automata(
    automata: HybridAutomata,
    psi: Tensor,
    H_t: Tensor,
    H_prev: Tensor,
    dt: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """Wrapper for gradient checkpointing hybrid automata.

    Use with torch.utils.checkpoint.checkpoint() to trade compute for memory.

    Example:
        from torch.utils.checkpoint import checkpoint
        psi_out, jumped = checkpoint(
            checkpoint_hybrid_automata,
            automata_layer,
            psi,
            H_t,
            H_prev,
            dt,
            use_reentrant=False
        )
    """
    return automata(psi, H_t, H_prev, dt, return_diagnostics=False)


# Utility: Construct effective Hamiltonian from attention patterns
def attention_to_hamiltonian(
    attn_weights: Tensor,
    make_hermitian: bool = True
) -> Tensor:
    """Convert attention weights to effective Hamiltonian.

    Args:
        attn_weights: [B, S, S] float32 - Attention matrix (softmax output)
        make_hermitian: If True, symmetrize to ensure Hermitian (default True)

    Returns:
        H: [B, S, S] complex64 - Effective Hamiltonian

    Note:
        Attention weights are typically asymmetric and real-valued.
        To use as a Hamiltonian (which must be Hermitian for unitary evolution),
        we symmetrize: H = (A + A†)/2 and optionally add imaginary antisymmetric
        part for non-conservative dynamics.
    """
    B, S, _ = attn_weights.shape

    if make_hermitian:
        # Hermitian symmetrization: H = (A + A†)/2
        attn_t = attn_weights.transpose(-2, -1)
        H_real = 0.5 * (attn_weights + attn_t)

        # Optional: Add small imaginary antisymmetric part for dissipation
        # H_imag = 0.5 * (A - A†)
        H_imag = 0.5 * (attn_weights - attn_t)

        H = safe_complex(H_real, H_imag)
    else:
        # Direct cast (non-Hermitian)
        H = safe_complex(
            attn_weights,
            torch.zeros_like(attn_weights)
        )

    return H
