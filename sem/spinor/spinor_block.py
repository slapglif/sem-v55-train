"""Block-diagonal complex transform with Cayley-parameterized unitarity.

Implements a block-diagonal *unitary* complex map using num_blocks independent
block_size×block_size complex matrices via Cayley parameterization:

    U = (I - A)(I + A)^{-1}

where A is anti-Hermitian (real part skew-symmetric, imaginary part symmetric).
This guarantees U is unitary for any A, without projection or regularization.

This factorization is a computational/memory tradeoff:
- Dense D×D complex matmul is O(D^2)
- Block-diagonal is O(num_blocks * block_size^2)

The Cayley map avoids torch.matrix_exp (no complex codegen in Inductor) and
maintains Real-Block Isomorphism throughout: the 2B×2B real-block linear system
is solved via torch.linalg.solve, which is well-supported on all backends.

[DERIVATION] Anti-Hermitian A has real part skew-symmetric (A_r^T = -A_r)
and imaginary part symmetric (A_i^T = A_i). The Cayley map U = (I-A)(I+A)^{-1}
is unitary for any such A. Near-identity init: A ≈ 0 → U ≈ I.
In Real-Block Isomorphism, the complex linear system becomes a 2B×2B real solve.
"""

import torch
import torch.nn as nn
from torch import Tensor

from sem.utils.complex_ops import safe_complex

# block_diagonal_complex_matmul no longer needed — einsum done inline with real-block params


def _cayley_real_block(A_r: Tensor, A_i: Tensor, eye: Tensor) -> tuple[Tensor, Tensor]:
    """Compute Cayley map U = (I - A)(I + A)^{-1} via Real-Block Isomorphism.

    Args:
        A_r: [N, B, B] skew-symmetric (real part of anti-Hermitian generator)
        A_i: [N, B, B] symmetric (imag part of anti-Hermitian generator)
        eye: [1, B, B] identity matrix

    Returns:
        (U_real, U_imag): each [N, B, B], the real and imaginary parts of
        the unitary matrix U.

    The complex system (I + A) U = (I - A) is mapped to a 2B×2B real system:

        [[I + A_r, -A_i], [A_i, I + A_r]] @ [[U_r], [U_i]] = [[I - A_r, A_i], [-A_i, I - A_r]]

    Solved via torch.linalg.solve for compile-friendly, complex-free forward pass.
    """
    N, B, _ = A_r.shape

    # I + A and I - A (real and imag parts)
    IpA_r = eye + A_r  # [N, B, B]
    IpA_i = A_i  # [N, B, B]
    ImA_r = eye - A_r  # [N, B, B]
    ImA_i = -A_i  # [N, B, B]

    # Build real-block 2x2 matrices: LHS = [[I+Ar, -Ai], [Ai, I+Ar]]
    # Shape: [N, 2B, 2B]
    LHS_top = torch.cat([IpA_r, -IpA_i], dim=-1)  # [N, B, 2B]
    LHS_bot = torch.cat([IpA_i, IpA_r], dim=-1)  # [N, B, 2B]
    LHS = torch.cat([LHS_top, LHS_bot], dim=-2)  # [N, 2B, 2B]

    # RHS = [[I-Ar, Ai], [-Ai, I-Ar]]
    RHS_top = torch.cat([ImA_r, -ImA_i], dim=-1)  # [N, B, 2B]
    RHS_bot = torch.cat([ImA_i, ImA_r], dim=-1)  # [N, B, 2B]
    RHS = torch.cat([RHS_top, RHS_bot], dim=-2)  # [N, 2B, 2B]

    # Solve LHS @ U_rb = RHS  →  U_rb = LHS^{-1} @ RHS
    U_rb = torch.linalg.solve(LHS, RHS)  # [N, 2B, 2B]

    # Extract real and imaginary parts
    U_real = U_rb[:, :B, :B]  # [N, B, B]
    U_imag = U_rb[:, B:, :B]  # [N, B, B]

    return U_real, U_imag


class SpinorBlock(nn.Module):
    """Block-diagonal unitary transformation layer via Cayley parameterization.

    Factorizes a D×D complex unitary transform into num_blocks independent
    block_size×block_size unitary matrices. Each block U is parameterized as:

        U = (I - A)(I + A)^{-1}

    where A is anti-Hermitian (A_real skew-symmetric, A_imag symmetric).
    This guarantees U†U = I structurally for any parameter values.

    Parameters are stored as float32 (real, imag) pairs for Real-Block
    Isomorphism compatibility with torch.compile and XPU backends.
    """

    def __init__(self, hidden_dim: int, block_size: int = 8):
        super().__init__()
        assert hidden_dim % block_size == 0, (
            f"hidden_dim {hidden_dim} must be divisible by block_size {block_size}"
        )

        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.num_blocks = hidden_dim // block_size

        # Anti-Hermitian generator parameters:
        # A_real will be enforced skew-symmetric: (A - A^T) / 2
        # A_imag will be enforced symmetric: (A + A^T) / 2
        # Init small so Cayley(A) ≈ I + 2A (near-identity)
        A_real = torch.randn(self.num_blocks, block_size, block_size) * 0.02
        A_imag = torch.randn(self.num_blocks, block_size, block_size) * 0.02

        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)

        # Tag for ComplexAdamW coupled momentum.
        # Use setattr to keep type checkers happy.
        setattr(self.A_real, "_is_complex_real", True)
        setattr(self.A_imag, "_is_complex_imag", True)
        setattr(self.A_real, "_complex_partner", self.A_imag)
        setattr(self.A_imag, "_complex_partner", self.A_real)

        # Cache identity matrix (registered as buffer for device tracking)
        self.register_buffer(
            "_eye",
            torch.eye(block_size).unsqueeze(0),  # [1, B, B]
        )

    def _get_skew_sym(self) -> tuple[Tensor, Tensor]:
        """Enforce anti-Hermitian structure on generator A.

        Returns:
            (A_r, A_i) where A_r is skew-symmetric and A_i is symmetric.
        """
        A_r = (self.A_real - self.A_real.transpose(-2, -1)) / 2
        A_i = (self.A_imag + self.A_imag.transpose(-2, -1)) / 2
        return A_r, A_i

    def forward(self, x: Tensor) -> Tensor:
        """Apply block-diagonal unitary transform via Cayley map.

        Args:
            x: [B, S, D] complex64
        Returns:
            [B, S, D] complex64

        The Cayley parameterization guarantees U†U = I structurally,
        so no regularization is needed for unitarity.
        """
        B, S, D = x.shape

        # Enforce anti-Hermitian structure
        A_r, A_i = self._get_skew_sym()

        # Compute unitary via Cayley map in real-block form
        U_real, U_imag = _cayley_real_block(A_r, A_i, self._eye)

        # Reshape input to blocks: [B, S, num_blocks, block_size]
        x_blocks = x.view(B, S, self.num_blocks, self.block_size)

        # Decompose input
        xr, xi = x_blocks.real, x_blocks.imag

        # Block-diagonal complex matmul via Real-Block Isomorphism
        # (U_r + i·U_i) @ (xr + i·xi) = (U_r@xr - U_i@xi) + i·(U_r@xi + U_i@xr)
        out_real = torch.einsum("noi,bsni->bsno", U_real, xr) - torch.einsum(
            "noi,bsni->bsno", U_imag, xi
        )
        out_imag = torch.einsum("noi,bsni->bsno", U_real, xi) + torch.einsum(
            "noi,bsni->bsno", U_imag, xr
        )

        out = safe_complex(out_real, out_imag)

        # Reshape back: [B, S, D]
        return out.reshape(B, S, D)

    def unitarity_loss(self) -> Tensor:
        """Compute ||U†U - I||²_F as a diagnostic (should be ~0).

        With Cayley parameterization, U is structurally unitary,
        so this should return a value near machine epsilon. Useful
        as a sanity check / monitoring metric rather than a loss term.
        """
        A_r, A_i = self._get_skew_sym()
        U_real, U_imag = _cayley_real_block(A_r, A_i, self._eye)

        W = safe_complex(U_real, U_imag)  # [N, B, B]
        W_dag = W.conj().transpose(-2, -1)
        WdW = torch.bmm(W_dag, W)  # [N, B, B]
        I = torch.eye(self.block_size, device=W.device, dtype=W.dtype).unsqueeze(0)
        return ((WdW - I).abs() ** 2).mean()


class SpinorGate(nn.Module):
    """Gated unitary transformation with selective activation via Cayley map.

    Instead of interpolating between identity and a fixed U (which breaks
    unitarity), this module scales the anti-Hermitian generator:

        U(g) = Cayley(g * A)

    where g ∈ [0, 1] is a per-feature sigmoid gate. This guarantees:
    - U(g=0) = Cayley(0) = I (identity)
    - U(g=1) = Cayley(A) = full unitary rotation
    - U(g) is unitary for ALL values of g

    The gate g is computed from the input via a linear projection + scaled
    sigmoid (logistic-probit approximation to Gaussian CDF).
    """

    def __init__(self, hidden_dim: int, block_size: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.num_blocks = hidden_dim // block_size

        # Anti-Hermitian generator (shared with the Cayley computation)
        A_real = torch.randn(self.num_blocks, block_size, block_size) * 0.02
        A_imag = torch.randn(self.num_blocks, block_size, block_size) * 0.02

        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)

        # Tag for ComplexAdamW coupled momentum
        setattr(self.A_real, "_is_complex_real", True)
        setattr(self.A_imag, "_is_complex_imag", True)
        setattr(self.A_real, "_complex_partner", self.A_imag)
        setattr(self.A_imag, "_complex_partner", self.A_real)

        # Gate projection (real-valued sigmoid gate)
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for real+imag input

        # Cache identity matrix
        self.register_buffer(
            "_eye",
            torch.eye(block_size).unsqueeze(0),  # [1, B, B]
        )

    def _get_skew_sym(self) -> tuple[Tensor, Tensor]:
        """Enforce anti-Hermitian structure on generator A."""
        A_r = (self.A_real - self.A_real.transpose(-2, -1)) / 2
        A_i = (self.A_imag + self.A_imag.transpose(-2, -1)) / 2
        return A_r, A_i

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated unitary spinor transformation via Cayley(g * A).

        Args:
            x: [B, S, D] complex64
        Returns:
            [B, S, D] complex64
        """
        B, S, D = x.shape

        # Compute gate from real representation
        x_real_cat = torch.cat([x.real, x.imag], dim=-1)  # [B, S, 2D]
        # SEOP Fix 18: Scaled sigmoid ≈ Gaussian CDF, but ~3x faster than erf on CPU
        # sigmoid(x * 1.7015) is a logistic-probit approximation to Φ(x).
        gate = torch.sigmoid(self.gate_proj(x_real_cat) * 1.7015)  # [B, S, D]

        # Reshape gate to block form and average within each block
        # [B, S, num_blocks, block_size] → mean over block_size → [B, S, N, 1, 1]
        gate_blocks = gate.view(B, S, self.num_blocks, self.block_size)
        g_block = gate_blocks.mean(dim=-1, keepdim=True).unsqueeze(
            -1
        )  # [B, S, N, 1, 1]

        # Enforce anti-Hermitian structure
        A_r, A_i = self._get_skew_sym()  # [N, B_s, B_s]

        # Scale generator by gate: gA_r, gA_i have shape [B, S, N, B_s, B_s]
        # Broadcasting: g_block [B, S, N, 1, 1] * A [N, B_s, B_s]
        gA_r = g_block * A_r  # [B, S, N, B_s, B_s]
        gA_i = g_block * A_i  # [B, S, N, B_s, B_s]

        # Compute Cayley(gA) for each (batch, seq, block)
        # Flatten batch dims for solve: [B*S*N, B_s, B_s]
        BS = B * S
        N = self.num_blocks
        Bs = self.block_size

        gA_r_flat = gA_r.reshape(BS * N, Bs, Bs)
        gA_i_flat = gA_i.reshape(BS * N, Bs, Bs)

        # Expand eye for the batch
        eye_flat = self._eye.expand(BS * N, -1, -1)  # [BS*N, Bs, Bs]

        # Cayley map via real-block solve
        U_real, U_imag = _cayley_real_block(gA_r_flat, gA_i_flat, eye_flat)

        # Reshape U back: [B, S, N, Bs, Bs]
        U_real = U_real.view(B, S, N, Bs, Bs)
        U_imag = U_imag.view(B, S, N, Bs, Bs)

        # Reshape input to blocks: [B, S, N, Bs]
        x_blocks = x.view(B, S, N, Bs)
        xr, xi = x_blocks.real, x_blocks.imag

        # Apply per-(batch, seq, block) unitary: U @ x
        # U [B, S, N, Bs, Bs] @ x [B, S, N, Bs] → [B, S, N, Bs]
        out_real = torch.einsum("bsnoi,bsni->bsno", U_real, xr) - torch.einsum(
            "bsnoi,bsni->bsno", U_imag, xi
        )
        out_imag = torch.einsum("bsnoi,bsni->bsno", U_real, xi) + torch.einsum(
            "bsnoi,bsni->bsno", U_imag, xr
        )

        out = safe_complex(out_real, out_imag)
        return out.reshape(B, S, D)
