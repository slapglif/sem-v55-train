"""Block-diagonal complex transform with Real-Block Isomorphism.

Implements a block-diagonal *complex linear* map using num_blocks independent
block_size×block_size complex matrices.

This factorization is a computational/memory tradeoff:
- Dense D×D complex matmul is O(D^2)
- Block-diagonal is O(num_blocks * block_size^2)

Important: the blocks are not constrained to be unitary (or SU(n)) during
training. If you want the per-block matrix U to be close to unitary, use the
provided unitarity regularizer (SpinorBlock.unitarity_loss) and/or an external
projection step in the training loop.

[MISMATCH] Previous version stored weights as complex64 nn.Parameter,
breaking torch.compile (inductor can't codegen complex ops) and
missing ComplexAdamW _complex_partner coupling. The forward() already
decomposed into real/imag for einsum — weight storage was the only
complex64 holdout.

[DERIVATION] Store weight_real, weight_imag as separate float32 params.
The block_diagonal_complex_matmul already accepts .real/.imag decomposition.
"""

import torch
import torch.nn as nn
from torch import Tensor

from sem.utils.complex_ops import safe_complex

# block_diagonal_complex_matmul no longer needed — einsum done inline with real-block params


class SpinorBlock(nn.Module):
    """Block-diagonal complex transformation layer.

    Factorizes a D×D complex transform into num_blocks independent
    block_size×block_size complex matrices, stored as float32 (real, imag)
    parameter pairs.

    Note: this parameterization is *not* unitary by construction. Use
    `unitarity_loss()` as a regularizer if you want to bias the blocks toward
    unitarity.
    """

    def __init__(self, hidden_dim: int, block_size: int = 8):
        super().__init__()
        assert hidden_dim % block_size == 0, (
            f"hidden_dim {hidden_dim} must be divisible by block_size {block_size}"
        )

        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.num_blocks = hidden_dim // block_size

        # Real-block storage (float32, compile-friendly)
        weight_real = torch.zeros(self.num_blocks, block_size, block_size)
        weight_imag = torch.zeros(self.num_blocks, block_size, block_size)

        for b in range(self.num_blocks):
            # Near-identity init: real part = I + noise
            weight_real[b] = (
                torch.eye(block_size) + torch.randn(block_size, block_size) * 0.02
            )
            # Antisymmetric imaginary part is a heuristic for a near-identity
            # complex transform. It does not guarantee (or preserve) unitarity.
            skew = torch.randn(block_size, block_size) * 0.02
            weight_imag[b] = skew - skew.T

        self.weight_real = nn.Parameter(weight_real)
        self.weight_imag = nn.Parameter(weight_imag)

        # Tag for ComplexAdamW coupled momentum.
        # Use setattr to keep type checkers happy.
        setattr(self.weight_real, "_is_complex_real", True)
        setattr(self.weight_imag, "_is_complex_imag", True)
        setattr(self.weight_real, "_complex_partner", self.weight_imag)
        setattr(self.weight_imag, "_complex_partner", self.weight_real)

    def forward(self, x: Tensor) -> Tensor:
        """Apply block-diagonal complex transform.

        Args:
            x: [B, S, D] complex64
        Returns:
            [B, S, D] complex64

        Note:
            This forward pass does not enforce unitarity. If strict unitarity is
            desired, consider adding `unitarity_loss()` to your training loss and
            optionally projecting each block matrix to the nearest unitary (e.g.
            via polar/QR) occasionally in the training loop.
        """
        B, S, D = x.shape

        # Reshape to blocks: [B, S, num_blocks, block_size]
        x_blocks = x.view(B, S, self.num_blocks, self.block_size)

        # Decompose input
        xr, xi = x_blocks.real, x_blocks.imag
        br, bi = self.weight_real, self.weight_imag

        # Block-diagonal complex matmul via Real-Block Isomorphism
        # (br + i·bi) @ (xr + i·xi) = (br@xr - bi@xi) + i·(br@xi + bi@xr)
        #
        # We keep the standard 4-einsum decomposition. While one can sometimes
        # restructure this into fewer calls by stacking real/imag parts, in
        # practice it typically does not reduce FLOPs and can regress kernel
        # selection or memory traffic depending on backend.
        out_real = torch.einsum("noi,bsni->bsno", br, xr) - torch.einsum(
            "noi,bsni->bsno", bi, xi
        )
        out_imag = torch.einsum("noi,bsni->bsno", br, xi) + torch.einsum(
            "noi,bsni->bsno", bi, xr
        )

        out = safe_complex(out_real, out_imag)

        # Reshape back: [B, S, D]
        return out.reshape(B, S, D)

    def unitarity_loss(self) -> Tensor:
        """Compute ||U^\u2020 U - I||^2_F for unitarity regularization.

        Returns a scalar loss that encourages each per-block complex matrix U
        to be unitary. This does not make the forward pass unitary by itself;
        it only provides a differentiable penalty that can be added to the
        training objective.
        """

        W = torch.complex(self.weight_real, self.weight_imag)  # [N, B, B]
        W_dag = W.conj().transpose(-2, -1)
        WdW = torch.bmm(W_dag, W)  # [N, B, B]
        I = torch.eye(self.block_size, device=W.device, dtype=W.dtype).unsqueeze(0)
        return ((WdW - I).abs() ** 2).mean()


class SpinorGate(nn.Module):
    """Gated complex transformation with selective activation.

    Applies a per-feature interpolation between identity and the SpinorBlock
    transform:

        output = x + gate * (SpinorBlock(x) - x)

    Even if the underlying SpinorBlock matrices were unitary, this gating is
    intentionally *not* a unitary operation for gate values in (0, 1). This is
    by design (selectivity is the point). If you regularize unitarity, the
    penalty should be applied to the SpinorBlock's matrix U itself (via
    `SpinorBlock.unitarity_loss()`), not to the gated output.
    """

    def __init__(self, hidden_dim: int, block_size: int = 8):
        super().__init__()
        self.spinor = SpinorBlock(hidden_dim, block_size)
        # Gate projection (real-valued sigmoid gate)
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for real+imag input

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated spinor transformation.

        Args:
            x: [B, S, D] complex64
        Returns:
            [B, S, D] complex64
        """
        # Compute gate from real representation
        x_real = torch.cat([x.real, x.imag], dim=-1)  # [B, S, 2D]
        # SEOP Fix 18: Scaled sigmoid ≈ Gaussian CDF, but ~3x faster than erf on CPU
        # Input to gate_proj is Gaussian (from ComplexRMSNorm). Linear projection preserves Gaussian.
        # sigmoid(Gaussian) with unit scale → bimodal at {0,1}, wastes interpolation range.
        # sigmoid(x * 1.7015) is a logistic-probit approximation to Φ(x).
        # Max absolute error is approximately 0.0095 vs Gaussian CDF Φ(x).
        # Maps Gaussian → ~Uniform(0,1), maximizing gate entropy. Uses HW-accelerated sigmoid.
        gate = torch.sigmoid(self.gate_proj(x_real) * 1.7015)  # [B, S, D]

        # Selective application (lerp form: avoids (1-gate) allocation + extra complex cast)
        rotated = self.spinor(x)
        return x + gate.to(x.dtype) * (rotated - x)
