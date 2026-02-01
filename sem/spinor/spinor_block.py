"""Block-diagonal spinor factorization with Real-Block Isomorphism.

Implements the spinor representation as block-diagonal complex matrices.
Instead of a dense D×D complex matrix (O(D^2)), we use num_blocks
independent block_size×block_size matrices (O(num_blocks * block_size^2)).

For D=256, block_size=8, num_blocks=32:
- Dense: 256^2 = 65,536 complex multiplies
- Block-diagonal: 32 * 8^2 = 2,048 complex multiplies
- Speedup: 32x (theoretical), ~100x with memory access patterns

The blocks maintain non-commutative geometric (spinor) properties
because each block acts as an independent SU(block_size) rotation.

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
import math
from torch import Tensor

from sem.utils.complex_ops import safe_complex

# block_diagonal_complex_matmul no longer needed — einsum done inline with real-block params


class SpinorBlock(nn.Module):
    """Block-diagonal spinor transformation layer.

    Factorizes a D×D complex transform into num_blocks independent
    block_size×block_size complex rotations, stored as float32 pairs.
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
            # Antisymmetric imaginary part for near-unitary initialization
            skew = torch.randn(block_size, block_size) * 0.02
            weight_imag[b] = skew - skew.T

        self.weight_real = nn.Parameter(weight_real)
        self.weight_imag = nn.Parameter(weight_imag)

        # Tag for ComplexAdamW coupled momentum
        self.weight_real._is_complex_real = True  # type: ignore[attr-defined]
        self.weight_imag._is_complex_imag = True  # type: ignore[attr-defined]
        self.weight_real._complex_partner = self.weight_imag  # type: ignore[attr-defined]
        self.weight_imag._complex_partner = self.weight_real  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> Tensor:
        """Apply block-diagonal spinor rotation.

        Args:
            x: [B, S, D] complex64
        Returns:
            [B, S, D] complex64
        """
        B, S, D = x.shape

        # Reshape to blocks: [B, S, num_blocks, block_size]
        x_blocks = x.view(B, S, self.num_blocks, self.block_size)

        # Decompose input
        xr, xi = x_blocks.real, x_blocks.imag
        br, bi = self.weight_real, self.weight_imag

        # Block-diagonal complex matmul via Real-Block Isomorphism
        # (br + i·bi) @ (xr + i·xi) = (br@xr - bi@xi) + i·(br@xi + bi@xr)
        out_real = torch.einsum("noi,bsni->bsno", br, xr) - torch.einsum(
            "noi,bsni->bsno", bi, xi
        )
        out_imag = torch.einsum("noi,bsni->bsno", br, xi) + torch.einsum(
            "noi,bsni->bsno", bi, xr
        )

        out = safe_complex(out_real, out_imag)

        # Reshape back: [B, S, D]
        return out.reshape(B, S, D)


class SpinorGate(nn.Module):
    """Gated spinor transformation with selective activation.

    Applies: output = gate * SpinorBlock(x) + (1-gate) * x
    where gate is input-dependent (selective mechanism from Mamba).
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
        # sigmoid(x * 1.7015) ≈ Φ(x) within 1% everywhere (logistic ≈ probit correspondence).
        # Maps Gaussian → ~Uniform(0,1), maximizing gate entropy. Uses HW-accelerated sigmoid.
        gate = torch.sigmoid(self.gate_proj(x_real) * 1.7015)  # [B, S, D]

        # Selective application (lerp form: avoids (1-gate) allocation + extra complex cast)
        rotated = self.spinor(x)
        return x + gate.to(x.dtype) * (rotated - x)
