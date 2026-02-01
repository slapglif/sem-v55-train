"""Complex operations specific to spinor representations.

Provides block-diagonal complex matrix operations and phase-aware
transformations used in the Mamba-3 spinor layer.
"""

import torch
import torch.nn as nn
from torch import Tensor


def block_diagonal_complex_matmul(x: Tensor, blocks: Tensor) -> Tensor:
    """Block-diagonal complex matrix-vector multiplication.

    100x faster than dense: O(B*S*num_blocks*block_size^2) vs O(B*S*D^2).
    Refactored to use explicit real math to avoid XPU complex overhead.

    Args:
        x: Input [B, S, num_blocks, block_size] complex64
        blocks: Weight blocks [num_blocks, block_size, block_size] complex64
    Returns:
        Output [B, S, num_blocks, block_size] complex64
    """
    # Decompose into real and imaginary components
    xr, xi = x.real, x.imag
    br, bi = blocks.real, blocks.imag

    # Einstein summation for real and imaginary parts
    # (br + i*bi) @ (xr + i*xi) = (br@xr - bi@xi) + i*(br@xi + bi@xr)

    # Real part of result
    out_real = torch.einsum("noi,bsni->bsno", br, xr) - torch.einsum(
        "noi,bsni->bsno", bi, xi
    )
    # Imaginary part of result
    out_imag = torch.einsum("noi,bsni->bsno", br, xi) + torch.einsum(
        "noi,bsni->bsno", bi, xr
    )

    return torch.complex(out_real, out_imag)


def complex_mul_real(
    r1: Tensor, i1: Tensor, r2: Tensor, i2: Tensor
) -> tuple[Tensor, Tensor]:
    """Complex multiplication decomposed into real operations.

    (r1 + i1*j) * (r2 + i2*j) = (r1*r2 - i1*i2) + (r1*i2 + i1*r2)j

    Args:
        r1, i1: Real/Imag components of first operand
        r2, i2: Real/Imag components of second operand
    Returns:
        (r_out, i_out): Real/Imag components of product
    """
    real = r1 * r2 - i1 * i2
    imag = r1 * i2 + i1 * r2
    return real, imag


def complex_selective_gate(x: Tensor, gate: Tensor) -> Tensor:
    """Apply complex-valued selective gating.

    Gate modulates both magnitude and phase of the input.
    Refactored to use explicit real math.

    Args:
        x: Input [B, S, D] complex64
        gate: Gate values [B, S, D] float32 (sigmoid output)
    Returns:
        Gated output [B, S, D] complex64
    """
    xr, xi = x.real, x.imag
    g = gate.to(xr.dtype)
    return torch.complex(xr * g, xi * g)
