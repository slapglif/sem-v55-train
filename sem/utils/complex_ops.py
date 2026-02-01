"""Complex tensor operations for SEM V5.5.

Provides efficient complex arithmetic primitives used throughout
the architecture: phase extraction, magnitude, complex multiplication,
Hermitian inner products, and real-complex conversions.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def complex_magnitude(z: Tensor) -> Tensor:
    """Compute magnitude |z| of complex tensor."""
    return z.abs()


def complex_phase(z: Tensor) -> Tensor:
    """Compute phase angle(z) of complex tensor."""
    return torch.angle(z)


def complex_from_polar(magnitude: Tensor, phase: Tensor) -> Tensor:
    """Construct complex tensor from polar form: z = r * exp(i*theta)."""
    return torch.polar(magnitude, phase)


def phase_rotate(z: Tensor, theta: Tensor) -> Tensor:
    """Rotate complex tensor by phase angle: z * exp(i*theta)."""
    return z * torch.polar(torch.ones_like(theta), theta)


def hermitian_inner_product(a: Tensor, b: Tensor, dim: int = -1) -> Tensor:
    """Compute <a, b> = sum(conj(a) * b) along dim."""
    return (a.conj() * b).sum(dim=dim)


def complex_to_real_pair(z: Tensor) -> tuple[Tensor, Tensor]:
    """Split complex tensor into (real, imag) pair."""
    return z.real, z.imag


def real_pair_to_complex(real: Tensor, imag: Tensor) -> Tensor:
    """Combine real and imag tensors into complex tensor."""
    return torch.complex(real, imag)


def complex_linear(input: Tensor, weight_real: Tensor, weight_imag: Tensor,
                    bias: Tensor | None = None) -> Tensor:
    """Complex-valued linear projection using real weight pairs.

    Computes: (W_r + i*W_i)(x_r + i*x_i) = (W_r*x_r - W_i*x_i) + i*(W_r*x_i + W_i*x_r)

    Args:
        input: Complex tensor [..., in_features]
        weight_real: Real weight [out_features, in_features]
        weight_imag: Imaginary weight [out_features, in_features]
        bias: Optional complex bias [out_features]
    """
    x_r, x_i = input.real, input.imag
    out_r = F.linear(x_r, weight_real) - F.linear(x_i, weight_imag)
    out_i = F.linear(x_r, weight_imag) + F.linear(x_i, weight_real)
    result = torch.complex(out_r, out_i)
    if bias is not None:
        result = result + bias
    return result


def ensure_complex(x: Tensor) -> Tensor:
    """Convert real tensor to complex if needed."""
    if x.is_complex():
        return x
    return torch.complex(x, torch.zeros_like(x))
