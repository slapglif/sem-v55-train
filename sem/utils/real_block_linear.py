"""
Real-Block Linear Layer for XPU compatibility.

[AUDIT]
Input: Complex signal x = x_r + j*x_i. Distributions x_r, x_i ~ Gaussian(0, sigma^2/2) for circular symmetry.
Parameters: weight_real, weight_imag ~ float32 parameters.
Target: Perform complex multiplication while bypassing lack of complex optimizer support on Intel XPU/NPU.

[MISMATCH]
Standard `nn.Linear` with `complex64` parameters suffers from Distributional Impedance Mismatch on XPU/NPU
hardware where optimizers lack complex kernels, leading to suboptimal convergence or fallback to slow CPU paths.

[DERIVATION]
The complex multiplication y = Wx is isomorphic to the real block transformation:
    Re(y) = W_r * x_r - W_i * x_i
    Im(y) = W_r * x_i + W_i * x_r
This derivation allows using standard real-valued optimizers (AdamW, etc.) on separate float32 weights
while mathematically preserving the complex signal integrity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor


class RealBlockLinear(nn.Module):
    """
    Implements a complex linear layer using real-valued float32 parameters.

    This layer stores weight_real and weight_imag separately to remain compatible
    with optimizers that do not support complex parameters on XPU/NPU backends.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Real and imaginary components stored as float32 parameters
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))

        # Tag for ComplexAdamW coupling
        self.weight_real._is_complex_real = True
        self.weight_imag._is_complex_imag = True
        self.weight_real._complex_partner = self.weight_imag
        self.weight_imag._complex_partner = self.weight_real

        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
            self.bias_real._is_complex_real = True
            self.bias_imag._is_complex_imag = True
            self.bias_real._complex_partner = self.bias_imag
            self.bias_imag._complex_partner = self.bias_real
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Applies Kaiming initialization to both real and imaginary components.
        """
        # Initialize real and imag parts independently
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))

        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the Real-Block Isomorphism transformation.

        $Re(y) = W_r x_r - W_i x_i$
        $Im(y) = W_r x_i + W_i x_r$

        Args:
            x: Input tensor, expected complex64.

        Returns:
            Output tensor, complex64.
        """
        if not x.is_complex():
            # For robustness, we handle non-complex input if it's the right shape
            # but primary use case is complex64.
            x = x.to(torch.complex64)

        xr = x.real
        xi = x.imag

        # Real-Block Isomorphism:
        # Re(y) = Wr*xr - Wi*xi
        # Im(y) = Wr*xi + Wi*xr

        # F.linear is used for optimized execution on both CPU and XPU/NPU
        yr = F.linear(xr, self.weight_real) - F.linear(xi, self.weight_imag)
        yi = F.linear(xi, self.weight_real) + F.linear(xr, self.weight_imag)

        if self.bias_real is not None:
            yr = yr + self.bias_real
            yi = yi + self.bias_imag

        return torch.complex(yr, yi)
