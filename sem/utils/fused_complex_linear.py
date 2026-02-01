"""Complex Linear Layer using Real-Block Isomorphism.

[AUDIT]
Input: z = z_r + i·z_i ∈ ℂ^{in}, components ~ Gaussian(0, σ²/2) for circular symmetry.
Weight: W = W_r + i·W_i ∈ ℂ^{out×in}.
Output: y = Wz ∈ ℂ^{out}.

[MISMATCH]
Previous implementation used native complex64 F.linear, which:
1. Breaks torch.compile (inductor has no complex codegen → eager fallback)
2. Breaks bf16 AMP (autocast ignores complex64 → stays fp32)
3. Misses ComplexAdamW _complex_partner coupling (uses view_as_real instead)
4. 24 instances × 8 layers = graph breaks across entire Mamba block

[DERIVATION]
The complex product y = Wz decomposes exactly:
    y_r = W_r·z_r - W_i·z_i
    y_i = W_r·z_i + W_i·z_r
This is the Real-Block Isomorphism — zero information loss, pure float32 ops.
For complex circular Gaussian inputs, correct init variance is 1/(2·fan_in)
per component (not 1/fan_in), so we scale Kaiming by 1/√2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from sem.utils.complex_ops import safe_complex


class FusedComplexLinear(nn.Module):
    """Complex linear layer using Real-Block Isomorphism.

    Stores weight_real and weight_imag as separate float32 parameters,
    enabling torch.compile fusion, bf16 AMP, and proper ComplexAdamW coupling.
    Drop-in replacement: same API (accepts complex64 input, returns complex64).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Real-block parameters (float32, compile-friendly)
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))

        # Tag for ComplexAdamW coupled momentum
        self.weight_real._is_complex_real = True  # type: ignore[attr-defined]
        self.weight_imag._is_complex_imag = True  # type: ignore[attr-defined]
        self.weight_real._complex_partner = self.weight_imag  # type: ignore[attr-defined]
        self.weight_imag._complex_partner = self.weight_real  # type: ignore[attr-defined]

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
            self.bias_real._is_complex_real = True  # type: ignore[attr-defined]
            self.bias_imag._is_complex_imag = True  # type: ignore[attr-defined]
            self.bias_real._complex_partner = self.bias_imag  # type: ignore[attr-defined]
            self.bias_imag._complex_partner = self.bias_real  # type: ignore[attr-defined]
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        self._reset_parameters()

    def _reset_parameters(self):
        """Kaiming init scaled for complex circular symmetry.

        For W = W_r + iW_i acting on z = z_r + iz_i:
          Var(y) = 2·Var(W_component)·Var(z)
        For unit output variance: Var(W_component) = 1/(2·fan_in).
        Standard Kaiming gives 1/fan_in, so we scale by 1/√2.
        """
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
            # Correct for complex fan-in: total variance should be 1/fan_in,
            # each component contributes 1/(2·fan_in)
            self.weight_real.mul_(1.0 / math.sqrt(2.0))
            self.weight_imag.mul_(1.0 / math.sqrt(2.0))

        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, z: Tensor) -> Tensor:
        """Apply complex linear via Real-Block Isomorphism.

        y_r = W_r·z_r - W_i·z_i
        y_i = W_r·z_i + W_i·z_r

        All operations are float32 → torch.compile can fuse the entire graph.
        bf16 AMP will autocast these F.linear calls correctly.

        Args:
            z: [..., in_features] complex64
        Returns:
            [..., out_features] complex64
        """
        zr = z.real
        zi = z.imag

        # Real-Block Isomorphism: 4 real matmuls = 1 complex matmul
        yr = F.linear(zr, self.weight_real) - F.linear(zi, self.weight_imag)
        yi = F.linear(zi, self.weight_real) + F.linear(zr, self.weight_imag)

        if self.bias_real is not None:
            yr = yr + self.bias_real
            yi = yi + self.bias_imag

        return safe_complex(yr, yi)
