"""Complex-valued Mamba-3 layer using official Mamba2 backend.

Wraps the mamba-ssm Mamba2 implementation with complex tensor support:
- Input: complex64 tensors [B, S, D]
- Conversion: complex → real (interleaved) for Mamba2
- Output: real → complex reconstruction
- Residual connection with learnable scaling

The key insight: meaning is encoded in the PHASE of the complex state.
Context changes ROTATE the phase while preserving magnitude (information density).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from torch import Tensor
from ..utils.complex_layernorm import ComplexRMSNorm
from ..utils.fused_complex_linear import FusedComplexLinear
from ..utils.complex_ops import safe_complex

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None


logger = logging.getLogger(__name__)


class ComplexMamba3Layer(nn.Module):
    """Single Complex Mamba-3 layer with Mamba2 backend.

    Architecture:
    1. ComplexRMSNorm
    2. Mamba2 (official mamba-ssm implementation)
    3. Residual connection
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 64,
        mimo_groups: int = 8,
        block_size: int = 8,
        d_conv: int = 4,
        num_layers: int = 1,
        use_mixed_precision_a: bool = True,
        memory_horizon_ratio: float = 0.0,
        max_seq_length: int = 256,
        use_mamba2: bool = True,
    ):
        """
        Args:
            hidden_dim: Model hidden dimension
            state_dim: SSM state dimension
            mimo_groups: Number of MIMO groups for parallel processing
            block_size: Spinor block size (unused, kept for backwards compatibility)
            d_conv: Kernel size for depthwise convolution
            num_layers: Total number of layers (for residual scaling)
            use_mixed_precision_a: Use bfloat16 for A tensor (unused, kept for backwards compatibility)
            memory_horizon_ratio: τ/S ratio (unused, kept for backwards compatibility)
            max_seq_length: Sequence length (unused, kept for backwards compatibility)
            use_mamba2: Must be True (Mamba2 is the only path now)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mimo_groups = mimo_groups
        self.group_dim = hidden_dim // mimo_groups

        assert hidden_dim % mimo_groups == 0, (
            f"hidden_dim {hidden_dim} must be divisible by mimo_groups {mimo_groups}"
        )

        # Pre-norm
        self.norm = ComplexRMSNorm(hidden_dim)

        # Mamba2 is required
        if Mamba2 is None:
            raise ImportError(
                "mamba-ssm package is required. Install with: uv pip install mamba-ssm"
            )

        self.mamba2 = Mamba2(
            d_model=hidden_dim * 2,
            d_state=state_dim,
            d_conv=d_conv,
            expand=2,
            headdim=64,
            chunk_size=256,
        )

        # Learnable magnitude gate threshold (SEOP Fix 12: χ²(2)-CDF matched)
        # For complex Gaussian z, |z|² ~ Exponential(1/2σ²). Using CDF gate
        # 1-exp(-|z|²·β) instead of sigmoid gives uniformly distributed gate values,
        # maximizing gradient information. Initialize β=exp(0)=1 for unit-variance match.
        self.activation_threshold = nn.Parameter(
            torch.tensor(0.0)
        )  # Reverted: sparse gate is a feature, not a bug

        # SEOP Fix 17+52: Per-dimension SSM output scaling
        # Init to 1.0 (not 2.5) — with scaled embedding init (Fix 52), the SSM output
        # magnitude is already well-matched. 2.5x amplification adds excessive noise
        # relative to the smaller embedding signal.
        self.ssm_output_scale = nn.Parameter(torch.ones(hidden_dim))

        # SEOP Fix 22+53: Learnable residual scaling initialized to 1/√L
        # Without scaling, variance grows as 1 + L·σ² across L layers.
        # DeepSeek/GPT-2 uses 1/√(2L) for transformer (2 branches: attn+FFN).
        # Mamba has 1 branch per layer, so correct scaling is 1/√L.
        self.residual_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(num_layers)))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of Complex Mamba-3 layer.

        Args:
            x: [B, S, D] complex64
        Returns:
            [B, S, D] complex64 (with residual)
        """
        B, S, D = x.shape
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Mamba2 forward pass (only path)
        x_real = torch.cat([x.real, x.imag], dim=-1)
        y_real = self.mamba2(x_real)
        y = safe_complex(y_real[..., :D], y_real[..., D:])
        y = y * self.ssm_output_scale
        return residual + self.residual_scale * y
