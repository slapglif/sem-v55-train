"""Phase-preserving normalization for complex tensors.

Standard LayerNorm destroys phase information in complex tensors.
This module normalizes only the magnitude while preserving phase angles,
which is essential for the spinor representation where meaning is
encoded in the phase.
"""

import torch
import torch.nn as nn
from sem.utils.complex_ops import safe_complex
from torch import Tensor


class ComplexRMSNorm(nn.Module):
    """RMS normalization for complex tensors, preserving phase.

    Simpler than full LayerNorm: z_out = z / rms(|z|) * gamma
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, z: Tensor) -> Tensor:
        # SEOP Fix 4: Sparse-aware RMS normalization
        # For K-sparse signals in D dims, standard RMS = √(Σ|z|²/D) underestimates
        # by factor √(K/D), amplifying noise in inactive entries.
        # Fix: compute RMS only over active (nonzero) entries.
        # Efficiency: compute |z|² directly, avoiding redundant sqrt→square.
        mag_sq = z.real * z.real + z.imag * z.imag  # |z|², no sqrt
        active_mask = mag_sq > 1e-12  # (1e-6)² threshold on mag_sq
        active_count = active_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        # SEOP Fix 86: Sum only active elements to match active_count denominator
        # Was: mag_sq.sum() / active_count — numerator summed ALL elements (including inactive)
        # but denominator only counted active ones, inflating RMS for sparse signals.
        active_mag_sq = mag_sq * active_mask.float()
        rms = (active_mag_sq.sum(dim=-1, keepdim=True) / active_count).sqrt()
        # Real-math normalization to bypass complex division/multiplication on XPU
        # scale is real-valued, shape (..., dim)
        scale = self.gamma / (rms + self.eps)
        return safe_complex(z.real * scale, z.imag * scale)
