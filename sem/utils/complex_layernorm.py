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


class ComplexLayerNorm(nn.Module):
    """Phase-preserving layer normalization for complex tensors.

    Normalizes magnitude: z_out = z / (mean(|z|) + eps) * gamma
    Phase is completely preserved.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # Learnable scale (real-valued, applied to magnitude)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        # Learnable phase shift
        self.beta_phase = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, z: Tensor) -> Tensor:
        """Normalize complex tensor preserving phase.

        Args:
            z: Complex tensor [..., normalized_shape]
        Returns:
            Normalized complex tensor, same shape
        """
        # Compute magnitude
        mag = z.abs()  # [..., D]

        # Normalize magnitude (mean over last dim)
        mean_mag = mag.mean(dim=-1, keepdim=True)
        var_mag = ((mag - mean_mag) ** 2).mean(dim=-1, keepdim=True)
        mag_normalized = (mag - mean_mag) / (var_mag.sqrt() + self.eps)

        # Apply learnable scale
        mag_scaled = mag_normalized * self.gamma

        # Reconstruct: preserve original phase, use normalized magnitude
        phase = torch.angle(z)
        phase = phase + self.beta_phase  # learnable phase shift

        return torch.polar(mag_scaled.abs() + self.eps, phase)


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
        rms = (mag_sq.sum(dim=-1, keepdim=True) / active_count).sqrt()
        # Real-math normalization to bypass complex division/multiplication on XPU
        # scale is real-valued, shape (..., dim)
        scale = self.gamma / (rms + self.eps)
        return safe_complex(z.real * scale, z.imag * scale)
