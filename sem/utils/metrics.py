"""Metrics for monitoring SEM V5.5 signal-entropic properties.

Tracks information-theoretic and physical quantities that validate
the architecture's core invariants: unitarity, entropy balance,
and information density.
"""
import torch
from torch import Tensor
import math


def unitarity_deviation(psi_in: Tensor, psi_out: Tensor) -> Tensor:
    """Measure deviation from unitarity: ||psi_out||^2 / ||psi_in||^2 - 1.

    Perfect unitarity gives 0.0. Values > 1e-6 indicate energy leak.

    Args:
        psi_in: Input wavefunction [..., D] complex
        psi_out: Output wavefunction [..., D] complex
    Returns:
        Scalar deviation (should be < 1e-6)
    """
    norm_in = (psi_in.abs() ** 2).sum(dim=-1)
    norm_out = (psi_out.abs() ** 2).sum(dim=-1)
    return (norm_out / (norm_in + 1e-12) - 1.0).abs().mean()


def information_density(z: Tensor) -> Tensor:
    """Compute information density as normalized magnitude entropy.

    High density = uniform magnitude distribution (maximum information).
    Low density = peaked distribution (redundant encoding).

    Args:
        z: Complex tensor [..., D]
    Returns:
        Scalar information density in [0, 1]
    """
    mag = z.abs()
    p = mag / (mag.sum(dim=-1, keepdim=True) + 1e-12)
    entropy = -(p * (p + 1e-12).log()).sum(dim=-1)
    max_entropy = math.log(z.shape[-1])
    return (entropy / max_entropy).mean()


def sparsity_ratio(x: Tensor, threshold: float = 1e-6) -> Tensor:
    """Fraction of near-zero elements.

    Args:
        x: Input tensor
        threshold: Values below this are considered zero
    Returns:
        Sparsity ratio in [0, 1] (1 = fully sparse)
    """
    if x.is_complex():
        x = x.abs()
    return (x < threshold).float().mean()


def phase_coherence(z: Tensor) -> Tensor:
    """Measure phase coherence across the hidden dimension.

    High coherence = phases are aligned (correlated representation).
    Low coherence = phases are random (independent features).

    Args:
        z: Complex tensor [..., D]
    Returns:
        Coherence in [0, 1]
    """
    unit_phases = z / (z.abs() + 1e-12)  # Normalize to unit magnitude
    mean_phase = unit_phases.mean(dim=-1)
    return mean_phase.abs().mean()  # |mean of unit vectors|


def bits_per_parameter(num_outliers: int, num_bulk: int,
                       outlier_bits: int = 16,
                       codebook_bits: int = 8,
                       group_size: int = 2) -> float:
    """Compute effective bits per parameter for HAS-VQ.

    Args:
        num_outliers: Parameters stored at high precision
        num_bulk: Parameters in VQ codebook
        outlier_bits: Bits for outlier storage
        codebook_bits: Bits for codebook index
        group_size: Parameters per VQ group
    Returns:
        Effective bits per parameter
    """
    total_params = num_outliers + num_bulk
    outlier_cost = num_outliers * outlier_bits
    bulk_cost = (num_bulk / group_size) * codebook_bits  # One index per group
    return (outlier_cost + bulk_cost) / total_params


def sinkhorn_convergence(T: Tensor, target_row: Tensor | None = None,
                         target_col: Tensor | None = None) -> Tensor:
    """Measure Sinkhorn transport plan convergence.

    Checks if marginals of T match target distributions.

    Args:
        T: Transport plan [..., N, M]
        target_row: Target row marginals [..., N] (default: uniform)
        target_col: Target column marginals [..., M] (default: uniform)
    Returns:
        Mean L1 error of marginals
    """
    row_marginal = T.sum(dim=-1)
    col_marginal = T.sum(dim=-2)

    if target_row is None:
        target_row = torch.ones_like(row_marginal) / row_marginal.shape[-1]
    if target_col is None:
        target_col = torch.ones_like(col_marginal) / col_marginal.shape[-1]

    row_error = (row_marginal - target_row).abs().mean()
    col_error = (col_marginal - target_col).abs().mean()
    return (row_error + col_error) / 2
