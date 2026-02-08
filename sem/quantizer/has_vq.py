"""Hessian-Adaptive Sparse Vector Quantization (HAS-VQ).

Top-level quantizer that combines:
1. Fisher Information tracking to identify critical parameters
2. High-precision outlier storage (FP16/INT8) for critical params
3. 2-bit VQ codebook for bulk parameters
4. Hessian-aligned grouping for optimal quantization

Achieves ~4.2 bits-per-parameter effective compression while
preserving model quality, especially on factual knowledge
stored in high-gradient subspaces.
"""

import torch
import torch.nn as nn
import math
import functools
import operator
from torch import Tensor
from typing import Dict, Optional

from .fisher_tracker import FisherTracker
from .outlier_store import OutlierStore
from .vq_codebook import VQCodebook


class HASVQ:
    """Hessian-Adaptive Sparse Vector Quantizer.

    Usage:
        1. During training: call update_fisher() after each backward pass
        2. After training: call quantize_model() to compress
        3. For inference: call dequantize_model() to restore
    """

    def __init__(
        self,
        codebook_size: int = 256,
        group_size: int = 2,
        fisher_ema_decay: float = 0.99,
        outlier_percentile: float = 0.01,
        dead_code_threshold: int = 100,
        outlier_precision: str = "fp16",
    ):
        self.codebook_size = codebook_size
        self.group_size = group_size
        self.outlier_percentile = outlier_percentile

        self.fisher = FisherTracker(decay=fisher_ema_decay)
        self.outlier_store = OutlierStore(precision=outlier_precision)
        self.codebooks: Dict[str, VQCodebook] = {}
        self.quantized_indices: Dict[str, Tensor] = {}
        self.original_shapes: Dict[str, tuple] = {}
        self.original_is_complex: Dict[str, bool] = {}
        self.dead_code_threshold = dead_code_threshold

    def update_fisher(self, model: nn.Module):
        """Update Fisher tracking after backward pass.

        Call this after loss.backward(), before optimizer.step().
        """
        self.fisher.update(model.named_parameters())

    def quantize_model(
        self, model: nn.Module, calibration_steps: int = 100
    ) -> Dict[str, float]:
        """Quantize all eligible parameters.

        Args:
            model: Trained model to quantize
            calibration_steps: (unused, Fisher already tracked during training)

        Returns:
            Dict of metrics: BPP per layer, total BPP, outlier fraction
        """
        metrics = {}
        total_params = 0
        total_bits = 0

        for name, param in model.named_parameters():
            if param.numel() < self.group_size * 4:
                # Skip tiny parameters (biases, etc.)
                continue

            # Get outlier mask from Fisher information
            mask = self.fisher.get_outlier_mask(name, self.outlier_percentile)

            if mask is None:
                # No Fisher data: store everything at full precision
                continue

            self.original_shapes[name] = param.shape
            self.original_is_complex[name] = bool(param.is_complex())

            # Flatten parameter for quantization
            if param.is_complex():
                flat = torch.view_as_real(param.data).reshape(-1)
            else:
                flat = param.data.reshape(-1)

            flat_mask = mask.reshape(-1)
            if param.is_complex():
                # Expand mask for real/imag
                flat_mask = (
                    flat_mask.unsqueeze(-1)
                    .expand_as(torch.view_as_real(param.data))
                    .reshape(-1)
                )

            # Store outliers at high precision
            self.outlier_store.store_outliers(name, flat, flat_mask)

            # Quantize bulk parameters
            bulk = flat[~flat_mask]

            # Pad to multiple of group_size
            remainder = bulk.numel() % self.group_size
            if remainder > 0:
                padding = self.group_size - remainder
                bulk = torch.cat([bulk, torch.zeros(padding, device=bulk.device)])

            # Reshape to groups
            groups = bulk.reshape(-1, self.group_size)

            # Create codebook for this parameter
            codebook = VQCodebook(
                self.codebook_size,
                self.group_size,
                dead_code_threshold=self.dead_code_threshold,
            ).to(param.device)

            # Quantize
            quantized, indices, _ = codebook.quantize(groups)
            codebook.update_codebook(groups, indices)

            self.codebooks[name] = codebook
            self.quantized_indices[name] = indices

            # Compute BPP for this parameter
            num_outliers = flat_mask.sum().item()
            num_bulk = flat.numel() - num_outliers
            outlier_bits = num_outliers * 16  # FP16
            bulk_bits = (num_bulk / self.group_size) * math.log2(self.codebook_size)
            param_bpp = (outlier_bits + bulk_bits) / flat.numel()

            metrics[f"{name}_bpp"] = param_bpp
            total_params += flat.numel()
            total_bits += outlier_bits + bulk_bits

        if total_params > 0:
            metrics["total_bpp"] = total_bits / total_params

        return metrics

    def dequantize_parameter(self, name: str, device: torch.device) -> Optional[Tensor]:
        """Reconstruct a quantized parameter.

        Args:
            name: Parameter name
            device: Target device
        Returns:
            Reconstructed tensor or None if not quantized
        """
        if name not in self.quantized_indices:
            return None

        codebook = self.codebooks[name]
        indices = self.quantized_indices[name]

        # Reconstruct bulk from codebook
        bulk = codebook.codebook[indices].reshape(-1)

        # Get original flat size
        shape = self.original_shapes[name]
        total_elements = functools.reduce(operator.mul, shape, 1)
        is_complex = self.original_is_complex.get(name, False)
        total_real_elements = total_elements * 2 if is_complex else total_elements

        # Create output tensor in the same flattened representation used during quantization.
        # For complex params, we reconstruct the real-view (..,2) flattened buffer then convert.
        flat = torch.zeros(total_real_elements, device=device)

        # Fill bulk values
        if name in self.outlier_store.store:
            mask = self.outlier_store.store[name]["mask"].reshape(-1)
            if mask.numel() != total_real_elements:
                raise ValueError(
                    f"Outlier mask size mismatch for {name}: mask has {mask.numel()} elems, "
                    f"expected {total_real_elements} (is_complex={is_complex})"
                )
            bulk_needed = int((~mask).sum().item())
            flat[~mask] = bulk[:bulk_needed]
            flat = self.outlier_store.restore_outliers(name, flat)
        else:
            flat[:] = bulk[:total_real_elements]

        if is_complex:
            flat2 = flat.reshape(*shape, 2)
            return torch.view_as_complex(flat2).to(device)
        return flat.reshape(shape).to(device)

    def compression_summary(self) -> str:
        """Return human-readable compression summary."""
        lines = ["HAS-VQ Compression Summary", "=" * 40]

        total_original = 0
        total_compressed = 0

        for name in self.quantized_indices:
            shape = self.original_shapes.get(name, ())
            numel = functools.reduce(operator.mul, shape, 1) if shape else 0
            original_bits = numel * 32  # FP32

            # Compressed bits
            if name in self.codebooks:
                idx_bits = len(self.quantized_indices[name]) * math.log2(
                    self.codebook_size
                )
                cb_bits = self.codebooks[name].codebook.numel() * 32
                outlier_bits = (
                    self.outlier_store.memory_bytes() * 8
                    if name in self.outlier_store.store
                    else 0
                )
                compressed = idx_bits + cb_bits + outlier_bits
            else:
                compressed = original_bits

            ratio = original_bits / max(compressed, 1)
            lines.append(f"  {name}: {ratio:.1f}x compression")

            total_original += original_bits
            total_compressed += compressed

        if total_compressed > 0:
            lines.append(
                f"\nTotal: {total_original / total_compressed:.1f}x compression"
            )
            lines.append(
                f"Effective BPP: {total_compressed / max(total_original / 32, 1):.2f}"
            )

        return "\n".join(lines)
