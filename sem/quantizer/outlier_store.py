"""High-precision storage for Fisher Information outliers.

Parameters identified as "outliers" (high Fisher information = critical
knowledge) are stored at FP16 or INT8 precision instead of being
vector-quantized. This preserves the fidelity of the most important
parameters while allowing aggressive compression of the bulk.
"""
import torch
from torch import Tensor
from typing import Dict, Tuple


class OutlierStore:
    """Stores high-Fisher-information parameter outliers at high precision.

    Supports FP16 storage with per-tensor scale factors for
    numerical stability.
    """

    def __init__(self, precision: str = 'fp16'):
        """
        Args:
            precision: 'fp16' or 'int8'
        """
        self.precision = precision
        self.store: Dict[str, Dict[str, Tensor]] = {}

    def store_outliers(self, name: str, values: Tensor, mask: Tensor):
        """Store outlier values at high precision.

        Args:
            name: Parameter name
            values: Full parameter tensor
            mask: Boolean mask (True = outlier to store)
        """
        outlier_values = values[mask]

        if self.precision == 'fp16':
            if outlier_values.is_complex():
                # Store real and imag as fp16 separately
                stored = torch.view_as_real(outlier_values).half()
            else:
                stored = outlier_values.half()
        elif self.precision == 'int8':
            # INT8 with per-tensor scaling
            if outlier_values.is_complex():
                real_view = torch.view_as_real(outlier_values).float()
            else:
                real_view = outlier_values.float()
            scale = real_view.abs().max() / 127.0
            stored = (real_view / (scale + 1e-12)).round().to(torch.int8)
            self.store[name] = {
                'values': stored,
                'mask': mask,
                'scale': scale,
                'is_complex': outlier_values.is_complex(),
            }
            return

        self.store[name] = {
            'values': stored,
            'mask': mask,
            'is_complex': outlier_values.is_complex(),
        }

    def restore_outliers(self, name: str, target: Tensor) -> Tensor:
        """Restore outlier values into a target tensor.

        Args:
            name: Parameter name
            target: Tensor to write outliers into (modified in-place)
        Returns:
            Modified target tensor
        """
        if name not in self.store:
            return target

        entry = self.store[name]
        mask = entry['mask']

        if self.precision == 'fp16':
            values = entry['values'].float()
            if entry['is_complex']:
                values = torch.view_as_complex(values)
        elif self.precision == 'int8':
            values = entry['values'].float() * entry['scale']
            if entry['is_complex']:
                values = torch.view_as_complex(values)

        target[mask] = values.to(target.dtype)
        return target

    def memory_bytes(self) -> int:
        """Total memory used by outlier store."""
        total = 0
        for entry in self.store.values():
            total += entry['values'].nelement() * entry['values'].element_size()
        return total

    def state_dict(self) -> dict:
        return {k: {kk: vv.cpu() if isinstance(vv, Tensor) else vv
                     for kk, vv in v.items()}
                for k, v in self.store.items()}

    def load_state_dict(self, state: dict):
        self.store = state
