"""2-bit Vector Quantized codebook with EMA learning.

The VQ codebook maps groups of parameters to discrete codes.
With group_size=2 and codebook_size=256 (8-bit index), the effective
rate is 8 bits / 2 params = 4 bits per parameter for the bulk.

Combined with 1% outliers at 16 bits:
    BPP = 0.99 * 4 + 0.01 * 16 = 4.12 bits per parameter

Codebook learning uses Exponential Moving Average (EMA) updates
instead of gradient descent, which is more stable for VQ.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class VQCodebook(nn.Module):
    """Vector Quantization codebook with EMA updates.

    Maps groups of parameters to nearest codebook entries.
    Dead codes are revived by reinitializing from high-error groups.
    """

    def __init__(self, codebook_size: int = 256, group_size: int = 2,
                 ema_decay: float = 0.99, dead_code_threshold: int = 100):
        super().__init__()
        self.codebook_size = codebook_size
        self.group_size = group_size
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook vectors
        self.register_buffer(
            'codebook', torch.randn(codebook_size, group_size) * 0.02
        )

        # EMA tracking
        self.register_buffer(
            'ema_count', torch.zeros(codebook_size)
        )
        self.register_buffer(
            'ema_sum', torch.zeros(codebook_size, group_size)
        )

        # Dead code tracking
        self.register_buffer(
            'usage_count', torch.zeros(codebook_size, dtype=torch.long)
        )
        self.register_buffer(
            'steps_since_use', torch.zeros(codebook_size, dtype=torch.long)
        )

    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Quantize parameter groups to nearest codebook entry.

        Args:
            x: [num_groups, group_size] float32
        Returns:
            quantized: [num_groups, group_size] quantized values
            indices: [num_groups] int64 codebook indices
            commit_loss: Scalar commitment loss
        """
        # Compute distances: ||x - codebook||^2
        # = ||x||^2 + ||codebook||^2 - 2 * x @ codebook^T
        dists = (
            (x ** 2).sum(dim=-1, keepdim=True)
            + (self.codebook ** 2).sum(dim=-1)
            - 2 * x @ self.codebook.T
        )  # [num_groups, codebook_size]

        # Nearest code
        indices = dists.argmin(dim=-1)  # [num_groups]
        quantized = self.codebook[indices]  # [num_groups, group_size]

        # Commitment loss: encourage encoder to commit to codes
        commit_loss = ((x.detach() - quantized) ** 2).mean()

        # Straight-through estimator for gradient
        quantized_st = x + (quantized - x).detach()

        return quantized_st, indices, commit_loss

    @torch.no_grad()
    def update_codebook(self, x: Tensor, indices: Tensor):
        """Update codebook via EMA.

        Args:
            x: [num_groups, group_size] input groups
            indices: [num_groups] assigned codebook indices
        """
        # Count assignments per code
        one_hot = torch.zeros(
            x.shape[0], self.codebook_size, device=x.device
        )
        one_hot.scatter_(1, indices.unsqueeze(1), 1.0)

        # EMA updates
        counts = one_hot.sum(dim=0)
        sums = one_hot.T @ x  # [codebook_size, group_size]

        self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_sum.mul_(self.ema_decay).add_(sums, alpha=1 - self.ema_decay)

        # Update codebook
        n = self.ema_count.unsqueeze(1).clamp(min=1e-5)
        self.codebook.copy_(self.ema_sum / n)

        # Track dead codes
        self.usage_count += counts.long()
        used = counts > 0
        self.steps_since_use[used] = 0
        self.steps_since_use[~used] += 1

    @torch.no_grad()
    def revive_dead_codes(self, x: Tensor):
        """Reinitialize dead codes from high-error input groups.

        Args:
            x: [num_groups, group_size] current input batch
        """
        dead = self.steps_since_use >= self.dead_code_threshold
        num_dead = dead.sum().item()

        if num_dead == 0:
            return

        # Find high-error groups
        dists = ((x.unsqueeze(1) - self.codebook.unsqueeze(0)) ** 2).sum(dim=-1)
        min_dists = dists.min(dim=1).values  # [num_groups]

        # Pick top-k highest error groups
        k = min(num_dead, x.shape[0])
        _, top_idx = torch.topk(min_dists, k)

        # Reinitialize dead codes
        dead_indices = dead.nonzero(as_tuple=True)[0][:k]
        self.codebook[dead_indices] = x[top_idx]
        self.steps_since_use[dead_indices] = 0
        self.ema_count[dead_indices] = 1.0
        self.ema_sum[dead_indices] = x[top_idx]

    def compute_bpp(self, num_groups: int) -> float:
        """Compute bits per parameter.

        Args:
            num_groups: Total number of parameter groups
        Returns:
            Bits per parameter
        """
        import math
        bits_per_index = math.log2(self.codebook_size)  # 8 for 256 codes
        total_bits = num_groups * bits_per_index
        total_params = num_groups * self.group_size
        return total_bits / total_params
