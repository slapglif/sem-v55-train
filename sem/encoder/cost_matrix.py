"""Learned cost matrix for MESH-SDR optimal transport.

The cost matrix C[i,j] represents the "distance" between input token
embedding i and SDR codebook entry j. It is learned end-to-end,
allowing the encoder to discover the optimal mapping from dense
embeddings to sparse distributed representations.
"""
import torch
import torch.nn as nn
from torch import Tensor


class LearnedCostMatrix(nn.Module):
    """Compute cost matrix between input embeddings and SDR codebook.

    Uses a learned codebook of SDR prototypes. Cost is squared L2
    distance in a projected space.
    """

    def __init__(self, input_dim: int, num_candidates: int, proj_dim: int = 64):
        """
        Args:
            input_dim: Dimension of input embeddings
            num_candidates: Number of SDR candidate positions (sdr_candidates)
            proj_dim: Dimension of projection space for cost computation
        """
        super().__init__()
        self.proj_dim = proj_dim

        # Project input to cost space
        self.input_proj = nn.Linear(input_dim, proj_dim)

        # Learnable SDR codebook prototypes
        self.codebook = nn.Parameter(torch.randn(num_candidates, proj_dim) * 0.02)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Compute cost matrix between embeddings and codebook.

        Args:
            embeddings: [B, S, input_dim] float32
        Returns:
            cost: [B, S, num_candidates] float32 (non-negative)
        """
        # Project input
        proj = self.input_proj(embeddings)  # [B, S, proj_dim]

        # Squared L2 distance: ||proj - codebook||^2
        # = ||proj||^2 + ||codebook||^2 - 2 * proj @ codebook^T
        proj_sq = (proj ** 2).sum(dim=-1, keepdim=True)  # [B, S, 1]
        codebook_sq = (self.codebook ** 2).sum(dim=-1)    # [num_candidates]
        cross = torch.matmul(proj, self.codebook.T)       # [B, S, num_candidates]

        cost = proj_sq + codebook_sq.unsqueeze(0).unsqueeze(0) - 2 * cross

        # Ensure non-negative (numerical safety)
        return cost.clamp(min=0.0)
