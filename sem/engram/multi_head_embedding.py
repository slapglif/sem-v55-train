"""Multi-Head Embedding - Offset-based multi-head embedding lookup.

Combines multiple embeddings with different vocabulary sizes into a single
embedding layer using offset indexing. Each head has its own vocabulary
partition within the combined embedding table.
"""

import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadEmbedding(nn.Module):
    """Multi-head embedding with offset indexing.

    Instead of creating separate nn.Embedding for each head, uses a single
    combined embedding table with offset indexing to partition vocabularies.

    Attributes:
        num_heads: Number of embedding heads
        embedding_dim: Embedding dimension per head
        offsets: Tensor[num_heads] of cumulative vocabulary offsets
        embedding: Combined embedding table
    """

    def __init__(self, list_of_N: list[int], D: int):
        """Initialize multi-head embedding.

        Args:
            list_of_N: Vocabulary size for each head
            D: Embedding dimension per head
        """
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        # Calculate cumulative offsets for each head
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        # Create combined embedding table
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Lookup embeddings for multi-head input IDs.

        Args:
            input_ids: [B, T, num_heads] int64 hash IDs for each head

        Returns:
            embeddings: [B, T, num_heads, D] float32 embeddings
        """
        # Add offsets to partition vocabularies: input_ids[..., h] += offsets[h]
        shifted_input_ids = input_ids + self.offsets

        # Lookup embeddings
        output = self.embedding(shifted_input_ids)

        return output  # [B, T, num_heads, D]
