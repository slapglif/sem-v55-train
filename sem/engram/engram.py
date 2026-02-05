"""Engram - N-gram hash-based embedding augmentation module.

Augments hidden states with n-gram context via hash-based embeddings.
Uses gated mixing to selectively incorporate n-gram information.

Architecture:
1. Generate n-gram hash IDs from input tokens
2. Lookup multi-head embeddings for each hash
3. Flatten embeddings across n-gram sizes and heads
4. Gate mixing: hidden_state ⊙ sigmoid(query · key)
5. Add short convolution for local smoothing
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import math

import torch
import torch.nn as nn
from torch import Tensor

from .hash_mapping import NgramHashMapping
from .multi_head_embedding import MultiHeadEmbedding


@dataclass
class EngramConfig:
    """Engram configuration.

    Attributes:
        tokenizer: Tokenizer instance (required)
        engram_vocab_size: Base vocab sizes for each n-gram size [bigram_vocab, trigram_vocab, ...]
        max_ngram_size: Maximum n-gram size (2 = bigrams only, 3 = bigrams + trigrams)
        n_embed_per_ngram: Embedding dimension per n-gram (divided among heads)
        n_head_per_ngram: Number of hash heads per n-gram size
        layer_ids: Layer IDs where Engram is inserted
        pad_id: Padding token ID
        seed: Random seed for hash multipliers
        kernel_size: Convolution kernel size
        hc_mult: Hierarchical capacity multiplier (number of gated channels)
    """

    tokenizer: Any = None
    engram_vocab_size: list[int] = field(default_factory=lambda: [129280 * 5, 129280 * 5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: list[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    hc_mult: int = 4  # Hierarchical capacity multiplier


class ShortConv(nn.Module):
    """Short convolution for local smoothing.

    Applies grouped 1D convolution across time dimension with RMSNorm
    and optional SiLU activation.

    Attributes:
        hc_mult: Number of channels (hierarchical capacity multiplier)
        activation: Whether to apply SiLU activation
        conv: Grouped 1D convolution
        norms: RMSNorm for each channel
        act_fn: SiLU activation function
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        """Initialize short convolution.

        Args:
            hidden_size: Hidden dimension
            kernel_size: Convolution kernel size
            dilation: Dilation factor
            norm_eps: RMSNorm epsilon
            hc_mult: Number of channels
            activation: Whether to apply SiLU activation
        """
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult

        # Grouped convolution (each channel processed separately)
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        # RMSNorm for each channel
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)])

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply short convolution.

        Args:
            x: [B, T, hc_mult, D] input tensor

        Returns:
            output: [B, T, hc_mult, D] convolved tensor
        """
        B, T, G, C = x.shape
        assert G == self.hc_mult, f"Expected {self.hc_mult} channels, got {G}"

        # Apply RMSNorm to each channel separately
        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]  # [B, T, C]
            normed_chunks.append(self.norms[i](chunk))

        # Concatenate channels: [B, T, G*C]
        x_norm = torch.cat(normed_chunks, dim=-1)

        # Transpose to [B, G*C, T] for Conv1d
        x_bct = x_norm.transpose(1, 2)

        # Apply convolution
        y_bct = self.conv(x_bct)

        # Truncate padding
        y_bct = y_bct[..., :T]

        # Apply activation
        if self.activation:
            y_bct = self.act_fn(y_bct)

        # Transpose back to [B, T, G*C] and reshape to [B, T, G, C]
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()

        return y


class Engram(nn.Module):
    """Engram module - n-gram hash-based embedding augmentation.

    Augments hidden states with n-gram context via:
    1. N-gram hash embeddings (multi-head, multi-layer)
    2. Gated mixing based on query-key attention
    3. Short convolution for local smoothing

    Attributes:
        layer_id: Layer ID for this Engram instance
        hidden_size: Hidden dimension
        hc_mult: Hierarchical capacity multiplier
        hash_mapping: N-gram hash mapping generator
        multi_head_embedding: Multi-head embedding lookup
        short_conv: Short convolution module
        value_proj: Linear projection for value embeddings
        key_projs: Linear projections for keys (one per channel)
        norm1: RMSNorm for keys
        norm2: RMSNorm for queries
    """

    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        config: EngramConfig,
    ):
        """Initialize Engram module.

        Args:
            layer_id: Layer ID for this instance
            hidden_size: Hidden dimension (must match encoder output)
            config: EngramConfig instance
        """
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hc_mult = config.hc_mult

        if config.tokenizer is None:
            raise ValueError("EngramConfig.tokenizer must be provided")

        # Hash mapping (shared across all layers for efficiency)
        # In SEM V8.0, we'll create one instance and share it
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=config.engram_vocab_size,
            max_ngram_size=config.max_ngram_size,
            n_embed_per_ngram=config.n_embed_per_ngram,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.layer_ids,
            tokenizer=config.tokenizer,
            pad_id=config.pad_id,
            seed=config.seed,
        )

        # Multi-head embedding
        # Flatten all n-gram sizes and heads into single list
        list_of_N = [
            x
            for y in self.hash_mapping.vocab_size_across_layers[self.layer_id]
            for x in y
        ]
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=list_of_N,
            D=config.n_embed_per_ngram // config.n_head_per_ngram,
        )

        # Short convolution
        self.short_conv = ShortConv(
            hidden_size=hidden_size,
            kernel_size=config.kernel_size,
            dilation=config.max_ngram_size,
            hc_mult=config.hc_mult,
        )

        # Projection layers
        engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, hidden_size)
        self.key_projs = nn.ModuleList([
            nn.Linear(engram_hidden_size, hidden_size)
            for _ in range(config.hc_mult)
        ])

        # Normalization layers
        self.norm1 = nn.ModuleList([
            nn.RMSNorm(hidden_size)
            for _ in range(config.hc_mult)
        ])
        self.norm2 = nn.ModuleList([
            nn.RMSNorm(hidden_size)
            for _ in range(config.hc_mult)
        ])

    def forward(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        """Apply Engram augmentation to hidden states.

        Args:
            hidden_states: [B, L, D] encoder output (REAL tensor, not complex)
            input_ids: [B, L] token IDs

        Returns:
            output: [B, L, D] augmented hidden states (REAL tensor)
        """
        B, L = input_ids.shape

        # Generate n-gram hash IDs
        # hash_input_ids: [B, L, num_heads_total]
        hash_input_ids = torch.from_numpy(
            self.hash_mapping.hash(input_ids.cpu().numpy())[self.layer_id]
        ).to(hidden_states.device)

        # Lookup embeddings and flatten across heads
        # embeddings: [B, L, num_heads_total, D_per_head]
        # -> flatten to [B, L, engram_hidden_size]
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)

        # Handle case where hidden_states is 3D (no hc_mult dimension)
        if hidden_states.dim() == 3:
            # Expand to [B, L, hc_mult, D]
            hidden_states_expanded = hidden_states.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        else:
            hidden_states_expanded = hidden_states

        # Compute gated mixing for each channel
        gates = []
        for hc_idx in range(self.hc_mult):
            # Key from n-gram embeddings
            key = self.key_projs[hc_idx](embeddings)  # [B, L, D]
            normed_key = self.norm1[hc_idx](key)

            # Query from hidden states
            query = hidden_states_expanded[:, :, hc_idx, :]  # [B, L, D]
            normed_query = self.norm2[hc_idx](query)

            # Gate = sigmoid(softplus(|query · key|) * sign(query · key))
            # This provides smooth gating with sign preservation
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)  # [B, L, 1]

            gates.append(gate)

        # Stack gates: [B, L, hc_mult, 1]
        gates = torch.stack(gates, dim=2)

        # Gated value projection: value * gate
        # value: [B, L, 1, D] -> broadcast to [B, L, hc_mult, D]
        value = gates * self.value_proj(embeddings).unsqueeze(2)  # [B, L, hc_mult, D]

        # Add short convolution
        output = value + self.short_conv(value)  # [B, L, hc_mult, D]

        # Reduce hc_mult dimension by summing
        return output.sum(dim=2)  # [B, L, D]
