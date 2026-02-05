"""Engram module - N-gram hash-based embedding augmentation.

Based on DeepSeek Engram architecture. Provides auxiliary context via
n-gram hash embeddings that augment the main encoder output.

Key components:
- CompressedTokenizer: Normalizes and compresses vocabulary
- NgramHashMapping: Generates multi-layer n-gram hash IDs
- MultiHeadEmbedding: Multi-head embedding lookup with offset indexing
- Engram: Main module that combines hash embeddings with hidden states
"""

from .compressed_tokenizer import CompressedTokenizer
from .hash_mapping import NgramHashMapping
from .multi_head_embedding import MultiHeadEmbedding
from .engram import Engram, EngramConfig

__all__ = [
    "CompressedTokenizer",
    "NgramHashMapping",
    "MultiHeadEmbedding",
    "Engram",
    "EngramConfig",
]
