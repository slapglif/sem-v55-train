"""N-gram Hash Mapping - Multi-layer n-gram hash ID generation.

Generates hash IDs for n-grams (bigrams, trigrams, etc.) using layer-specific
random multipliers and modulo arithmetic. Each layer uses different multipliers
to create layer-specific hash spaces, and each n-gram size has multiple heads
with different vocabulary sizes (prime numbers for collision resistance).
"""

from typing import Any, Optional
import numpy as np
from sympy import isprime

from .compressed_tokenizer import CompressedTokenizer


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Find next prime number after start that hasn't been used.

    Args:
        start: Starting value
        seen_primes: Set of already-used primes

    Returns:
        Next unused prime number
    """
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    """N-gram hash mapping generator.

    For each layer and n-gram size, generates hash IDs using:
    - Layer-specific random multipliers (for different hash spaces per layer)
    - Multi-head hashing (multiple hash functions per n-gram size)
    - Prime moduli (for collision resistance)

    Attributes:
        vocab_size_per_ngram: List of base vocab sizes for each n-gram size
        max_ngram_size: Maximum n-gram size (e.g., 3 for bigrams + trigrams)
        n_embed_per_ngram: Embedding dimension per n-gram
        n_head_per_ngram: Number of hash heads per n-gram size
        pad_id: Padding token ID (compressed)
        layer_ids: List of layer IDs that use Engram
        compressed_tokenizer: CompressedTokenizer instance
        tokenizer_vocab_size: Size of compressed vocabulary
        layer_multipliers: Dict[layer_id -> np.ndarray[max_ngram_size] of multipliers]
        vocab_size_across_layers: Dict[layer_id -> List[List[int]]] of prime vocab sizes
    """

    def __init__(
        self,
        engram_vocab_size: list[int],
        max_ngram_size: int,
        n_embed_per_ngram: int,
        n_head_per_ngram: int,
        layer_ids: list[int],
        tokenizer: Any,
        pad_id: Optional[int] = None,
        seed: int = 0,
    ):
        """Initialize n-gram hash mapping.

        Args:
            engram_vocab_size: List of base vocab sizes for each n-gram (len = max_ngram_size - 1)
            max_ngram_size: Maximum n-gram size (e.g., 3 for bigrams + trigrams)
            n_embed_per_ngram: Embedding dimension per n-gram
            n_head_per_ngram: Number of hash heads per n-gram size
            layer_ids: Layer IDs that use Engram
            tokenizer: Tokenizer instance
            pad_id: Padding token ID (in original vocab), will be compressed
            seed: Random seed for layer multipliers
        """
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.layer_ids = layer_ids

        # Build compressed tokenizer
        self.compressed_tokenizer = CompressedTokenizer(tokenizer=tokenizer)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)

        # Compress pad_id
        if pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[pad_id])
        else:
            self.pad_id = 0  # Default to 0 if not specified

        # Generate layer-specific multipliers
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers: dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            # Ensure odd multipliers for better hash distribution
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        # Calculate prime vocab sizes for each layer
        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self) -> dict[int, list[list[int]]]:
        """Calculate prime vocabulary sizes for each layer and n-gram size.

        For each layer and n-gram size, finds n_head_per_ngram distinct primes
        >= base vocab size to use as moduli for hash functions.

        Returns:
            Dict[layer_id -> List[List[int]]]
            Outer list: n-gram sizes (bigram, trigram, ...)
            Inner list: prime vocab sizes for each head
        """
        seen_primes: set[int] = set()
        vocab_size_across_layers: dict[int, list[list[int]]] = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes: list[list[int]] = []

            # For each n-gram size (2, 3, ..., max_ngram_size)
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes: list[int] = []
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram

                # Find num_head distinct primes >= vocab_size
                current_prime_search_start = vocab_size - 1
                for _ in range(num_head):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)

            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        """Generate n-gram hash IDs for a specific layer.

        Args:
            input_ids: [B, T] compressed token IDs
            layer_id: Layer ID

        Returns:
            hash_ids: [B, T, num_heads_total] where num_heads_total = sum over n-gram sizes
        """
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        multipliers = self.layer_multipliers[layer_id]

        # Precompute shifted versions for all n-gram positions
        def shift_k(k: int) -> np.ndarray:
            """Shift input_ids by k positions, padding with pad_id."""
            if k == 0:
                return x
            shifted = np.pad(x, ((0, 0), (k, 0)), mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes: list[np.ndarray] = []

        # For each n-gram size (bigram, trigram, ...)
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2

            # Get n tokens (current, prev, prev-prev, ...)
            tokens = base_shifts[:n]

            # Hash via XOR of (token * multiplier)
            # mix = t0*m0 XOR t1*m1 XOR t2*m2 ...
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])

            # Generate hash IDs for each head using different prime moduli
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        # Stack all heads: [B, T, num_heads_total]
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids: np.ndarray) -> dict[int, np.ndarray]:
        """Generate n-gram hash IDs for all layers.

        Args:
            input_ids: [B, T] token IDs (original vocabulary)

        Returns:
            Dict[layer_id -> np.ndarray[B, T, num_heads_total]]
        """
        # Compress token IDs first
        input_ids = self.compressed_tokenizer(input_ids)

        hash_ids_for_all_layers: dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)

        return hash_ids_for_all_layers
