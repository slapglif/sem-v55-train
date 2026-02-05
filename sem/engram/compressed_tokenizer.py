"""Compressed Tokenizer - Vocabulary normalization and compression.

Maps original tokenizer vocabulary to a smaller normalized vocabulary by
applying Unicode normalization (NFKC, NFD, accents stripping, lowercasing)
and whitespace normalization.

This reduces vocabulary size and improves n-gram hash collision rates
for semantically similar tokens.
"""

from typing import Any, Optional
import numpy as np
from tokenizers import normalizers, Regex


class CompressedTokenizer:
    """Compresses tokenizer vocabulary via normalization.

    Attributes:
        tokenizer: Original tokenizer instance
        normalizer: Tokenizers normalizer pipeline
        lookup_table: np.ndarray[int64] mapping old token ID → new compressed ID
        num_new_token: Number of unique normalized tokens
    """

    def __init__(self, tokenizer: Any):
        """Initialize compressed tokenizer.

        Args:
            tokenizer: Tokenizer instance with decode() and convert_ids_to_tokens() methods
        """
        self.tokenizer = tokenizer

        # Normalization pipeline:
        # 1. NFKC (compatibility decomposition + composition)
        # 2. NFD (canonical decomposition)
        # 3. StripAccents (remove diacritics)
        # 4. Lowercase
        # 5. Normalize whitespace (collapse runs of [ \t\r\n]+ to single space)
        # 6. Preserve single space (use sentinel to avoid stripping it)
        # 7. Strip leading/trailing whitespace
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),  # Protect single space
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),  # Restore protected space
        ])

        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self) -> int:
        """Return number of unique normalized tokens."""
        return self.num_new_token

    def _build_lookup_table(self) -> tuple[np.ndarray, int]:
        """Build mapping from original token IDs to compressed IDs.

        Returns:
            lookup_table: np.ndarray[vocab_size] of int64
            num_new_tokens: int
        """
        old2new: dict[int, int] = {}
        key2new: dict[str, int] = {}
        new_tokens: list[str] = []

        vocab_size = len(self.tokenizer)

        for tid in range(vocab_size):
            # Decode token to text
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            # If decode produces replacement character, use raw token
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                # Normalize text
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            # Lookup or create new compressed ID
            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)

            old2new[tid] = nid

        # Build lookup table
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def _compress(self, input_ids: np.ndarray) -> np.ndarray:
        """Compress token IDs using lookup table.

        Args:
            input_ids: np.ndarray of token IDs

        Returns:
            compressed_ids: np.ndarray of compressed token IDs
        """
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        """Compress token IDs.

        Args:
            input_ids: Token IDs to compress

        Returns:
            Compressed token IDs
        """
        return self._compress(input_ids)
