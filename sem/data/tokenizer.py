"""Tokenizer wrapper for SEM V5.5 training.

Provides a BPE tokenizer targeting vocab_size=32768 for use with
the MESH-SDR encoder (sem/encoder/mesh_sdr.py).
"""
from pathlib import Path
from typing import List, Optional
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing


class SEMTokenizer:
    """BPE tokenizer wrapper for SEM V5.5.

    Special tokens:
        0: <pad>
        1: <eos>
        2: <bos>
        3: <unk>
        4: <doc_boundary>
    """

    SPECIAL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>", "<doc_boundary>"]

    def __init__(self, tokenizer_path: str):
        """Load a pre-trained tokenizer from disk."""
        self.tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
        self._ensure_special_tokens()
        self._setup_ids()

    def _ensure_special_tokens(self):
        """Add special tokens if they don't exist in the tokenizer."""
        from tokenizers import AddedToken

        existing_vocab = self.tokenizer.get_vocab()
        tokens_to_add = []

        for token in self.SPECIAL_TOKENS:
            if token not in existing_vocab:
                tokens_to_add.append(AddedToken(token, special=True))

        if tokens_to_add:
            self.tokenizer.add_special_tokens(tokens_to_add)

    def _setup_ids(self):
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
        self.doc_boundary_id = self.tokenizer.token_to_id("<doc_boundary>")

        # Validate all special tokens have valid IDs
        for name in ["pad_id", "eos_id", "bos_id", "unk_id", "doc_boundary_id"]:
            if getattr(self, name) is None:
                raise ValueError(
                    f"Special token {name} has no ID. "
                    f"Tokenizer vocab_size={self.tokenizer.get_vocab_size()}"
                )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [enc.ids for enc in self.tokenizer.encode_batch(texts)]

    @classmethod
    def train(cls, output_path: str, texts_iterator, vocab_size: int = 32768) -> "SEMTokenizer":
        """Train a new BPE tokenizer from a text iterator.

        Args:
            output_path: Directory to save tokenizer files
            texts_iterator: Iterator yielding text strings
            vocab_size: Target vocabulary size
        Returns:
            Trained SEMTokenizer instance
        """
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=cls.SPECIAL_TOKENS,
            show_progress=True,
            min_frequency=2,
        )

        tokenizer.train_from_iterator(texts_iterator, trainer=trainer)

        # Save
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(out / "tokenizer.json"))

        return cls(output_path)
