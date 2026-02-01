"""Streaming data pipeline for FineWeb-Edu.

Provides an IterableDataset that streams from HuggingFace,
filters by education quality score, tokenizes on-the-fly,
and packs sequences with document boundary tokens.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Iterator, Tuple
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


class FineWebEduStream:
    """Streams documents from FineWeb-Edu with quality filtering.

    Args:
        min_score: Minimum education quality score (0-5)
        split: Dataset split
        shuffle_buffer: Number of documents to buffer for shuffling
        dataset_name: HuggingFace dataset name
    """

    def __init__(
        self,
        min_score: int = 2,
        split: str = "train",
        shuffle_buffer: int = 10000,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
    ):
        self.min_score = min_score
        self.split = split
        self.shuffle_buffer = shuffle_buffer
        self.dataset_name = dataset_name

    def __iter__(self) -> Iterator[str]:
        """Yield document texts that meet the quality threshold."""
        from datasets import load_dataset

        logger.info(f"[DATA] Starting FineWeb-Edu stream from {self.dataset_name}")
        logger.info(f"[DATA] Split: {self.split}, min_score: {self.min_score}")
        start_time = time.time()

        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
        )
        logger.info(f"[DATA] Dataset loaded in {time.time() - start_time:.2f}s")

        # Filter by education quality score
        if self.min_score > 0:
            logger.info(f"[DATA] Filtering by min_score >= {self.min_score}")
            ds = ds.filter(lambda x: x.get("score", 0) >= self.min_score)

        # Shuffle within buffer
        if self.shuffle_buffer > 0:
            logger.info(f"[DATA] Shuffling with buffer_size={self.shuffle_buffer}")
            ds = ds.shuffle(buffer_size=self.shuffle_buffer)

        doc_count = 0
        logger.info("[DATA] Streaming documents...")

        for example in ds:
            text = example.get("text", "")
            if text.strip():
                doc_count += 1
                if doc_count % 1000 == 0:
                    logger.info(f"[DATA] Streamed {doc_count} documents...")
                yield text


class SequencePacker:
    """Packs tokenized documents into fixed-length sequences.

    Concatenates documents with <doc_boundary> separator tokens,
    yielding sequences of exactly seq_len tokens.

    Args:
        tokenizer: SEMTokenizer instance
        seq_len: Target sequence length
        ema_beta: EMA decay factor for token frequencies
    """

    def __init__(self, tokenizer, seq_len: int, ema_beta: float = 0.99):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.doc_boundary_id = tokenizer.doc_boundary_id
        self.vocab_size = tokenizer.vocab_size
        self.ema_beta = ema_beta
        # Initialize frequencies as uniform distribution
        self.token_freqs = torch.ones(self.vocab_size) / self.vocab_size

    def pack(
        self, text_iterator: Iterator[str]
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Pack texts into fixed-length token sequences.

        Args:
            text_iterator: Iterator yielding document texts
        Yields:
            tuple: (torch.LongTensor of shape [seq_len], torch.Tensor of shape [vocab_size])
        """
        buffer = []
        docs_processed = 0
        sequences_yielded = 0

        logger.info(f"[PACK] Starting sequence packing (seq_len={self.seq_len})...")

        for text in text_iterator:
            docs_processed += 1

            # Tokenize document
            tokens = self.tokenizer.encode(text)
            if not tokens:
                continue

            # Update EMA of token frequencies (SEOP: frequency-aware encoding)
            # Use bincount for efficient frequency tracking
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            counts = torch.bincount(token_tensor, minlength=self.vocab_size).float()
            batch_freqs = counts / len(tokens)
            self.token_freqs = (
                self.ema_beta * self.token_freqs + (1 - self.ema_beta) * batch_freqs
            )

            # Append doc boundary + tokens to buffer
            if buffer:  # Add boundary between documents
                buffer.append(self.doc_boundary_id)
            buffer.extend(tokens)

            # Yield complete sequences from buffer
            while len(buffer) >= self.seq_len:
                sequences_yielded += 1
                if sequences_yielded % 100 == 0:
                    logger.info(
                        f"[PACK] Yielded {sequences_yielded} sequences from {docs_processed} docs"
                    )
                yield (
                    torch.tensor(buffer[: self.seq_len], dtype=torch.long),
                    self.token_freqs.clone(),
                )
                buffer = buffer[self.seq_len :]


class PackedStreamingDataset(IterableDataset):
    """PyTorch IterableDataset wrapping FineWeb-Edu streaming with packing.

    Args:
        tokenizer: SEMTokenizer instance
        seq_len: Sequence length for packing
        min_score: Minimum FineWeb-Edu quality score
        shuffle_buffer: Shuffle buffer size
        dataset_name: HuggingFace dataset name
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        min_score: int = 2,
        shuffle_buffer: int = 10000,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_score = min_score
        self.shuffle_buffer = shuffle_buffer
        self.dataset_name = dataset_name

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        stream = FineWebEduStream(
            min_score=self.min_score,
            shuffle_buffer=self.shuffle_buffer,
            dataset_name=self.dataset_name,
        )
        packer = SequencePacker(self.tokenizer, self.seq_len)
        yield from packer.pack(iter(stream))

    def create_dataloader(
        self, batch_size: int, num_workers: int = 4, pin_memory: bool = True
    ) -> DataLoader:
        """Create a DataLoader for this dataset.

        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Pin memory for CUDA
        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=True,
        )
