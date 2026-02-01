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
        shuffle_buffer: int = 1000,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
    ):
        self.min_score = min_score
        self.split = split
        self.shuffle_buffer = shuffle_buffer
        self.dataset_name = dataset_name

    def __iter__(self) -> Iterator[str]:
        from datasets import load_dataset
        import os

        logger.info(f"[DATA] Starting FineWeb-Edu stream from {self.dataset_name}")
        logger.info(f"[DATA] Split: {self.split}, min_score: {self.min_score}")

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            logger.info("[DATA] Using HF_TOKEN from environment")
        else:
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                logger.info(f"[DATA] Using cached HF token from {token_path}")
            else:
                logger.warning(
                    "[DATA] No HF token found — gated datasets will fail. "
                    "Set HF_TOKEN env var or run `huggingface-cli login`."
                )

        t_load = time.perf_counter()
        try:
            ds = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=True,
                token=hf_token,
            )
        except Exception as e:
            logger.error(f"[DATA] load_dataset failed: {e}")
            raise RuntimeError(
                f"Failed to load {self.dataset_name}: {e}. "
                f"Check network connectivity and HF_TOKEN."
            ) from e
        t_load_done = time.perf_counter()
        logger.info(f"[DATA] Dataset loaded in {t_load_done - t_load:.2f}s")

        # Filter by education quality score
        t_filter = time.perf_counter()
        if self.min_score > 0:
            logger.info(f"[DATA] Filtering by min_score >= {self.min_score}")
            ds = ds.filter(lambda x: x.get("score", 0) >= self.min_score)
        t_filter_done = time.perf_counter()

        # Shuffle within buffer
        t_shuffle = time.perf_counter()
        if self.shuffle_buffer > 0:
            logger.info(f"[DATA] Shuffling with buffer_size={self.shuffle_buffer}")
            ds = ds.shuffle(buffer_size=self.shuffle_buffer)
        t_shuffle_done = time.perf_counter()

        logger.info(
            f"[DATA] Pipeline setup: load={t_load_done - t_load:.3f}s "
            f"filter={t_filter_done - t_filter:.3f}s "
            f"shuffle={t_shuffle_done - t_shuffle:.3f}s"
        )

        doc_count = 0
        t_first_doc = time.perf_counter()
        logger.info("[DATA] Waiting for first document from stream...")

        for example in ds:
            text = example.get("text", "")
            if text.strip():
                doc_count += 1
                if doc_count == 1:
                    latency = time.perf_counter() - t_first_doc
                    logger.info(f"[DATA] First doc received ({latency:.1f}s)")
                if doc_count % 1000 == 0:
                    elapsed = time.perf_counter() - t_first_doc
                    logger.info(
                        f"[DATA] {doc_count} docs streamed "
                        f"({doc_count / elapsed:.0f} docs/s)"
                    )
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

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        ema_beta: float = 0.99,
        timing_enabled: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.doc_boundary_id = tokenizer.doc_boundary_id
        self.vocab_size = tokenizer.vocab_size
        self.ema_beta = ema_beta
        self.timing_enabled = timing_enabled
        self.token_freqs = torch.ones(self.vocab_size) / self.vocab_size
        self._timing_accum = {"tokenize": [], "freq_update": [], "pack_yield": []}

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

            t_tok_start = time.perf_counter()
            tokens = self.tokenizer.encode(text)
            if self.timing_enabled:
                self._timing_accum["tokenize"].append(time.perf_counter() - t_tok_start)

            if not tokens:
                continue

            t_freq_start = time.perf_counter()
            token_tensor = torch.tensor(tokens, dtype=torch.long)
            counts = torch.bincount(token_tensor, minlength=self.vocab_size).float()
            batch_freqs = counts / len(tokens)
            self.token_freqs = (
                self.ema_beta * self.token_freqs + (1 - self.ema_beta) * batch_freqs
            )
            if self.timing_enabled:
                self._timing_accum["freq_update"].append(
                    time.perf_counter() - t_freq_start
                )

            if buffer:
                buffer.append(self.doc_boundary_id)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len:
                sequences_yielded += 1
                if sequences_yielded % 100 == 0:
                    logger.info(
                        f"[PACK] Yielded {sequences_yielded} sequences from {docs_processed} docs"
                    )
                    if self.timing_enabled and self._timing_accum["tokenize"]:
                        avg_tok = (
                            sum(self._timing_accum["tokenize"])
                            / len(self._timing_accum["tokenize"])
                            * 1000
                        )
                        avg_freq = (
                            sum(self._timing_accum["freq_update"])
                            / len(self._timing_accum["freq_update"])
                            * 1000
                        )
                        logger.info(
                            f"[PACK] Avg timing: tokenize={avg_tok:.2f}ms, freq_update={avg_freq:.2f}ms"
                        )
                        self._timing_accum = {
                            "tokenize": [],
                            "freq_update": [],
                            "pack_yield": [],
                        }

                t_yield_start = time.perf_counter()
                yield (
                    torch.tensor(buffer[: self.seq_len], dtype=torch.long),
                    self.token_freqs.clone(),
                )
                if self.timing_enabled:
                    self._timing_accum["pack_yield"].append(
                        time.perf_counter() - t_yield_start
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
        timing_enabled: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.min_score = min_score
        self.shuffle_buffer = shuffle_buffer
        self.dataset_name = dataset_name
        self.timing_enabled = timing_enabled

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        stream = FineWebEduStream(
            min_score=self.min_score,
            shuffle_buffer=self.shuffle_buffer,
            dataset_name=self.dataset_name,
        )
        packer = SequencePacker(
            self.tokenizer, self.seq_len, timing_enabled=self.timing_enabled
        )
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
        t_dl = time.perf_counter()
        # Note: num_workers>0 requires the dataset to be picklable.
        # IterableDataset with HF streaming is NOT fork-safe on Windows,
        # so we cap at 0 on Windows and use the caller's value elsewhere.
        import sys

        effective_workers = num_workers if sys.platform != "win32" else 0
        dl = DataLoader(
            self,
            batch_size=batch_size,
            num_workers=effective_workers,
            pin_memory=pin_memory and batch_size > 0,
            drop_last=True,
        )
        logger.info(
            f"[DATA] DataLoader created in {time.perf_counter() - t_dl:.3f}s "
            f"(batch_size={batch_size}, workers={effective_workers}, pin_memory={pin_memory and batch_size > 0})"
        )
        return dl
