import pytorch_lightning as L
from torch.utils.data import DataLoader
from typing import Any, Optional
import logging

from .streaming import PackedStreamingDataset

logger = logging.getLogger(__name__)


class SEMDataModule(L.LightningDataModule):
    def __init__(self, config: Any, tokenizer: Any):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        seq_len = self.config.model.max_seq_length
        min_score = 2
        if self.config.curriculum.enabled:
            seq_len = self.config.curriculum.stages[0].get("seq_len", seq_len)
            min_score = self.config.curriculum.stages[0].get("min_score", min_score)

        self.dataset = PackedStreamingDataset(
            tokenizer=self.tokenizer,
            seq_len=seq_len,
            min_score=min_score,
            shuffle_buffer=self.config.training.shuffle_buffer_size,
            dataset_name=self.config.training.dataset_name,
            timing_enabled=self.config.training.timing_enabled,
        )

    def train_dataloader(self):
        return self.dataset.create_dataloader(
            batch_size=self.config.training.micro_batch_size,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            prefetch_factor=self.config.training.prefetch_factor,
        )

    def update_seq_len(self, new_seq_len: int, new_min_score: Optional[int] = None):
        if self.dataset:
            self.dataset.seq_len = new_seq_len
            if new_min_score is not None:
                self.dataset.min_score = new_min_score
            logger.info(
                f"DataModule: Updated seq_len to {new_seq_len}, min_score to {self.dataset.min_score}"
            )
