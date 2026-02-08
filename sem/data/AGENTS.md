<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-03 -->

# data

## Purpose

Data loading and preprocessing infrastructure. Handles streaming datasets (FineWeb-Edu), tokenization, and PyTorch Lightning DataModule integration.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `streaming.py` | `StreamingDataset`, `create_streaming_dataloader` - HuggingFace streaming |
| `tokenizer.py` | `SEMTokenizer` - BPE tokenizer wrapper |
| `lightning_datamodule.py` | `SEMDataModule` - Lightning DataModule for training |

## For AI Agents

### Working In This Directory

- **Streaming**: Uses HuggingFace `datasets` with `streaming=True`
- **Tokenization**: BPE tokenizer trained on FineWeb-Edu (32k vocab)
- **Batching**: Dynamic batching with sequence length padding
- **Curriculum**: Supports variable sequence lengths per stage

### Key Components

```python
class StreamingDataset(IterableDataset):
    """Streaming dataset from HuggingFace Hub.

    Features:
    - On-the-fly tokenization
    - Configurable sequence length
    - Token frequency tracking for Zipf weighting
    - Shuffle buffer for randomization
    """

class SEMDataModule(LightningDataModule):
    """Lightning DataModule for SEM training.

    Handles:
    - Train/val split
    - Distributed sampling
    - Automatic worker setup
    """
```

### Data Format

```python
# Batch structure
{
    "input_ids": Tensor[B, S],      # Token IDs
    "attention_mask": Tensor[B, S], # Padding mask (optional)
    "token_freqs": Tensor[V],       # EMA token frequencies (for Zipf weighting)
}
```

### Testing Requirements

```bash
# Test with real data (requires HF_TOKEN)
uv run pytest -k "test_real_data" -v
```

### Common Patterns

- **Dataset**: `HuggingFaceFW/fineweb-edu` (default)
- **Streaming**: `streaming=True` for memory efficiency
- **Workers**: `num_workers=0` for streaming (IterableDataset)
- **Shuffle buffer**: `shuffle_buffer_size` for randomization

## Dependencies

### Internal
- None (leaf module)

### External
- `datasets` - HuggingFace datasets
- `tokenizers` - BPE tokenization
- `torch.utils.data` - DataLoader

<!-- MANUAL: -->
