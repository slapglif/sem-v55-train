<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# data

## Purpose

Data loading and preprocessing infrastructure. Handles streaming datasets (FineWeb-Edu), tokenization, and PyTorch Lightning DataModule integration.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `streaming.py` | `FineWebEduStream` + `PackedStreamingDataset` (stream/filter/tokenize/pack) |
| `tokenizer.py` | `SEMTokenizer` - BPE tokenizer wrapper (loads/saves `tokenizer.json`) |
| `lightning_datamodule.py` | `SEMDataModule` - Lightning DataModule wrapping `PackedStreamingDataset` |

## For AI Agents

### Working In This Directory

- **Streaming**: Uses HuggingFace `datasets` with `streaming=True`
- **Tokenization**: BPE tokenizer trained on FineWeb-Edu (32k vocab)
- **Packing**: `PackedStreamingDataset` packs documents into fixed-length sequences w/ `<doc_boundary>` token
- **Curriculum**: Supports variable sequence lengths per stage

### Key Components

```python
class FineWebEduStream:
    """Stream raw documents from FineWeb-Edu with min_score filtering.

    Features:
    - HF streaming with retry logic
    - Optional local JSONL cache for fast restarts
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
(token_ids, token_freqs)
# token_ids:   Tensor[B, S]
# token_freqs: Tensor[V] (EMA unigram frequencies; may be None in low_vram_mode)
```

### Testing Requirements

```bash
# Integration-level smoke test
uv run pytest tests/test_integration.py -v
```

### Common Patterns

- **Dataset**: `HuggingFaceFW/fineweb-edu` (default)
- **Streaming**: `streaming=True` for memory efficiency
- **Workers**: `num_workers=0` is safest for HF streaming; higher values require careful worker cache handling
- **Local cache**: `FineWebEduStream(cache_dir=...)` enables JSONL reuse across restarts

## Dependencies

### Internal
- None (leaf module)

### External
- `datasets` - HuggingFace datasets
- `tokenizers` - BPE tokenization
- `torch.utils.data` - DataLoader

<!-- MANUAL: -->
