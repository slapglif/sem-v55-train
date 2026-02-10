<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# quantizer

## Purpose

Vector quantization modules for compression and regularization. Includes codebook learning, Fisher Information tracking for importance-weighted quantization, and outlier handling for extreme values.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `vq_codebook.py` | `VQCodebook` - learnable vector quantization codebook |
| `has_vq.py` | `HasVQ` - wrapper adding VQ to any module |
| `fisher_tracker.py` | `FisherTracker` - EMA of Fisher Information for importance weighting |
| `outlier_store.py` | `OutlierStore` - handles extreme activation values |

## For AI Agents

### Working In This Directory

- **VQ**: Discrete bottleneck via nearest-neighbor lookup
- **EMA updates**: Codebook updated via exponential moving average
- **Dead code revival**: Unused codes reinitialized from active data
- **Fisher weighting**: Important dimensions quantized more carefully

### Key Components

```python
class VQCodebook(nn.Module):
    """Vector Quantization with EMA codebook updates.

    Features:
    - Nearest neighbor assignment (L2 distance)
    - EMA codebook updates (not gradient-based)
    - Dead code detection and revival
    - Commitment loss for encoder training
    """

class FisherTracker(nn.Module):
    """Track Fisher Information for importance weighting.

    Fisher ≈ E[(∂L/∂θ)²] - high Fisher = important parameter
    Used to prioritize quantization precision.
    """
```

### Testing Requirements

```bash
uv run pytest tests/test_has_vq.py -v
```

### Common Patterns

- **Codebook size**: `codebook_size` (default 256) entries
- **Group size**: `group_size` for grouped quantization
- **EMA decay**: `fisher_ema_decay` controls tracking smoothness
- **Dead threshold**: Codes unused for `dead_code_threshold` steps get revived

## Dependencies

### Internal
- None (leaf module)

### External
- `torch` - Tensor operations

<!-- MANUAL: -->

## Required MCP Tools

All agents working in this directory MUST use these MCP tools during multi-stage workflows:

### Sequential Thinking
Use `sequential-thinking` MCP for any multi-step reasoning, planning, or debugging:
- Break complex problems into explicit sequential steps
- Revise thinking when new information emerges
- Branch to explore alternative approaches
- Invoke at the START of any non-trivial task

### Context7
Use Context7 MCP tools to resolve library documentation before writing code:
- `context7_resolve-library-id` — Find the correct library identifier
- `context7_query-docs` — Query up-to-date documentation for that library

Only skip these tools when the task is trivially simple and would not benefit from structured reasoning or documentation lookup.
