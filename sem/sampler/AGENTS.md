<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# sampler

## Purpose

Collapse/sampling head converting complex hidden states into vocabulary logits and sampled tokens. Current implementation uses a log-linear projection over real/imag parts (SEOP Fix 48) rather than a quadratic Born rule.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `born_collapse.py` | `BornCollapseSampler` - wavefunction → probabilities → tokens |

## For AI Agents

### Working In This Directory

- **Log-linear head**: `logits = W_r·Re(psi) + W_i·Im(psi) + b`
- **Sampling**: Temperature scaling, top-k filtering, top-p (nucleus) filtering
- **Weight tying**: `SEMModel` ties `proj_real.weight` to `encoder.embedding.weight`

### Key Components

```python
class BornCollapseSampler(nn.Module):
    """Log-linear collapse: ψ → logits → probabilities → tokens.

    Steps:
    1. Project Re/Im(ψ) to logits
    2. Apply temperature, top-k, top-p filtering
    3. Softmax + sample (or return logits/log_probs)
    """
```

### Testing Requirements

```bash
uv run pytest tests/test_born.py -v
```

### Common Patterns

- **Temperature**: `temperature` scales logits before softmax
- **Top-k**: Keep only `top_k` highest probability tokens
- **Top-p**: Keep tokens until cumulative probability ≥ `top_p`
- **Training loss**: computed in `sem/model.py` via `torch.nn.functional.cross_entropy`

## Dependencies

### Internal
- `sem/model.py` - Computes CE loss from logits + optional unitary regularization

### External
- `torch` - Tensor operations

<!-- MANUAL: -->
