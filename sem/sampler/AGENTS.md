<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-03 -->

# sampler

## Purpose

Born Collapse sampler implementing quantum-inspired probability extraction from complex wavefunctions. Converts complex amplitudes to token probabilities via the Born rule: P(i) = |ψ_i|².

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `born_collapse.py` | `BornCollapseSampler` - wavefunction → probabilities → tokens |

## For AI Agents

### Working In This Directory

- **Born rule**: `P(token) = |⟨token|ψ⟩|² / Σ|ψ|²`
- **Normalization**: Enforced via amplitude squared sum
- **Sampling**: Top-k, top-p (nucleus), temperature scaling

### Key Components

```python
class BornCollapseSampler(nn.Module):
    """Born rule collapse: ψ → probabilities → tokens.

    Steps:
    1. Project ψ to vocabulary space (complex linear)
    2. Compute |ψ|² for each token
    3. Normalize: p_i = |ψ_i|² / Σ|ψ_j|²
    4. Apply temperature, top-k, top-p filtering
    5. Sample or return logits
    """
```

### Loss Computation

```python
# NLL loss from Born probabilities
target_amp_sq = gather(amp_sq, targets)  # P(target)
nll_loss = -log(target_amp_sq + eps).mean()

# Unitary regularization (Σ|ψ|² ≈ 1)
unitary_div = (log(amp_sq.sum(-1)))².mean()
loss = nll_loss + unitary_lambda * unitary_div
```

### Testing Requirements

```bash
uv run pytest tests/test_born.py -v
```

### Common Patterns

- **Temperature**: `temperature` scales logits before softmax
- **Top-k**: Keep only `top_k` highest probability tokens
- **Top-p**: Keep tokens until cumulative probability ≥ `top_p`
- **Chunk size**: `born_chunk_size` for low VRAM mode

## Dependencies

### Internal
- `utils/complex_ops.py` - Complex magnitude operations

### External
- `torch` - Tensor operations

<!-- MANUAL: -->
