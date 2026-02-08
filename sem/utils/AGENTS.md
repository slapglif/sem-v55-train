<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# utils

## Purpose

Utility modules for complex arithmetic, custom layers, optimizers, and metrics. Provides building blocks used throughout the SEM architecture.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `complex_ops.py` | Complex arithmetic: multiply, conjugate, magnitude |
| `complex_layernorm.py` | `ComplexRMSNorm` - RMS normalization for complex tensors |
| `complex_adamw.py` | `ComplexAdamW` - AdamW optimizer for complex parameters |
| `real_block_linear.py` | `RealBlockLinear` - Real-Block Isomorphism for XPU |
| `fused_complex_linear.py` | `FusedComplexLinear` - fused complex linear layer |
| `sparse_utils.py` | Sparse matrix utilities for Laplacian construction |
| `metrics.py` | Training metrics: perplexity, unitarity, gradient norms |

## For AI Agents

### Working In This Directory

- **Complex ops**: Fundamental building blocks for complex arithmetic
- **Real-Block**: Alternative representation for XPU/hardware without complex support
- **Optimizers**: Custom AdamW supporting complex parameters

### Key Components

```python
class ComplexRMSNorm(nn.Module):
    """RMS Normalization for complex tensors.

    norm = x / sqrt(mean(|x|²) + eps)
    Maintains phase while normalizing magnitude.
    """

class ComplexAdamW(Optimizer):
    """AdamW optimizer for complex parameters.

    Handles complex tensors by operating on real/imag
    components separately with proper momentum tracking.
    """

class RealBlockLinear(nn.Module):
    """Real-Block Isomorphism: C → R².

    Represents complex [a + bi] as [[a, -b], [b, a]].
    Enables complex arithmetic on hardware without
    native complex64 support (Intel XPU).
    """
```

### Real-Block Isomorphism

```
Complex:     z = a + bi
Real-Block:  [[a, -b],
              [b,  a]]

Multiplication: z₁ × z₂ via matrix multiply
Maintains: |z₁ × z₂| = |z₁| × |z₂|
```

### Testing Requirements

```bash
# Test complex operations
uv run pytest -k "complex" -v
```

### Common Patterns

- **Magnitude**: `torch.abs(z)` or `(z.real² + z.imag²).sqrt()`
- **Phase**: `torch.angle(z)`
- **Conjugate**: `torch.conj(z)` or `z.conj()`

## Dependencies

### Internal
- None (leaf module)

### External
- `torch` - Tensor operations
- `scipy.sparse` - Sparse matrix construction

<!-- MANUAL: -->
