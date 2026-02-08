<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-03 -->

# spinor

## Purpose

Complex Mamba-3 layers implementing sequential context integration via spinor rotations. Uses State Space Models (SSM) operating in complex64 space with O(N) sequential processing.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `complex_mamba3.py` | `ComplexMamba3Layer` - main SSM layer with MIMO structure |
| `spinor_block.py` | `SpinorBlock` - gated spinor rotation unit |
| `complex_ops.py` | Complex arithmetic helpers (multiply, normalize) |

## For AI Agents

### Working In This Directory

- **SSM variant**: Mamba-style selective state space model
- **Complex arithmetic**: All operations in complex64
- **Sequential scan**: O(S) processing for numerical stability
- **MIMO groups**: Parallel independent SSM channels

### Key Components

```python
class ComplexMamba3Layer(nn.Module):
    """Complex-valued Mamba layer with spinor rotations.

    Components:
    - Input projection (complex linear)
    - Depthwise convolution (complex)
    - SSM scan (A, B, C, D matrices in complex)
    - Output projection (complex linear)
    """
```

### SSM Scan

**Current implementation**: Sequential O(S) loop for numerical stability.

```python
# Sequential scan (stable but slower)
for t in range(seq_len):
    h = A * h + B[:, t] * x[:, t]
    y[:, t] = C[:, t] * h + D * x[:, t]
```

**Note**: Mamba2 provides parallel scan - investigate for speed improvements.

### Testing Requirements

```bash
uv run pytest tests/test_complex_mamba.py -v
```

### Common Patterns

- **State dimension**: `state_dim` (default 64) controls memory capacity
- **MIMO groups**: `mimo_groups` (default 8) for parallel channels
- **Block size**: Determines granularity of spinor rotations

## Dependencies

### Internal
- `utils/complex_ops.py` - Complex arithmetic

### External
- `torch` - Tensor operations
- `einops` - Tensor rearrangement

<!-- MANUAL:
TODO: Investigate Mamba2 official parallel scan implementation
-->
