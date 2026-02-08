<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-03 -->

# encoder

## Purpose

MESH-SDR (Minimum Entropy Sparse Hyperdimensional) encoder that transforms discrete tokens into sparse complex representations on the Crystal Manifold. Uses Sinkhorn Optimal Transport for differentiable sparse activation selection.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `mesh_sdr.py` | `MESHEncoder` - main encoder class with OT-based sparse activation |
| `sinkhorn.py` | `sinkhorn_log_stabilized` - numerically stable Sinkhorn-Knopp algorithm |
| `cost_matrix.py` | Cost matrix computation for optimal transport |

## For AI Agents

### Working In This Directory

- **Core algorithm**: Sinkhorn OT produces doubly-stochastic transport plan
- **Sparsity**: Top-k selection from transport marginals
- **Complex output**: Returns `complex64` tensor on Crystal Manifold

### Key Components

```python
class MESHEncoder(nn.Module):
    """Token IDs → Sparse Complex SDR.

    Flow:
    1. Token embedding lookup (vocab_size → hidden_dim)
    2. Cost matrix computation (embedding × codebook)
    3. Sinkhorn OT → transport plan
    4. Top-k sparse activation selection
    5. Lift to complex via learned phase
    """
```

### Mathematical Foundation

- **Sinkhorn algorithm**: Iteratively normalizes rows/columns to approach doubly-stochastic
- **Log-stabilization**: Prevents numerical underflow in softmax operations
- **SDR sparsity**: Only `sdr_sparsity` (default 32) elements are active per position

### Testing Requirements

```bash
uv run pytest tests/test_sinkhorn.py -v
```

### Common Patterns

- **Epsilon tuning**: `sinkhorn_epsilon` controls entropy regularization
- **Convergence**: `sinkhorn_max_iter=50`, `sinkhorn_tol=1e-3`
- **Low VRAM**: `low_vram_mode` processes in chunks

## Dependencies

### Internal
- None (leaf module)

### External
- `torch` - Tensor operations
- `einops` - Tensor rearrangement

<!-- MANUAL: -->
