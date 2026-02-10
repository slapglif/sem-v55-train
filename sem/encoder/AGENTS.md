<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# encoder

## Purpose

MESH-SDR (Minimum Entropy Sparse Hyperdimensional) encoder that transforms discrete tokens into sparse complex representations on the Crystal Manifold. Uses Sinkhorn Optimal Transport for differentiable sparse activation selection.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `mesh_sdr.py` | `MESHEncoder` - token embedding → Sinkhorn OT → sparse SDR → complex lift (plus `simple_mode`) |
| `sinkhorn.py` | `LogSinkhorn` - log-domain Sinkhorn with optional auto-ε scaling |
| `cost_matrix.py` | `LearnedCostMatrix` - learned cost / codebook projection for OT |

## For AI Agents

### Working In This Directory

- **Core algorithm**: Sinkhorn OT produces transport plan `T` over SDR candidates
- **Sparsity modes**: hard top-k (sparse) vs `soft_sparse=True` (gradient-preserving)
- **Simple mode**: `simple_mode=True` bypasses OT/SDR and returns complex embedding for weight tying
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
- **Weight tying**: prefer `simple_mode=True` when debugging output-projection / embedding coupling

## Dependencies

### Internal
- None (leaf module)

### External
- `torch` - Tensor operations
- `einops` - Tensor rearrangement

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
