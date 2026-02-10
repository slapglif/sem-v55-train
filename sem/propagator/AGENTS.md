<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# propagator

## Purpose

Cayley-Soliton propagator implementing unitary wave propagation through a learned Graph Laplacian. Uses Conjugate Gradient (CG) solver for implicit Cayley transform, preserving unitarity while allowing nonlinear dynamics.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `cayley_soliton.py` | `CayleySolitonPropagator` + `CayleySolitonStack` (phase rotation + Cayley diffusion) |
| `cg_solver.py` | `cg_solve` / `cg_solve_sparse` with implicit-diff backward (real-block matvec) |
| `hamiltonian.py` | `GraphLaplacianHamiltonian` + `MultiScaleHamiltonian` (small-world sparse Laplacian pyramid) |
| `unitarity_check.py` | Unitarity verification utilities |

## For AI Agents

### Working In This Directory

- **Cayley transform**: `U = (I - iH/2)(I + iH/2)^{-1}` preserves unitarity
- **CG solver**: Iterative solve of `(I + iH/2)x = (I - iH/2)ψ`
- **Nonlinear phase step**: uses per-dim `alpha` + PIT-like `pit_gamma` phase shaping (SEOP Fix 29)
- **Warm-start**: `_psi_cache` stores last CG solution as `x0` to reduce iterations
- **Implicit differentiation**: CG gradients via implicit differentiation in `cg_solver.py`

### Key Components

```python
class CayleySolitonStack(nn.Module):
    """Stack of Cayley-Soliton propagation layers.

    Each layer:
    1. Build sparse Hamiltonian H from learned parameters
    2. Apply soliton envelope: sech((x-μ)/γ) modulation
    3. Solve (I + iH/2)ψ' = (I - iH/2)ψ via CG
    4. Optional nonlinear phase rotation
    """
```

### CG Solver Details

- **Max iterations**: `cg_max_iter=5` for training, 20 for inference
- **Tolerance**: `cg_tol=1e-6`
- **Lazy CG**: Skip iterations if residual already small
- **Direct solve**: Optional for small systems
- **Autocast**: CG explicitly disables AMP autocast to keep float32 precision

### Testing Requirements

```bash
uv run pytest tests/test_cayley.py -v
uv run pytest tests/test_unitarity.py -v
```

### Common Patterns

- **Laplacian sparsity**: `laplacian_sparsity` controls graph connectivity
- **Time step**: `cayley_dt` scales Hamiltonian strength
- **Nonlinear alpha**: Learnable per-dimension envelope strength (`CayleySolitonPropagator.alpha`)

## Dependencies

### Internal
- `sem/utils/complex_ops.py` - Safe complex construction utilities
- `sem/spinor/complex_ops.py` - Complex helpers used by phase rotation

### External
- `torch` - Tensor operations

<!-- MANUAL:
SEOP Fix 29: `pit_gamma` phase shaping + bounded envelope for XPU stability
-->

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
