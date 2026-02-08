<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-03 -->

# propagator

## Purpose

Cayley-Soliton propagator implementing unitary wave propagation through a learned Graph Laplacian. Uses Conjugate Gradient (CG) solver for implicit Cayley transform, preserving unitarity while allowing nonlinear dynamics.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `cayley_soliton.py` | `CayleySolitonStack` - main propagator with soliton envelope |
| `cg_solver.py` | `cg_solve_complex` - CG solver for complex linear systems |
| `hamiltonian.py` | `SparseHamiltonian` - learned sparse Graph Laplacian |
| `unitarity_check.py` | Unitarity verification utilities |

## For AI Agents

### Working In This Directory

- **Cayley transform**: `U = (I - iH/2)(I + iH/2)^{-1}` preserves unitarity
- **CG solver**: Iterative solve of `(I + iH/2)x = (I - iH/2)ψ`
- **Soliton envelope**: `pit_gamma` controls sech-envelope width (SEOP Fix 29)
- **Implicit differentiation**: CG gradients via implicit function theorem

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

### Testing Requirements

```bash
uv run pytest tests/test_cayley.py -v
uv run pytest tests/test_unitarity.py -v
```

### Common Patterns

- **Laplacian sparsity**: `laplacian_sparsity` controls graph connectivity
- **Time step**: `cayley_dt` scales Hamiltonian strength
- **Nonlinear alpha**: Phase-dependent rotation strength

## Dependencies

### Internal
- `utils/sparse_utils.py` - Sparse matrix operations

### External
- `torch` - Tensor operations
- `scipy.sparse` - Sparse Laplacian construction

<!-- MANUAL:
SEOP Fix 29: Added pit_gamma soliton envelope for XPU stability
-->
