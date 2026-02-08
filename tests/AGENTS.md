<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-03 -->

# tests

## Purpose

Pytest test suite validating core SEM components. Tests verify mathematical correctness (unitarity, probability normalization), numerical stability (no NaN/Inf), and integration between components.

## Key Files

| File | Description |
|------|-------------|
| `test_integration.py` | End-to-end model tests: forward pass, gradient flow, generation |
| `test_cayley.py` | Cayley-Soliton propagator: unitarity preservation, CG convergence |
| `test_born.py` | Born collapse sampler: probability normalization, gradient flow |
| `test_sinkhorn.py` | Sinkhorn OT encoder: doubly-stochastic convergence |
| `test_complex_mamba.py` | Complex Mamba-3 layers: SSM correctness, spinor rotations |
| `test_unitarity.py` | Unitarity checks across full pipeline |
| `test_has_vq.py` | Vector quantization: codebook updates, EMA behavior |

## For AI Agents

### Working In This Directory

- **Run before commits**: All tests must pass
- **Add tests for new features**: Follow existing patterns
- **Test isolation**: Each test should be independent

### Testing Requirements

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_cayley.py -v

# Run with coverage
uv run pytest tests/ --cov=sem --cov-report=term-missing
```

### Common Patterns

- **Fixtures**: Use `@pytest.fixture` for model/config setup
- **Tolerance checks**: `torch.allclose(a, b, atol=1e-5, rtol=1e-4)`
- **Gradient verification**: Check `loss.backward()` doesn't raise
- **Device agnostic**: Tests should work on CPU/CUDA/XPU

### Test Categories

| Category | Tests | What They Verify |
|----------|-------|------------------|
| Unitarity | `test_unitarity.py`, `test_cayley.py` | |ψ|² ≈ 1 preserved |
| Probability | `test_born.py` | Σ p_i = 1, p_i >= 0 |
| Convergence | `test_sinkhorn.py`, `test_cayley.py` | Iterative algorithms converge |
| Gradients | `test_integration.py` | Backprop produces valid gradients |
| Numerical | All tests | No NaN/Inf in forward/backward |

## Dependencies

### Internal
- All tests import from `sem.*` package

### External
- `pytest` - Test framework
- `pytest-benchmark` - Performance benchmarks (optional)
- `torch` - Tensor operations

<!-- MANUAL: -->
