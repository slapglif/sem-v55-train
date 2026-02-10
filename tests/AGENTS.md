<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# tests

## Purpose

Pytest test suite validating core SEM components. Tests verify numerical stability (no NaN/Inf), unitarity/energy monitoring, mHC wiring, V8 feature integration, and end-to-end forward/backward behavior.

## Key Files

| File | Description |
|------|-------------|
| `test_integration.py` | End-to-end model tests: forward pass, gradient flow, generation |
| `test_cayley.py` | Cayley-Soliton propagator: unitarity preservation, CG convergence |
| `test_born.py` | Sampler head: logits shape/sanity + sampling invariants |
| `test_sinkhorn.py` | Sinkhorn OT encoder: doubly-stochastic convergence |
| `test_complex_mamba.py` | Complex Mamba-3 layers: SSM correctness, spinor rotations |
| `test_unitarity.py` | Unitarity checks across full pipeline |
| `test_has_vq.py` | Vector quantization: codebook updates, EMA behavior |
| `test_mhc.py` | mHC (manifold-constrained hyper-connections) sinkhorn projection + residual behavior |
| `test_layer_quality.py` | Qualitative layer tests (includes a battery of component-level invariants; ~37 tests) |
| `test_adaptive_chunk_pscan.py` | Parallel scan chunking heuristics / correctness checks |
| `test_gradient_checkpointing.py` | Gradient checkpointing regression coverage |
| `test_quick_convergence.py` | Short-run convergence smoke test for major wiring regressions |

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
| Sampling/Logits | `test_born.py` | Filtering + sampling invariants; logits finite |
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
