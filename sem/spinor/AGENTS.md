<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# spinor

## Purpose

Complex Mamba-3 layers implementing sequential context integration via spinor rotations. Uses State Space Models (SSM) operating in complex64 space with O(N) sequential processing.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `complex_mamba3.py` | `ComplexMamba3Layer` (V5.5) + core selective SSM building blocks |
| `spinor_block.py` | `SpinorBlock` - gated spinor rotation unit used inside Mamba |
| `complex_ops.py` | Low-level complex helpers (e.g., `complex_mul_real`) |
| `complex_pscan.py` | Parallel scan (Blelloch) implementation for complex/real-block SSM experiments |
| `lindblad.py` | `LindbladDissipation` (V8) selective forgetting term |
| `hybrid_automata.py` | `HybridAutomata` (V8) curvature monitor + Landau-Zener-style jump |
| `quaternion.py` | `QuaternionicEscape` (V8) singularity escape mechanism |

## For AI Agents

### Working In This Directory

- **SSM variant**: Mamba-style selective state space model (complex-valued)
- **Complex arithmetic**: Most paths operate in `torch.complex64`
- **Scan choices**: V5.5 uses sequential scan for stability; `complex_pscan.py` contains parallel scan utilities
- **V8 hooks**: Lindblad / HybridAutomata / Quaternionic are used via `sem.model.ComplexMamba3LayerV8`

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
uv run pytest tests/test_quick_convergence.py -v
```

### Common Patterns

- **State dimension**: `state_dim` (default 64) controls memory capacity
- **MIMO groups**: `mimo_groups` (default 8) for parallel channels
- **Block size**: Determines granularity of spinor rotations

## Dependencies

### Internal
- `sem/utils/complex_ops.py` - Shared helpers (`safe_complex`, magnitude utilities)

### External
- `torch` - Tensor operations
- `einops` - Tensor rearrangement

<!-- MANUAL:
TODO: Investigate Mamba2 official parallel scan implementation
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
