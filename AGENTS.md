<!-- Generated: 2026-02-03 | Updated: 2026-02-03 -->

# SEM V5.5 "Lean Crystal" - Signal-Entropic Model

## Purpose

A novel language model architecture implementing **Signal-Entropic principles** with:
- **MESH-SDR Encoder**: Tokens → sparse complex SDR on Crystal Manifold via Sinkhorn OT
- **Complex Mamba-3 Layers**: Sequential context integration via spinor rotations (O(N) SSM)
- **Cayley-Soliton Propagator**: Unitary wave propagation through learned Graph Laplacian
- **Born Collapse Sampler**: Wavefunction → token probabilities via |ψ|² (quantum-inspired)

The architecture operates entirely in **complex64** space (or Real-Block Isomorphism for XPU support), maintaining unitarity constraints throughout the forward pass.

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Package definition, dependencies (torch, scipy, einops) |
| `hf_train.py` | Standalone HuggingFace training script (non-Lightning) |
| `hf_train_lightning.py` | PyTorch Lightning training with XPU/accelerator support |
| `launch_job.py` | Launch training jobs on HuggingFace Spaces |
| `push_to_hub.py` | Upload code/checkpoints to HuggingFace Hub |
| `test_xpu_local.py` | Local validation script for XPU/CPU testing |
| `Dockerfile` | Container build for cloud training |
| `uv.lock` | Locked dependencies for reproducible builds |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `sem/` | Core Python package - model, training, components (see `sem/AGENTS.md`) |
| `configs/` | YAML configuration files for different hardware profiles (see `configs/AGENTS.md`) |
| `tests/` | Pytest test suite for components (see `tests/AGENTS.md`) |
| `lean4/` | Lean 4 formal verification proofs (see `lean4/AGENTS.md`) |
| `.github/` | CI/CD workflows |
| `.sisyphus/` | Persistent planning documents (SEOP fixes, execution plans) |

## For AI Agents

### Working In This Directory

- **ALWAYS use `uv`** for package management: `uv pip install`, `uv run`, `uv sync`
- **PREFER `task` tool** for complex multi-file work
- **Training scripts**: Use `hf_train_lightning.py` for production (XPU/accelerator support)
- **Config selection**: Match config to target hardware (see `configs/`)

### Architecture Overview

```
Token IDs → [MESH Encoder] → ψ (complex) → [Mamba Layers] → ψ' → [Cayley Propagator] → ψ'' → [Born Collapse] → Logits
              Sinkhorn OT      B,S,D          SSM scan       B,S,D    CG solve + unitary    B,S,D    |ψ|²         B,S,V
```

### Testing Requirements

```bash
# Run full test suite
uv run pytest tests/ -v

# Run specific component tests
uv run pytest tests/test_cayley.py -v
uv run pytest tests/test_born.py -v
```

### Common Patterns

- **Complex tensors**: Use `torch.complex64` dtype, Real-Block for XPU
- **Config loading**: `SEMConfig.from_yaml("configs/default.yaml")`
- **Model instantiation**: `SEMModel(config)`
- **Unitary loss**: `unitary_lambda * (log(sum(|ψ|²)))²` term in loss

## Dependencies

### Internal
- `sem/` - All model components
- `configs/` - Hardware-specific configurations

### External
- `torch>=2.2.0` - Core tensor operations
- `scipy>=1.11.0` - Sparse matrix operations (Laplacian)
- `einops>=0.7.0` - Tensor rearrangement
- `lightning` - Training framework (optional)
- `datasets`, `transformers`, `tokenizers` - HuggingFace ecosystem

## Hardware Support

| Target | Config | Notes |
|--------|--------|-------|
| Intel XPU | `xpu_optimized.yaml` | Real-Block Isomorphism, eager mode |
| NVIDIA A10G | `a10g_speed.yaml` | torch.compile, mixed precision |
| NVIDIA L40S | `l40s_full.yaml` | Full precision, large batch |
| NVIDIA A100 | `a100_optimized.yaml` | Maximum throughput |
| CPU/Edge | `edge.yaml` | Reduced dimensions, no compile |

<!-- MANUAL: Task tracking and project-specific notes below -->

## Task Tracker (Historical)

### Project: Real-Block Isomorphism (XPU Support) - COMPLETED

All waves completed successfully:
- Wave 0: Environment (XPU torch installed)
- Wave 1-5: Real-Block Refactor (all primitives, layers, solvers)
- Wave 6: Training Infrastructure
- Wave 7: Post-SEOP Stabilization

**Status: READY FOR HF DEPLOYMENT**

### Recent SEOP Fixes

- **SEOP Fix 29**: XPU stability - pit_gamma soliton envelope, precision plugin, warm restarts
- CG solver implicit differentiation restored
- Sequential Mamba scan (O(S)) - numerical stability over parallel
