<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# sem

## Purpose

Core Python package implementing the Signal-Entropic Model V5.5 "Lean Crystal" architecture. Contains all neural network modules, training infrastructure, and utilities for building and training the SEM language model.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package version (`5.5.0`) and exports |
| `model.py` | `SEMModel` and V8 wiring (mHC/Lindblad/HybridAutomata/Quaternionic/Engram hooks) |
| `config.py` | `SEMConfig` dataclasses (incl. `V8Config`) + YAML loader |
| `train.py` | CLI training entry point using `SEMTrainer` (non-Lightning) |
| `generate.py` | Checkpoint load + sampling-based generation helper |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `encoder/` | MESH-SDR encoder: tokens → sparse complex SDR via Sinkhorn OT (see `encoder/AGENTS.md`) |
| `spinor/` | Complex Mamba-3 layers: sequential context via spinor rotations (see `spinor/AGENTS.md`) |
| `propagator/` | Cayley-Soliton propagator: unitary wave diffusion via CG solver (see `propagator/AGENTS.md`) |
| `sampler/` | Collapse sampler: complex state → vocab logits + sampling (see `sampler/AGENTS.md`) |
| `quantizer/` | Vector quantization, Fisher tracking, outlier handling (see `quantizer/AGENTS.md`) |
| `data/` | Data loading, streaming, tokenization (see `data/AGENTS.md`) |
| `training/` | Training infrastructure: Lightning module, callbacks, schedulers (see `training/AGENTS.md`) |
| `utils/` | Utilities: complex ops, layer norms, optimizers, metrics (see `utils/AGENTS.md`) |
| `engram/` | N-gram hash-based embedding augmentation (DeepSeek-style), conv-gated mixing (see `engram/AGENTS.md`) |
| `hyper_connections/` | Manifold-constrained residuals (mHC) via Sinkhorn projection (see `hyper_connections/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- **Entry point**: `SEMModel` in `model.py` is the main interface
- **Config loading**: `SEMConfig.from_yaml(path)` to load YAML configs
- **Pipeline flow**: encoder → spinor layers → propagator → final_norm → sampler
- **Complex arithmetic**: All intermediate tensors are `complex64` (or Real-Block pairs)

### Architecture Flow

```python
# Forward pass through SEM
psi = self.encoder(token_ids)           # [B, S, D] complex64
for layer in self.mamba_layers:
    psi = layer(psi)                    # Spinor rotation
psi = self.propagator(psi)              # Cayley-Soliton diffusion
psi = self.final_norm(psi)              # ComplexRMSNorm
output = self.sampler(psi)              # Collapse → logits (+ optional sampling)
```

### Testing Requirements

```bash
# Run all tests
uv run pytest tests/ -v

# Run integration test
uv run pytest tests/test_integration.py -v
```

### Common Patterns

- **Config access**: `config.model.hidden_dim`, `config.training.learning_rate`
- **Loss computation**: CE on `output['logits']` + optional unitary regularization (see `SEMModel.forward`)
- **Parameter counting**: `model.count_parameters()` returns dict by module

## Dependencies

### Internal
- All subdirectories are imported in `model.py`
- `config.py` defines dataclasses used throughout

### External
- `torch` - Tensor operations, autograd
- `einops` - Tensor rearrangement (`rearrange`, `repeat`)
- `scipy.sparse` - Sparse Laplacian construction

<!-- MANUAL: -->
