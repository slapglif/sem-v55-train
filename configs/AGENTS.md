<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-03 -->

# configs

## Purpose

YAML configuration files for different hardware targets and training scenarios. Each config defines model architecture, training hyperparameters, curriculum settings, and hardware-specific optimizations.

## Key Files

| File | Description |
|------|-------------|
| `default.yaml` | Baseline configuration for development/testing |
| `max_aggression.yaml` | Maximum resource utilization (16GB RAM + 8GB VRAM targets) |
| `a100_optimized.yaml` | NVIDIA A100 - large batch, high throughput |
| `a10g_speed.yaml` | NVIDIA A10G - balanced speed/memory |
| `l40s_full.yaml` | NVIDIA L40S - full precision, production |
| `xpu_optimized.yaml` | Intel XPU - Real-Block mode, eager execution |
| `edge.yaml` | CPU/Edge devices - reduced dimensions |
| `cloud_fast.yaml` | Quick cloud training runs |
| `cloud_a10g.yaml` | Cloud A10G specific settings |
| `test.yaml` | Minimal config for unit tests |
| `test_real_data.yaml` | Test config with real FineWeb data |
| `training.yaml` | Reference training configuration |

## For AI Agents

### Working In This Directory

- **Select config by hardware**: Match target GPU/accelerator to config file
- **Modify carefully**: Changes affect model architecture and training dynamics
- **Validate**: Run `test_xpu_local.py` after config changes

### Config Structure

```yaml
model:
  hidden_dim: 256      # Complex hidden dimension
  num_layers: 8        # Number of Mamba + Propagator layers
  vocab_size: 32768    # Tokenizer vocabulary size
  max_seq_length: 2048 # Maximum sequence length

encoder:
  sdr_sparsity: 32     # Active elements in SDR
  sinkhorn_epsilon: 0.05

spinor:
  state_dim: 64        # SSM state dimension
  mimo_groups: 8       # MIMO parallelism

propagator:
  cayley_dt: 0.1       # Cayley transform time step
  cg_max_iter: 5       # CG solver iterations
  pit_gamma: 1.0       # Soliton envelope width

training:
  batch_size: 32
  learning_rate: 3e-4
  unitary_lambda: 0.1  # Unitarity regularization

curriculum:
  enabled: true
  stages: [...]        # Progressive seq length increase
```

### Testing Requirements

```bash
# Validate config loads correctly
python -c "from sem.config import SEMConfig; SEMConfig.from_yaml('configs/default.yaml')"
```

### Common Patterns

- **Curriculum stages**: `min_score`, `seq_len`, `min_steps` per stage
- **Hardware flags**: `no_compile`, `no_amp` for XPU/CPU
- **Low VRAM**: `low_vram_mode: true`, `born_chunk_size` for memory efficiency

## Dependencies

### Internal
- Loaded by `SEMConfig.from_yaml()` in `sem/config.py`
- Used by training scripts (`hf_train.py`, `hf_train_lightning.py`)

### External
- `pyyaml` - YAML parsing

<!-- MANUAL: -->
