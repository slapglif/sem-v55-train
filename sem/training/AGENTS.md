<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# training

## Purpose

Training infrastructure including PyTorch Lightning module, custom callbacks, learning rate schedulers, curriculum learning, distillation, and health monitoring.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `lightning_module.py` | `SEMLightningModule` - main Lightning training module |
| `trainer.py` | `SEMTrainer` - standalone trainer (non-Lightning) |
| `callbacks.py` | Lightning callbacks: `SEMConsoleLogger`, `SEMCurriculumCallback`, `SEMHealthCallback`, wandb helpers |
| `scheduler.py` | `WSDScheduler` - warmup-stable-decay learning rate schedule |
| `curriculum.py` | `CurriculumManager` - progressive training stages |
| `distillation.py` | `EMATeacher` - self-distillation with EMA teacher |
| `checkpoint.py` | `CheckpointManager` - save/load + retention policy |
| `health.py` | `HealthMonitor` - training health metrics + NaN/grad checks |
| `precision_plugin.py` | `CustomMixedPrecisionPlugin` - XPU-safe mixed precision |
| `xpu_accelerator.py` | `XPUAccelerator` - Intel XPU support for Lightning |

## For AI Agents

### Working In This Directory

- **Lightning module**: `SEMLightningModule` is the main training interface
- **Callbacks**: Curriculum, health checks, gradient monitoring
- **XPU support**: Custom accelerator and precision plugin
- **Gradient checkpointing**: optional monkey-patching in `SEMLightningModule.__init__` (see config)

### Key Components

```python
class SEMLightningModule(LightningModule):
    """PyTorch Lightning module for SEM training.

    Features:
    - Automatic mixed precision
    - Gradient clipping and monitoring
    - WSD learning rate schedule
    - Curriculum learning integration
    - Health monitoring
    """

class SEMCurriculumCallback(Callback):
    """Progressive training with increasing sequence length.

    Stages:
    1. Short sequences (512), high learning rate
    2. Medium sequences (1024), reduced LR
    3. Full sequences (2048), final LR
    """
```

### Scheduler: Warmup-Stable-Decay (WSD)

```
LR
 ^
 |     /‾‾‾‾‾‾‾‾‾‾‾\
 |    /             \
 |   /               \
 |  /                 \
 +--|-------|---------|---> Steps
    warmup  stable    decay
```

### Testing Requirements

```bash
# Integration test
uv run pytest tests/test_integration.py -v
```

### Common Patterns

- **Unitary lambda**: `config.training.unitary_lambda` regularization strength
- **Curriculum stages**: `config.curriculum.stages` list of stage configs
- **Health thresholds**: Alerts on NaN, gradient explosion, loss plateau

## Dependencies

### Internal
- `../model.py` - SEMModel
- `../config.py` - SEMConfig
- `../data/` - Data loading

### External
- `pytorch-lightning` - Training framework (LightningModule, callbacks)
- `wandb` - Experiment tracking (optional)
- `torch.optim` - Optimizers

<!-- MANUAL:
SEOP Fix 29: Added precision_plugin.py and xpu_accelerator.py for Intel XPU stability
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
