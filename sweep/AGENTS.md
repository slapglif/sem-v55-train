<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-08 | Updated: 2026-02-08 -->

# sweep

## Purpose

Optuna-based hyperparameter sweep and neural architecture search (NAS) utilities for SEM V5.5. These scripts run short synthetic training loops as a fast proxy for stability/convergence when tuning V8 features (mHC/Lindblad/HybridAutomata/Quaternionic) and core model hyperparameters.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Marks `sweep` as a Python package |
| `hyperparam_sweep.py` | Optuna sweep over training + V8 parameters using short-run loss as objective |
| `nas_search.py` | Optuna search over architecture (dims/layers/SSM sizes) + V8 feature toggles |
| `hp_sweep.log` | Example/working log output from a sweep run (artifact) |
| `hyperparam_real_mhc.db` | Example Optuna SQLite study DB (artifact) |

## For AI Agents

### Working In This Directory

- These scripts instantiate `SEMModel` directly and run `n_steps` of toy next-token training on random tokens.
- The objective is a proxy: final loss after N steps (optionally with metadata such as steps/sec).
- Results are stored in a SQLite Optuna study DB (see `--db`).

### Common Commands

```bash
uv run python sweep/hyperparam_sweep.py --n-trials 50 --n-steps 100 --device cpu
uv run python sweep/nas_search.py --n-trials 100 --n-steps 100 --device cpu
```

### Gotchas

- Optuna storage uses SQLite URIs like `sqlite:///sweep/hyperparam.db`.
- These runs do not represent real data quality; treat them as wiring/stability filters before expensive training.

## Dependencies

### Internal

- `sem/config.py` and `sem/model.py`

### External

- `optuna` - sweep/NAS framework
- `torch` - model + training loop
