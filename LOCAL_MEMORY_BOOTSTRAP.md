## Local Memory Bootstrap (Temporary)

This file stores project memory locally until Supermemory authentication is fixed.

Created: 2026-02-05
Scope: sem-v55-train repository

### User Preferences

- Communication preference for this task: detailed updates.
- No extra repo rules beyond AGENTS/docs.
- Prefer exhaustive, parallel context gathering before implementation.

### Project Configuration and Tooling

- Package/build config is Python setuptools (`pyproject.toml`).
- Primary runtime and workflow manager should be `uv` (`uv run`, `uv pip install`, `uv sync`).
- Test framework is `pytest` with test discovery under `tests/`.
- Main CI workflow currently focuses on Docker build/push to GHCR on `main`/`master` pushes.
- Branching observed: active default branch is `master`.

### Core Architecture

- Top-level model pipeline is:
  1. `MESHEncoder`
  2. `ComplexMamba3Layer` stack
  3. `CayleySolitonStack`
  4. `BornCollapseSampler`
- Composition root is `sem/model.py`.
- Config source of truth is dataclass-based `SEMConfig` in `sem/config.py`, loaded via `SEMConfig.from_yaml(...)`.

### Training Entry Points

- Standalone training path:
  - `python -m sem.train --config ...`
  - Orchestrated by `sem/training/trainer.py` (`SEMTrainer`).
- Lightning training path:
  - `python hf_train_lightning.py --config ...`
  - Uses `SEMLightningModule` + `SEMDataModule`.
- HF job launcher path:
  - `launch_job.py` builds a remote command that installs deps and runs `hf_train_lightning.py`.

### Data and Tokenization

- Dataset default is `HuggingFaceFW/fineweb-edu` with streaming enabled in data pipeline.
- Data path includes retry/backoff logic and optional local JSONL caching in `sem/data/streaming.py`.
- Token frequency EMA is tracked during packing and used downstream.

### Hardware and Numerical Patterns

- Project is complex-tensor-centric (`complex64`) with XPU-oriented real/complex fallback patterns.
- CG solver enforces autocast exclusion for numerical stability in critical paths.
- XPU path includes memory-sensitive runtime downscaling in Lightning entrypoint.
- Gradient checkpointing has historical instability notes; config defaults often keep it disabled.

### Testing and Validation Commands

- Canonical full suite: `uv run pytest tests/ -v`
- Common focused runs:
  - `uv run pytest tests/test_cayley.py -v`
  - `uv run pytest tests/test_born.py -v`
  - `uv run pytest tests/test_integration.py -v`

### Conventions and Patterns

- Config sections are stable and mirrored across YAML and dataclasses (`model`, `encoder`, `spinor`, `propagator`, `training`, `curriculum`, `distillation`, `v8`).
- Commit style in recent history is mostly Conventional Commit-like (`fix:`, `feat:`, `docs:`, `chore:`, `test:`).
- Repo has recurring SEOP fix notes documenting numerical/training stabilization history.

### Known Hotspots (from docs/history)

- NaN loss and training divergence.
- CG solver precision/convergence behavior.
- Mixed precision interactions with propagator math.
- Gradient checkpointing determinism mismatch in some paths.
- XPU low-VRAM stability and throughput tuning.

### Team/History Snapshot

- Contributor distribution in current history sample is heavily concentrated in one primary contributor.
- Recent work clusters around SEOP stability fixes, Lightning migration, and XPU support hardening.

### Local Persistence Note

- Supermemory tool calls currently fail with `401 Unauthorized` in this environment.
- When auth is restored, migrate these items into Supermemory as project-scoped memories.
