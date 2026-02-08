<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-08 | Updated: 2026-02-08 -->

# hyper_connections

## Purpose

Manifold-Constrained Hyper-Connections (mHC) implementation used to stabilize residual connections by projecting a stream-mixing matrix onto the Birkhoff polytope (doubly stochastic) via Sinkhorn. In SEM V8 wiring, mHC is enabled by default with `mhc_streams=4`.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports (`sinkhorn_log*`, `MHCResidual`, `SimpleMHC`, `mhc_residual`) |
| `sinkhorn.py` | `sinkhorn_log` and `sinkhorn_log_complex` (log-domain Sinkhorn projection) |
| `mhc.py` | `MHCResidual` / `SimpleMHC` modules + `mhc_residual` functional API |

## For AI Agents

### Working In This Directory

- SEM integrates mHC in `sem/model.py` inside `ComplexMamba3LayerV8` (see `use_mhc`, `mhc_streams`, `mhc_num_iters`, `mhc_tau`).
- `SimpleMHC` partitions the channel dimension into `num_streams` and mixes across streams with a projected `H_res`.
- `sinkhorn_log_complex` projects the magnitude and re-applies a masked phase to avoid gradient explosions near zero magnitude.

### Common Commands

```bash
uv run pytest tests/test_mhc.py -v
uv run pytest tests/test_layer_quality.py -v
```

### Gotchas

- Very low `mhc_tau` produces near-permutation matrices; Sinkhorn may need more iterations to converge.
- Complex Sinkhorn masking is intentional: phase is set to 0 where magnitude is tiny (phase is undefined there).
- There is a special fast-path for `num_streams == 1` that skips Sinkhorn entirely.

## Dependencies

### Internal

- Used by `sem/model.py` (V8 residual wiring).

### External

- `torch` - tensor ops + autograd
