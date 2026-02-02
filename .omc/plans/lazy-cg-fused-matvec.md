# Plan: Residual-Gated Lazy CG + Fused Real-Block Matvec

## Context

### Original Request
Optimize the CG solver in the Cayley-Soliton propagator with two techniques:
1. **Residual-Gated Lazy CG** — skip CG when cached solution is still valid (saves ~60% forward FLOPs)
2. **Fused Real-Block Matvec** — batch real/imag parts into single sparse_mm call (saves kernel launches)

### Research Findings
- System A_minus = I + i·dt/2·H is near-identity (κ ≈ 1.5)
- Between training steps, H changes by O(3e-4), so cached solutions are usually still valid
- 8-layer stack × 3-5 CG iterations = 24-40 matvecs per forward; lazy gate costs 1 matvec per layer
- `matvec_real()` is called twice per complex matvec (real + imag); fusing halves sparse_mm launches
- Gradient trick `x = x_star + (b - A·x_star)` already exists in `cg_solve_sparse` — lazy CG reuses the same mechanism with 0 iterations

### Current Code Structure
- `CayleySolitonPropagator.forward()` (cayley_soliton.py:69-154): Already has `_psi_cache` for warm-start
- `cg_solve_sparse()` (cg_solver.py:142-173): torch.no_grad CG + gradient trick
- `GraphLaplacianHamiltonian.matvec_real()` (hamiltonian.py:154-203): Builds sparse COO per call, does `sparse.mm(A, v.t()).t()`
- `MultiScaleHamiltonian.matvec_real()` (hamiltonian.py:286-294): Calls child `matvec_real` per scale
- `PropagatorConfig` (config.py:37-42): `cayley_dt`, `cg_max_iter`, `cg_tol`, etc.
- `model.py:66-74`: Constructs `CayleySolitonStack` from `PropagatorConfig`

---

## Work Objectives

### Core Objective
Reduce forward-pass FLOP cost of the Cayley-Soliton propagator by ~50% through residual-gated CG skipping and fused sparse matvec, with zero impact on gradient correctness or unitarity.

### Deliverables
1. Config fields `lazy_cg` and `lazy_cg_tol` in `PropagatorConfig`
2. Residual gate + skip logic in `CayleySolitonPropagator.forward()`
3. Per-layer skip rate tracking (accessible for health monitoring)
4. `matvec_real_fused(vr, vi)` method on both `GraphLaplacianHamiltonian` and `MultiScaleHamiltonian`
5. All existing matvec callsites updated to use fused path
6. Health monitor extended with `cg_skip_rate` metric

### Definition of Done
- [ ] `lazy_cg=False` reproduces bit-identical results to current code
- [ ] `lazy_cg=True` skips CG when residual < tol (verified via skip counter)
- [ ] Gradient flows to all H parameters in both skip and solve paths
- [ ] Fused matvec produces identical output to separate real/imag calls
- [ ] Health monitor logs `cg_skip_rate` per health check
- [ ] No new dependencies; all real-block math (no complex64 in solver)

---

## Guardrails

### Must Have
- Configurable: `lazy_cg: bool` toggle (default True)
- Gradient correctness: `∂x/∂θ_H` flows through A_minus matvec in skip path
- Unitarity preserved: Cayley transform structure unchanged
- Skip rate tracking: EMA counter per layer, aggregated for logging
- Fused matvec: identical numerics to current separate calls

### Must NOT Have
- New Python dependencies
- Changes to CG algorithm itself (only gating around it)
- Complex64 tensors in the solver hot path
- Changes to the backward pass of `cg_solve_sparse`
- Breaking changes to `CayleySolitonStack` public API

---

## Task Flow

```
Wave 1 (Foundation - parallel):
  T1: Config + Fused Matvec     ──┐
  T2: Lazy CG gate logic         ──┤
                                    │
Wave 2 (Integration):              │
  T3: Wire everything + health  ◄──┘
```

---

## Detailed TODOs

### Wave 1: Foundation (parallel tasks, no dependencies between them)

#### T1: Config Fields + Fused Real-Block Matvec
**File:** `sem/config.py`, `sem/propagator/hamiltonian.py`
**Agent:** executor-high
**Acceptance:** Fused method returns identical output to two separate `matvec_real` calls

1. **sem/config.py** — Add to `PropagatorConfig`:
   ```python
   lazy_cg: bool = True
   lazy_cg_tol: float = 1e-6  # Residual gate tolerance
   ```

2. **sem/propagator/hamiltonian.py** — Add `GraphLaplacianHamiltonian.matvec_real_fused(vr, vi)`:
   - Stack `v_flat = torch.cat([vr_flat, vi_flat], dim=0)` along batch dim (double the batch)
   - Single `torch.sparse.mm(A_sparse, v_flat.t()).t()` call
   - Split result back: `Avr_flat, Avi_flat = result.split(N, dim=0)` where N = original batch size
   - Apply `degree * v - Av` for each part separately
   - Return `(Hvr, Hvi)` tuple

3. **sem/propagator/hamiltonian.py** — Add `MultiScaleHamiltonian.matvec_real_fused(vr, vi)`:
   - Loop over scales, call `scale.matvec_real_fused(vr, vi)`
   - Weight and sum both parts with cached scale weights
   - Return `(Hvr, Hvi)` tuple

#### T2: Residual-Gated Lazy CG Logic
**File:** `sem/propagator/cayley_soliton.py`, `sem/propagator/cg_solver.py`
**Agent:** executor-high
**Acceptance:** With lazy_cg=True, CG is skipped when relative residual < tol; gradient trick still applied

1. **sem/propagator/cayley_soliton.py** — Modify `CayleySolitonPropagator.__init__()`:
   - Add `lazy_cg: bool = True` and `lazy_cg_tol: float = 1e-6` parameters
   - Store as `self.lazy_cg`, `self.lazy_cg_tol`
   - Add skip tracking: `self._cg_skip_count = 0`, `self._cg_total_count = 0`

2. **sem/propagator/cayley_soliton.py** — Modify `CayleySolitonPropagator.forward()`:
   - After computing `rhs_real_block` and before CG solve, add residual gate:
   ```python
   skip_cg = False
   if self.lazy_cg and x0 is not None:
       # Gate cost: 1 matvec (vs 3-5 for full CG)
       Ax0 = a_minus_matvec_wrapped(x0)
       residual = Ax0 - rhs_real_block
       rel_residual = residual.norm() / (rhs_real_block.norm() + 1e-12)
       skip_cg = rel_residual.item() < self.lazy_cg_tol
   
   self._cg_total_count += 1
   if skip_cg:
       self._cg_skip_count += 1
       # Use cached solution with gradient trick (same as cg_solve_sparse)
       # x_star = x0 (detached cached solution)
       # x = x_star + (b - A·x_star) gives correct gradients through A and b
       if torch.is_grad_enabled() and rhs_real_block.requires_grad:
           Ax = a_minus_matvec_wrapped(x0)  # reuse Ax0 from gate
           psi_out_real_block = x0 + (rhs_real_block - Ax)
       else:
           psi_out_real_block = x0
   else:
       psi_out_real_block = cg_solve_sparse(
           a_minus_matvec_wrapped, rhs_real_block, self.cg_max_iter, self.cg_tol, x0=x0
       )
   ```
   - **Important**: When `skip_cg=True` and grad enabled, reuse `Ax0` from the gate check (already computed) to avoid a redundant matvec. The variable `Ax0` from the gate and `Ax` in the gradient trick are the same computation — assign once, use twice.

3. **sem/propagator/cayley_soliton.py** — Add skip rate property:
   ```python
   @property
   def cg_skip_rate(self) -> float:
       if self._cg_total_count == 0:
           return 0.0
       return self._cg_skip_count / self._cg_total_count
   
   def reset_cg_stats(self):
       self._cg_skip_count = 0
       self._cg_total_count = 0
   ```

4. **sem/propagator/cayley_soliton.py** — Update `CayleySolitonStack.__init__()`:
   - Add `lazy_cg` and `lazy_cg_tol` params, pass to each layer

5. **sem/propagator/cayley_soliton.py** — Add `CayleySolitonStack` aggregate properties:
   ```python
   @property
   def cg_skip_rate(self) -> float:
       rates = [l.cg_skip_rate for l in self.layers]
       return sum(rates) / len(rates) if rates else 0.0
   
   def reset_cg_stats(self):
       for layer in self.layers:
           layer.reset_cg_stats()
   ```

---

### Wave 2: Integration (depends on T1 + T2)

#### T3: Wire Config → Model, Use Fused Matvec, Health Monitoring
**File:** `sem/model.py`, `sem/propagator/cayley_soliton.py`, `sem/training/health.py`
**Agent:** executor
**Acceptance:** Config flows end-to-end; fused matvec used in forward; health monitor tracks skip rate

1. **sem/model.py** — Thread new config fields to `CayleySolitonStack`:
   - Add `lazy_cg=c.propagator.lazy_cg, lazy_cg_tol=c.propagator.lazy_cg_tol` to constructor call

2. **sem/propagator/cayley_soliton.py** — Replace separate matvec calls with fused:
   - In `a_plus_matvec` and `a_minus_matvec`, replace:
     ```python
     # OLD: two separate calls
     Hvr = self.hamiltonian.matvec_real(vr)
     Hvi = self.hamiltonian.matvec_real(vi)
     ```
     with:
     ```python
     # NEW: single fused call
     Hvr, Hvi = self.hamiltonian.matvec_real_fused(vr, vi)
     ```

3. **sem/training/health.py** — Add CG skip rate tracking:
   - Add `cg_skip_rate: float = 0.0` field to `HealthReport`
   - In `HealthMonitor.check()`, after propagator check:
     ```python
     if hasattr(model, 'propagator') and hasattr(model.propagator, 'cg_skip_rate'):
         report.cg_skip_rate = model.propagator.cg_skip_rate
     ```
   - Add `"health/cg_skip_rate": report.cg_skip_rate` to `get_metrics_dict()`
   - Add `cg_skip_rate` to `state_dict()` serialization

4. **sem/training/health.py** — Add skip rate warning:
   - If `cg_skip_rate < 0.3` after 1000+ steps, emit warning: "Low CG skip rate — lazy_cg may not be helping"
   - If `cg_skip_rate > 0.99`, emit info: "Very high CG skip rate — consider lowering lazy_cg_tol for accuracy"

---

## Commit Strategy

| Wave | Commit Message |
|------|---------------|
| 1 | `feat(propagator): add fused real-block matvec and lazy CG residual gate` |
| 2 | `feat(propagator): wire lazy CG config, use fused matvec, add skip rate monitoring` |

Single squash commit also acceptable:
`feat(propagator): residual-gated lazy CG + fused real-block matvec optimization`

---

## Success Criteria

1. **Correctness**: `lazy_cg=False` is bit-identical to current behavior
2. **Skip Rate**: >50% CG skips after warmup (first ~100 steps will be lower)
3. **Speed**: Measurable wall-clock improvement (expect ~30-50% faster forward pass)
4. **Gradient Flow**: `torch.autograd.gradcheck` or manual verification that H parameters receive gradients in skip path
5. **Monitoring**: `health/cg_skip_rate` appears in training logs
6. **No Regression**: Unitarity deviation unchanged; loss curve shape preserved
