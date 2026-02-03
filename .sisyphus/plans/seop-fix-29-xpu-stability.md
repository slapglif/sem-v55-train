# SEOP Fix 29: XPU Numerical Stability & Information Density

## TL;DR

> **Quick Summary**: Fix numerical instability (unitarity explosion 2.66 → 1.71e+09) in HF Training Job #6 by implementing Phase Information Transfer (PIT) in the Cayley propagator, hardening Lightning precision control to prevent bf16 poisoning of the CG solver, and integrating IPEX for XPU performance.
>
> **Deliverables**:
> - Full PIT transform (phase density equalization + envelope correction) in `cayley_soliton.py`
> - Custom Lightning precision plugin excluding Cayley propagator from autocast
> - Fix `trainer.py` bf16 auto-enable bug that ignores config `dtype: float32`
> - IPEX `ipex.optimize()` integration in Lightning training path
> - Updated `max_aggression.yaml` with precision control
> - Validated dry run on CUDA (L40S target)
>
> **Estimated Effort**: Medium (6 tasks, ~3-4 hours)
> **Parallel Execution**: YES - 2 waves + 1 sequential validation wave
> **Critical Path**: Task 1 (PIT) + Task 3 (Precision Plugin) → Task 5 (Config) → Task 6 (Dry Run)

---

## Context

### Original Request
Fix numerical instability in SEM training on L40S (HF Job #6) caused by bf16 precision poisoning the CG solver and missing phase density normalization in the Cayley propagator. Implement PIT transform, harden precision, integrate IPEX.

### Interview Summary
**Key Decisions**:
- PIT Transform: Full PIT (phase density equalization + soliton envelope correction)
- Precision Strategy: Custom Lightning MixedPrecision plugin (exclude Cayley from autocast)
- IPEX: Minimal integration (`ipex.optimize()` wrapper only)

**Research Findings**:
- SEOP Fix 27 (float32 exclusion zone) is already implemented in `cg_solver.py` and `cayley_soliton.py`
- SEOP Fix 28 (block-Jacobi preconditioner) is already implemented in `cayley_soliton.py:149-160`
- IPEX is imported in `sem/train.py:17-20` but `ipex.optimize()` is NEVER called
- **ROOT CAUSE BUG**: `sem/training/trainer.py:92-99` auto-enables bf16 AMP on CUDA even when config says `dtype: float32`
- `launch_job.py` does NOT pass `--precision` flag → defaults to "32-true" in Lightning path, but the non-Lightning `trainer.py` path ignores this and auto-detects bf16
- Phase rotation in `cayley_soliton.py:104-111` normalizes intensity by mean but has NO phase density correction

### Metis Review
**Identified Gaps** (addressed):
- `trainer.py` bf16 auto-enable ignores config dtype → Fixed in Task 4
- Lightning has no `exclude_modules` API for MixedPrecision → Custom plugin in Task 3
- IPEX imported but never used → Task 4 adds `ipex.optimize()`
- No phase distribution normalization after Kerr rotation → PIT in Task 1

---

## Work Objectives

### Core Objective
Eliminate numerical instability in the CG solver path by (1) adding Phase Information Transfer to maximize entropy density in the Cayley propagator, (2) making precision control bulletproof against bf16 autocast leaking into float32-critical paths, and (3) integrating IPEX for XPU performance.

### Concrete Deliverables
- Modified `sem/propagator/cayley_soliton.py` with full PIT transform
- New `sem/training/precision_plugin.py` custom Lightning precision plugin
- Modified `sem/training/trainer.py` respecting config dtype + IPEX optimize
- Modified `hf_train_lightning.py` using custom precision plugin
- Updated `configs/max_aggression.yaml` with precision control section
- Passing dry run: unitarity < 2.0, CG residual < 1e-4, no NaN

### Definition of Done
- [ ] Unitarity divergence stays < 2.0 for at least 50 training steps
- [ ] CG residual < 1e-4 at convergence
- [ ] SDR sparsity > 0.3 (not collapsed)
- [ ] No NaN/Inf in loss or gradients
- [ ] `ipex.optimize()` called when IPEX available + XPU device

### Must Have
- Float32 precision guaranteed for CG solver regardless of trainer-level AMP settings
- PIT phase density equalization after every Kerr rotation
- Soliton envelope correction preserving |ψ| profile
- Config-driven precision control (no bf16 auto-detection overriding config)

### Must NOT Have (Guardrails)
- Do NOT change CG solver algorithm (implicit differentiation must remain)
- Do NOT modify the Hamiltonian sparse matvec (SEOP Fix 23/24 are stable)
- Do NOT change `cg_max_iter` or `cg_tol` (50 iterations at 1e-5 is correct)
- Do NOT introduce new dependencies beyond `intel_extension_for_pytorch` (already imported)
- Do NOT add complex64 arithmetic (Real-Block Isomorphism must remain)
- Do NOT modify the block-Jacobi preconditioner (SEOP Fix 28 is correct)
- Do NOT break the non-Lightning training path (`sem/train.py` / `hf_train.py`)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES — `test_xpu_local.py` exists for validation
- **User wants tests**: Manual-only (dry run validation, not TDD)
- **Framework**: Bash commands + Python script execution

### Automated Verification (Agent-Executable)

Each TODO includes executable verification. Primary method:
```bash
# Quick stability check (10 steps, synthetic data)
.venv/Scripts/python test_xpu_local.py
```

For full validation:
```bash
# Lightning path with explicit precision
python hf_train_lightning.py --config configs/max_aggression.yaml --precision 32-true --no-compile
```

**Evidence Requirements**:
- Terminal output showing loss values (no NaN)
- Unitarity divergence metric (< 2.0)
- CG residual reported (< 1e-4)

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately) — Core Math + Infrastructure:
├── Task 1: PIT Transform in Cayley Propagator [no dependencies]
├── Task 2: Verify & Harden CG Float32 Path [no dependencies]
├── Task 3: Custom Lightning Precision Plugin [no dependencies]
└── Task 4: IPEX Integration + Trainer bf16 Bug Fix [no dependencies]

Wave 2 (After Wave 1) — Config + Integration:
└── Task 5: Config Updates & Integration Wiring [depends: 1, 2, 3, 4]

Wave 3 (After Wave 2) — Validation:
└── Task 6: XPU/CUDA Dry Run Validation [depends: 5]

Critical Path: Task 1 + Task 3 → Task 5 → Task 6
Parallel Speedup: ~60% faster than sequential (4 tasks in Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 5 | 2, 3, 4 |
| 2 | None | 5 | 1, 3, 4 |
| 3 | None | 5 | 1, 2, 4 |
| 4 | None | 5 | 1, 2, 3 |
| 5 | 1, 2, 3, 4 | 6 | None |
| 6 | 5 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Dispatch |
|------|-------|---------------------|
| 1 | 1, 2, 3, 4 | 4 parallel agents: `run_in_background=true` for each |
| 2 | 5 | 1 sequential agent after Wave 1 completes |
| 3 | 6 | 1 sequential agent after Task 5 completes |

---

## TODOs

- [ ] 1. Implement Full PIT Transform in Cayley Propagator

  **What to do**:
  
  Add Phase Information Transfer (PIT) after the Kerr nonlinear rotation in `CayleySolitonPropagator.forward()`. This consists of two sub-steps:
  
  **1a. Phase Density Equalization** (after line 111 in `cayley_soliton.py`):
  - Compute the phase of each component: `theta = atan2(psi_rot_i, psi_rot_r)` 
  - Compute phase density: soft-histogram of theta values over [-π, π]
  - Compute equalization correction: `correction = uniform_density / (phase_density + eps)`
  - Apply correction to phase: `theta_eq = theta + small_correction_factor * (correction - 1.0)`
  - Reconstruct: `psi_eq_r = amplitude * cos(theta_eq)`, `psi_eq_i = amplitude * sin(theta_eq)`
  
  **Implementation approach**: Use differentiable soft-histogram via kernel density estimation (KDE) with Gaussian kernels. This keeps gradients flowing. The correction strength should be controlled by a learnable parameter `pit_strength` (init=0.01) so the model can learn how much equalization to apply.
  
  ```python
  # Pseudocode for phase density equalization
  amplitude = torch.sqrt(psi_rot_r**2 + psi_rot_i**2 + 1e-12)
  theta = torch.atan2(psi_rot_i, psi_rot_r)  # [-pi, pi]
  
  # Soft histogram via KDE (differentiable)
  num_bins = 32
  bin_centers = torch.linspace(-math.pi, math.pi, num_bins, device=theta.device)
  # sigma controls smoothness of density estimate
  sigma = 2 * math.pi / num_bins
  # [B, S, D, 1] vs [num_bins] → [B, S, D, num_bins]
  diffs = theta.unsqueeze(-1) - bin_centers
  weights = torch.exp(-0.5 * (diffs / sigma)**2)
  density = weights.sum(dim=-2, keepdim=True)  # [B, S, 1, num_bins] per-sequence density
  density = density / (density.sum(dim=-1, keepdim=True) + 1e-8)  # normalize to PDF
  
  # Equalization: correction per dimension based on its bin
  uniform = 1.0 / num_bins
  # Interpolate correction for each theta value
  correction = uniform / (density_at_theta + 1e-8)
  correction = torch.clamp(correction, 0.5, 2.0)  # Prevent extreme corrections
  
  # Apply soft correction
  theta_eq = theta + self.pit_strength * (correction - 1.0) * sigma
  ```
  
  **1b. Soliton Envelope Correction** (after phase equalization):
  - The amplitude should be preserved through the PIT step
  - Add a soft normalization: `amplitude_corrected = amplitude * (target_rms / (rms + eps))`
  - Where `target_rms = rms(psi_input)` (preserve the input RMS through the nonlinear step)
  - This prevents the soliton from gaining/losing energy through the phase manipulation
  
  ```python
  # Envelope correction: preserve RMS through nonlinear step
  input_rms = torch.sqrt((psi_r**2 + psi_i**2).mean(dim=-1, keepdim=True) + 1e-12)
  output_rms = torch.sqrt((amplitude**2).mean(dim=-1, keepdim=True) + 1e-12)
  envelope_scale = input_rms / (output_rms + 1e-12)
  amplitude_corrected = amplitude * envelope_scale
  
  # Reconstruct from corrected phase + amplitude
  psi_rot_r = amplitude_corrected * torch.cos(theta_eq)
  psi_rot_i = amplitude_corrected * torch.sin(theta_eq)
  ```
  
  **New __init__ parameters** (add to `CayleySolitonPropagator.__init__`):
  - `self.pit_strength = nn.Parameter(torch.tensor(0.01))` — learnable PIT correction strength
  - `self.pit_num_bins = 32` — number of KDE bins (non-learnable)
  - `self.pit_enabled = True` — flag to disable PIT if needed
  
  **Must NOT do**:
  - Do NOT use non-differentiable histogram (must use KDE for gradient flow)
  - Do NOT clamp theta_eq to [-π, π] hard (use soft wrapping if needed)
  - Do NOT modify the CG diffusion step (Step 2) — PIT only applies to Step 1
  - Do NOT change the existing intensity normalization (line 106) — PIT is ADDITIONAL

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
    - Reason: Numerically sensitive math with differentiable KDE, phase geometry, and gradient flow requirements
  - **Skills**: [`git-master`]
    - `git-master`: Clean atomic commit of the PIT implementation
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: No UI component
    - `playwright`: No browser testing

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Task 5 (config integration)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `sem/propagator/cayley_soliton.py:82-233` — Full `forward()` method. The PIT code goes AFTER line 111 (after `complex_mul_real` rotation) and BEFORE line 113 (Step 2: Cayley diffusion). The new code must produce `psi_rot_r, psi_rot_i` that replace the existing ones.
  - `sem/propagator/cayley_soliton.py:41-68` — `__init__` method where new `pit_strength` parameter must be added alongside `self.alpha`
  - `sem/propagator/cayley_soliton.py:104-111` — Existing Kerr rotation to understand the input/output contract: takes `(psi_r, psi_i)` → produces `(psi_rot_r, psi_rot_i)`

  **API/Type References**:
  - `sem/spinor/complex_ops.py:complex_mul_real` — The function used for Kerr rotation. Its signature: `complex_mul_real(ar, ai, br, bi) -> (cr, ci)`. PIT output must be compatible.
  - `sem/propagator/cayley_soliton.py:248-280` — `CayleySolitonStack.__init__` must also accept `pit_strength`, `pit_num_bins`, `pit_enabled` and pass them to each `CayleySolitonPropagator`

  **Documentation References**:
  - Phase density equalization is analogous to histogram equalization in image processing, but applied to complex phase distributions on the unit circle
  - The KDE approach ensures differentiability: `density(θ) = (1/N) Σ K((θ - θ_i)/σ)` where K is Gaussian kernel

  **Acceptance Criteria**:

  ```bash
  # Agent runs quick validation:
  .venv/Scripts/python -c "
  import torch, math
  from sem.propagator.cayley_soliton import CayleySolitonPropagator
  prop = CayleySolitonPropagator(dim=64, cg_max_iter=10, cg_tol=1e-5)
  psi = torch.randn(2, 16, 64, dtype=torch.complex64)
  psi_out = prop(psi)
  # Check norm preservation (unitarity)
  norm_in = psi.abs().pow(2).sum(dim=-1).mean()
  norm_out = psi_out.abs().pow(2).sum(dim=-1).mean()
  ratio = (norm_out / norm_in).item()
  print(f'Norm ratio: {ratio:.6f}')
  assert 0.8 < ratio < 1.2, f'Unitarity violated: {ratio}'
  # Check PIT parameter exists
  assert hasattr(prop, 'pit_strength'), 'pit_strength parameter missing'
  print('PIT transform: PASS')
  "
  # Assert: Output contains "PIT transform: PASS"
  # Assert: Norm ratio between 0.8 and 1.2
  ```

  **Evidence to Capture**:
  - [ ] Terminal output showing norm ratio and PASS message
  - [ ] `pit_strength` parameter confirmed in model

  **Commit**: YES
  - Message: `feat(propagator): implement Phase Information Transfer (PIT) in Cayley nonlinear step`
  - Files: `sem/propagator/cayley_soliton.py`
  - Pre-commit: quick validation script above

---

- [ ] 2. Verify & Harden CG Float32 Exclusion Path

  **What to do**:
  
  The CG solver already has SEOP Fix 27 (`torch.autocast(enabled=False)`), but we need to harden it against edge cases where Lightning's precision plugin might re-enter autocast.
  
  **2a. Add explicit dtype assertions** in `_cg_solve_impl()` (after line 85 in `cg_solver.py`):
  ```python
  # After b = b.float() / b.to(torch.complex64) casting
  assert b.dtype in (torch.float32, torch.complex64), \
      f"CG solver requires float32 precision, got {b.dtype}"
  ```
  
  **2b. Add runtime dtype check at CG entry** in `_cg_solve()` (after line 72):
  - Before entering the autocast-disabled context, log a warning if autocast was active
  - This helps diagnose if Lightning is wrapping the CG solver in autocast
  ```python
  if torch.is_autocast_enabled():
      import logging
      logging.getLogger(__name__).warning(
          "CG solver entered with autocast ENABLED — disabling for float32 precision"
      )
  ```
  
  **2c. Harden the Cayley forward() autocast context** in `cayley_soliton.py`:
  - Replace the manual `__enter__`/`__exit__` pattern (lines 94-95, 232) with a proper `with` statement
  - The current pattern is fragile — if an exception occurs between `__enter__` and `__exit__`, the context won't be cleaned up
  ```python
  # BEFORE (fragile):
  _autocast_ctx = torch.autocast(device_type=_device_type, enabled=False)
  _autocast_ctx.__enter__()
  ... # 140 lines of code
  _autocast_ctx.__exit__(None, None, None)
  return psi_out
  
  # AFTER (robust):
  # Wrap the entire body in a helper or use try/finally
  ```
  - Use `try/finally` to ensure cleanup, OR refactor the inner logic into `_forward_impl()` and wrap with `with` statement
  
  **Must NOT do**:
  - Do NOT change the CG algorithm (implicit differentiation remains)
  - Do NOT add overhead to the hot path (assertions only in debug mode, use `__debug__` guard)
  - Do NOT change `cg_max_iter` or `cg_tol` values

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small targeted changes (assertions + context manager refactor), clear scope
  - **Skills**: [`git-master`]
    - `git-master`: Atomic commit
  - **Skills Evaluated but Omitted**:
    - `ultrabrain`: Not needed — straightforward defensive coding

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `sem/propagator/cg_solver.py:54-73` — `_cg_solve()` wrapper with autocast disable. The warning log goes after line 71 (`_device_type = b.device.type`), before line 72 (`with torch.autocast...`)
  - `sem/propagator/cg_solver.py:76-165` — `_cg_solve_impl()` where dtype assertion goes after line 85 (`b = b.float()...`)
  - `sem/propagator/cg_solver.py:173-214` — `cg_solve_sparse()` also has autocast disable (line 199), same hardening needed
  - `sem/propagator/cayley_soliton.py:91-95,232` — The fragile `__enter__`/`__exit__` pattern to refactor

  **Acceptance Criteria**:

  ```bash
  # Agent verifies assertions don't break normal operation:
  .venv/Scripts/python -c "
  import torch
  from sem.propagator.cg_solver import cg_solve_sparse
  
  def dummy_matvec(v):
      return v * 2.0  # Simple diagonal system
  
  b = torch.randn(4, 32, 2)  # [B, D, 2] real-block format
  x = cg_solve_sparse(dummy_matvec, b, max_iter=10, tol=1e-5)
  assert x.dtype == b.dtype, f'dtype mismatch: {x.dtype} vs {b.dtype}'
  print(f'CG solve dtype: {x.dtype} — PASS')
  
  # Test with autocast active (simulating Lightning bf16)
  with torch.autocast('cpu', dtype=torch.bfloat16):
      x2 = cg_solve_sparse(dummy_matvec, b, max_iter=10, tol=1e-5)
      assert x2.dtype == b.dtype, f'CG leaked bf16: {x2.dtype}'
      print(f'CG under autocast: {x2.dtype} — PASS')
  "
  # Assert: Both PASS messages appear
  ```

  **Commit**: YES (groups with Task 1)
  - Message: `fix(cg_solver): harden float32 exclusion with assertions and robust context management`
  - Files: `sem/propagator/cg_solver.py`, `sem/propagator/cayley_soliton.py`
  - Pre-commit: validation script above

---

- [ ] 3. Create Custom Lightning Precision Plugin

  **What to do**:
  
  Create `sem/training/precision_plugin.py` — a custom Lightning `MixedPrecision` plugin that applies bf16 autocast to everything EXCEPT the Cayley propagator stack.
  
  **3a. Create the plugin class**:
  ```python
  # sem/training/precision_plugin.py
  import torch
  from pytorch_lightning.plugins.precision import MixedPrecision
  from contextlib import contextmanager
  
  class SEMPrecisionPlugin(MixedPrecision):
      """Mixed precision plugin that forces float32 for numerically-critical paths.
      
      The CG solver and Cayley propagator require float32 precision.
      bf16 has ~3 decimal digits — CG needs ~7 to converge.
      
      This plugin works as belt-and-suspenders with the module-level
      torch.autocast(enabled=False) already in cg_solver.py and cayley_soliton.py.
      """
      
      def __init__(self, precision: str = "bf16-mixed", device: str = "cuda"):
          super().__init__(precision, device)
      
      # Override the forward context to disable autocast for specific modules
      # Lightning calls this around training_step
  ```
  
  **Key implementation detail**: Lightning's `MixedPrecision` plugin wraps `training_step` with `torch.autocast`. The existing `torch.autocast(enabled=False)` in the Cayley propagator SHOULD nest correctly (inner disable overrides outer enable). The plugin's main value is:
  1. Making the precision intent explicit and documented
  2. Logging when autocast is active to aid debugging
  3. Ensuring the `precision` arg in config is respected (not auto-detected)
  
  **3b. Integrate into `hf_train_lightning.py`**:
  - Import the plugin
  - If `args.precision` is `"bf16-mixed"`, use `SEMPrecisionPlugin` instead of default
  - Add `plugins=[precision_plugin]` to `L.Trainer()`
  
  **Must NOT do**:
  - Do NOT disable bf16 globally (we WANT bf16 for encoder, mamba, final_norm)
  - Do NOT modify Lightning internals or monkey-patch
  - Do NOT break `--precision 32-true` mode (should still work)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: New file creation with clear API contract, straightforward Lightning plugin
  - **Skills**: [`git-master`]
    - `git-master`: Atomic commit for new file
  - **Skills Evaluated but Omitted**:
    - `ultrabrain`: Lightning plugin API is well-documented, not complex

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `hf_train_lightning.py:44-48` — Precision argument definition. The plugin replaces `precision=args.precision` on line 106
  - `hf_train_lightning.py:102-115` — `L.Trainer()` constructor where `plugins=` parameter must be added
  - `sem/training/lightning_module.py:62-97` — `training_step()` that the plugin wraps with autocast context

  **API/Type References**:
  - PyTorch Lightning `MixedPrecision` plugin: subclass of `PrecisionPlugin`, override `forward_context()` and `optimizer_step()`
  - `torch.autocast` context manager: `torch.autocast(device_type, dtype, enabled)`

  **External References**:
  - Lightning precision docs: https://lightning.ai/docs/pytorch/stable/extensions/precision.html

  **Acceptance Criteria**:

  ```bash
  # Agent verifies plugin imports and initializes:
  .venv/Scripts/python -c "
  from sem.training.precision_plugin import SEMPrecisionPlugin
  plugin = SEMPrecisionPlugin(precision='bf16-mixed', device='cuda')
  print(f'Plugin created: {type(plugin).__name__}')
  print(f'Precision: {plugin.precision}')
  print('Precision plugin: PASS')
  "
  # Assert: Output contains "Precision plugin: PASS"
  ```

  **Commit**: YES
  - Message: `feat(training): add custom Lightning precision plugin excluding Cayley from autocast`
  - Files: `sem/training/precision_plugin.py`, `hf_train_lightning.py`
  - Pre-commit: validation script above

---

- [ ] 4. IPEX Integration + Trainer bf16 Bug Fix

  **What to do**:
  
  **4a. Fix the bf16 auto-enable bug in `sem/training/trainer.py`**:
  - Lines 92-99 auto-enable bf16 AMP on CUDA even when config says `dtype: float32`
  - Fix: Check `config.training.dtype` BEFORE auto-detecting bf16 support
  ```python
  # BEFORE (bug):
  elif torch.cuda.is_bf16_supported():
      self._use_amp = True
      self._amp_dtype = torch.bfloat16
  
  # AFTER (fixed):
  elif config.training.dtype != "float32" and torch.cuda.is_bf16_supported():
      self._use_amp = True
      self._amp_dtype = torch.bfloat16
  else:
      self._use_amp = False
      logger.info(f"AMP disabled: config dtype={config.training.dtype}")
  ```
  
  **4b. Add `ipex.optimize()` to Lightning training path** in `hf_train_lightning.py`:
  - After model and optimizer creation, before `trainer.fit()`
  - Only when IPEX is available AND device is XPU
  ```python
  # In hf_train_lightning.py, after model creation (line 57):
  try:
      import intel_extension_for_pytorch as ipex
      if accelerator == "xpu":
          model, optimizer = ipex.optimize(
              model, 
              optimizer=model.configure_optimizers()["optimizer"],
              dtype=torch.float32  # Respect config dtype
          )
          logger.info("IPEX optimize applied for XPU")
  except ImportError:
      pass
  ```
  
  **Important caveat**: `ipex.optimize()` must be called BEFORE `trainer.fit()` and the optimized model must be passed to the trainer. However, Lightning manages optimizer creation internally via `configure_optimizers()`. The correct approach is:
  - For Lightning path: Use `ipex.optimize()` on the model only (no optimizer), OR
  - Override `configure_optimizers()` in `SEMLightningModule` to apply IPEX after optimizer creation
  
  **Recommended approach**: Add IPEX model optimization in `SEMLightningModule.__init__()`:
  ```python
  # In sem/training/lightning_module.py __init__:
  try:
      import intel_extension_for_pytorch as ipex
      if hasattr(torch, "xpu") and torch.xpu.is_available():
          self.model = ipex.optimize(self.model, dtype=torch.float32)
          logger.info("IPEX model optimization applied")
  except ImportError:
      pass
  ```
  
  **Must NOT do**:
  - Do NOT break CUDA training (IPEX only for XPU)
  - Do NOT change dtype to bf16 in IPEX optimize (must be float32 for CG stability)
  - Do NOT modify the non-Lightning trainer path (`sem/train.py` already has IPEX import)
  - Do NOT add IPEX as a hard dependency (graceful fallback)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Bug fix + minimal integration wrapper, clear scope
  - **Skills**: [`git-master`]
    - `git-master`: Atomic commit
  - **Skills Evaluated but Omitted**:
    - `ultrabrain`: Straightforward bug fix and API call

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Task 5
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `sem/training/trainer.py:70-104` — AMP configuration logic. The bf16 bug is at lines 92-99 where `torch.cuda.is_bf16_supported()` overrides config dtype
  - `sem/train.py:17-20` — Existing IPEX import pattern (graceful fallback)
  - `sem/train.py:39-67` — `_configure_cpu_for_intel()` showing current IPEX environment setup
  - `sem/training/lightning_module.py:16-20` — `__init__` where IPEX model optimization should go

  **API/Type References**:
  - `ipex.optimize(model, optimizer=None, dtype=torch.float32)` → returns optimized (model, optimizer) or just model
  - `config.training.dtype` — string field in `SEMConfig`, currently "float32"

  **Acceptance Criteria**:

  ```bash
  # Agent verifies bf16 bug fix:
  .venv/Scripts/python -c "
  from sem.config import SEMConfig
  config = SEMConfig.from_yaml('configs/max_aggression.yaml')
  print(f'Config dtype: {config.training.dtype}')
  assert config.training.dtype == 'float32', 'Config should specify float32'
  print('Config dtype check: PASS')
  "
  # Assert: Output contains "Config dtype check: PASS"
  
  # Agent verifies IPEX import doesn't crash:
  .venv/Scripts/python -c "
  try:
      import intel_extension_for_pytorch as ipex
      print(f'IPEX available: {ipex.__version__}')
  except ImportError:
      print('IPEX not installed (expected on non-XPU systems) — PASS')
  "
  # Assert: Either shows version or shows "PASS"
  ```

  **Commit**: YES
  - Message: `fix(trainer): respect config dtype for AMP + add IPEX optimize for XPU`
  - Files: `sem/training/trainer.py`, `sem/training/lightning_module.py`
  - Pre-commit: validation scripts above

---

- [ ] 5. Config Updates & Integration Wiring

  **What to do**:
  
  **5a. Update `configs/max_aggression.yaml`** with new PIT and precision sections:
  ```yaml
  propagator:
    cayley_dt: 0.01
    cg_max_iter: 50
    cg_tol: 1.0e-5
    nonlinear_alpha: 0.01
    laplacian_sparsity: 8
    # SEOP Fix 29: Phase Information Transfer
    pit_enabled: true
    pit_num_bins: 32
    pit_strength_init: 0.01
  
  training:
    # ... existing fields ...
    dtype: float32
    # SEOP Fix 29: Precision control
    precision_mode: "selective-bf16"  # "32-true" | "bf16-mixed" | "selective-bf16"
  ```
  
  **5b. Update `sem/config.py`** to parse new fields:
  - Add `pit_enabled`, `pit_num_bins`, `pit_strength_init` to propagator config
  - Add `precision_mode` to training config
  - Ensure backward compatibility (defaults if fields not present)
  
  **5c. Wire PIT config into `CayleySolitonPropagator` construction**:
  - In `sem/model.py` where `CayleySolitonStack` is created, pass PIT params from config
  - Both `CayleySolitonStack.__init__` and `CayleySolitonPropagator.__init__` must accept the new params
  
  **5d. Wire precision plugin into `hf_train_lightning.py`**:
  - If `config.training.precision_mode == "selective-bf16"`, use `SEMPrecisionPlugin`
  - If `"32-true"`, use default Lightning precision
  - If `"bf16-mixed"`, use default Lightning bf16 (user explicitly wants full bf16)
  
  **5e. Update `launch_job.py`** to pass precision mode from config:
  - Currently does not pass `--precision` flag
  - Add: read config, extract precision_mode, pass as `--precision` arg
  
  **Must NOT do**:
  - Do NOT break backward compatibility with existing configs (default values for all new fields)
  - Do NOT change the `CayleySolitonStack` constructor signature incompatibly

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Config wiring is straightforward integration, connecting already-implemented pieces
  - **Skills**: [`git-master`]
    - `git-master`: Atomic commit spanning multiple files
  - **Skills Evaluated but Omitted**:
    - `ultrabrain`: No complex logic, just plumbing

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Wave 1)
  - **Blocks**: Task 6
  - **Blocked By**: Tasks 1, 2, 3, 4

  **References**:

  **Pattern References**:
  - `configs/max_aggression.yaml:25-30` — Existing propagator config section to extend
  - `configs/max_aggression.yaml:44-53` — Existing training config section to extend
  - `sem/config.py` — Config parser. Find the `propagator` and `training` dataclass/dict definitions. New fields go here.
  - `sem/model.py` — Where `CayleySolitonStack` is constructed. Find the line and add PIT params from config.
  - `hf_train_lightning.py:102-115` — Trainer constructor to wire precision plugin
  - `launch_job.py:21-49` — Launch script to update with precision flag

  **API/Type References**:
  - `sem/propagator/cayley_soliton.py:41-68` — `CayleySolitonPropagator.__init__` signature (Task 1 adds PIT params here)
  - `sem/propagator/cayley_soliton.py:255-280` — `CayleySolitonStack.__init__` signature (must pass through PIT params)

  **Acceptance Criteria**:

  ```bash
  # Agent verifies config parsing:
  .venv/Scripts/python -c "
  from sem.config import SEMConfig
  config = SEMConfig.from_yaml('configs/max_aggression.yaml')
  
  # Check propagator PIT config
  assert hasattr(config.propagator, 'pit_enabled'), 'pit_enabled missing from config'
  assert config.propagator.pit_enabled == True, 'pit_enabled should be True'
  print(f'PIT config: enabled={config.propagator.pit_enabled}, bins={config.propagator.pit_num_bins}')
  
  # Check precision config
  assert hasattr(config.training, 'precision_mode'), 'precision_mode missing'
  print(f'Precision mode: {config.training.precision_mode}')
  
  print('Config integration: PASS')
  "
  # Assert: Output contains "Config integration: PASS"
  
  # Agent verifies model construction with PIT:
  .venv/Scripts/python -c "
  from sem.config import SEMConfig
  from sem.model import SEMModel
  config = SEMConfig.from_yaml('configs/max_aggression.yaml')
  model = SEMModel(config)
  # Check PIT is wired into propagator
  prop_layer = model.propagator.layers[0]
  assert hasattr(prop_layer, 'pit_strength'), 'PIT not wired into propagator'
  print(f'Propagator PIT strength: {prop_layer.pit_strength.item():.4f}')
  print('Model construction with PIT: PASS')
  "
  # Assert: Output contains "Model construction with PIT: PASS"
  ```

  **Commit**: YES
  - Message: `chore(config): wire PIT transform and precision control into config and model construction`
  - Files: `configs/max_aggression.yaml`, `sem/config.py`, `sem/model.py`, `hf_train_lightning.py`, `launch_job.py`
  - Pre-commit: validation scripts above

---

- [ ] 6. XPU/CUDA Dry Run Validation

  **What to do**:
  
  **6a. Update `test_xpu_local.py`** to validate all SEOP Fix 29 changes:
  - Test PIT transform (phase density should be more uniform after PIT)
  - Test CG float32 enforcement (no bf16 leakage)
  - Test full forward-backward pass (no NaN/Inf)
  - Test unitarity (divergence < 2.0)
  - Test 10 training steps with loss decreasing (no explosion)
  
  **6b. Run the dry run**:
  ```bash
  # Local test (CPU or XPU if available):
  .venv/Scripts/python test_xpu_local.py
  
  # Lightning path test (if CUDA available):
  python hf_train_lightning.py --config configs/max_aggression.yaml --precision 32-true --no-compile
  # Kill after 10 steps, verify metrics
  ```
  
  **6c. Capture evidence**:
  - Loss values for steps 0-10 (should decrease, no NaN)
  - Unitarity divergence metric (should stay < 2.0)
  - CG residual (should be < 1e-4)
  - SDR sparsity (should be > 0.3)
  
  **Must NOT do**:
  - Do NOT run a full training job (just 10-step validation)
  - Do NOT modify any source code in this task (validation only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Script execution and metric validation, no code changes
  - **Skills**: [`git-master`]
    - `git-master`: Commit updated test script
  - **Skills Evaluated but Omitted**:
    - `playwright`: No browser testing needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential after Wave 2)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `test_xpu_local.py` — Existing validation script. Extend it with SEOP Fix 29 checks.
  - `sem/training/lightning_module.py:62-97` — `training_step()` that the dry run exercises
  - `sem/propagator/cayley_soliton.py:226-227` — Unitarity check in eval mode

  **Acceptance Criteria**:

  ```bash
  # Agent runs full validation:
  .venv/Scripts/python test_xpu_local.py 2>&1 | tee .sisyphus/evidence/task-6-dryrun.log
  
  # Assert from log:
  # 1. No "NaN" or "Inf" in output
  # 2. Loss values present and decreasing
  # 3. "PASS" or "SUCCESS" in final output
  # 4. Unitarity divergence < 2.0 if reported
  ```

  ```bash
  # Verify no NaN in output:
  .venv/Scripts/python -c "
  with open('.sisyphus/evidence/task-6-dryrun.log') as f:
      content = f.read()
  assert 'nan' not in content.lower() or 'no nan' in content.lower(), 'NaN detected in dry run!'
  print('Dry run NaN check: PASS')
  "
  ```

  **Evidence to Capture**:
  - [ ] Full dry run log saved to `.sisyphus/evidence/task-6-dryrun.log`
  - [ ] Loss trajectory (steps 0-10)
  - [ ] Unitarity metric
  - [ ] CG residual

  **Commit**: YES
  - Message: `test: validate SEOP Fix 29 (PIT + precision hardening + IPEX)`
  - Files: `test_xpu_local.py`
  - Pre-commit: dry run must pass

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(propagator): implement Phase Information Transfer (PIT) in Cayley nonlinear step` | `sem/propagator/cayley_soliton.py` | norm preservation test |
| 2 | `fix(cg_solver): harden float32 exclusion with assertions and robust context management` | `sem/propagator/cg_solver.py`, `sem/propagator/cayley_soliton.py` | dtype test under autocast |
| 3 | `feat(training): add custom Lightning precision plugin excluding Cayley from autocast` | `sem/training/precision_plugin.py`, `hf_train_lightning.py` | plugin init test |
| 4 | `fix(trainer): respect config dtype for AMP + add IPEX optimize for XPU` | `sem/training/trainer.py`, `sem/training/lightning_module.py` | config dtype check |
| 5 | `chore(config): wire PIT transform and precision control into config and model construction` | `configs/max_aggression.yaml`, `sem/config.py`, `sem/model.py`, `hf_train_lightning.py`, `launch_job.py` | config parse + model build |
| 6 | `test: validate SEOP Fix 29 (PIT + precision hardening + IPEX)` | `test_xpu_local.py` | 10-step dry run |

---

## Success Criteria

### Verification Commands
```bash
# Quick validation (< 1 min):
.venv/Scripts/python test_xpu_local.py

# Full Lightning validation (< 5 min):
timeout 300 python hf_train_lightning.py --config configs/max_aggression.yaml --precision 32-true --no-compile
```

### Final Checklist
- [ ] PIT transform implemented and wired into config
- [ ] CG solver hardened with dtype assertions and robust context
- [ ] Custom precision plugin created and integrated into Lightning trainer
- [ ] Trainer bf16 auto-enable bug fixed (respects config.training.dtype)
- [ ] IPEX optimize integrated for XPU path
- [ ] Config updated with PIT and precision control sections
- [ ] 10-step dry run passes: no NaN, loss decreasing, unitarity < 2.0
- [ ] All "Must NOT Have" constraints respected (no CG algorithm changes, no complex64 arithmetic)
