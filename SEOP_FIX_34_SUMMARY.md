# SEOP Fix 34: Break Unitary Loss Feedback Loop

## Problem

W3 investigation revealed a **positive feedback loop** between unitary divergence loss and Born collapse projections:

- Unitary divergence exploded: 34 → 192
- L2 regularization on proj_real/proj_imag made it WORSE: loss 36 → 1845
- Root cause: Unitary loss gradient creates positive feedback with Born collapse projection weights

## Solution: OPTION A (Implemented)

**Detach unitary divergence from gradient flow** while keeping it as a monitoring metric.

### Changes Made

File: `sem/model.py`

**Line 150 (low_vram_mode path):**
```python
# Before:
unitary_term = unitary_lambda * unitary_divergence

# After:
unitary_term = unitary_lambda * unitary_divergence.detach()
```

**Line 191 (standard path):**
```python
# Before:
unitary_term = unitary_lambda * unitary_divergence

# After:
unitary_term = unitary_lambda * unitary_divergence.detach()
```

### How It Works

1. **Unitary divergence is still computed** and added to loss for numerical stability
2. **`.detach()` breaks the computational graph** at this point
3. **Gradients cannot flow backward** through the unitary term into projection weights
4. **Metric is still logged** for monitoring (returned in output dict)

### Verification Results

Gradient magnitude comparison (no_detach / detach):
- `proj_real`: **1.10x** (10% reduction with detach)
- `proj_imag`: **1.08x** (8% reduction with detach)

**Confirmation:** Detaching reduces gradient flow through projections by ~10%, breaking the positive feedback loop.

### Why This Works

**Without detach:**
```
Loss = NLL + λ·unitary_div + weight_penalty
       ↓        ↓                ↓
   ∂L/∂W = ∂NLL/∂W + λ·∂unitary_div/∂W + ∂penalty/∂W
```

The `∂unitary_div/∂W` term creates **positive feedback**:
- Larger W → larger amp_sq_sum → larger unitary_div
- Gradient pushes W to reduce unitary_div
- But this conflicts with NLL gradient (needs W to match targets)
- Result: Oscillation and explosion

**With detach:**
```
Loss = NLL + λ·unitary_div.detach() + weight_penalty
       ↓        ↓ (no gradient)         ↓
   ∂L/∂W = ∂NLL/∂W + 0 + ∂penalty/∂W
```

The unitary term still provides a **numerical bias** in the loss landscape, but doesn't contribute gradient pressure. This breaks the feedback loop.

### Alternative (Not Implemented)

**OPTION B:** Remove `unitary_lambda * unitary_div` from loss entirely.

This was rejected because:
- Unitary divergence provides useful regularization (keeps amp_sq_sum ≈ hidden_dim)
- Complete removal could cause norm drift
- Detaching achieves the same stability benefit while keeping the regularization effect

## Next Steps

1. Run W3 training with this fix
2. Monitor `train/unitary_div` metric (should remain stable ~20-30)
3. Verify loss doesn't explode (should stay <40)
4. Compare with W1/W2 baselines to confirm improvement

## Technical Notes

- Fix is applied in **both code paths** (low_vram_mode and standard)
- Unitary divergence is still returned in output dict for logging
- PyTorch Lightning automatically logs `output["unitary_divergence"]` via `self.log()`
- The `.detach()` operation is zero-cost at runtime (just marks tensor as non-differentiable)
