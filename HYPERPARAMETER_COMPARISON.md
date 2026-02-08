# Hyperparameter Changes Summary

## Quick Reference Table

### test.yaml (100-step quick test)

| Parameter | BEFORE | AFTER | Change Factor | Impact |
|-----------|--------|-------|---------------|--------|
| `learning_rate` | 1.0e-4 | **3.0e-5** | **÷3.3** | Gentler updates, prevents unitarity violations |
| `weight_decay` | 0.1 | **0.01** | **÷10** | Less aggressive regularization early in training |
| `warmup_steps` | 50 | **200** | **×4** | Smoother LR ramp (even exceeds max_steps!) |
| `gradient_clip` | 1.0 | **0.3** | **÷3.3** | Tighter clipping to suppress gradient spikes |
| `unitary_lambda` | (none) | **0.01** | **NEW** | **CRITICAL**: Prevents Cayley propagator drift |
| `scheduler_type` | wsd | **cosine** | CHANGED | Simpler for initial debugging |

**Net Effect**: MUCH more conservative - prioritizes stability over speed

---

### max_aggression.yaml (Full 100K-step training)

| Parameter | BEFORE | AFTER | Change Factor | Impact |
|-----------|--------|-------|---------------|--------|
| `learning_rate` | 1.0e-6 | **5.0e-5** | **×50** | Actually enables learning (1e-6 was absurd) |
| `weight_decay` | 0.1 | **0.05** | **÷2** | Moderate reduction in regularization |
| `warmup_steps` | 8000 | **2000** | **÷4** | Faster warmup (8K was excessive) |
| `gradient_clip` | 0.2 | **0.5** | **×2.5** | More tolerance for natural gradient variance |
| `unitary_lambda` | 0.005 | **0.01** | **×2** | Stronger unitarity enforcement |
| `scheduler_type` | wsd | **wsd** | (unchanged) | Keep WSD for full training |

**Net Effect**: MUCH more aggressive - enables actual learning within reasonable time

---

## Expected Training Curves

### BEFORE (with bugs):
```
Step   0: loss=2.85
Step  50: loss=3.20  ← INCREASING
Step 100: loss=4.60  ← DIVERGENCE
```

### AFTER (expected):
```
Step   0: loss=2.85
Step  50: loss=2.90  ← Warmup still active
Step 100: loss=2.70  ← Gentle decrease
Step 200: loss=2.40  ← Real learning starts post-warmup
```

**Key Insight**: For test.yaml, warmup_steps=200 but max_steps=100, so the entire test run is WARMUP. This is intentional - we're testing stability, not convergence speed.

---

## Rationale by Parameter

### learning_rate
- **Complex-valued models need ~0.3-0.5× LR** of real-valued equivalents
- Real + Imag = 2× degrees of freedom per parameter
- Cayley constraint couples real/imag updates → overshoot risk

**Rule**: `lr_complex ≈ 0.3 × lr_real`

### weight_decay
- **L2 penalty can destroy complex phase structure**
- Example: If Re(w)→0 and Im(w)→0, then arg(w) is undefined
- Early in training, phase is MORE important than magnitude

**Rule**: Use 0.01-0.05 for complex models vs 0.1 for real models

### warmup_steps
- **Initial random Hamiltonian → chaotic eigenvalues**
- CG solver condition number κ is high → needs many iterations or diverges
- Warmup allows model to "find" a reasonable Hamiltonian first

**Rule**: Warmup should be ≥1% of total steps for complex SSMs

### gradient_clip
- **Complex gradients have spurious phase components**
- ∂L/∂θ where θ is angular can be high-magnitude but zero actual effect
- Tight clipping (0.3-0.5) suppresses these phantom gradients

**Rule**: Use 0.3-0.5 for complex vs 1.0-5.0 for real models

### unitary_lambda
- **MOST CRITICAL PARAMETER**
- Cayley (I - iH/2)^(-1)(I + iH/2) is unitary IFF CG converges exactly
- Floating-point errors accumulate: ||U†U - I|| can reach O(1e-2)
- Unitarity loss λ||U†U - I||² forces gradients to maintain constraint

**Rule**: Always use 0.01-0.05 for Cayley-based architectures

---

## Validation Checklist

After running with new configs, verify:

- [ ] Loss decreases (or stays flat during warmup)
- [ ] No NaN or Inf values
- [ ] Unitarity deviation <1e-4 (check health monitor)
- [ ] CG solver converges in <10 iterations
- [ ] Gradient norm doesn't explode (should stay <10)
- [ ] Perplexity decreases over time (after warmup)

If ANY of these fail → use emergency fallback settings (see OPTIMIZER_TUNING_SUMMARY.md)

---

## SEOP Perspective: Entropy Flow Control

**Before fixes**: Entropy leaking through non-unitary propagator
- Loss divergence = Latent entropy explosion
- Random walk in activation space
- Chaos

**After fixes**: Entropy bounded by unitarity constraint
- unitary_lambda enforces reversible evolution
- Reduced LR prevents entropy injection via noisy gradients
- Information flows COHERENTLY through Cayley layers

**Expected outcome**: Structured latent representations instead of noise
