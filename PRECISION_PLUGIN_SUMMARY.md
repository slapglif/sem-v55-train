# Custom Lightning Precision Plugin - Implementation Summary

## Problem Statement
The CayleySolitonPropagator requires float32 precision for:
1. **CG solver convergence**: Needs ~7 decimal digits of precision; bf16 has only ~3
2. **Trigonometric functions**: cos/sin for phase rotation require high precision
3. **Hamiltonian matvec operations**: Sparse matrix arithmetic accumulates errors in lower precision

When Lightning's `bf16-mixed` precision is enabled, it applies autocast globally, which can poison the CG solver and cause:
- Unitarity violations (2.66 → 1.71e+09)
- SDR sparsity collapse (0.5 → 0.0)
- CG residual explosion (3.72e+04 vs target < 1e-4)

## Solution: SEMPrecisionPlugin

### Files Created/Modified

#### 1. `sem/training/precision_plugin.py` (NEW)
Custom Lightning precision plugin that:
- Inherits from `pytorch_lightning.plugins.precision.MixedPrecisionPlugin`
- Caches references to all `CayleySolitonPropagator` instances during setup
- Wraps the forward pass context to disable autocast for propagator layers
- Maintains Lightning 2.x compatibility

**Key Classes:**
- `SEMPrecisionPlugin`: Main plugin class
  - `__init__(precision)`: Initialize with precision string
  - `setup_module(module)`: Cache propagator references
  - `forward_context()`: Return custom context manager

- `_PropagatorExclusionContext`: Context manager that:
  - Enters base Lightning autocast context
  - Wraps propagator forward methods to disable autocast
  - Restores original forwards on exit

#### 2. `hf_train_lightning.py` (MODIFIED)
Updated to use the precision plugin:

```python
# Import the plugin
from sem.training.precision_plugin import SEMPrecisionPlugin

# In main():
# Create plugin only for mixed precision modes
plugins = []
if args.precision in ["bf16-mixed", "16-mixed"]:
    plugins.append(SEMPrecisionPlugin(precision=args.precision))
    logger.info(
        f"Using SEMPrecisionPlugin to exclude CayleySolitonPropagator from {args.precision} autocast"
    )

# Pass to Trainer
trainer = L.Trainer(
    ...
    plugins=plugins if plugins else None,
    ...
)
```

## How It Works

### Execution Flow
1. **Trainer initialization**: Plugin is passed to `L.Trainer` if using mixed precision
2. **Model setup**: Plugin's `setup_module()` is called, caching all propagator instances
3. **Training step**: 
   - Plugin's `forward_context()` returns custom context manager
   - Context manager enters base Lightning autocast (enables bf16 for most layers)
   - Context manager wraps propagator forwards to disable autocast locally
   - Propagator runs in float32 while rest of model uses bf16
   - Context manager restores original forwards on exit

### Precision Guarantees
- **Propagator layers**: Always float32 (no autocast)
- **Other layers**: Use specified precision (bf16-mixed, 16-mixed, etc.)
- **Backward pass**: Gradients computed with appropriate precision

## Usage

### Command Line
```bash
# With bf16-mixed precision (plugin active)
python hf_train_lightning.py --config configs/max_aggression.yaml --precision bf16-mixed

# With float32 (plugin inactive)
python hf_train_lightning.py --config configs/max_aggression.yaml --precision 32-true

# With 16-mixed (plugin active)
python hf_train_lightning.py --config configs/max_aggression.yaml --precision 16-mixed
```

### Logging
When mixed precision is used, you'll see:
```
[INFO] Using SEMPrecisionPlugin to exclude CayleySolitonPropagator from bf16-mixed autocast
```

## Backward Compatibility
- ✓ Existing `--precision` argument handling unchanged
- ✓ Plugin only activates for mixed precision modes
- ✓ Float32 training unaffected (plugin not used)
- ✓ Lightning 2.x API compliance maintained

## Testing Recommendations
1. Run training with `--precision bf16-mixed` and verify:
   - No NaN loss at step 0
   - CG residual < 1e-4
   - Unitarity divergence stable
   - SDR sparsity > 0.5

2. Compare metrics with float32 baseline:
   - Loss trajectory should match
   - Convergence speed should be similar or faster
   - Memory usage should be lower with bf16

## Future Enhancements
- Add per-layer precision control via config
- Support for other precision modes (fp8, etc.)
- Metrics tracking for autocast exclusion effectiveness
- Integration with torch.compile for XPU optimization
