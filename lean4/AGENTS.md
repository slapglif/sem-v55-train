<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-03 -->

# lean4

## Purpose

Lean 4 formal verification proofs for SEM mathematical foundations. Contains machine-checked proofs of key properties: unitarity preservation, probability normalization, and information-theoretic bounds.

## Key Files

| File | Description |
|------|-------------|
| `lakefile.lean` | Lake build configuration for the Lean project |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `SEM/` | Core SEM proof modules (see `SEM/AGENTS.md` if exists) |

## For AI Agents

### Working In This Directory

- **Lean 4 syntax**: Different from Python - functional, dependently typed
- **Lake build**: Use `lake build` to compile and check proofs
- **Proof strategy**: State theorems, provide constructive proofs

### Building Proofs

```bash
# Build all proofs
lake build

# Check specific file
lake build SEM.Unitarity
```

### Mathematical Foundations

The Lean proofs formalize:
1. **Unitarity**: Cayley transform preserves norm (|U·ψ| = |ψ|)
2. **Probability**: Born rule produces valid probability distribution
3. **Information bounds**: Entropy constraints on SDR encoding

## Dependencies

### Internal
- Self-contained Lean project

### External
- `lean4` - Lean 4 compiler/prover
- `mathlib4` - Mathematical library (if used)

<!-- MANUAL: -->
