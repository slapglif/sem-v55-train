<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# lean4

## Purpose

Lean 4 formal verification proofs for SEM mathematical foundations. Contains machine-checked proofs of key properties: unitarity preservation, probability normalization, and information-theoretic bounds.

## Key Files

| File | Description |
|------|-------------|
| `lakefile.lean` | Lake build configuration for the Lean project |
| `SEM.lean` | Root module importing all SEM proof files |
| `lake-manifest.json` | Pinned Lean dependencies resolved by Lake |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `SEM/` | Core SEM proof modules (see `SEM/AGENTS.md`) |

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
