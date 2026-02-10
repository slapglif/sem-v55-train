<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-08 | Updated: 2026-02-08 -->

# lean4/SEM

## Purpose

Lean 4 proof modules for SEM V5.5 "Lean Crystal". These files provide a lightweight formal skeleton (often axiom-driven) for core claims like Cayley unitarity, information conservation, and supporting analytic statements.

## Key Files

| File | Description |
|------|-------------|
| `Basic.lean` | Foundational types (Complex, WaveFunction) and basic operations |
| `Unitarity.lean` | Cayley transform unitarity statement (axiomatic core) |
| `Conservation.lean` | Information conservation formulation using norm preservation |
| `Hermeneutic.lean` | Hermeneutic-circle fixed-point formalization (conceptual scaffolding) |
| `MemoryHorizon.lean` | Memory horizon theorem sketch for SSM exponential decay (tau ~ S/e) |
| `SinkhornEpsilon.lean` | Sinkhorn epsilon scaling theorem sketch (entropy/variance approximation) |

## For AI Agents

### Working In This Directory

- This Lean project is intentionally minimal and does not fully depend on Mathlib linear algebra.
- Several results are encoded as axioms; treat this as a formalized specification scaffold rather than a complete proof library.

### Common Commands

```bash
# From lean4/
lake build
lake build SEM
lake build SEM.Unitarity
```

### Gotchas

- The custom `Complex` type in `Basic.lean` is not Mathlib's `Complex`.
- Any attempt to strengthen axioms into proofs will likely require migrating to Mathlib matrix/vector definitions.

## Dependencies

### Internal

- Imported by `lean4/SEM.lean`.

### External

- `lean4` - compiler/prover
- `lake` - build tool

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
