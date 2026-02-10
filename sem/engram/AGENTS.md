<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-08 | Updated: 2026-02-08 -->

# engram

## Purpose

Optional Engram module that augments hidden states with n-gram context using hash-based embeddings (DeepSeek-style). In SEM this is wired as a lightweight, conv-gated additive feature (no attention) into `sem/model.py`.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Public exports (`Engram`, `EngramConfig`, helpers) |
| `engram.py` | `Engram` + `EngramConfig` + `ShortConv` conv-gated mixing block |
| `hash_mapping.py` | `NgramHashMapping` - layer-specific n-gram hashing (multi-head, prime moduli) |
| `multi_head_embedding.py` | `MultiHeadEmbedding` - offset-indexed multi-head embedding table |
| `compressed_tokenizer.py` | `CompressedTokenizer` - vocabulary normalization/compression for better hashing |

## For AI Agents

### Working In This Directory

- Engram is designed to be plug-in augmentation; the core model runs without it.
- Engram expects a tokenizer-like object (see `EngramConfig.tokenizer`) and uses NumPy for hashing.
- Integration point is `sem/model.py` (adds Engram contribution to the real part of the complex state).

### Common Commands

```bash
# Run integration tests that exercise optional wiring paths
uv run pytest tests/test_integration.py -v
uv run pytest tests/test_layer_quality.py -v
```

### Gotchas

- `hash_mapping.py` uses `sympy.isprime` to choose prime vocab sizes; this is a runtime dependency.
- Hashing runs on CPU (NumPy); keep batch/seq sizes small in quick experiments.
- Engram operates on real-valued features in SEM (currently `psi.abs()`), not directly on complex tensors.

## Dependencies

### Internal

- Used (optionally) from `sem/model.py`.

### External

- `torch` - Engram module and projections
- `numpy` - fast hashing / array ops
- `sympy` - prime selection for hash moduli
- `tokenizers` - normalizers used by `CompressedTokenizer`

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
