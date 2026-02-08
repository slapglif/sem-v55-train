<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-03 | Updated: 2026-02-08 -->

# sampler

## Purpose

Collapse/sampling head converting complex hidden states into vocabulary logits and sampled tokens. Uses log-linear projection (SEOP Fix 48) with a composable LogitsProcessor chain (SEOP Fix 59) for modern sampling methods.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Module exports |
| `born_collapse.py` | `BornCollapseSampler` - wavefunction → logits → processor chain → tokens |
| `logits_processors.py` | Composable `LogitsProcessor` classes + `build_processor_chain` factory |

## For AI Agents

### Working In This Directory

- **Log-linear head**: `logits = W_r·Re(psi) + W_i·Im(psi) + b`
- **Processor chain**: Composable processors applied in order: penalties → temperature → filters
- **Weight tying**: `SEMModel` ties `proj_real.weight` to `encoder.embedding.weight`

### Key Components

```python
class BornCollapseSampler(nn.Module):
    # Projects complex psi to logits, applies LogitsProcessorList, samples

class LogitsProcessorList:
    # Sequential chain of LogitsProcessor instances

def build_processor_chain(config: SamplerConfig) -> LogitsProcessorList:
    # Factory: builds chain from SamplerConfig, skipping disabled processors
```

### Available Processors

| Processor | Config Field | Disabled Value |
|-----------|-------------|---------------|
| RepetitionPenaltyProcessor | `repetition_penalty` | 1.0 |
| FrequencyPenaltyProcessor | `frequency_penalty` | 0.0 |
| PresencePenaltyProcessor | `presence_penalty` | 0.0 |
| NoRepeatNgramProcessor | `no_repeat_ngram_size` | 0 |
| TemperatureProcessor | `temperature` | 1.0 |
| TopKProcessor | `top_k` | 0 |
| TopPProcessor | `top_p` | 1.0 |
| MinPProcessor | `min_p` | 0.0 |
| TypicalProcessor | `typical_p` | 1.0 |
| TopAProcessor | `top_a` | 0.0 |
| EpsilonCutoffProcessor | `epsilon_cutoff` | 0.0 |
| EtaCutoffProcessor | `eta_cutoff` | 0.0 |

### Testing Requirements

```bash
uv run pytest tests/test_born.py -v
```

## Dependencies

### Internal
- `sem/config.py` - `SamplerConfig` dataclass
- `sem/model.py` - Wires processor chain, passes `input_ids` for repetition penalties

### External
- `torch` - Tensor operations

<!-- MANUAL: -->
