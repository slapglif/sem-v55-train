"""Composable logits processors for the Born Collapse Sampler.

SEOP Fix 59: Modern sampling chain matching llama.cpp / HuggingFace standards.
Processors are applied in sequence: penalties → filters → temperature.

Application order (matching HuggingFace convention):
  1. RepetitionPenaltyProcessor (multiplicative, on raw logits)
  2. FrequencyPenaltyProcessor (additive by count)
  3. PresencePenaltyProcessor (additive one-time)
  4. NoRepeatNgramProcessor (ban repeated n-grams)
  5. TemperatureProcessor (scale logits, unless temperature_last)
  6. TopKProcessor
  7. TopPProcessor (nucleus)
  8. MinPProcessor
  9. TypicalProcessor
 10. TopAProcessor
 11. EpsilonCutoffProcessor
 12. EtaCutoffProcessor
 13. TemperatureProcessor (if temperature_last)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from ..config import SamplerConfig


class LogitsProcessor:
    """Base class for logits processors."""

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        raise NotImplementedError


class LogitsProcessorList:
    """Applies a list of LogitsProcessor in sequence."""

    def __init__(self, processors: list[LogitsProcessor]) -> None:
        self.processors = processors

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        for processor in self.processors:
            logits = processor(input_ids, logits)
        return logits

    def __len__(self) -> int:
        return len(self.processors)

    def __iter__(self):
        return iter(self.processors)


# ---------------------------------------------------------------------------
# Penalty processors (applied first, on raw logits)
# ---------------------------------------------------------------------------


class RepetitionPenaltyProcessor(LogitsProcessor):
    """Multiplicative penalty for tokens already in input_ids.

    For each token in input_ids: if logit > 0, divide by penalty; if < 0, multiply.
    This is the HuggingFace / CTRL convention (Keskar et al., 2019).
    """

    def __init__(self, penalty: float) -> None:
        if penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.penalty == 1.0 or input_ids.numel() == 0:
            return logits
        score = torch.gather(logits, -1, input_ids.clamp(min=0))
        score = torch.where(score > 0, score / self.penalty, score * self.penalty)
        logits = logits.scatter(-1, input_ids.clamp(min=0), score)
        return logits


class FrequencyPenaltyProcessor(LogitsProcessor):
    """Additive penalty proportional to token frequency in context (OpenAI-style)."""

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.penalty == 0.0 or input_ids.numel() == 0:
            return logits
        V = logits.shape[-1]
        counts = torch.zeros_like(logits)
        # Count occurrences of each token, clamping indices to valid range
        ids_clamped = input_ids.clamp(min=0, max=V - 1)
        counts.scatter_add_(
            -1, ids_clamped, torch.ones_like(ids_clamped, dtype=logits.dtype)
        )
        logits = logits - self.penalty * counts
        return logits


class PresencePenaltyProcessor(LogitsProcessor):
    """Additive one-time penalty for tokens present in context (OpenAI-style)."""

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.penalty == 0.0 or input_ids.numel() == 0:
            return logits
        V = logits.shape[-1]
        mask = torch.zeros_like(logits)
        ids_clamped = input_ids.clamp(min=0, max=V - 1)
        mask.scatter_(-1, ids_clamped, 1.0)
        logits = logits - self.penalty * mask
        return logits


class NoRepeatNgramProcessor(LogitsProcessor):
    """Ban repeating n-grams by setting their logits to -inf."""

    def __init__(self, ngram_size: int) -> None:
        self.ngram_size = ngram_size

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.ngram_size <= 0 or input_ids.shape[-1] < self.ngram_size:
            return logits
        B = input_ids.shape[0]
        for b in range(B):
            ids = input_ids[b].tolist()
            # Build set of seen (n-1)-grams and their following tokens
            ngram_to_next: dict[tuple, set] = {}
            for i in range(len(ids) - self.ngram_size + 1):
                ngram = tuple(ids[i : i + self.ngram_size - 1])
                next_token = ids[i + self.ngram_size - 1]
                if ngram not in ngram_to_next:
                    ngram_to_next[ngram] = set()
                ngram_to_next[ngram].add(next_token)
            # Check if current context ends with a seen (n-1)-gram
            if len(ids) >= self.ngram_size - 1:
                current_ngram = tuple(ids[-(self.ngram_size - 1) :])
                banned = ngram_to_next.get(current_ngram, set())
                for token_id in banned:
                    if 0 <= token_id < logits.shape[-1]:
                        logits[b, token_id] = float("-inf")
        return logits


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


class TemperatureProcessor(LogitsProcessor):
    def __init__(self, temperature: float) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.temperature == 1.0:
            return logits
        return logits / self.temperature


# ---------------------------------------------------------------------------
# Filter processors (applied after temperature, before sampling)
# ---------------------------------------------------------------------------


class TopKProcessor(LogitsProcessor):
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        k = self.top_k
        if k <= 0 or k >= logits.shape[-1]:
            return logits
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[..., -1:]
        return logits.masked_fill(logits < threshold, float("-inf"))


class TopPProcessor(LogitsProcessor):
    """Nucleus (top-p) filtering. Keeps smallest set of tokens with cumulative P >= p."""

    def __init__(self, top_p: float) -> None:
        self.top_p = top_p

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        p = self.top_p
        if p >= 1.0:
            return logits
        p = max(p, 1e-6)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= p
        sorted_mask[..., 0] = False
        sorted_logits[sorted_mask] = float("-inf")
        return logits.scatter(-1, sorted_indices, sorted_logits)


class MinPProcessor(LogitsProcessor):
    """Min-P filtering: discard tokens with P < min_p * max(P).

    Widely adopted in llama.cpp, FlashInfer, SGLang, HuggingFace.
    Adaptive: threshold scales with the model's confidence.
    """

    def __init__(self, min_p: float) -> None:
        self.min_p = min_p

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.min_p <= 0.0:
            return logits
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        threshold = self.min_p * max_prob
        # Mask tokens below threshold, but always keep at least the top-1
        mask = probs < threshold
        top1 = probs.argmax(dim=-1, keepdim=True)
        mask.scatter_(-1, top1, False)
        return logits.masked_fill(mask, float("-inf"))


class TypicalProcessor(LogitsProcessor):
    """Typical sampling (Meister et al., 2022): keep tokens near expected information content.

    Filters by |info(token) - entropy| sorted ascending, keeping cumulative mass < typical_p.
    """

    def __init__(self, typical_p: float) -> None:
        self.typical_p = typical_p

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.typical_p >= 1.0:
            return logits
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # Information content of each token
        neg_entropy = (probs * log_probs).sum(dim=-1, keepdim=True)  # -H
        # Shifted log-probs: |log p(x) + H| = |log p(x) - (-H)|
        shifted = torch.abs(log_probs + neg_entropy)
        # Sort by typicality (ascending = most typical first)
        sorted_shifted, sorted_indices = torch.sort(shifted, dim=-1)
        sorted_probs = probs.gather(-1, sorted_indices)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Keep until cumulative mass exceeds typical_p
        sorted_mask = (cumulative_probs - sorted_probs) >= self.typical_p
        sorted_mask[..., 0] = False  # Always keep most typical token
        # Build logits mask in original order
        mask = sorted_mask.scatter(-1, sorted_indices, sorted_mask)
        return logits.masked_fill(mask, float("-inf"))


class TopAProcessor(LogitsProcessor):
    """Top-A filtering: keep tokens with P >= top_a * max(P)²."""

    def __init__(self, top_a: float) -> None:
        self.top_a = top_a

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.top_a <= 0.0:
            return logits
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        threshold = self.top_a * max_prob * max_prob
        mask = probs < threshold
        top1 = probs.argmax(dim=-1, keepdim=True)
        mask.scatter_(-1, top1, False)
        return logits.masked_fill(mask, float("-inf"))


class EpsilonCutoffProcessor(LogitsProcessor):
    """Hard probability floor: discard tokens with P < epsilon."""

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.epsilon <= 0.0:
            return logits
        probs = F.softmax(logits, dim=-1)
        mask = probs < self.epsilon
        top1 = probs.argmax(dim=-1, keepdim=True)
        mask.scatter_(-1, top1, False)
        return logits.masked_fill(mask, float("-inf"))


class EtaCutoffProcessor(LogitsProcessor):
    """Eta sampling: entropy-adaptive cutoff.

    threshold = min(eta, sqrt(eta) * exp(-H(p)))
    """

    def __init__(self, eta: float) -> None:
        self.eta = eta

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        if self.eta <= 0.0:
            return logits
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        threshold = torch.clamp_min(
            math.sqrt(self.eta) * torch.exp(-entropy),
            self.eta,
        )
        mask = probs < threshold
        top1 = probs.argmax(dim=-1, keepdim=True)
        mask.scatter_(-1, top1, False)
        return logits.masked_fill(mask, float("-inf"))


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------


def build_processor_chain(config: SamplerConfig) -> LogitsProcessorList:
    """Build the composable processor chain from SamplerConfig.

    Order follows HuggingFace convention:
    penalties → temperature (unless last) → filters → temperature (if last)
    """
    processors: list[LogitsProcessor] = []

    # 1. Penalty processors (on raw logits)
    if config.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyProcessor(config.repetition_penalty))
    if config.frequency_penalty != 0.0:
        processors.append(FrequencyPenaltyProcessor(config.frequency_penalty))
    if config.presence_penalty != 0.0:
        processors.append(PresencePenaltyProcessor(config.presence_penalty))
    if config.no_repeat_ngram_size > 0:
        processors.append(NoRepeatNgramProcessor(config.no_repeat_ngram_size))

    # 2. Temperature (before filters, unless temperature_last)
    if not config.temperature_last and config.temperature != 1.0:
        processors.append(TemperatureProcessor(config.temperature))

    # 3. Filter processors
    if config.top_k > 0:
        processors.append(TopKProcessor(config.top_k))
    if config.top_p < 1.0:
        processors.append(TopPProcessor(config.top_p))
    if config.min_p > 0.0:
        processors.append(MinPProcessor(config.min_p))
    if config.typical_p < 1.0:
        processors.append(TypicalProcessor(config.typical_p))
    if config.top_a > 0.0:
        processors.append(TopAProcessor(config.top_a))
    if config.epsilon_cutoff > 0.0:
        processors.append(EpsilonCutoffProcessor(config.epsilon_cutoff))
    if config.eta_cutoff > 0.0:
        processors.append(EtaCutoffProcessor(config.eta_cutoff))

    # 4. Temperature (after filters, if temperature_last)
    if config.temperature_last and config.temperature != 1.0:
        processors.append(TemperatureProcessor(config.temperature))

    return LogitsProcessorList(processors)
