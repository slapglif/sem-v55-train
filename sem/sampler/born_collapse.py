"""Log-Linear Collapse Sampler with composable LogitsProcessor chain.

SEOP Fix 48: Log-linear projection replaces quadratic Born rule.
SEOP Fix 59: Modern sampling chain (min-p, typical, repetition penalty, etc.)

logits = W_r @ Re(psi) + W_i @ Im(psi) + bias

Weight tying: proj_real.weight is tied to encoder.embedding.weight in SEMModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .logits_processors import LogitsProcessorList


class BornCollapseSampler(nn.Module):
    """Log-linear sampler: complex wavefunction -> vocabulary logits -> token probabilities.

    Pipeline:
    1. Linear projection of complex wavefunction to vocabulary logits
    2. LogitsProcessorList chain (penalties, temperature, filters)
    3. Softmax + categorical sampling or argmax
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        processors: Optional[LogitsProcessorList] = None,
        # Legacy parameters (used when processors is None)
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.processors = processors

        # Legacy fallback parameters
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        # logits = W_r @ Re(psi) + W_i @ Im(psi) + bias
        self.proj_real = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.proj_imag = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def compute_logits(self, psi: Tensor) -> Tensor:
        """Project complex wavefunction to vocabulary logits.

        Args:
            psi: [B, S, D] complex64 wavefunction
        Returns:
            logits: [B, S, V] float32
        """
        return self.proj_real(psi.real) + self.proj_imag(psi.imag) + self.output_bias

    def _legacy_filter(
        self, logits: Tensor, temperature: float, top_k: int, top_p: float
    ) -> Tensor:
        """Legacy filtering path when no processor chain is configured."""
        logits = logits / max(temperature, 1e-8)
        if 0 < top_k < logits.shape[-1]:
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            threshold = topk_vals[..., -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if top_p < 1.0:
            p = max(top_p, 1e-6)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= p
            sorted_mask[..., 0] = False
            sorted_logits[sorted_mask] = float("-inf")
            logits = logits.scatter(-1, sorted_indices, sorted_logits)
        return logits

    def forward(
        self,
        psi: Tensor,
        input_ids: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sample: bool = True,
    ) -> dict:
        """Log-linear collapse: wavefunction -> token.

        Args:
            psi: [B, S, D] complex64 wavefunction
            input_ids: [B, S_ctx] token history for repetition penalties
            temperature: Override temperature (legacy path only)
            top_k: Override top-k (legacy path only)
            top_p: Override top-p (legacy path only)
            sample: If True, sample; if False, return logits only

        Returns:
            dict with 'logits', 'log_probs', and optionally 'tokens', 'probs'
        """
        logits = self.compute_logits(psi)
        result: dict[str, Tensor] = {"logits": logits}

        if sample:
            if self.processors is not None:
                B, S, V = logits.shape
                sampling_logits = logits.reshape(-1, V)
                ids_for_proc = (
                    input_ids
                    if input_ids is not None
                    else torch.empty(B, 0, dtype=torch.long, device=logits.device)
                )
                if ids_for_proc.shape[0] != sampling_logits.shape[0]:
                    ids_for_proc = ids_for_proc.repeat_interleave(S, dim=0)
                sampling_logits = self.processors(ids_for_proc, sampling_logits)
                sampling_logits = sampling_logits.reshape(B, S, V)
            else:
                temp = temperature if temperature is not None else self._temperature
                k = top_k if top_k is not None else self._top_k
                p = top_p if top_p is not None else self._top_p
                sampling_logits = self._legacy_filter(logits, temp, k, p)

            probs = F.softmax(sampling_logits, dim=-1)
            tokens = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(
                probs.shape[:-1]
            )
            result["tokens"] = tokens
            result["probs"] = probs

        result["log_probs"] = F.log_softmax(logits, dim=-1)
        return result

    def training_forward(self, psi: Tensor) -> Tensor:
        """Simplified forward for training: return log_probs for cross-entropy."""
        logits = self.compute_logits(psi)
        return F.log_softmax(logits, dim=-1)
