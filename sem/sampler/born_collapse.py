"""Born Collapse Sampler for token generation.

Implements the Born rule from quantum mechanics for token sampling:
    P(token) = |ψ_token|²

The wavefunction ψ (complex-valued, from the Cayley propagator) is
projected onto the vocabulary space. The squared magnitude of each
component gives the probability of generating that token.

This is the final "measurement" step that collapses the continuous
crystal manifold representation into a discrete token.

Supports:
- Temperature-controlled sampling
- Top-k filtering (keep k highest probability tokens)
- Top-p (nucleus) filtering (keep tokens with cumulative p >= threshold)
- Greedy decoding (argmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class BornCollapseSampler(nn.Module):
    """Born rule sampler: complex wavefunction -> token probabilities.

    Architecture:
    1. Complex-to-real vocabulary projection
    2. Born rule: P = |amplitude|^2 (computed in log-space for stability)
    3. Temperature scaling
    4. Top-k / Top-p filtering
    5. Categorical sampling or argmax
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Complex-to-real vocabulary projection
        # ψ (complex, dim D) -> amplitude (complex, dim V)
        # Uses 2 real Linear layers: (W_r + iW_i) @ (ψ_re + iψ_im)
        # = W_r@ψ_re - W_i@ψ_im + i(W_r@ψ_im + W_i@ψ_re)
        # NOTE: For large V (32K+), 4 contiguous float32 GEMMs outperform
        # complex64 BLAS which has non-contiguous memory and 3 matmul overhead.
        self.proj_real = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.proj_imag = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Output bias (real-valued)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def compute_log_born_probs(self, psi: Tensor) -> Tuple[Tensor, Tensor]:
        """Project complex wavefunction and compute log|amplitude|^2.

        Numerically stable implementation:
        - Avoids sqrt() followed by log()
        - Directly computes log(re^2 + im^2) for Born rule

        Args:
            psi: [B, S, D] complex64 wavefunction
        Returns:
            log_probs_unnorm: [B, S, V] float32 unnormalized log-probabilities
        """
        # Complex projection: amp = (W_r + iW_i) @ (ψ_re + iψ_im)
        amp_real = self.proj_real(psi.real) - self.proj_imag(psi.imag)
        amp_imag = self.proj_real(psi.imag) + self.proj_imag(psi.real)

        # SEOP Fix 3: Rotationally-symmetric Born rule
        # Adding bias to real part only creates a preferential axis in the complex plane.
        # Instead, add bias in log-magnitude space to preserve rotational symmetry.
        # log P = log(re² + im²) + bias  (bias shifts log-probability uniformly)
        # SEOP Fix 16: Adaptive Born floor
        # Fixed 1e-12 creates a hard gradient wall at log(1e-12) ≈ -27.6 for rare tokens.
        # Adaptive floor = mean(amp²) * 1e-6 keeps floor 60dB below signal mean,
        # maintaining gradient flow for rare tokens proportional to actual amplitude scale.
        amp_sq = amp_real**2 + amp_imag**2
        floor = amp_sq.mean(dim=-1, keepdim=True) * 1e-6 + 1e-30  # 60dB below mean
        log_amp_sq = torch.log(amp_sq + floor)
        log_probs_unnorm = log_amp_sq + self.output_bias

        return log_probs_unnorm, amp_sq

    def apply_temperature(
        self, logits: Tensor, temperature: Optional[float] = None
    ) -> Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: [B, S, V] unnormalized log-probabilities
            temperature: Sampling temperature (default: self.temperature)
        Returns:
            scaled_logits: [B, S, V]
        """
        temp = temperature if temperature is not None else self.temperature
        return logits / max(temp, 1e-8)

    def apply_top_k(self, logits: Tensor, k: Optional[int] = None) -> Tensor:
        """Apply top-k filtering: zero out all but top-k logits.

        Args:
            logits: [B, S, V]
            k: Number of top tokens to keep
        Returns:
            Filtered logits
        """
        k = k if k is not None else self.top_k
        if k <= 0 or k >= logits.shape[-1]:
            return logits

        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[..., -1:]

        return logits.masked_fill(logits < threshold, float("-inf"))

    def apply_top_p(self, logits: Tensor, p: Optional[float] = None) -> Tensor:
        """Apply nucleus (top-p) filtering.

        Keep the smallest set of tokens whose cumulative probability >= p.

        Args:
            logits: [B, S, V]
            p: Cumulative probability threshold
        Returns:
            Filtered logits
        """
        p = p if p is not None else self.top_p
        if p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= p
        sorted_logits[sorted_mask] = float("-inf")

        logits = logits.scatter(-1, sorted_indices, sorted_logits)
        return logits

    def forward(
        self,
        psi: Tensor,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sample: bool = True,
    ) -> dict:
        """Full Born collapse: wavefunction -> token.

        Args:
            psi: [B, S, D] complex64 wavefunction
            temperature: Override temperature
            top_k: Override top-k
            top_p: Override top-p
            sample: If True, sample; if False, return logits only

        Returns:
            dict with:
                'logits': [B, S, V] raw logits (for training loss)
                'log_probs': [B, S, V] log softmax (for cross-entropy)
                'amp_sq': [B, S, V] squared amplitudes (for UnitaryBornLoss)
                'tokens': [B, S] sampled token indices (if sample=True)
                'probs': [B, S, V] probabilities (if sample=True)
        """
        # Step 1: Born rule projection -> log|amplitude|^2
        log_probs_unnorm, amp_sq = self.compute_log_born_probs(psi)  # [B, S, V]

        # Step 2: Temperature scaling
        logits = self.apply_temperature(log_probs_unnorm, temperature)  # [B, S, V]

        result = {
            "logits": logits,
            "log_probs": F.log_softmax(logits, dim=-1),
            "amp_sq": amp_sq,
        }

        if sample:
            # Step 3: Apply filtering
            filtered_logits = self.apply_top_k(logits, top_k)
            filtered_logits = self.apply_top_p(filtered_logits, top_p)

            # Step 4: Sample
            probs = F.softmax(filtered_logits, dim=-1)
            tokens = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(
                probs.shape[:-1]
            )

            result["tokens"] = tokens
            result["probs"] = probs

        return result

    def training_forward(self, psi: Tensor) -> Tensor:
        """Simplified forward for training: return log_probs for cross-entropy.

        Args:
            psi: [B, S, D] complex64
        Returns:
            log_probs: [B, S, V] float32
        """
        log_probs_unnorm, _ = self.compute_log_born_probs(psi)
        logits = self.apply_temperature(log_probs_unnorm)
        return F.log_softmax(logits, dim=-1)
