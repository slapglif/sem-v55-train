"""Log-Linear Collapse Sampler for token generation.

SEOP Fix 48: Replaced quadratic Born rule |W·psi|^2 with log-linear projection.

The Born rule P(token) = |W·psi|^2 has a fundamental rank deficiency:
    image_dim = D(2D+1) = 32,896 < V = 50,262 for D=128
This means the model CANNOT produce arbitrary probability distributions.

Log-linear projection: logits = W_r @ Re(psi) + W_i @ Im(psi) + bias
The projection has rank ≤ 2D (e.g., 256 for D=128), not full rank over V.
This is intentional: the bias term provides a full-rank offset (Zipf prior),
and the 2D-rank perturbation provides context-dependent adjustments.
Standard softmax cross-entropy loss applies directly.

Supports:
- Temperature-controlled sampling
- Top-k filtering
- Top-p (nucleus) filtering
- Greedy decoding (argmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional


class BornCollapseSampler(nn.Module):
    """Log-linear sampler: complex wavefunction -> vocabulary logits -> token probabilities.

    Architecture:
    1. Linear projection of complex wavefunction to vocabulary logits
    2. Temperature scaling
    3. Top-k / Top-p filtering
    4. Categorical sampling or argmax

    Weight tying: In SEM, proj_real.weight is tied to encoder.embedding.weight.
    proj_imag.weight is untied, giving the imaginary channel independent capacity.
    This asymmetry is an intentional inductive bias — Re(psi) carries the
    "semantic" signal (tied to token embeddings), Im(psi) carries "phase" info
    (position, context modulation).
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

        # Log-linear vocabulary projection
        # logits = W_r @ Re(psi) + W_i @ Im(psi) + bias
        self.proj_real = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.proj_imag = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def compute_logits(self, psi: Tensor) -> Tensor:
        """Project complex wavefunction to vocabulary logits.

        SEOP Fix 48: Linear projection replaces quadratic Born rule.
        logits = W_r @ Re(psi) + W_i @ Im(psi) + bias

        Args:
            psi: [B, S, D] complex64 wavefunction
        Returns:
            logits: [B, S, V] float32
        """
        return self.proj_real(psi.real) + self.proj_imag(psi.imag) + self.output_bias

    def apply_temperature(
        self, logits: Tensor, temperature: Optional[float] = None
    ) -> Tensor:
        """Apply temperature scaling to logits."""
        temp = temperature if temperature is not None else self.temperature
        return logits / max(temp, 1e-8)

    def apply_top_k(self, logits: Tensor, k: Optional[int] = None) -> Tensor:
        """Apply top-k filtering: mask all but top-k logits."""
        k = k if k is not None else self.top_k
        if k <= 0 or k >= logits.shape[-1]:
            return logits
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[..., -1:]
        return logits.masked_fill(logits < threshold, float("-inf"))

    def apply_top_p(self, logits: Tensor, p: Optional[float] = None) -> Tensor:
        """Apply nucleus (top-p) filtering.

        SEOP optimization note: When used after top_k filtering, most entries are -inf.
        torch.sort puts -inf last (descending), and softmax(-inf)=0, so cumulative_probs
        naturally excludes them. The O(V log V) sort is the main cost; for V=50k this is
        ~0.5ms, acceptable for inference. For training, top_p is not applied (sample=False).
        """
        p = p if p is not None else self.top_p
        if p >= 1.0:
            return logits
        p = max(p, 1e-6)  # Guard: p<=0 would mask ALL tokens → NaN
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = (cumulative_probs - F.softmax(sorted_logits, dim=-1)) >= p
        # Always keep at least the top-1 token to prevent all-masked → NaN
        sorted_mask[..., 0] = False
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
        """Log-linear collapse: wavefunction -> token.

        Args:
            psi: [B, S, D] complex64 wavefunction
            temperature: Override temperature
            top_k: Override top-k
            top_p: Override top-p
            sample: If True, sample; if False, return logits only

        Returns:
            dict with:
                'logits': [B, S, V] raw logits
                'log_probs': [B, S, V] log softmax
                'tokens': [B, S] sampled token indices (if sample=True)
                'probs': [B, S, V] probabilities (if sample=True)
        """
        logits = self.compute_logits(psi)
        scaled_logits = self.apply_temperature(logits, temperature)

        result = {
            "logits": logits,
            "log_probs": F.log_softmax(scaled_logits, dim=-1),
        }

        if sample:
            filtered_logits = self.apply_top_k(scaled_logits, top_k)
            filtered_logits = self.apply_top_p(filtered_logits, top_p)
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
        logits = self.compute_logits(psi)
        return F.log_softmax(logits, dim=-1)
