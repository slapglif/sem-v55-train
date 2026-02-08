"""MESH-SDR: Minimize Entropy of Sinkhorn - Sparse Distributed Representation.

The entropic encoder maps dense token embeddings to sparse complex
representations on the Crystal Manifold via attention-guided optimal
transport.

Pipeline:
1. Learned embedding -> Cost matrix computation
2. Log-domain Sinkhorn -> Transport plan T
3. Top-K sparsification -> Sparse SDR
4. Complex lift with positional phase -> Crystal Manifold point
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional

from .sinkhorn import LogSinkhorn
from .cost_matrix import LearnedCostMatrix
from sem.utils.complex_ops import safe_complex


class MESHEncoder(nn.Module):
    """MESH-SDR Entropic Encoder.

    Maps token IDs to sparse complex representations via
    entropy-regularized optimal transport.
    """

    positional_phase: Tensor

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        sdr_sparsity: int = 32,
        sdr_candidates: int = 128,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_max_iter: int = 50,
        sinkhorn_tol: float = 1e-3,
        sinkhorn_auto_epsilon: bool = False,
        sinkhorn_auto_epsilon_scale: float = 0.05,
        max_seq_length: int = 2048,
        low_vram_mode: bool = False,
        soft_sparse: bool = False,
        soft_sparse_temp: float = 0.1,
        simple_mode: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sdr_sparsity = sdr_sparsity
        self.sdr_candidates = sdr_candidates
        self.low_vram_mode = low_vram_mode
        self.soft_sparse = soft_sparse
        self.soft_sparse_temp = soft_sparse_temp
        self.simple_mode = simple_mode

        # Token embedding (real-valued)
        # SEOP Fix 52: Scale init to N(0, 1/√D) for weight tying compatibility.
        # Default N(0,1) gives ||e||²≈D=128, so self-similarity logit≈128 → softmax≈1.0
        # for current token. This traps the model in "repeat current token" local minimum.
        # With 1/√D: ||e||²≈1, self-similarity≈1 → near-uniform initial predictions.
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(hidden_dim))

        if simple_mode:
            # SEOP Fix 52: Direct embedding → complex, bypassing Sinkhorn/SDR.
            # Re(z) = embedding (stays in embedding space for weight tying).
            # Im(z) = learned projection (additional capacity for phase dynamics).
            self.imag_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.xavier_uniform_(
                self.imag_proj.weight, gain=0.3
            )  # Small Im channel — let model learn phase magnitude
        else:
            # Cost matrix module
            self.cost = LearnedCostMatrix(hidden_dim, sdr_candidates)

            self.sinkhorn = LogSinkhorn(
                epsilon=sinkhorn_epsilon,
                max_iter=sinkhorn_max_iter,
                tol=sinkhorn_tol,
                auto_epsilon=sinkhorn_auto_epsilon,
                auto_epsilon_scale=sinkhorn_auto_epsilon_scale,
            )

            # Projection from SDR candidates back to hidden_dim
            self.sdr_to_hidden = nn.Linear(sdr_candidates, hidden_dim)

            # Positional phase encoding for complex lift
            self.register_buffer(
                "positional_phase",
                self._build_positional_phase(max_seq_length, hidden_dim),
            )

    def _build_positional_phase(self, max_len: int, dim: int) -> Tensor:
        """Build positional phase angles for complex lift.

        Uses sinusoidal frequencies so each position has a unique
        phase pattern across the hidden dimension.
        """
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 1).float() * (-math.log(10000.0) / dim)
        )
        # Phase angles in [0, 2*pi)
        phase = position * div_term  # [max_len, dim]
        return phase

    def forward(
        self, token_ids: Tensor, token_freqs: Optional[Tensor] = None
    ) -> Tensor:
        """Encode tokens to sparse complex representations.

        Args:
            token_ids: [B, S] int64 token indices
            token_freqs: Optional [B, V] or [V] EMA of token frequencies
        Returns:
            z: [B, S, D] complex64 sparse representation on Crystal Manifold
        """
        B, S = token_ids.shape

        # Step 1: Dense embedding
        x = self.embedding(token_ids)  # [B, S, D] float32

        # SEOP Fix 52: Simple mode — Re(z) = embedding, Im(z) = learned projection
        # Keeps Re in embedding space so weight-tied output projection works correctly.
        # Mamba provides implicit positional encoding via sequential scan.
        if self.simple_mode:
            x_imag = self.imag_proj(x)
            return safe_complex(x, x_imag)

        # Step 2: Compute cost matrix
        C = self.cost(x)  # [B, S, sdr_candidates]

        if token_freqs is not None:
            if token_freqs.dim() == 1:
                f = token_freqs[token_ids]
            else:
                f = torch.gather(token_freqs, 1, token_ids)

            weight = 1.0 / torch.log(f + math.e)
            C = C * weight.unsqueeze(-1)

        # Step 3: Sinkhorn optimal transport

        if self.low_vram_mode and C.device.type == "xpu":
            C_cpu = C.float().cpu()
            T = self.sinkhorn(C_cpu).to(C.device)
        else:
            T = self.sinkhorn(C)  # [B, S, sdr_candidates]

        # Step 4: SEOP Fix 44 — Soft-sparse option preserves gradient flow
        # Hard top-k zeros 87.5% of dimensions, destroying Gaussian tail information.
        # Soft-threshold (shrinkage operator) preserves partial signal from near-threshold
        # components: sign(x)·max(|x|-τ, 0). τ adapts so ~k components survive per token.
        # This is the proximal operator of the L1 norm — optimal for sparse recovery.
        #
        # SEOP Fix 44: soft_sparse=True uses temperature-scaled softmax instead:
        # T_weights = softmax(|T| / τ) keeps all dimensions active with varying weight.
        # This prevents zero gradients and preserves Gaussian tail information.
        T_abs = T.abs()

        if self.soft_sparse:
            # Soft-sparse: temperature-scaled attention weighting (all dims active)
            T_weights = torch.softmax(T_abs / self.soft_sparse_temp, dim=-1)
            T_sparse = T * T_weights
        else:
            # Hard sparse: original top-k sparsification
            topk_vals, topk_idx = torch.topk(T_abs, self.sdr_sparsity, dim=-1)
            tau = topk_vals[..., -1:]
            T_sparse = torch.sign(T) * torch.relu(T_abs - tau)
            fallback = T_sparse.abs().sum(dim=-1, keepdim=True) <= 1e-12
            if fallback.any():
                mask = torch.zeros_like(T_abs).scatter(-1, topk_idx, 1.0)
                T_sparse = torch.where(fallback, T * mask, T_sparse)

        self._last_sdr_sparse = T_sparse.detach()

        # Step 5: Project sparse transport plan to hidden dim
        sdr = self.sdr_to_hidden(T_sparse)  # [B, S, D] float32

        # Step 6: Complex lift with positional phase (SEOP Fix 1: lossless sign encoding)
        # Instead of polar(|sdr|, phase) which destroys sign (1 bit/dim lost),
        # multiply real sdr by complex positional phasor: z = sdr * exp(i*phase)
        # This preserves the sign as a π phase flip — zero information loss.
        phase = self.positional_phase[:S, :]
        cos = torch.cos(phase).unsqueeze(0)
        sin = torch.sin(phase).unsqueeze(0)

        z_real = sdr * cos
        z_imag = sdr * sin
        z = safe_complex(z_real, z_imag)

        return z  # [B, S, D] complex64
