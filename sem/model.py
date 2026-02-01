"""SEM V5.5 'Lean Crystal' - Top-level Model.

Assembles the full Signal-Entropic Model:
    Encoder (MESH) -> Context (Spinors) -> Propagation (Cayley) -> Collapse (Born)

The model processes token sequences through:
1. MESH-SDR encoder: tokens -> sparse complex SDR on Crystal Manifold
2. Complex Mamba-3 layers: sequential context integration via spinor rotations
3. Cayley-Soliton propagator: unitary wave propagation through Graph Laplacian
4. Born Collapse: wavefunction -> token probabilities via |ψ|²
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .config import SEMConfig
from .encoder.mesh_sdr import MESHEncoder
from .spinor.complex_mamba3 import ComplexMamba3Layer
from .propagator.cayley_soliton import CayleySolitonStack
from .sampler.born_collapse import BornCollapseSampler
from .utils.complex_layernorm import ComplexRMSNorm


class SEMModel(nn.Module):
    """Signal-Entropic Model V5.5 'Lean Crystal'.

    Full architecture combining MESH encoding, complex Mamba-3 spinors,
    Cayley-Soliton propagation, and Born collapse sampling.
    """

    def __init__(self, config: SEMConfig):
        super().__init__()
        self.config = config
        c = config  # Shorthand

        # 1. MESH-SDR Encoder
        self.encoder = MESHEncoder(
            vocab_size=c.model.vocab_size,
            hidden_dim=c.model.hidden_dim,
            sdr_sparsity=c.encoder.sdr_sparsity,
            sdr_candidates=c.encoder.sdr_candidates,
            sinkhorn_epsilon=c.encoder.sinkhorn_epsilon,
            sinkhorn_max_iter=c.encoder.sinkhorn_max_iter,
            sinkhorn_tol=c.encoder.sinkhorn_tol,
            max_seq_length=c.model.max_seq_length,
        )

        # 2. Complex Mamba-3 Spinor Layers
        self.mamba_layers = nn.ModuleList(
            [
                ComplexMamba3Layer(
                    hidden_dim=c.model.hidden_dim,
                    state_dim=c.spinor.state_dim,
                    mimo_groups=c.spinor.mimo_groups,
                    block_size=c.spinor.block_size,
                    d_conv=c.spinor.d_conv,
                    num_layers=c.model.num_layers,
                )
                for _ in range(c.model.num_layers)
            ]
        )

        # 3. Cayley-Soliton Propagator Stack
        self.propagator = CayleySolitonStack(
            dim=c.model.hidden_dim,
            num_layers=c.model.num_layers,
            dt=c.propagator.cayley_dt,
            nonlinear_alpha=c.propagator.nonlinear_alpha,
            cg_max_iter=c.propagator.cg_max_iter,
            cg_tol=c.propagator.cg_tol,
            laplacian_sparsity=c.propagator.laplacian_sparsity,
        )

        # 4. Final normalization before collapse
        self.final_norm = ComplexRMSNorm(c.model.hidden_dim)

        # 5. Born Collapse Sampler
        self.sampler = BornCollapseSampler(
            hidden_dim=c.model.hidden_dim,
            vocab_size=c.model.vocab_size,
            temperature=c.sampler.temperature,
            top_k=c.sampler.top_k,
            top_p=c.sampler.top_p,
        )

    def forward(
        self,
        token_ids: Tensor,
        targets: Optional[Tensor] = None,
        token_freqs: Optional[Tensor] = None,
    ) -> dict:
        """Forward pass through full SEM pipeline.

        Args:
            token_ids: [B, S] input token indices
            targets: [B, S] target token indices (for training loss)
            token_freqs: [B, V] or [V] EMA of token frequencies

        Returns:
            dict with 'logits', 'log_probs', 'loss' (if targets given),
            and optionally 'tokens' during inference
        """
        # 1. Encode: tokens -> Crystal Manifold
        psi = self.encoder(token_ids, token_freqs=token_freqs)  # [B, S, D] complex64

        # 2. Context: spinor rotation layers
        for mamba_layer in self.mamba_layers:
            psi = mamba_layer(psi)  # [B, S, D] complex64

        # 3. Propagate: Cayley-Soliton diffusion
        psi = self.propagator(psi)  # [B, S, D] complex64

        # 4. Final norm
        psi = self.final_norm(psi)  # [B, S, D] complex64

        # 5. Collapse: Born rule sampling
        output = self.sampler(psi, sample=not self.training)

        # Compute loss if targets provided
        if targets is not None:
            # Shift for next-token prediction: predict token[t+1] from state[t]
            amp_sq = output["amp_sq"][:, :-1, :].contiguous()
            target_ids = targets[:, 1:].contiguous()

            target_amp_sq = torch.gather(amp_sq, -1, target_ids.unsqueeze(-1)).squeeze(
                -1
            )
            nll_term = -torch.log(target_amp_sq + 1e-12).mean()

            unitary_lambda = self.config.training.unitary_lambda
            unitary_divergence = ((amp_sq.sum(dim=-1) - 1) ** 2).mean()
            unitary_term = unitary_lambda * unitary_divergence

            loss = nll_term + unitary_term
            output["loss"] = loss
            output["unitary_divergence"] = unitary_divergence

        return output

    def count_parameters(self) -> dict:
        """Count parameters by module."""
        counts = {}
        for name, module in self.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            # Count complex params as 2x (real + imag)
            complex_count = sum(
                p.numel() * 2 if p.is_complex() else p.numel()
                for p in module.parameters()
            )
            counts[name] = {
                "total": total,
                "trainable": trainable,
                "effective_real": complex_count,
            }

        counts["total"] = {
            "total": sum(c["total"] for c in counts.values() if isinstance(c, dict)),
            "trainable": sum(
                c["trainable"] for c in counts.values() if isinstance(c, dict)
            ),
            "effective_real": sum(
                c["effective_real"] for c in counts.values() if isinstance(c, dict)
            ),
        }
        return counts

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> Tensor:
        """Autoregressive generation.

        Args:
            prompt_ids: [1, S] prompt token indices
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus filtering

        Returns:
            [1, S + max_new_tokens] generated token indices
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            input_ids = generated[:, -self.config.model.max_seq_length :]

            # Forward pass
            output = self.sampler(
                self.final_norm(self.propagator(self._encode_and_context(input_ids))),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample=True,
            )

            # Get next token (last position)
            next_token = output["tokens"][:, -1:]
            generated = torch.cat([generated, next_token], dim=1)

            # TODO: EOS detection

        return generated

    def _encode_and_context(
        self, token_ids: Tensor, token_freqs: Optional[Tensor] = None
    ) -> Tensor:
        """Encode and apply context layers (helper for generation)."""
        psi = self.encoder(token_ids, token_freqs=token_freqs)
        for mamba_layer in self.mamba_layers:
            psi = mamba_layer(psi)
        return psi
