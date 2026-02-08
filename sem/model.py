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
import torch.nn.functional as F
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

        # 1. MESH-SDR Encoder (or simple mode for direct embedding)
        self.encoder = MESHEncoder(
            vocab_size=c.model.vocab_size,
            hidden_dim=c.model.hidden_dim,
            sdr_sparsity=c.encoder.sdr_sparsity,
            sdr_candidates=c.encoder.sdr_candidates,
            sinkhorn_epsilon=c.encoder.sinkhorn_epsilon,
            sinkhorn_max_iter=c.encoder.sinkhorn_max_iter,
            sinkhorn_tol=c.encoder.sinkhorn_tol,
            sinkhorn_auto_epsilon=c.encoder.sinkhorn_auto_epsilon,
            sinkhorn_auto_epsilon_scale=c.encoder.sinkhorn_auto_epsilon_scale,
            max_seq_length=c.model.max_seq_length,
            low_vram_mode=c.training.low_vram_mode,
            soft_sparse=c.encoder.soft_sparse,
            soft_sparse_temp=c.encoder.soft_sparse_temp,
            simple_mode=c.encoder.simple_mode,
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
                    memory_horizon_ratio=c.spinor.memory_horizon_ratio,
                    max_seq_length=c.model.max_seq_length,
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
            lazy_cg=c.propagator.lazy_cg,
            lazy_cg_tol=c.propagator.lazy_cg_tol,
            direct_solve=c.propagator.direct_solve,
            pit_gamma=c.propagator.pit_gamma,
            adaptive_cg_tol=c.propagator.adaptive_cg_tol,
            cg_tol_warmup=c.propagator.cg_tol_warmup,
            cg_tol_mid=c.propagator.cg_tol_mid,
            cg_tol_late=c.propagator.cg_tol_late,
            cg_tol_warmup_end=c.propagator.cg_tol_warmup_end,
            cg_tol_mid_end=c.propagator.cg_tol_mid_end,
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

        # SEOP Fix 51: Weight tying — share embedding weights with output projection
        self.sampler.proj_real.weight = self.encoder.embedding.weight

        # SEOP: Initialize output bias to approximate Zipf distribution for faster convergence
        # This front-loads unigram knowledge so the model focuses on contextual learning
        # instead of spending thousands of steps learning token frequencies.
        with torch.no_grad():
            # Approximate Zipf: log(1/(rank+1)) for rank 0..V-1
            ranks = torch.arange(
                1, self.config.model.vocab_size + 1, dtype=torch.float32
            )
            log_freqs = -torch.log(ranks)
            log_freqs = log_freqs - log_freqs.logsumexp(
                0
            )  # normalize to valid log-probs
            self.sampler.output_bias.data.copy_(log_freqs)

        # SEOP Fix 45: Propagator warmup bypass for 3x throughput during early training
        # Set to True after warmup_steps to enable full propagation
        self.propagator_enabled = False
        self._propagator_warmup_steps = c.training.warmup_steps

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

        # 3. Propagate: Cayley-Soliton diffusion (SEOP Fix 45: skip during warmup for 3x throughput)
        if self.propagator_enabled:
            psi = self.propagator(psi)  # [B, S, D] complex64
        # else: identity pass during warmup - CG solve is expensive and not needed early

        # 4. Final norm
        psi = self.final_norm(psi)  # [B, S, D] complex64

        # Unitarity regularization / monitoring.
        # Keep the wavefunction energy per token near its expected scale.
        # We normalize by hidden_dim so the expected value is ~1.0.
        psi_energy = (psi.real * psi.real + psi.imag * psi.imag).sum(dim=-1)  # [B,S]
        psi_energy_norm = psi_energy / float(self.config.model.hidden_dim)
        # SEOP Fix 35: Clamp psi_energy_norm before log² to prevent quadratic
        # explosion when propagator amplifies ψ after warmup (gradient death at step 2180).
        # Without clamping, log(large)² grows without bound → dominates loss → clips all gradients.
        psi_energy_norm_clamped = torch.clamp(psi_energy_norm, min=1e-3, max=10.0)
        unitary_divergence = (torch.log(psi_energy_norm_clamped) ** 2).mean()

        # 5. Collapse: log-linear projection (SEOP Fix 48)
        output = self.sampler(psi, sample=not self.training)

        # Expose monitoring metric for trainers/curriculum.
        output["unitary_divergence"] = unitary_divergence

        # Compute loss if targets provided
        if targets is not None:
            # Shift for next-token prediction: predict token[t+1] from state[t]
            logits = output["logits"][:, :-1, :].contiguous()
            target_ids = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                target_ids.view(-1),
                label_smoothing=self.config.training.label_smoothing,
            )

            # Apply unitarity regularization strength from config.
            unitary_lambda = float(getattr(self.config.training, "unitary_lambda", 0.0))
            if unitary_lambda != 0.0:
                loss = loss + unitary_lambda * unitary_divergence
            output["loss"] = loss

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
        eos_token_id: Optional[int] = 1,
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

        finished = torch.zeros(
            generated.shape[0], device=generated.device, dtype=torch.bool
        )

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            input_ids = generated[:, -self.config.model.max_seq_length :]

            # Forward pass
            psi = self._encode_and_context(input_ids)
            if self.propagator_enabled:
                psi = self.propagator(psi)
            psi = self.final_norm(psi)
            output = self.sampler(
                psi,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample=True,
            )

            # Get next token (last position)
            next_token = output["tokens"][:, -1:]

            if eos_token_id is not None:
                # If some sequences already finished, keep emitting EOS for them.
                eos = torch.tensor(
                    int(eos_token_id), device=generated.device, dtype=next_token.dtype
                )
                next_token = torch.where(
                    finished.view(-1, 1),
                    eos.view(1, 1).expand_as(next_token),
                    next_token,
                )
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == int(eos_token_id))
                if bool(finished.all()):
                    break

        return generated

    def _encode_and_context(
        self, token_ids: Tensor, token_freqs: Optional[Tensor] = None
    ) -> Tensor:
        """Encode and apply context layers (helper for generation)."""
        psi = self.encoder(token_ids, token_freqs=token_freqs)
        for mamba_layer in self.mamba_layers:
            psi = mamba_layer(psi)
        return psi
