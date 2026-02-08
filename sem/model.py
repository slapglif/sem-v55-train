"""SEM - Signal-Entropic Model.

Assembles the full Signal-Entropic Model:
    Encoder (MESH) -> Context (Spinors) -> Propagation (Cayley) -> Collapse (Born)

V8 features (Lindblad, HybridAutomata, Quaternionic, Engram, mHC) are controlled
by config.v8 flags and default to enabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .config import SEMConfig
from .encoder.mesh_sdr import MESHEncoder
from .spinor.complex_mamba3 import ComplexMamba3Layer
from .spinor.lindblad import LindbladDissipation
from .spinor.hybrid_automata import HybridAutomata
from .spinor.quaternion import QuaternionicEscape
from .propagator.cayley_soliton import CayleySolitonStack
from .sampler.born_collapse import BornCollapseSampler
from .utils.complex_layernorm import ComplexRMSNorm

# Optional imports
try:
    from .engram import Engram, EngramConfig

    HAS_ENGRAM = True
except ImportError:
    HAS_ENGRAM = False

try:
    from .hyper_connections import mhc_residual

    HAS_MHC = True
except ImportError:
    HAS_MHC = False


class ComplexMamba3LayerV8(nn.Module):
    """V8.0 Mamba layer with Lindblad, Quaternionic escape, and optional mHC.

    Enhancements over V5.5:
    1. Lindblad dissipation for selective forgetting
    2. Hybrid automata for phase transition detection
    3. Quaternionic escape for singularity handling
    4. Optional mHC residual (if available)
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 64,
        mimo_groups: int = 8,
        block_size: int = 8,
        d_conv: int = 4,
        num_layers: int = 1,
        memory_horizon_ratio: float = 0.0,
        max_seq_length: int = 2048,
        # V8.0 additions
        use_mhc: bool = True,
        mhc_num_iters: int = 10,
        mhc_tau: float = 0.05,
        use_lindblad: bool = True,
        lindblad_gamma: float = 0.01,
        num_lindblad_ops: int = 4,
        use_hybrid_automata: bool = True,
        curvature_threshold: float = 0.1,
        use_quaternionic: bool = True,
        condition_threshold: float = 100.0,
    ):
        """Initialize V8.0 Mamba layer.

        Args:
            hidden_dim: Model hidden dimension
            state_dim: SSM state dimension
            mimo_groups: Number of MIMO groups
            block_size: Spinor block size
            d_conv: Convolution kernel size
            num_layers: Total number of layers (for residual scaling)
            memory_horizon_ratio: τ = ratio * max_seq_length; 0 = default init
            max_seq_length: Maximum sequence length for horizon scaling
            use_mhc: Use manifold-constrained residuals (requires hyper_connections module)
            mhc_num_iters: Sinkhorn iterations for Birkhoff projection
            mhc_tau: Temperature for Sinkhorn
            use_lindblad: Enable Lindblad dissipation
            lindblad_gamma: Dissipation strength
            num_lindblad_ops: Number of Lindblad operators
            use_hybrid_automata: Enable curvature-based transitions
            curvature_threshold: K threshold for transitions
            use_quaternionic: Enable quaternionic singularity escape
            condition_threshold: Condition number threshold for singularities
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_mhc = use_mhc and HAS_MHC
        self.use_lindblad = use_lindblad
        self.use_hybrid_automata = use_hybrid_automata
        self.use_quaternionic = use_quaternionic

        # Base Mamba layer (from V5.5)
        self.base_layer = ComplexMamba3Layer(
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            mimo_groups=mimo_groups,
            block_size=block_size,
            d_conv=d_conv,
            num_layers=num_layers,
            memory_horizon_ratio=memory_horizon_ratio,
            max_seq_length=max_seq_length,
        )

        # V8.0: mHC residual (replaces standard residual)
        if self.use_mhc:
            # For S=1, Sinkhorn is trivially [[1.0]] — use non-learnable buffers.
            # For S>1, use nn.Parameter for actual stream mixing.
            mhc_streams = 1
            if mhc_streams == 1:
                self.register_buffer("H_res_logits", torch.zeros(1, 1))
                self.register_buffer("H_res_logits_imag", torch.zeros(1, 1))
            else:
                self.H_res_logits = nn.Parameter(torch.zeros(mhc_streams, mhc_streams))
                self.H_res_logits_imag = nn.Parameter(
                    torch.zeros(mhc_streams, mhc_streams)
                )
            self.mhc_num_iters = mhc_num_iters
            self.mhc_tau = mhc_tau

        # V8.0: Lindblad dissipation
        if use_lindblad:
            self.lindblad = LindbladDissipation(
                dim=hidden_dim,
                num_lindblad_ops=num_lindblad_ops,
                gamma=lindblad_gamma,
            )

        # V8.0: Hybrid automata (singularity detection via Lie bracket curvature)
        if use_hybrid_automata:
            self.hybrid_automata = HybridAutomata(
                dim=hidden_dim,
                curvature_threshold=curvature_threshold,
                learnable_threshold=True,
            )
            # Use plain attribute (not register_buffer) to avoid gradient checkpointing
            # tensor metadata mismatch — _H_prev mutates shape during forward.
            self._H_prev = None

        # V8.0: Quaternionic escape
        if use_quaternionic:
            self.quaternionic = QuaternionicEscape(
                dim=hidden_dim,
                condition_threshold=condition_threshold,
            )

    def approximate_hamiltonian(self, psi: Tensor) -> Tensor:
        """Approximate effective Hamiltonian from state magnitude pattern.

        H_eff is used for:
        1. Hybrid automata curvature computation
        2. Quaternionic singularity detection

        Approximation: H ≈ (1/N) Σ_s |ψ_s⟩⟨ψ_s| (outer product density)

        Args:
            psi: [B, S, D] complex state
        Returns:
            H: [B, D, D] approximate Hamiltonian
        """
        # Use magnitude pattern as proxy for energy density
        mag = psi.abs()  # [B, S, D]

        # Compute density matrix approximation
        # H ≈ (1/S) Σ_s mag_s ⊗ mag_s
        H = torch.einsum("bsd,bse->bde", mag, mag) / mag.shape[1]  # [B, D, D]

        return H

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with V8.0 enhancements.

        Pipeline:
        1. Base Mamba processing
        2. Lindblad dissipation (selective forgetting)
        3. Hybrid automata (curvature-based transitions)
        4. Quaternionic escape (singularity handling)
        5. mHC or standard residual

        Args:
            x: [B, S, D] complex64
        Returns:
            [B, S, D] complex64
        """
        residual = x
        out = self.base_layer(x)

        # Extract branch = f(x) from base_layer output (residual + scale * f(x))
        scale = self.base_layer.residual_scale
        branch_out = (out - residual) / max(scale, 1e-8)

        if self.use_lindblad:
            branch_out = self.lindblad(branch_out)

        if self.use_hybrid_automata or self.use_quaternionic:
            with torch.no_grad():
                H_eff_real = self.approximate_hamiltonian(branch_out)
                H_eff = torch.complex(H_eff_real, torch.zeros_like(H_eff_real))

        if self.use_hybrid_automata:
            if self._H_prev is not None and self._H_prev.shape[0] == H_eff.shape[0]:
                branch_out, jumped = self.hybrid_automata(
                    branch_out, H_eff, self._H_prev
                )
            self._H_prev = H_eff.detach()

        if self.use_quaternionic:
            branch_out, escaped = self.quaternionic(branch_out, H_eff)

        # Re-scale branch back before residual combination to avoid gradient explosion
        branch_out = branch_out * scale

        if self.use_mhc:
            out = mhc_residual(
                residual,
                branch_out,
                self.H_res_logits,
                H_res_logits_imag=self.H_res_logits_imag,
                mhc_num_iters=self.mhc_num_iters,
                mhc_tau=self.mhc_tau,
            )
        else:
            out = residual + branch_out

        return out


class SEMModel(nn.Module):
    """Signal-Entropic Model.

    Full architecture combining MESH encoding, complex Mamba-3 spinors,
    Cayley-Soliton propagation, and Born collapse sampling.

    V8 features (Lindblad, HybridAutomata, Quaternionic, Engram, mHC) are
    activated when config.v8 flags are True.
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

        # Determine if V8 features are needed
        v8 = getattr(c, "v8", None)
        self._use_v8_layers = any(
            [
                getattr(v8, "use_lindblad", False) if v8 else False,
                getattr(v8, "use_hybrid_automata", False) if v8 else False,
                getattr(v8, "use_quaternionic", False) if v8 else False,
                getattr(v8, "use_mhc", False) if v8 else False,
            ]
        )

        # 2. Engram (optional - O(1) N-gram lookup)
        self.use_engram = False
        if HAS_ENGRAM and hasattr(c, "engram") and getattr(c.engram, "enabled", False):
            self.use_engram = True
            engram_config = EngramConfig(
                tokenizer=getattr(c.engram, "tokenizer", None),
                max_ngram_size=getattr(c.engram, "max_ngram_size", 3),
                layer_ids=list(c.engram.layer_ids),
            )
            self.engram_layers = nn.ModuleDict()
            for layer_id in engram_config.layer_ids:
                self.engram_layers[str(layer_id)] = Engram(
                    layer_id=layer_id,
                    hidden_size=c.model.hidden_dim,
                    config=engram_config,
                )

        # 3. Complex Mamba-3 Spinor Layers (V8 wrapper or plain V5.5)
        if self._use_v8_layers:
            use_lindblad = getattr(v8, "use_lindblad", False)
            use_hybrid = getattr(v8, "use_hybrid_automata", False)
            use_quat = getattr(v8, "use_quaternionic", False)
            use_mhc = getattr(v8, "use_mhc", False)

            self.mamba_layers = nn.ModuleList(
                [
                    ComplexMamba3LayerV8(
                        hidden_dim=c.model.hidden_dim,
                        state_dim=c.spinor.state_dim,
                        mimo_groups=c.spinor.mimo_groups,
                        block_size=c.spinor.block_size,
                        d_conv=c.spinor.d_conv,
                        num_layers=c.model.num_layers,
                        memory_horizon_ratio=c.spinor.memory_horizon_ratio,
                        max_seq_length=c.model.max_seq_length,
                        # V8.0 config
                        use_mhc=use_mhc,
                        use_lindblad=use_lindblad,
                        lindblad_gamma=getattr(v8, "lindblad_gamma", 0.01),
                        num_lindblad_ops=getattr(v8, "num_lindblad_ops", 4),
                        use_hybrid_automata=use_hybrid,
                        curvature_threshold=getattr(v8, "curvature_threshold", 0.1),
                        use_quaternionic=use_quat,
                        condition_threshold=getattr(v8, "condition_threshold", 100.0),
                    )
                    for _ in range(c.model.num_layers)
                ]
            )
        else:
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

        # 4. Cayley-Soliton Propagator Stack
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

        # 5. Final normalization before collapse
        self.final_norm = ComplexRMSNorm(c.model.hidden_dim)

        # 6. Born Collapse Sampler
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

    def enable_propagator(self, enable: bool = True):
        """Enable or disable Cayley-Soliton propagator.

        Typically disabled during warmup to avoid CG convergence issues,
        then enabled once the model learns reasonable representations.

        Args:
            enable: If True, propagator is applied. If False, skip.
        """
        self.propagator_enabled = enable

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

        # 2. Context: spinor rotation layers (with optional Engram)
        for i, mamba_layer in enumerate(self.mamba_layers):
            # Apply Engram if configured for this layer
            if self.use_engram and str(i) in self.engram_layers:
                # Engram expects [B, L, D] real features — use magnitude as proxy
                engram_out = self.engram_layers[str(i)](psi.abs(), token_ids)
                # Add engram contribution to real part
                psi = torch.complex(psi.real + engram_out, psi.imag)

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
        psi_energy_norm_clamped = torch.clamp(
            psi_energy_norm,
            min=self.config.training.unitary_clamp_min,
            max=self.config.training.unitary_clamp_max,
        )
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
                # SEOP Fix 34: Detach unitary_divergence from gradient graph.
                # Unitarity is enforced structurally by Cayley transform;
                # loss term is purely a monitoring/regularization signal.
                loss = loss + unitary_lambda * unitary_divergence.detach()

            # V8: Add unitarity regularization from HybridAutomata layers
            if self._use_v8_layers:
                for layer in self.mamba_layers:
                    if (
                        hasattr(layer, "use_hybrid_automata")
                        and layer.use_hybrid_automata
                    ):
                        if unitary_lambda != 0.0:
                            loss = (
                                loss
                                + unitary_lambda
                                * layer.hybrid_automata.compute_unitarity_loss()
                            )
                        else:
                            loss = loss + layer.hybrid_automata.compute_unitarity_loss()

            output["loss"] = loss

        return output

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about V8 components.

        Returns:
            dict with:
                'lindblad_gamma': List of dissipation strengths per layer
                'num_quaternionic_escapes': How many times escape was triggered
                'num_hybrid_jumps': How many times transitions occurred
                'engram_active': Which layers have Engram
                'mhc_enabled': Which layers use mHC
        """
        if not self._use_v8_layers and not self.use_engram:
            return {}

        diagnostics = {
            "lindblad_gamma": [],
            "num_quaternionic_escapes": 0,
            "num_hybrid_jumps": 0,
            "engram_active": [],
            "mhc_enabled": [],
        }

        if self._use_v8_layers:
            for i, layer in enumerate(self.mamba_layers):
                if hasattr(layer, "use_lindblad") and layer.use_lindblad:
                    diagnostics["lindblad_gamma"].append(layer.lindblad.gamma)

                if hasattr(layer, "use_mhc") and layer.use_mhc:
                    diagnostics["mhc_enabled"].append(i)

        if self.use_engram:
            diagnostics["engram_active"] = list(self.engram_layers.keys())

        return diagnostics

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
        for i, mamba_layer in enumerate(self.mamba_layers):
            if self.use_engram and str(i) in self.engram_layers:
                engram_out = self.engram_layers[str(i)](psi.abs(), token_ids)
                psi = torch.complex(psi.real + engram_out, psi.imag)
            psi = mamba_layer(psi)
        return psi
