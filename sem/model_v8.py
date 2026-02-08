"""SEM V8.0 - Grand Unified Model.

Integrates:
- Gemini theory: Lindblad dissipation, Quaternionic escape, Hybrid automata
- DeepSeek innovations: Engram (O(1) lookup), mHC (manifold-constrained residuals)
- V5.5 foundation: MESH-SDR encoder, Complex Mamba-3, Born collapse

Architecture combines multiple innovations to achieve:
1. Selective forgetting (Lindblad) - avoid catastrophic retention
2. Singularity handling (Quaternionic) - avoid NaN crashes
3. Curvature detection (Hybrid Automata) - detect phase transitions
4. Fast lookups (Engram) - O(1) N-gram memory
5. Manifold residuals (mHC) - information-preserving skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

from .config import SEMConfig
from .encoder.mesh_sdr import MESHEncoder
from .spinor.complex_mamba3 import ComplexMamba3Layer
from .spinor.lindblad import LindbladDissipation
from .spinor.hybrid_automata import HybridAutomata
from .spinor.quaternion import QuaternionicEscape
from .propagator.cayley_soliton import CayleySolitonStack
from .sampler.born_collapse import BornCollapseSampler
from .utils.complex_layernorm import ComplexRMSNorm

# Optional imports (may not exist yet)
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
        # V8.0 additions
        use_mhc: bool = False,
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
        )

        # V8.0: mHC residual (replaces standard residual)
        if self.use_mhc:
            # H_res logits for Sinkhorn projection -- [S, S] where S=1 (single-stream)
            # Sinkhorn projects onto Birkhoff Polytope; for S=1 this is trivially [[1.0]]
            # For multi-stream, increase S to 2-8 for actual mixing benefit
            self.H_res_logits = nn.Parameter(torch.zeros(1, 1))
            self.H_res_logits_imag = nn.Parameter(torch.zeros(1, 1))
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
            # Store previous H for curvature computation (as buffer for DDP compat)
            self.register_buffer("_H_prev", None)

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

        # Base Mamba processing (includes residual in V5.5)
        out = self.base_layer(x)

        # Extract branch output (remove residual to apply V8 enhancements)
        # base_layer returns: residual + scale * f(x), so f(x) = (out - residual) / scale
        scale = self.base_layer.residual_scale
        branch_out = (out - residual) / max(scale, 1e-8)

        # Apply Lindblad dissipation (selective forgetting)
        if self.use_lindblad:
            branch_out = self.lindblad(branch_out)

        # Compute effective Hamiltonian for hybrid automata/quaternionic
        if self.use_hybrid_automata or self.use_quaternionic:
            H_eff_real = self.approximate_hamiltonian(branch_out)
            # Cast to complex for standalone HybridAutomata (expects complex64 H)
            H_eff = torch.complex(H_eff_real, torch.zeros_like(H_eff_real))

        # Hybrid automata: check for curvature spikes
        if self.use_hybrid_automata:
            if self._H_prev is not None and self._H_prev.shape[0] == H_eff.shape[0]:
                branch_out, jumped = self.hybrid_automata(
                    branch_out, H_eff, self._H_prev
                )
            self._H_prev = H_eff.detach()

        # Quaternionic escape at singularities
        if self.use_quaternionic:
            branch_out, escaped = self.quaternionic(branch_out, H_eff)

        # mHC residual (Birkhoff polytope projection) or standard residual
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
            # Standard residual (fallback)
            out = residual + branch_out * self.base_layer.residual_scale

        return out


class SEMV8Model(nn.Module):
    """SEM V8.0 Grand Unified Model.

    Combines innovations from multiple sources:
    - Engram (DeepSeek): O(1) N-gram memory lookup
    - mHC (DeepSeek): Manifold-constrained residuals via Birkhoff projection
    - Lindblad (Gemini): Selective forgetting through dissipation
    - Hybrid Automata (Gemini): Curvature-based phase transitions
    - Quaternionic Escape (Gemini): Navigate around Cayley poles
    - V5.5 Foundation: MESH-SDR, Complex Mamba-3, Born Collapse

    Architecture:
        Token → Encoder → [Engram + Mamba(V8) × L] → Propagator → Sampler → Token

    Key Features:
    - Selective forgetting (Lindblad) prevents catastrophic retention
    - Singularity handling (Quaternionic) avoids NaN crashes
    - Phase transitions (Hybrid Automata) adapt to regime changes
    - Fast lookups (Engram) reduce sequential bottleneck
    - Information-preserving residuals (mHC) improve gradient flow
    """

    def __init__(self, config: SEMConfig):
        """Initialize SEM V8.0 model.

        Args:
            config: SEMConfig with model, encoder, spinor, propagator settings
        """
        super().__init__()
        self.config = config
        c = config

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
            low_vram_mode=c.training.low_vram_mode,
            soft_sparse=c.encoder.soft_sparse,
            soft_sparse_temp=c.encoder.soft_sparse_temp,
            simple_mode=c.encoder.simple_mode,
        )

        # 2. Engram (optional - O(1) N-gram lookup)
        # Configuration not yet in SEMConfig, so disabled for now
        self.use_engram = False
        if HAS_ENGRAM and hasattr(c, "engram") and c.engram.enabled:
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

        # 3. V8.0 Mamba layers (with Lindblad, Hybrid Automata, Quaternionic)
        # Read V8 config if available, else use defaults
        v8_config = getattr(c, "v8", None)
        use_lindblad = getattr(v8_config, "use_lindblad", True) if v8_config else True
        use_hybrid = (
            getattr(v8_config, "use_hybrid_automata", True) if v8_config else True
        )
        use_quat = getattr(v8_config, "use_quaternionic", True) if v8_config else True
        use_mhc = getattr(v8_config, "use_mhc", False) if v8_config else False

        self.mamba_layers = nn.ModuleList(
            [
                ComplexMamba3LayerV8(
                    hidden_dim=c.model.hidden_dim,
                    state_dim=c.spinor.state_dim,
                    mimo_groups=c.spinor.mimo_groups,
                    block_size=c.spinor.block_size,
                    d_conv=c.spinor.d_conv,
                    num_layers=c.model.num_layers,
                    # V8.0 config (use defaults if not specified)
                    use_mhc=use_mhc,
                    use_lindblad=use_lindblad,
                    lindblad_gamma=getattr(v8_config, "lindblad_gamma", 0.01)
                    if v8_config
                    else 0.01,
                    num_lindblad_ops=getattr(v8_config, "num_lindblad_ops", 4)
                    if v8_config
                    else 4,
                    use_hybrid_automata=use_hybrid,
                    curvature_threshold=getattr(v8_config, "curvature_threshold", 0.1)
                    if v8_config
                    else 0.1,
                    use_quaternionic=use_quat,
                    condition_threshold=getattr(v8_config, "condition_threshold", 100.0)
                    if v8_config
                    else 100.0,
                )
                for _ in range(c.model.num_layers)
            ]
        )

        # 4. Cayley-Soliton Propagator (optional, skip during warmup)
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
        )
        self.propagator_enabled = False

        # 5. Final norm
        self.final_norm = ComplexRMSNorm(c.model.hidden_dim)

        # 6. Born Collapse Sampler
        self.sampler = BornCollapseSampler(
            hidden_dim=c.model.hidden_dim,
            vocab_size=c.model.vocab_size,
            temperature=c.sampler.temperature,
            top_k=c.sampler.top_k,
            top_p=c.sampler.top_p,
        )

        # Weight tying (encoder embedding = sampler projection)
        self.sampler.proj_real.weight = self.encoder.embedding.weight

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
        """Forward pass through SEM V8.0.

        Pipeline:
            1. Encode tokens → complex spinors
            2. Process through Mamba layers with V8 enhancements
            3. Optional Cayley propagator (if enabled)
            4. Final normalization
            5. Born collapse to vocabulary
            6. Compute loss if targets provided

        Args:
            token_ids: [B, S] input tokens
            targets: [B, S] target tokens (for loss computation)
            token_freqs: Optional token frequency EMA for entropy-weighted encoding

        Returns:
            dict with:
                'logits': [B, S, V] vocabulary logits
                'log_probs': [B, S, V] log probabilities
                'loss': scalar (if targets provided)
                Optional: 'tokens', 'probs' (if sampling)
        """
        # 1. Encode
        psi = self.encoder(token_ids, token_freqs=token_freqs)  # [B, S, D] complex64

        # 2. Mamba layers with V8.0 enhancements
        for i, mamba_layer in enumerate(self.mamba_layers):
            # Apply Engram if configured for this layer
            if self.use_engram and str(i) in self.engram_layers:
                # Engram expects [B, L, D] real features — use magnitude as proxy
                engram_out = self.engram_layers[str(i)](psi.abs(), token_ids)
                # Add engram contribution to real part
                psi = torch.complex(psi.real + engram_out, psi.imag)

            psi = mamba_layer(psi)

        # 3. Propagator (skip during warmup)
        if self.propagator_enabled:
            psi = self.propagator(psi)

        # 4. Final norm
        psi = self.final_norm(psi)

        # Unitarity monitoring/regularization (shared with v5.5).
        psi_energy = (psi.real * psi.real + psi.imag * psi.imag).sum(dim=-1)  # [B,S]
        psi_energy_norm = psi_energy / float(self.config.model.hidden_dim)
        # SEOP Fix 35: Clamp psi_energy_norm before log² to prevent quadratic
        # explosion when propagator amplifies ψ after warmup (gradient death at step 2180).
        psi_energy_norm_clamped = torch.clamp(
            psi_energy_norm,
            min=self.config.training.unitary_clamp_min,
            max=self.config.training.unitary_clamp_max,
        )
        unitary_divergence = (torch.log(psi_energy_norm_clamped) ** 2).mean()

        # 5. Born collapse
        output = self.sampler(psi, sample=not self.training)

        output["unitary_divergence"] = unitary_divergence

        # 6. Loss computation
        if targets is not None:
            logits = output["logits"][:, :-1, :].contiguous()
            target_ids = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                target_ids.view(-1),
                label_smoothing=self.config.training.label_smoothing,
            )

            unitary_lambda = float(getattr(self.config.training, "unitary_lambda", 0.0))
            if unitary_lambda != 0.0:
                # SEOP Fix 34: Detach unitary_divergence from gradient graph.
                loss = loss + unitary_lambda * unitary_divergence.detach()

            # Add unitarity regularization from HybridAutomata layers
            for layer in self.mamba_layers:
                if layer.use_hybrid_automata:
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
        """Get diagnostic information about V8.0 components.

        Returns:
            dict with:
                'lindblad_gamma': List of dissipation strengths per layer
                'num_quaternionic_escapes': How many times escape was triggered
                'num_hybrid_jumps': How many times transitions occurred
                'engram_active': Which layers have Engram
                'mhc_enabled': Which layers use mHC
        """
        diagnostics = {
            "lindblad_gamma": [],
            "num_quaternionic_escapes": 0,
            "num_hybrid_jumps": 0,
            "engram_active": [],
            "mhc_enabled": [],
        }

        for i, layer in enumerate(self.mamba_layers):
            if layer.use_lindblad:
                diagnostics["lindblad_gamma"].append(layer.lindblad.gamma)

            if layer.use_mhc:
                diagnostics["mhc_enabled"].append(i)

        if self.use_engram:
            diagnostics["engram_active"] = list(self.engram_layers.keys())

        return diagnostics

    def count_parameters(self) -> dict:
        """Count parameters by module.

        Similar to V5.5 but includes V8.0 sub-modules
        (lindblad, hybrid_automata, quaternionic) within each mamba layer.
        """
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
            generated = torch.cat([generated, next_token], dim=1)

        return generated
