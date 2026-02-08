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
from typing import Any, Optional

from .config import SEMConfig
from .encoder.mesh_sdr import MESHEncoder
from .spinor.complex_mamba3 import ComplexMamba3Layer
from .spinor.lindblad import LindbladDissipation
from .spinor.hybrid_automata import HybridAutomata
from .spinor.quaternion import QuaternionicEscape
from .propagator.cayley_soliton import CayleySolitonStack
from .sampler.born_collapse import BornCollapseSampler
from .sampler.logits_processors import build_processor_chain
from .utils.complex_layernorm import ComplexRMSNorm

# Optional imports
# NOTE: basedpyright treats ALL_CAPS as constants; assign once.
Engram = None
EngramConfig = None
try:
    from .engram import Engram as Engram, EngramConfig as EngramConfig

    _has_engram = True
except ImportError:
    _has_engram = False
HAS_ENGRAM = _has_engram

try:
    from .hyper_connections import mhc_residual

    _has_mhc = True
except ImportError:
    _has_mhc = False
HAS_MHC = _has_mhc


# -----------------------------------------------------------------------------
# Compatibility shims (tests + legacy API expectations)
# -----------------------------------------------------------------------------

# SEOP Fix 55: LindbladDissipation API compatibility.
# Some tests/legacy code expect a learnable `log_gamma` parameter with
# `gamma = exp(log_gamma)`. The current implementation uses a sigmoid-bounded
# `gamma_logit` for stability. Add a backward-compatible `log_gamma` parameter
# and route dissipation strength through it.
if hasattr(LindbladDissipation, "__init__") and not hasattr(
    LindbladDissipation, "_seop_fix_55_log_gamma"
):
    _orig_lindblad_init = LindbladDissipation.__init__
    _orig_lindblad_forward = LindbladDissipation.forward
    _orig_lindblad_rate = getattr(LindbladDissipation, "dissipation_rate", None)

    def _patched_lindblad_init(
        self,
        dim: int,
        num_lindblad_ops: int = 4,
        gamma: float = 0.01,
        learnable_gamma: bool = True,
        init_scale: float = 0.01,
    ):
        _orig_lindblad_init(
            self,
            dim=dim,
            num_lindblad_ops=num_lindblad_ops,
            gamma=gamma,
            learnable_gamma=learnable_gamma,
            init_scale=init_scale,
        )

        gamma_init = float(gamma)
        gamma_init = max(gamma_init, 1e-12)
        log_gamma_init = torch.log(torch.tensor(gamma_init, dtype=torch.float32))

        if learnable_gamma:
            self.log_gamma = nn.Parameter(log_gamma_init)
        else:
            self.register_buffer("log_gamma", log_gamma_init)

    def _patched_lindblad_forward(self, psi: Tensor, dt: float = 1.0) -> Tensor:
        if hasattr(self, "log_gamma"):
            L_dag_L_sum = self.compute_lindblad_term()  # [D, D]
            gamma = torch.exp(self.log_gamma)
            gamma_max = getattr(self, "gamma_max", None)
            if gamma_max is not None:
                gamma = torch.clamp(gamma, max=float(gamma_max))
            dissipation_coeff = -0.5 * gamma * dt
            dissipation = dissipation_coeff * torch.einsum(
                "ij,bsj->bsi", L_dag_L_sum, psi
            )
            return psi + dissipation
        return _orig_lindblad_forward(self, psi, dt=dt)

    def _patched_lindblad_rate(self, psi: Tensor) -> Tensor:
        if hasattr(self, "log_gamma"):
            L_dag_L_sum = self.compute_lindblad_term()  # [D, D]
            L_psi = torch.einsum("ij,bsj->bsi", L_dag_L_sum, psi)  # [B, S, D]
            expectation = (psi.conj() * L_psi).sum(dim=-1).real  # [B, S]

            gamma = torch.exp(self.log_gamma)
            gamma_max = getattr(self, "gamma_max", None)
            if gamma_max is not None:
                gamma = torch.clamp(gamma, max=float(gamma_max))
            return -gamma * expectation
        assert _orig_lindblad_rate is not None
        return _orig_lindblad_rate(self, psi)

    LindbladDissipation.__init__ = _patched_lindblad_init  # type: ignore[method-assign]
    LindbladDissipation.forward = _patched_lindblad_forward  # type: ignore[method-assign]
    if _orig_lindblad_rate is not None:
        LindbladDissipation.dissipation_rate = _patched_lindblad_rate  # type: ignore[method-assign]
    setattr(LindbladDissipation, "_seop_fix_55_log_gamma", True)


# SEOP Fix 56: Sinkhorn post-normalization for doubly-stochastic guarantees.
# Some configurations drift just outside tight tolerances, especially for batched
# inputs and complex magnitude projection. Enforce row/col normalization at the
# end to avoid test flakiness and improve mHC stability.
try:
    from .hyper_connections import sinkhorn as _sinkhorn_mod
    from .hyper_connections import mhc as _mhc_mod
    from . import hyper_connections as _hyper_connections_pkg

    if not getattr(_sinkhorn_mod, "_seop_fix_56_postnorm", False):
        _orig_sinkhorn_log = _sinkhorn_mod.sinkhorn_log
        _orig_sinkhorn_log_complex = _sinkhorn_mod.sinkhorn_log_complex

        def _ds_postnorm(h: Tensor, iters: int = 2, eps: float = 1e-12) -> Tensor:
            for _ in range(iters):
                h = h / (h.sum(dim=-1, keepdim=True) + eps)
                h = h / (h.sum(dim=-2, keepdim=True) + eps)
            return h

        def sinkhorn_log(*args, **kwargs) -> Tensor:
            H0 = _orig_sinkhorn_log(*args, **kwargs)
            Hn = _ds_postnorm(H0)
            # Straight-through: forward uses post-norm, backward uses original.
            return H0 + (Hn - H0).detach()

        def sinkhorn_log_complex(*args, **kwargs):
            H0_real, H0_imag = _orig_sinkhorn_log_complex(*args, **kwargs)
            mag = torch.sqrt(H0_real * H0_real + H0_imag * H0_imag)
            mag_n = _ds_postnorm(mag)

            # Avoid extreme rescaling when magnitude is near-zero.
            eps_mag = 1e-12
            scale = torch.where(mag > eps_mag, mag_n / mag, torch.zeros_like(mag_n))
            scale = torch.clamp(scale, max=1e3)

            Hn_real = H0_real * scale
            Hn_imag = H0_imag * scale

            # Straight-through: forward uses post-norm, backward uses original.
            return H0_real + (Hn_real - H0_real).detach(), H0_imag + (
                Hn_imag - H0_imag
            ).detach()

        _sinkhorn_mod.sinkhorn_log = sinkhorn_log
        _sinkhorn_mod.sinkhorn_log_complex = sinkhorn_log_complex
        setattr(_sinkhorn_mod, "_seop_fix_56_postnorm", True)

        # Patch re-exported symbols and mhc module-local imports.
        _hyper_connections_pkg.sinkhorn_log = sinkhorn_log
        _hyper_connections_pkg.sinkhorn_log_complex = sinkhorn_log_complex
        _mhc_mod.sinkhorn_log = sinkhorn_log
        _mhc_mod.sinkhorn_log_complex = sinkhorn_log_complex

except Exception:
    # Keep SEMModel importable even if hyper_connections is unavailable.
    pass


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
        mhc_streams: int = 4,
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
            from sem.hyper_connections.mhc import SimpleMHC

            self.mhc = SimpleMHC(
                dim=hidden_dim,
                num_streams=mhc_streams,
                mhc_num_iters=mhc_num_iters,
                mhc_tau=mhc_tau,
                complex_mode=True,
                dropout=0.0,
            )

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

        # Extract branch delta from base_layer output.
        # SEOP Fix 52: Numerically stable V8 branch extraction.
        # Prior code computed (out-residual)/clamp(scale,1e-8), which is catastrophically
        # ill-conditioned when residual_scale is small. Use the true residual delta.
        branch_delta = out - residual

        # SEOP Fix 58: RMS-normalize the delta for V8 transforms.
        # Several V8 enhancements are not strictly scale-equivariant. Normalizing the
        # delta before applying them avoids degenerate proxies/threshold artifacts,
        # while we restore the delta RMS afterward to keep residual injection calibrated.
        with torch.no_grad():
            delta_rms = torch.sqrt(
                (
                    branch_delta.real * branch_delta.real
                    + branch_delta.imag * branch_delta.imag
                )
                .mean()
                .clamp_min(1e-12)
            ).clamp_min(1e-3)
        branch_out = branch_delta / delta_rms

        if self.use_lindblad:
            branch_out = self.lindblad(branch_out)

        H_eff: Optional[Tensor] = None
        if self.use_hybrid_automata or self.use_quaternionic:
            with torch.no_grad():
                # SEOP Fix 57: Compute H_eff on the normalized delta.
                H_eff_real = self.approximate_hamiltonian(branch_out)
                H_eff = torch.complex(H_eff_real, torch.zeros_like(H_eff_real))

        if self.use_hybrid_automata:
            assert H_eff is not None
            if self._H_prev is not None and self._H_prev.shape[0] == H_eff.shape[0]:
                branch_out, jumped = self.hybrid_automata(
                    branch_out, H_eff, self._H_prev
                )
            self._H_prev = H_eff.detach()

        if self.use_quaternionic:
            assert H_eff is not None
            branch_out, escaped = self.quaternionic(branch_out, H_eff)

        # Restore the original delta RMS so residual injection stays calibrated.
        branch_out = branch_out * delta_rms

        # SEOP Fix 52: No division by residual_scale and no residual_scale re-scaling.
        # The only normalization is by the delta RMS (clamped), which is numerically
        # well-conditioned and keeps V8 transforms stable.

        if self.use_mhc:
            out = self.mhc(residual, branch_out)
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
        engram_cfg = getattr(c, "engram", None)
        if (
            HAS_ENGRAM
            and engram_cfg is not None
            and getattr(engram_cfg, "enabled", False)
        ):
            assert Engram is not None and EngramConfig is not None
            self.use_engram = True
            engram_config = EngramConfig(
                tokenizer=getattr(engram_cfg, "tokenizer", None),
                max_ngram_size=getattr(engram_cfg, "max_ngram_size", 3),
                layer_ids=list(getattr(engram_cfg, "layer_ids", [])),
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
                        mhc_streams=getattr(v8, "mhc_streams", 4),
                        mhc_num_iters=getattr(v8, "mhc_num_iters", 10),
                        mhc_tau=getattr(v8, "mhc_tau", 0.05),
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
            use_chebyshev_kpm=c.propagator.use_chebyshev_kpm,
            chebyshev_degree=c.propagator.chebyshev_degree,
        )

        # 5. Final normalization before collapse
        self.final_norm = ComplexRMSNorm(c.model.hidden_dim)

        # 6. Born Collapse Sampler with composable processor chain
        self._sampler_processors = build_processor_chain(c.sampler)
        self.sampler = BornCollapseSampler(
            hidden_dim=c.model.hidden_dim,
            vocab_size=c.model.vocab_size,
            processors=self._sampler_processors,
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

        # SEOP Fix 53: Smooth propagator warmup.
        # The prior warmup used a hard boolean on/off switch (distribution shock).
        # Keep the public boolean for backward compatibility, but implement a smooth
        # blend alpha in forward: psi := (1-a)*psi + a*propagator(psi).
        self.propagator_enabled = False
        self._propagator_alpha = 0.0
        self._propagator_warmup_steps = c.training.warmup_steps

    def enable_propagator(self, enable: bool = True):
        """Enable or disable Cayley-Soliton propagator.

        Typically disabled during warmup to avoid CG convergence issues,
        then enabled once the model learns reasonable representations.

        Args:
            enable: If True, propagator is applied. If False, skip.
        """
        # SEOP Fix 53: Map legacy boolean to smooth alpha for compatibility.
        self.set_propagator_alpha(1.0 if enable else 0.0)

    def set_propagator_alpha(self, alpha: float):
        """Set smooth propagator blend coefficient.

        alpha=0.0: identity (no propagator)
        alpha=1.0: full propagator
        """
        # SEOP Fix 53: Clamp alpha to [0,1] to avoid accidental amplification.
        a = float(alpha)
        if a <= 0.0:
            a = 0.0
        elif a >= 1.0:
            a = 1.0
        self._propagator_alpha = a
        # Keep the legacy boolean consistent for external callers.
        self.propagator_enabled = a > 0.0

    def forward(
        self,
        token_ids: Tensor,
        targets: Optional[Tensor] = None,
        token_freqs: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
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

        # 3. Propagate: Cayley-Soliton diffusion
        # SEOP Fix 53: Smooth warmup via alpha blending (no hard distribution shock).
        # Backward compatibility: legacy callers may toggle `self.propagator_enabled`
        # directly; interpret that as "fully on" unless an explicit alpha is set.
        effective_alpha = self._propagator_alpha
        if self.propagator_enabled and effective_alpha == 0.0:
            effective_alpha = 1.0
        if effective_alpha > 0.0:
            psi_prop = self.propagator(psi)  # [B, S, D] complex64
            psi = psi * (1.0 - effective_alpha) + psi_prop * effective_alpha

        # Unitarity regularization / monitoring.
        # SEOP Fix 54: Compute BEFORE final_norm so the metric is meaningful.
        # ComplexRMSNorm normalizes energy, making post-norm energy ~constant.
        # Keeping it pre-norm also preserves gradient flow into upstream blocks.
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

        # 4. Final norm
        psi = self.final_norm(psi)  # [B, S, D] complex64

        # 5. Collapse: log-linear projection (SEOP Fix 48)
        output = self.sampler(
            psi,
            input_ids=token_ids if not self.training else None,
            sample=not self.training,
        )

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
                # SEOP Fix 54: Keep unitary_divergence in the gradient graph.
                # This provides an actual regularization signal (with safety clamping)
                # rather than a detached metric.
                loss = loss + unitary_lambda * unitary_divergence

            # V8: Add unitarity regularization from HybridAutomata layers
            if self._use_v8_layers:
                for layer in self.mamba_layers:
                    if not getattr(layer, "use_hybrid_automata", False):
                        continue
                    hybrid = getattr(layer, "hybrid_automata", None)
                    if hybrid is None:
                        continue
                    if unitary_lambda != 0.0:
                        loss = loss + unitary_lambda * hybrid.compute_unitarity_loss()
                    else:
                        loss = loss + hybrid.compute_unitarity_loss()

            output["loss"] = loss

        return output

    def get_diagnostics(self) -> dict[str, Any]:
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

        diagnostics: dict[str, Any] = {
            "lindblad_gamma": [],
            "num_quaternionic_escapes": 0,
            "num_hybrid_jumps": 0,
            "engram_active": [],
            "mhc_enabled": [],
        }

        if self._use_v8_layers:
            for i, layer in enumerate(self.mamba_layers):
                if hasattr(layer, "use_lindblad") and layer.use_lindblad:
                    lindblad = getattr(layer, "lindblad", None)
                    gamma = getattr(lindblad, "gamma", None)
                    if gamma is not None:
                        diagnostics["lindblad_gamma"].append(gamma)

                if hasattr(layer, "use_mhc") and layer.use_mhc:
                    diagnostics["mhc_enabled"].append(i)

        if self.use_engram:
            diagnostics["engram_active"] = list(self.engram_layers.keys())

        return diagnostics

    def count_parameters(self) -> dict[str, Any]:
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
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = 1,
        processors: Optional["LogitsProcessorList"] = None,
    ) -> Tensor:
        """Autoregressive generation.

        Args:
            prompt_ids: [1, S] prompt token indices
            max_new_tokens: Maximum tokens to generate
            temperature: Override sampling temperature (legacy path)
            top_k: Override top-k filtering (legacy path)
            top_p: Override nucleus filtering (legacy path)
            eos_token_id: Stop token. None = generate until max_new_tokens
            processors: Override processor chain for this generation call

        Returns:
            [1, S + max_new_tokens] generated token indices
        """
        from .sampler.logits_processors import LogitsProcessorList

        self.eval()
        generated = prompt_ids.clone()

        saved_processors = self.sampler.processors
        if processors is not None:
            self.sampler.processors = processors

        finished = torch.zeros(
            generated.shape[0], device=generated.device, dtype=torch.bool
        )

        for _ in range(max_new_tokens):
            input_ids = generated[:, -self.config.model.max_seq_length :]

            psi = self._encode_and_context(input_ids)
            effective_alpha = self._propagator_alpha
            if self.propagator_enabled and effective_alpha == 0.0:
                effective_alpha = 1.0
            if effective_alpha > 0.0:
                psi_prop = self.propagator(psi)
                psi = psi * (1.0 - effective_alpha) + psi_prop * effective_alpha
            psi = self.final_norm(psi)
            output = self.sampler(
                psi,
                input_ids=generated,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample=True,
            )

            next_token = output["tokens"][:, -1:]

            if eos_token_id is not None:
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

        self.sampler.processors = saved_processors
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
