"""Cayley-Soliton Propagator for unitary wave evolution.

Two-step propagator that preserves information (norm) exactly:

Step 1: Nonlinear Phase Rotation (pointwise, O(D))
    psi_rot = psi * exp(i * alpha * |psi|^2)
    This is a Kerr-like self-phase modulation. The phase of each
    component rotates proportionally to its own intensity.

Step 2: Sparse Cayley Diffusion (O(sparsity * D) via CG)
    Solve: (I + i*dt/2 * H) psi_out = (I - i*dt/2 * H) psi_rot
    This is the Crank-Nicolson scheme applied to the Schrodinger
    equation i*dpsi/dt = H*psi, using the Cayley transform to
    guarantee unitarity.

The "soliton" is a self-sustaining wave packet that maintains its
form through the balance of nonlinear self-focusing (Step 1) and
linear diffusion (Step 2).
"""

import time
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

from .hamiltonian import MultiScaleHamiltonian
from . import cg_solver as _cg_mod
from .cg_solver import cg_solve, cg_solve_sparse
from .unitarity_check import check_unitarity
from ..utils.complex_ops import safe_complex
from sem.spinor.complex_ops import complex_mul_real
from .chebyshev_kpm import (
    chebyshev_kpm_solve,
    estimate_spectral_radius,
    resolvent_chebyshev_coeffs,
)


class CayleySolitonPropagator(nn.Module):
    """Single Cayley-Soliton propagation step.

    Combines nonlinear phase rotation with unitary Cayley diffusion.
    """

    def __init__(
        self,
        dim: int,
        dt: float = 0.1,
        nonlinear_alpha: float = 0.1,
        cg_max_iter: int = 20,
        cg_tol: float = 1e-6,
        laplacian_sparsity: int = 5,
        num_scales: int = 3,
        check_unitarity_flag: bool = True,
        lazy_cg: bool = True,
        lazy_cg_tol: float = 1e-6,
        direct_solve: bool = False,
        pit_gamma: float = 1.0,
        adaptive_cg_tol: bool = False,
        cg_tol_warmup: float = 1e-4,
        cg_tol_mid: float = 1e-5,
        cg_tol_late: float = 1e-6,
        cg_tol_warmup_end: int = 2000,
        cg_tol_mid_end: int = 50000,
        use_chebyshev_kpm: bool = False,
        chebyshev_degree: int = 12,
    ):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.adaptive_cg_tol = adaptive_cg_tol
        self._cg_tol_schedule = (
            (cg_tol_warmup_end, cg_tol_warmup),
            (cg_tol_mid_end, cg_tol_mid),
            (float("inf"), cg_tol_late),
        )
        self._training_step: int = 0
        self.check_unitarity_flag = check_unitarity_flag
        # Warm-start cache for faster CG convergence
        self._psi_cache = None
        self.lazy_cg = lazy_cg
        self.lazy_cg_tol = lazy_cg_tol
        self.direct_solve = direct_solve
        self.pit_gamma = pit_gamma
        self.use_chebyshev_kpm = use_chebyshev_kpm
        self.chebyshev_degree = chebyshev_degree
        # CG skip statistics for monitoring
        self._cg_skip_count: int = 0
        self._cg_total_count: int = 0
        self.timing_enabled: bool = False
        self._timing_stats: dict[str, Any] = {}

        # SEOP Fix 9: Per-dimension Kerr coefficient
        # Scalar α applies uniform nonlinear coupling to all feature channels.
        # Different dimensions encode different features with different magnitude
        # distributions — each channel needs its own coupling strength.
        # Vector α ∈ ℝ^D lets each dimension learn its optimal rotation rate.
        self.alpha = nn.Parameter(torch.full((dim,), nonlinear_alpha))

        # Multi-scale Hamiltonian (Graph Laplacian Pyramid)
        self.hamiltonian = MultiScaleHamiltonian(
            dim, num_scales=num_scales, base_sparsity=laplacian_sparsity
        )

        if self.use_chebyshev_kpm:
            # Compute initial λ_max from Gershgorin bound on initial weights
            self.hamiltonian.cache_weights()
            _lmax = estimate_spectral_radius(self.hamiltonian)
            self.hamiltonian.clear_cache()
            _lmax = max(_lmax, 1e-8)
            # Initialize Chebyshev coefficients from analytical resolvent
            init_coeffs = resolvent_chebyshev_coeffs(
                alpha=dt / 2.0,
                lambda_max=_lmax,
                degree=chebyshev_degree,
            )
            self.cheby_coeffs = nn.Parameter(init_coeffs)
            self.register_buffer("_cheby_lambda_max", torch.tensor(_lmax))

    def get_effective_cg_tol(self) -> float:
        """Return CG tolerance for the current training step."""
        if not self.adaptive_cg_tol:
            return self.cg_tol
        for step_boundary, tol in self._cg_tol_schedule:
            if self._training_step < step_boundary:
                return tol
        return self.cg_tol

    def set_training_step(self, step: int) -> None:
        self._training_step = step

    def forward(self, psi: Tensor) -> Tensor:
        """Propagate wavefunction through one Cayley-Soliton step.

        Args:
            psi: [B, S, D] complex64 input wavefunction
        Returns:
            psi_out: [B, S, D] complex64 propagated wavefunction
        """
        psi = psi.to(torch.complex64) if psi.dtype != torch.complex64 else psi
        # SEOP Fix 27: Disable AMP autocast for entire Cayley step.
        # CG solver, phase rotation (cos/sin), and Hamiltonian matvecs all need float32.
        with torch.autocast(device_type=psi.device.type, enabled=False):
            _t = self.timing_enabled
            t0 = t_cache = t_rhs = t_gate = t_cg = 0.0
            if _t:
                t0 = time.perf_counter()

            psi_real_block = torch.view_as_real(psi)
            psi_r, psi_i = psi_real_block.unbind(-1)

            # Step 1: Nonlinear phase rotation (Kerr effect)
            intensity = psi_r * psi_r + psi_i * psi_i
            intensity = intensity / intensity.mean(dim=-1, keepdim=True).clamp(
                min=1e-6
            )  # SEOP Fix 78: Safe normalization when mean near-zero
            # PIT Transform: Normalize phase distribution to prevent bunching
            phase_shift = torch.pi * (
                1.0 - 2.0 * torch.exp(-self.pit_gamma * intensity)
            )

            cos_p = torch.cos(phase_shift)
            sin_p = torch.sin(phase_shift)
            psi_rot_r, psi_rot_i = complex_mul_real(psi_r, psi_i, cos_p, sin_p)
            # SEOP Fix 42: Bounded rational envelope instead of catastrophic cosh()
            # cosh(10) = 11013 destroys signal. Rational: max attenuation ~10x not 11000x
            # f(x) = 1 + α·x² keeps gradients flowing through high-intensity regions
            # SEOP Fix 9: Per-dimension Kerr coefficient modulates envelope strength
            alpha = torch.abs(self.alpha)  # Ensure non-negative coupling
            envelope = 1.0 + alpha * intensity * intensity  # Bounded rational envelope
            envelope = envelope.clamp(
                max=10.0
            )  # SEOP Fix 78: Cap envelope to prevent vanishing gradients
            psi_rot_r = psi_rot_r / envelope
            psi_rot_i = psi_rot_i / envelope
            norm_in = (psi_r * psi_r + psi_i * psi_i).sum(dim=-1, keepdim=True)
            norm_rot = (psi_rot_r * psi_rot_r + psi_rot_i * psi_rot_i).sum(
                dim=-1, keepdim=True
            )
            scale = torch.sqrt((norm_in + 1e-8) / (norm_rot + 1e-8))
            scale = scale.clamp(
                max=10.0
            )  # SEOP Fix 78: Prevent explosive norm rescaling
            psi_rot_r = psi_rot_r * scale
            psi_rot_i = psi_rot_i * scale

            # Step 2: Cayley diffusion via sparse Hamiltonian matvec
            half_dt = self.dt / 2

            if _t:
                t_cache = time.perf_counter()
            self.hamiltonian.cache_weights()
            if _t:
                self._timing_stats.setdefault("cache_weights", []).append(
                    time.perf_counter() - t_cache
                )
            D = self.dim

            def a_plus_matvec(v_pair):
                vr, vi = v_pair
                Hvr, Hvi = self.hamiltonian.matvec_real_fused(vr, vi)
                return (vr + half_dt * Hvi, vi - half_dt * Hvr)

            def a_minus_matvec(v_pair):
                """Apply (I + i*dt/2*H) in real-block form.

                For symmetric H (graph Laplacian), the complex operator A = I + i*d*H
                is normal: A^H A = I + d^2 H^2, so it is invertible and typically
                well-conditioned. In real-block coordinates (vr, vi), the matrix is:

                    [[I, -d*H],
                     [d*H,  I]]

                which is not symmetric in the standard Euclidean inner product.
                Vanilla CG therefore has no formal SPD guarantee here; we rely on a
                block-Jacobi preconditioner and residual monitoring/fallback.
                """
                vr, vi = v_pair
                Hvr, Hvi = self.hamiltonian.matvec_real_fused(vr, vi)
                return (vr - half_dt * Hvi, vi + half_dt * Hvr)

            # Compute RHS: A_plus @ psi_rot
            if _t:
                t_rhs = time.perf_counter()
            rhs_r, rhs_i = a_plus_matvec((psi_rot_r, psi_rot_i))
            if _t:
                self._timing_stats.setdefault("rhs_matvec", []).append(
                    time.perf_counter() - t_rhs
                )

            if self.use_chebyshev_kpm:
                # Update λ_max from current Hamiltonian weights (Gershgorin bound)
                # SEOP Fix: Cache lambda_max to avoid GPU-CPU sync every step
                # Only update every 10 steps (weights change slowly)
                if self._training_step % 10 == 0 or self._training_step < 5:
                    _diag = self.hamiltonian.get_diagonal().real
                    # Keep as tensor on GPU, avoid .item() sync
                    # _diag.max() returns a 0-dim tensor
                    _lmax = 2.0 * _diag.max()
                    self._cheby_lambda_max.copy_(_lmax.detach())

                # Use cached tensor value
                _lmax_tensor = self._cheby_lambda_max

                out_r, out_i = chebyshev_kpm_solve(
                    matvec_fn=self.hamiltonian.matvec_real_fused,
                    rhs_r=rhs_r,
                    rhs_i=rhs_i,
                    coeffs=self.cheby_coeffs,
                    lambda_max=_lmax_tensor,
                )

                psi_out = torch.view_as_complex(
                    torch.stack([out_r, out_i], dim=-1).contiguous()
                )
                # SEOP Fix 81: In-place cache update (no .clone())
                if self._psi_cache is None or self._psi_cache.shape != psi_out.shape:
                    self._psi_cache = psi_out.detach()
                else:
                    self._psi_cache.copy_(psi_out.detach())
                self.hamiltonian.clear_cache()

                if self.check_unitarity_flag and not self.training:
                    check_unitarity(psi, psi_out)

                if _t:
                    self._timing_stats.setdefault("total", []).append(
                        time.perf_counter() - t0
                    )

                return psi_out

            if self.direct_solve:
                H = self.hamiltonian.get_hamiltonian_dense()
                I = torch.eye(D, dtype=H.dtype, device=H.device)
                A_plus = I + 1j * half_dt * H
                rhs = safe_complex(rhs_r, rhs_i)
                batch_size, seq_len, _ = rhs.shape
                rhs_flat = rhs.reshape(batch_size * seq_len, D)
                # Solve in float64: Graph Laplacians have a zero eigenvalue, so
                # when dt*λ_max is large, float32 loses the I term in I+iHdt/2,
                # making the matrix numerically singular. float64 preserves it.
                orig_dtype = rhs_flat.dtype
                A64 = A_plus.to(torch.complex128)
                rhs64 = rhs_flat.to(torch.complex128)
                try:
                    psi_out_flat = torch.linalg.solve(A64, rhs64.T).T.to(orig_dtype)
                except torch._C._LinAlgError:
                    import warnings

                    warnings.warn(
                        "Cayley direct_solve: singular even in float64, returning identity",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    psi_out_flat = rhs_flat
                psi_out = psi_out_flat.reshape(batch_size, seq_len, D)
                # SEOP Fix 81: In-place cache update (no .clone())
                if self._psi_cache is None or self._psi_cache.shape != psi_out.shape:
                    self._psi_cache = psi_out.detach()
                else:
                    self._psi_cache.copy_(psi_out.detach())

                self.hamiltonian.clear_cache()

                if self.check_unitarity_flag and not self.training:
                    check_unitarity(psi, psi_out)

                if _t:
                    self._timing_stats.setdefault("total", []).append(
                        time.perf_counter() - t0
                    )

                return psi_out

            def a_minus_matvec_wrapped(v_real_block):
                """matvec wrapper for CG in real-block representation.

                The CG solver operates on tensors shaped [..., D, 2] where the last
                dimension stores (real, imag). This wrapper maps that real-block
                tensor through the Cayley system operator (I + i*dt/2*H).

                The resulting real-block operator is invertible but not symmetric;
                CG is used pragmatically with preconditioning and a residual-based
                safety fallback in `sem.propagator.cg_solver.cg_solve_sparse`.
                """
                vr, vi = v_real_block.unbind(-1)
                out_r, out_i = a_minus_matvec((vr, vi))
                return torch.stack([out_r, out_i], dim=-1)

            # SEOP Fix 28: Block-Jacobi preconditioner for CG convergence.
            # A_minus = [[I, d*H], [-d*H, I]] per-dim 2x2 block inverse is O(D).
            # Without this, growing Hamiltonian eigenvalues make CG diverge.
            diag_H = self.hamiltonian.get_diagonal().real
            d_k = half_dt * diag_H
            inv_s2 = 1.0 / (1.0 + d_k * d_k)

            def block_jacobi_precond(v_real_block):
                vr, vi = v_real_block.unbind(-1)
                out_r = (vr - d_k * vi) * inv_s2
                out_i = (d_k * vr + vi) * inv_s2
                return torch.stack([out_r, out_i], dim=-1)

            batch_size, seq_len, _ = rhs_r.shape
            rhs_real_block = torch.stack([rhs_r, rhs_i], dim=-1).reshape(-1, D, 2)

            x0 = None
            if self._psi_cache is not None and self._psi_cache.shape == (
                batch_size,
                seq_len,
                D,
            ):
                x0 = torch.view_as_real(self._psi_cache).reshape(-1, D, 2)
                if _t:
                    self._timing_stats["cache_hits"] = (
                        self._timing_stats.get("cache_hits", 0) + 1
                    )

            # Residual-gated lazy CG
            skip_cg = False
            Ax0 = None
            if self.lazy_cg and x0 is not None:
                if _t:
                    t_gate = time.perf_counter()
                Ax0 = a_minus_matvec_wrapped(x0)
                residual = Ax0 - rhs_real_block
                rel_residual = residual.norm() / (rhs_real_block.norm() + 1e-12)
                if torch.compiler.is_compiling():
                    # During torch.compile: always run CG (no graph break)
                    skip_cg = False
                else:
                    # Eager mode optimization: GPU sync (.item()) costs ~4000ms.
                    # With cg_max_iter=5 and D=256, full CG (<1s) is faster than sync (4s).
                    # Skip the check and always run CG to keep CPU/GPU async.
                    skip_cg = False
                if _t:
                    self._timing_stats.setdefault("lazy_gate", []).append(
                        time.perf_counter() - t_gate
                    )

            self._cg_total_count += 1
            if _t:
                self._timing_stats["cg_calls"] = (
                    self._timing_stats.get("cg_calls", 0) + 1
                )

            if skip_cg and x0 is not None and Ax0 is not None:
                self._cg_skip_count += 1
                if _t:
                    self._timing_stats["cg_skips"] = (
                        self._timing_stats.get("cg_skips", 0) + 1
                    )
                if torch.is_grad_enabled() and rhs_real_block.requires_grad:
                    psi_out_real_block = x0 + (rhs_real_block - Ax0)
                else:
                    psi_out_real_block = x0
            else:
                if _t:
                    t_cg = time.perf_counter()
                effective_tol = self.get_effective_cg_tol()
                psi_out_real_block = cg_solve_sparse(
                    a_minus_matvec_wrapped,
                    rhs_real_block,
                    self.cg_max_iter,
                    effective_tol,
                    precond=block_jacobi_precond,
                    x0=x0,
                )
                if _t:
                    self._timing_stats.setdefault("cg_solve", []).append(
                        time.perf_counter() - t_cg
                    )
                    self._timing_stats.setdefault("cg_iterations", []).append(
                        _cg_mod._last_cg_iterations
                    )

            psi_out = torch.view_as_complex(
                psi_out_real_block.reshape(batch_size, seq_len, D, 2)
            )
            # SEOP Fix 81: In-place cache update (no .clone())
            if self._psi_cache is None or self._psi_cache.shape != psi_out.shape:
                self._psi_cache = psi_out.detach()
            else:
                self._psi_cache.copy_(psi_out.detach())

            self.hamiltonian.clear_cache()

            if self.check_unitarity_flag and not self.training:
                check_unitarity(psi, psi_out)

            if _t:
                self._timing_stats.setdefault("total", []).append(
                    time.perf_counter() - t0
                )

            return psi_out

    @property
    def cg_skip_rate(self) -> float:
        """Fraction of forward passes where CG was skipped (lazy gate passed)."""
        if self._cg_total_count == 0:
            return 0.0
        return self._cg_skip_count / self._cg_total_count

    def reset_cg_stats(self):
        """Reset CG skip statistics (call at health check intervals)."""
        self._cg_skip_count = 0
        self._cg_total_count = 0


class CayleySolitonStack(nn.Module):
    """Stack of Cayley-Soliton propagation layers.

    Each layer has its own learnable Hamiltonian and nonlinear strength,
    allowing the model to learn different propagation dynamics at each depth.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 8,
        dt: float = 0.1,
        nonlinear_alpha: float = 0.1,
        cg_max_iter: int = 20,
        cg_tol: float = 1e-6,
        laplacian_sparsity: int = 5,
        lazy_cg: bool = True,
        lazy_cg_tol: float = 1e-6,
        direct_solve: bool = False,
        pit_gamma: float = 1.0,
        adaptive_cg_tol: bool = False,
        cg_tol_warmup: float = 1e-4,
        cg_tol_mid: float = 1e-5,
        cg_tol_late: float = 1e-6,
        cg_tol_warmup_end: int = 2000,
        cg_tol_mid_end: int = 50000,
        use_chebyshev_kpm: bool = False,
        chebyshev_degree: int = 12,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CayleySolitonPropagator(
                    dim=dim,
                    dt=dt,
                    nonlinear_alpha=nonlinear_alpha,
                    cg_max_iter=cg_max_iter,
                    cg_tol=cg_tol,
                    laplacian_sparsity=laplacian_sparsity,
                    lazy_cg=lazy_cg,
                    lazy_cg_tol=lazy_cg_tol,
                    direct_solve=direct_solve,
                    pit_gamma=pit_gamma,
                    adaptive_cg_tol=adaptive_cg_tol,
                    cg_tol_warmup=cg_tol_warmup,
                    cg_tol_mid=cg_tol_mid,
                    cg_tol_late=cg_tol_late,
                    cg_tol_warmup_end=cg_tol_warmup_end,
                    cg_tol_mid_end=cg_tol_mid_end,
                    use_chebyshev_kpm=use_chebyshev_kpm,
                    chebyshev_degree=chebyshev_degree,
                )
                for _ in range(num_layers)
            ]
        )

    def set_training_step(self, step: int) -> None:
        for layer in self.layers:
            cast(CayleySolitonPropagator, layer).set_training_step(step)

    def set_timing(self, enabled: bool) -> None:
        for layer in self.layers:
            cast(CayleySolitonPropagator, layer).timing_enabled = enabled

    def collect_and_clear_timing(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for layer in self.layers:
            prop = cast(CayleySolitonPropagator, layer)
            stats = prop._timing_stats
            for k, v in stats.items():
                if isinstance(v, list):
                    merged.setdefault(k, []).extend(v)
                else:
                    merged[k] = merged.get(k, 0) + v
            prop._timing_stats = {}
        return merged

    def forward(self, psi: Tensor) -> Tensor:
        for layer in self.layers:
            psi = cast(CayleySolitonPropagator, layer)(psi)
        return psi

    @property
    def cg_skip_rate(self) -> float:
        rates = [
            cast(CayleySolitonPropagator, layer).cg_skip_rate for layer in self.layers
        ]
        return sum(rates) / len(rates) if rates else 0.0

    def reset_cg_stats(self):
        for layer in self.layers:
            cast(CayleySolitonPropagator, layer).reset_cg_stats()
