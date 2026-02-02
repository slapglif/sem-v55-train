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
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from .hamiltonian import MultiScaleHamiltonian
from . import cg_solver as _cg_mod
from .cg_solver import cg_solve, cg_solve_sparse
from .unitarity_check import check_unitarity
from sem.spinor.complex_ops import complex_mul_real


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
    ):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.check_unitarity_flag = check_unitarity_flag
        # Warm-start cache for faster CG convergence
        self._psi_cache = None
        self.lazy_cg = lazy_cg
        self.lazy_cg_tol = lazy_cg_tol
        # CG skip statistics for monitoring
        self._cg_skip_count: int = 0
        self._cg_total_count: int = 0
        self.timing_enabled: bool = False
        self._timing_stats: dict = {}

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
        _device_type = psi.device.type
        _autocast_ctx = torch.autocast(device_type=_device_type, enabled=False)
        _autocast_ctx.__enter__()
        _t = self.timing_enabled
        t0 = t_cache = t_rhs = t_gate = t_cg = 0.0
        if _t:
            t0 = time.perf_counter()

        psi_real_block = torch.view_as_real(psi)
        psi_r, psi_i = psi_real_block.unbind(-1)

        # Step 1: Nonlinear phase rotation (Kerr effect)
        intensity = psi_r * psi_r + psi_i * psi_i
        intensity = intensity / (intensity.mean(dim=-1, keepdim=True) + 1e-8)
        phase_shift = self.alpha * intensity

        cos_p = torch.cos(phase_shift)
        sin_p = torch.sin(phase_shift)
        psi_rot_r, psi_rot_i = complex_mul_real(psi_r, psi_i, cos_p, sin_p)

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

        def a_minus_matvec_wrapped(v_real_block):
            vr, vi = v_real_block.unbind(-1)
            out_r, out_i = a_minus_matvec((vr, vi))
            return torch.stack([out_r, out_i], dim=-1)

        B, S, _ = rhs_r.shape
        rhs_real_block = torch.stack([rhs_r, rhs_i], dim=-1).reshape(-1, D, 2)

        x0 = None
        if self._psi_cache is not None and self._psi_cache.shape == (B, S, D):
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
            skip_cg = rel_residual.item() < self.lazy_cg_tol
            if _t:
                self._timing_stats.setdefault("lazy_gate", []).append(
                    time.perf_counter() - t_gate
                )

        self._cg_total_count += 1
        if _t:
            self._timing_stats["cg_calls"] = self._timing_stats.get("cg_calls", 0) + 1

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
            psi_out_real_block = cg_solve_sparse(
                a_minus_matvec_wrapped,
                rhs_real_block,
                self.cg_max_iter,
                self.cg_tol,
                x0=x0,
            )
            if _t:
                self._timing_stats.setdefault("cg_solve", []).append(
                    time.perf_counter() - t_cg
                )
                self._timing_stats.setdefault("cg_iterations", []).append(
                    _cg_mod._last_cg_iterations
                )

        psi_out = torch.view_as_complex(psi_out_real_block.reshape(B, S, D, 2))
        self._psi_cache = psi_out.detach().clone()

        self.hamiltonian.clear_cache()

        if self.check_unitarity_flag and not self.training:
            check_unitarity(psi, psi_out)

        if _t:
            self._timing_stats.setdefault("total", []).append(time.perf_counter() - t0)

        _autocast_ctx.__exit__(None, None, None)
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
                )
                for _ in range(num_layers)
            ]
        )

    def set_timing(self, enabled: bool) -> None:
        for layer in self.layers:
            cast(CayleySolitonPropagator, layer).timing_enabled = enabled

    def collect_and_clear_timing(self) -> dict:
        merged: dict = {}
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
