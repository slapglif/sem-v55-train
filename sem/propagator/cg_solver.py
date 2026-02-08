"""Conjugate Gradient (CG) solver used by the Cayley-Soliton propagator.

This module contains two CG entry points with different backward behavior:

- `cg_solve` (dense matrix): standard CG on an explicit (Hermitian) SPD matrix.
  The custom autograd Function implements an implicit-diff style backward by
  solving the adjoint system.

- `cg_solve_sparse` (matvec): forward runs CG under `torch.no_grad()` and the
  backward uses a straight-through estimator (STE) that keeps gradients flowing
  without storing CG iterates:

      x ~= x_star + (b - A(x_star))

  This is *not* true implicit differentiation; it is a first-order / Neumann-
  style approximation to A^{-1} (treating A^{-1} ~= I in the backward).

CG guarantees apply to symmetric/Hermitian positive definite systems. The
Cayley diffusion operator (in real-block form) is not symmetric in the
standard Euclidean inner product, but it is normal and well-conditioned.
We therefore monitor the achieved residual and can fall back to a direct
solve if convergence is poor.
"""

import warnings

import torch
from torch import Tensor
from torch.autograd import Function
from typing import Optional, Callable, Any, cast

_last_cg_iterations: int = 0


class CGSolverFunction(Function):
    """Custom autograd Function for CG solver with implicit differentiation."""

    @staticmethod
    def forward(ctx, rhs: Tensor, A_dense: Tensor, max_iter: int, tol: float) -> Tensor:
        """Solve A @ x = rhs via Conjugate Gradient (dense path).

        Args:
            rhs: Right-hand side [..., D] complex64
            A_dense: System matrix [D, D] complex64 (Hermitian positive definite)
            max_iter: Maximum CG iterations
            tol: Residual tolerance

        Returns:
            x: Solution [..., D] complex64
        """
        x = _cg_solve(A_dense, rhs, max_iter, tol)
        ctx.save_for_backward(x, A_dense)
        ctx.max_iter = max_iter
        ctx.tol = tol
        return x

    @staticmethod
    def backward(ctx, *grad_outputs: Any) -> Any:
        """Backward pass using implicit differentiation."""
        grad_output = grad_outputs[0]
        x, A_dense = ctx.saved_tensors
        # Solves A^H @ grad_rhs = grad_output. Since A is Hermitian, A^H = A.
        grad_rhs = _cg_solve(A_dense, grad_output, ctx.max_iter, ctx.tol)
        grad_A = -torch.einsum("...i,...j->ij", grad_rhs, x.conj())
        return grad_rhs, grad_A, None, None


def _cg_solve(
    A: Any,
    b: Tensor,
    max_iter: int,
    tol: float,
    precond: Optional[Callable[[Tensor], Tensor]] = None,
    x0: Optional[Tensor] = None,
) -> Tensor:
    """Conjugate Gradient solver.

    CG has formal convergence guarantees only for (Hermitian) SPD operators.
    The Cayley propagator uses CG on a real-block operator with a large
    skew-symmetric component (i.e. it is not symmetric in the usual inner
    product), so those guarantees do not strictly apply. In that case we rely
    on the operator's structure (normal + well-conditioned) and add residual
    monitoring / safety fallback in `cg_solve_sparse`.

    NOTE: This function disables AMP autocast to ensure float32 precision.
    bf16 has only ~3 decimal digits -- CG needs ~7 to converge to tol=1e-7.
    Without this, the Cayley propagator's unitarity guarantee breaks.
    """
    # SEOP Fix 27: Force float32 precision for CG solver.
    # bf16 autocast from the trainer poisons matmul ops inside matvec callbacks,
    # preventing convergence (residual stuck at ~1.0 instead of <1e-4).
    _device_type = b.device.type
    with torch.autocast(device_type=_device_type, enabled=False):
        return _cg_solve_impl(A, b, max_iter, tol, precond, x0)


def _cg_solve_impl(
    A: Any,
    b: Tensor,
    max_iter: int,
    tol: float,
    precond: Optional[Callable[[Tensor], Tensor]] = None,
    x0: Optional[Tensor] = None,
) -> Tensor:
    """CG solver implementation (always runs in float32)."""
    assert not torch.is_autocast_enabled()
    if torch.is_complex(b):
        assert b.dtype == torch.complex64
    else:
        assert b.dtype == torch.float32
    b = b.float() if not torch.is_complex(b) else b.to(torch.complex64)
    if x0 is not None:
        x0 = x0.float() if not torch.is_complex(x0) else x0.to(torch.complex64)
    is_complex = torch.is_complex(b)
    b_real = torch.view_as_real(b) if is_complex else b

    if callable(A) and not isinstance(A, Tensor):
        if is_complex:

            def matvec(v: Tensor) -> Tensor:
                v_c = torch.view_as_complex(v)
                res_c = A(v_c)
                return torch.view_as_real(cast(Tensor, res_c))
        else:

            def matvec(v: Tensor) -> Tensor:
                return cast(Tensor, A(v))

    else:
        if torch.is_complex(A):
            A_real = torch.view_as_real(A)
            Ar, Ai = A_real[..., 0], A_real[..., 1]

            def matvec(v: Tensor) -> Tensor:
                vr, vi = v[..., 0:1], v[..., 1:2]
                # (Ar + iAi)(vr + ivi) = (Ar vr - Ai vi) + i(Ar vi + Ai vr)
                res_r = torch.matmul(Ar, vr) - torch.matmul(Ai, vi)
                res_i = torch.matmul(Ar, vi) + torch.matmul(Ai, vr)
                return torch.cat([res_r, res_i], dim=-1)
        else:
            matvec = lambda v: torch.matmul(A, v)

    p_fn: Optional[Callable[[Tensor], Tensor]] = None
    if precond is not None:
        if is_complex:
            raw_precond = precond

            def _p_fn(v: Tensor) -> Tensor:
                v_c = torch.view_as_complex(v)
                res_c = raw_precond(v_c)
                return torch.view_as_real(cast(Tensor, res_c))

            p_fn = _p_fn
        else:
            p_fn = precond

    if x0 is not None:
        x = x0
        r = b_real - matvec(x)
    else:
        x = torch.zeros_like(b_real)
        r = b_real.clone()

    z = p_fn(r) if p_fn is not None else r
    p = z.clone()

    # Hermitian inner product: Re(<a, b>) = sum(a_r * b_r + a_i * b_i)
    rz_old = (r * z).sum(dim=(-2, -1), keepdim=True)
    b_norm = torch.norm(b_real, dim=(-2, -1), keepdim=True) + 1e-12

    global _last_cg_iterations
    iters_done = 0
    for _ in range(max_iter):
        iters_done += 1
        Ap = matvec(p)
        pAp = (p * Ap).sum(dim=(-2, -1), keepdim=True)
        alpha = rz_old / (pAp + 1e-12)

        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = torch.norm(r, dim=(-2, -1), keepdim=True)
        if (r_norm / b_norm).max() < tol:
            break

        z = p_fn(r) if p_fn is not None else r
        rz_new = (r * z).sum(dim=(-2, -1), keepdim=True)
        beta = rz_new / (rz_old + 1e-12)
        p = z + beta * p
        rz_old = rz_new

    _last_cg_iterations = iters_done
    return torch.view_as_complex(x) if is_complex else x


def cg_solve(A: Tensor, b: Tensor, max_iter: int = 20, tol: float = 1e-6) -> Tensor:
    """Differentiable CG solve with dense matrix (public API)."""
    return cast(Tensor, CGSolverFunction.apply(b, A, max_iter, tol))


def cg_solve_sparse(
    matvec_fn: Callable[[Tensor], Tensor],
    b: Tensor,
    max_iter: int = 20,
    tol: float = 1e-6,
    precond=None,
    x0: Optional[Tensor] = None,
) -> Tensor:
    """CG solve using a matvec callback.

    Forward pass:
      Runs CG under `torch.no_grad()` to compute `x_star`.

    Backward pass:
      Uses a straight-through estimator (STE):

          x = x_star + (b - A(x_star))

      This keeps gradients flowing into `b` (identity) and into parameters
      captured by `matvec_fn` through the residual term, but it is *not* true
      implicit differentiation (no adjoint solve; effectively A^{-1} ~= I).

    Args:
        matvec_fn: Callable v -> A @ v (must be differentiable)
        b: [..., D] complex64 right-hand side
        max_iter: Maximum CG iterations (use 3 for fast training, 40 for accuracy)
        tol: Convergence tolerance
        precond: Optional preconditioner callable(v) -> M^{-1} @ v
        x0: Optional warm-start initial guess

    Returns:
        x: [..., D] complex64 solution
    """
    # SEOP Fix 27: Float32 exclusion zone (same as _cg_solve)
    _device_type = b.device.type
    with torch.autocast(device_type=_device_type, enabled=False):
        b = b.float() if not torch.is_complex(b) else b.to(torch.complex64)
        if x0 is not None:
            x0 = x0.float() if not torch.is_complex(x0) else x0.to(torch.complex64)

        with torch.no_grad():
            x_star = _cg_solve(matvec_fn, b, max_iter, tol, precond=precond, x0=x0)

            # Convergence safety: CG is only guaranteed for SPD systems. When CG is
            # applied to the Cayley real-block operator (non-symmetric), it usually
            # converges due to structure + preconditioning, but we still sanity-check
            # the achieved relative residual and fall back if it is poor.
            warn_thresh = 0.1
            Ax_star_ng = matvec_fn(x_star)
            residual_ng = b - Ax_star_ng

            if torch.is_complex(b):
                dim = b.shape[-1]
                residual_rb = torch.view_as_real(residual_ng).reshape(-1, dim, 2)
                b_rb = torch.view_as_real(b).reshape(-1, dim, 2)
            else:
                dim = b.shape[-2]
                residual_rb = residual_ng.reshape(-1, dim, 2)
                b_rb = b.reshape(-1, dim, 2)

            r_norm = torch.norm(residual_rb, dim=(-2, -1))
            b_norm = torch.norm(b_rb, dim=(-2, -1)) + 1e-12
            rel_r = r_norm / b_norm

            need_fallback = (~torch.isfinite(rel_r)) | (rel_r > warn_thresh)

            if bool((rel_r > warn_thresh).any()):
                warnings.warn(
                    (
                        "cg_solve_sparse: CG did not reach an acceptable residual "
                        f"(max rel_residual={rel_r.max().item():.3e} > {warn_thresh}). "
                        "Falling back to torch.linalg.solve for the failing systems."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )

            if bool(need_fallback.any()):
                # Build a dense matrix A from the matvec by probing basis vectors.
                # This is expensive, but only used as a rare safety fallback when
                # CG fails to converge.
                idx = need_fallback.nonzero(as_tuple=False).flatten()

                if torch.is_complex(b):
                    # Complex linear system: build A by applying matvec to I.
                    A_cols = matvec_fn(torch.eye(dim, device=b.device, dtype=b.dtype))
                    A_dense = A_cols.transpose(0, 1).contiguous()  # [D, D]

                    b_flat = b.reshape(-1, dim)
                    x_flat = x_star.reshape(-1, dim).clone()
                    b_bad = b_flat.index_select(0, idx)
                    x_bad = torch.linalg.solve(A_dense, b_bad.T).T
                    x_flat.index_copy_(0, idx, x_bad)
                    x_star = x_flat.reshape_as(x_star)
                else:
                    # Real-block representation of a complex operator: build a
                    # complex A from real basis probes.
                    eye = torch.eye(dim, device=b.device, dtype=b.dtype)
                    basis = torch.zeros((dim, dim, 2), device=b.device, dtype=b.dtype)
                    basis[:, :, 0] = eye

                    A_cols = torch.view_as_complex(matvec_fn(basis).contiguous())
                    A_dense = A_cols.transpose(0, 1).contiguous()  # [D, D]

                    b_c = torch.view_as_complex(b_rb.contiguous())
                    x_c = torch.view_as_complex(x_star.reshape(-1, dim, 2).contiguous())

                    b_bad = b_c.index_select(0, idx)
                    x_bad = torch.linalg.solve(A_dense, b_bad.T).T

                    x_c = x_c.clone()
                    x_c.index_copy_(0, idx, x_bad)
                    x_star = torch.view_as_real(x_c).reshape_as(x_star)

        if torch.is_grad_enabled() and b.requires_grad:
            Ax = matvec_fn(x_star)
            residual = b - Ax
            # STE (straight-through estimator): treat x_star as a constant but allow
            # gradients to flow through the residual term.
            x = x_star + residual
        else:
            x = x_star

        return x
