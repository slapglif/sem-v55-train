"""Conjugate Gradient solver with differentiable backward pass.

Solves the linear system Ax = b where A = (I + i*dt/2 * H) for the
Cayley diffusion step. The backward pass uses implicit differentiation:

If f(x, theta) = Ax - b = 0, then:
    dx/dtheta = -A^{-1} (dA/dtheta * x)

This requires another CG solve in the backward pass, but avoids
storing all intermediate CG iterates (memory-efficient).
"""

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
    """Conjugate Gradient solver for Hermitian positive definite systems.

    NOTE: This function disables AMP autocast to ensure float32 precision.
    bf16 has only ~3 decimal digits — CG needs ~7 to converge to tol=1e-7.
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
            matvec = A

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
    matvec_fn: Callable,
    b: Tensor,
    max_iter: int = 20,
    tol: float = 1e-6,
    precond=None,
    x0: Optional[Tensor] = None,
) -> Tensor:
    """CG solve using sparse matvec with implicit differentiation.

    Uses the mathematical trick: x = x_star + (b - A @ x_star)
    to avoid storing all CG iterates while maintaining gradient flow.

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

        if torch.is_grad_enabled() and b.requires_grad:
            Ax = matvec_fn(x_star)
            residual = b - Ax
            x = x_star + residual
        else:
            x = x_star

        return x
