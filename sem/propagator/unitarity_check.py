"""Runtime unitarity validation for the Cayley propagator.

Provides assertions and metrics to verify that the propagator
preserves the L2 norm of the wavefunction (energy conservation).
"""
import torch
from torch import Tensor
import warnings


def check_unitarity(psi_in: Tensor, psi_out: Tensor,
                    tol: float = 1e-6, raise_error: bool = False) -> float:
    """Check if propagation preserves norm (unitarity).

    Args:
        psi_in: Input wavefunction [..., D] complex
        psi_out: Output wavefunction [..., D] complex
        tol: Maximum allowable deviation
        raise_error: If True, raise AssertionError on violation

    Returns:
        Maximum unitarity deviation across batch
    """
    norm_in_sq = (psi_in.abs() ** 2).sum(dim=-1)
    norm_out_sq = (psi_out.abs() ** 2).sum(dim=-1)

    deviation = ((norm_out_sq / (norm_in_sq + 1e-12)) - 1.0).abs()
    max_dev = deviation.max().item()

    if max_dev > tol:
        msg = f"Unitarity violation: max deviation {max_dev:.2e} > tol {tol:.2e}"
        if raise_error:
            raise AssertionError(msg)
        else:
            warnings.warn(msg)

    return max_dev


def cayley_unitarity_check(H: Tensor, dt: float, tol: float = 1e-6) -> float:
    """Verify that the Cayley operator U = (I - i*dt/2*H)^{-1}(I + i*dt/2*H) is unitary.

    Args:
        H: Hamiltonian [D, D] complex, must be Hermitian
        dt: Time step
        tol: Tolerance for ||U^H U - I||_F

    Returns:
        Frobenius norm of U^H U - I
    """
    D = H.shape[0]
    I = torch.eye(D, dtype=H.dtype, device=H.device)

    # Build Cayley matrices
    A_minus = I - 1j * (dt / 2) * H  # (I - i*dt/2*H)
    A_plus = I + 1j * (dt / 2) * H   # (I + i*dt/2*H)

    # U = A_minus^{-1} @ A_plus
    U = torch.linalg.solve(A_minus, A_plus)

    # Check unitarity: U^H @ U should be I
    UhU = U.conj().T @ U
    deviation = (UhU - I).norm().item()

    if deviation > tol:
        warnings.warn(f"Cayley operator unitarity deviation: {deviation:.2e}")

    return deviation
