"""Quaternionic Escape (Imaginary Attenuation) for Cayley-Solve Stability.

SEM uses Cayley-style linear solves of the form (I + iH) x = b. For the
Hermitian/real PSD H used in practice, (I + iH) is always invertible, so there
is no literal Cayley singularity to "go around".

This module instead implements a lightweight, differentiable stability heuristic:

- Estimate ||H||_2 via power iteration (cheap O(D^2) proxy for solve difficulty)
- Use a soft sigmoid gate (learnable via log_threshold) to decide escape strength
- Apply the effective action of the quaternionic lift+projection:
  attenuate Im(psi) and renormalize per token

The Quaternion class provides reusable quaternion algebra and is intentionally
kept independent of this heuristic.

Author: SEM V8.0 Team
Date: Feb 5, 2026
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import math


class Quaternion:
    """Quaternion representation for numerical stability at singularities.

    A quaternion q = w + xi + yj + zk where:
    - w is the real/scalar part
    - (x, y, z) is the vector/imaginary part
    - i² = j² = k² = ijk = -1

    Quaternions form a 4D normed division algebra - the only one beyond complex
    numbers. This property allows us to "rotate around" singularities that would
    crash in the complex plane.

    Args:
        w: Real/scalar component [...]
        x: i coefficient [...]
        y: j coefficient [...]
        z: k coefficient [...]
    """

    def __init__(self, w: Tensor, x: Tensor, y: Tensor, z: Tensor):
        """Initialize quaternion from 4 real components."""
        self.w = w  # Real part
        self.x = x  # i coefficient (complex imaginary)
        self.y = y  # j coefficient (1st extra dimension)
        self.z = z  # k coefficient (2nd extra dimension)

    @classmethod
    def from_complex(cls, z: Tensor) -> "Quaternion":
        """Lift complex tensor to quaternion (embed in i plane, j=k=0).

        Args:
            z: [...] complex tensor
        Returns:
            Quaternion with w=Re(z), x=Im(z), y=0, z=0
        """
        return cls(z.real, z.imag, torch.zeros_like(z.real), torch.zeros_like(z.real))

    def to_complex(self) -> Tensor:
        """Project quaternion back to complex plane (discard j,k components).

        Returns:
            [...] complex tensor (w + xi)
        """
        return torch.complex(self.w, self.x)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Quaternion multiplication via Hamilton product.

        Non-commutative: q1 * q2 ≠ q2 * q1 in general
        Associative: (q1 * q2) * q3 = q1 * (q2 * q3)

        Args:
            other: Quaternion to multiply
        Returns:
            Product quaternion
        """
        # Hamilton product formula (derived from i²=j²=k²=ijk=-1)
        # w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        # x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        # y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        # z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w

        # Expand all products
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z
        z = self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x
        return Quaternion(w, x, y, z)

    def conj(self) -> "Quaternion":
        """Quaternion conjugate: q* = w - xi - yj - zk.

        Property: q * q* = |q|² (scalar)
        """
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self) -> Tensor:
        """Quaternion norm: |q| = sqrt(w² + x² + y² + z²).

        Property: |q1 * q2| = |q1| * |q2| (multiplicative norm)
        """
        return torch.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Quaternion":
        """Return unit quaternion (norm = 1).

        Unit quaternions form SU(2) and represent 3D rotations via:
        v' = q * v * q*
        """
        n = self.norm()
        n = n.clamp(min=1e-8)  # Prevent division by zero

        # Ensure n has same shape as components for broadcasting
        while n.dim() < self.w.dim():
            n = n.unsqueeze(-1)

        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)


class QuaternionicEscape(nn.Module):
    """Differentiable escape heuristic for numerically difficult Cayley solves.

    For the SEM Hamiltonians used in practice (real symmetric PSD), (I + iH) is
    always invertible and does not have a true singularity. However, large
    ||H||_2 can still correlate with harder iterative solves and numerical
    sensitivity.

    This module:
    - estimates ||H||_2 with a few power-iteration steps
    - computes a soft risk gate via a sigmoid around a learnable threshold
    - applies imaginary-axis attenuation (Im(psi) *= cos(angle)) + renorm
    - blends original and attenuated states using the soft gate

    Args:
        dim: state dimension (for buffer allocation)
        condition_threshold: threshold on ||H||_2 (stored in log space)
        learnable: if True, threshold and angle are learnable parameters
    """

    def __init__(
        self, dim: int, condition_threshold: float = 100.0, learnable: bool = True
    ):
        super().__init__()
        self.dim = dim

        # Store threshold in log space for numerical stability
        if learnable:
            self.log_threshold = nn.Parameter(
                torch.tensor(math.log(condition_threshold))
            )
        else:
            self.register_buffer(
                "log_threshold", torch.tensor(math.log(condition_threshold))
            )

        # Escape rotation angle (initialized to π/2 = 90 degrees)
        # This rotates the state "perpendicular" to the complex plane
        if learnable:
            self.escape_angle = nn.Parameter(torch.tensor(math.pi / 2))
        else:
            self.register_buffer("escape_angle", torch.tensor(math.pi / 2))

    def estimate_spectral_norm(self, H: Tensor, n_iters: int = 5) -> Tensor:
        """Estimate ||H||_2 via power iteration.

        This estimates the largest singular value of H using a small number of
        power-iteration steps. Complexity is O(n_iters * D^2) per batch.

        Args:
            H: [B, D, D] real or complex matrix
            n_iters: number of power iteration steps (5 is usually sufficient)
        Returns:
            spectral_norm: [B] estimated largest singular value
        """
        if H.dim() != 3:
            raise ValueError(f"H must be [B, D, D], got shape={tuple(H.shape)}")

        B, D, _ = H.shape

        # Random init vector (right singular vector estimate)
        v = torch.randn(B, D, 1, device=H.device, dtype=H.dtype)
        v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp(min=1e-8)

        # Power iteration on H^* H (implemented as alternating H and H^*)
        H_h = H.transpose(-2, -1).conj() if H.is_complex() else H.transpose(-2, -1)
        for _ in range(n_iters):
            u = torch.bmm(H, v)  # [B, D, 1]
            u = u / torch.linalg.norm(u, dim=1, keepdim=True).clamp(min=1e-8)
            v = torch.bmm(H_h, u)
            v = v / torch.linalg.norm(v, dim=1, keepdim=True).clamp(min=1e-8)

        # Final spectral norm estimate
        Hv = torch.bmm(H, v)
        sigma = torch.linalg.norm(Hv, dim=1).squeeze(-1)  # [B]
        return sigma.real if sigma.is_complex() else sigma

    def detect_singularity(self, H: Tensor) -> Tuple[Tensor, Tensor]:
        """Detect instability via spectral norm of H (proxy for CG difficulty).

        For Hermitian/real PSD H, (I + iH) is always invertible and its smallest
        singular value is >= 1, so a condition-number test of (I + iH) is not a
        meaningful proxy for the numerical difficulty of the Cayley solve.

        Instead we use ||H||_2: large spectral norm generally correlates with
        harder linear solves and worse numerical stability in iterative methods.

        Args:
            H: [B, D, D] effective Hamiltonian matrix (real or complex)
        Returns:
            singular_mask: [B] bool, True if instability proxy exceeds threshold
            spectral_norm: [B] estimated ||H||_2
        """
        spectral_norm = self.estimate_spectral_norm(H)
        threshold = self.log_threshold.exp()
        singular_mask = spectral_norm > threshold
        return singular_mask, spectral_norm

    def quaternionic_rotation(self, psi: Tensor, angle: Tensor) -> Tensor:
        """Rotate complex state in quaternionic j-k plane.

        Note: when this result is projected back to the complex plane
        (discarding j,k), the effect is exactly:

            Re(psi) unchanged
            Im(psi) scaled by cos(angle)

        The forward() path implements this attenuation directly for clarity and
        speed; this method is retained for reference/experimentation.

        Args:
            psi: [B, S, D] or [B, D] complex64 state vector
            angle: rotation angle (scalar or [B])
        Returns:
            psi_rotated: same shape as psi, complex64
        """
        # Lift complex state to quaternion (j=k=0 initially)
        q = Quaternion.from_complex(psi)

        # Construct rotation quaternion: exp(angle*j/2) = cos(a/2) + sin(a/2)*j
        half_angle = angle / 2
        rot_w = torch.cos(half_angle)  # Real part
        rot_x = torch.zeros_like(rot_w)  # i component = 0
        rot_y = torch.sin(half_angle)  # j component (rotation axis)
        rot_z = torch.zeros_like(rot_w)  # k component = 0

        # Broadcast rotation quaternion to match psi shape
        # angle can be scalar or [B], psi is [B, ...] or [B, S, D]
        if rot_w.dim() == 0:
            # Scalar angle → broadcast to batch
            rot_w = rot_w.expand(psi.shape[0])

        # Expand to match all dimensions of psi
        expand_shape = [-1] + [1] * (psi.dim() - 1)
        rot_w = rot_w.view(*expand_shape).expand_as(psi.real)
        rot_x = torch.zeros_like(rot_w)
        rot_y = rot_y.view(*expand_shape).expand_as(psi.real)
        rot_z = torch.zeros_like(rot_w)

        # Create rotation quaternion and normalize it (ensure it's a unit quaternion)
        rot = Quaternion(rot_w, rot_x, rot_y, rot_z)
        rot = rot.normalize()  # Unit quaternion: |rot| = 1
        rot_conj = rot.conj()

        # Apply rotation: q' = rot * q * rot_conj
        # This is a similarity transformation that preserves norm when rot is unit
        q_rotated = rot * q * rot_conj

        # Project back to complex plane (discard j, k components)
        return q_rotated.to_complex()

    def forward(self, psi: Tensor, H: Tensor) -> Tuple[Tensor, Tensor]:
        """Attenuate the imaginary axis when the CG solve is likely unstable.

        This module does not attempt to detect a literal Cayley singularity.
        For the Hermitian/real PSD H used in SEM, (I + iH) is always invertible.

        Instead, we use ||H||_2 as an instability proxy (large ||H||_2 tends to
        make iterative solves harder), and apply a smooth escape that matches the
        *effective* action of the quaternionic lift+projection:

        - attenuate Im(psi) by cos(escape_angle)
        - renormalize per-token to preserve ||psi||_2
        - blend with a soft sigmoid gate so log_threshold is learnable

        Args:
            psi: [B, S, D] or [B, D] complex64 spinor state
            H: [B, D, D] effective Hamiltonian (for risk estimation)
        Returns:
            psi_safe: same shape as psi
            escaped_mask: [B] bool diagnostic mask (hard threshold; not used for gating)
        """
        spectral_norm = self.estimate_spectral_norm(H)

        # Soft gate (differentiable): gate -> 0 below threshold, gate -> 1 above.
        log_risk = torch.log(spectral_norm.clamp(min=1e-8)) - self.log_threshold
        gate = torch.sigmoid(5.0 * log_risk)

        # Diagnostic mask (hard threshold for logging only)
        threshold = self.log_threshold.exp()
        escaped_mask = spectral_norm > threshold

        # Imaginary-axis attenuation (equivalent to quaternionic j-axis rotation
        # followed by projection back to the complex plane).
        cos_angle = torch.cos(self.escape_angle)
        psi_attenuated = torch.complex(psi.real, psi.imag * cos_angle)

        # Renormalize to preserve L2 norm per token
        norm_before = torch.linalg.norm(psi, dim=-1, keepdim=True)
        norm_after = torch.linalg.norm(psi_attenuated, dim=-1, keepdim=True).clamp(
            min=1e-8
        )
        psi_attenuated = psi_attenuated * (norm_before / norm_after)

        # Soft blend
        gate_expanded = gate.view(-1, *([1] * (psi.dim() - 1)))
        psi_safe = (1 - gate_expanded) * psi + gate_expanded * psi_attenuated

        return psi_safe, escaped_mask


def test_quaternion_algebra():
    """Unit tests for quaternion operations."""
    print("Testing quaternion algebra...")

    # Test 1: Multiplication identity
    q1 = Quaternion(
        torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0)
    )
    identity = Quaternion(
        torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    )
    result = q1 * identity
    assert torch.allclose(result.w, q1.w)
    assert torch.allclose(result.x, q1.x)
    assert torch.allclose(result.y, q1.y)
    assert torch.allclose(result.z, q1.z)
    print("✓ Identity property")

    # Test 2: Conjugate property q * q* = |q|²
    q_conj = q1.conj()
    product = q1 * q_conj
    norm_sq = q1.norm() ** 2
    assert torch.allclose(product.w, norm_sq)
    assert torch.allclose(product.x, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(product.y, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(product.z, torch.tensor(0.0), atol=1e-6)
    print("✓ Conjugate property")

    # Test 3: Norm preservation |q1 * q2| = |q1| * |q2|
    q2 = Quaternion(
        torch.tensor(0.5), torch.tensor(-1.0), torch.tensor(2.0), torch.tensor(-0.5)
    )
    product = q1 * q2
    lhs = product.norm()
    rhs = q1.norm() * q2.norm()
    assert torch.allclose(lhs, rhs, rtol=1e-5)
    print("✓ Norm preservation")

    # Test 4: Complex lifting and projection
    z = torch.tensor([1.0 + 2.0j, 3.0 - 1.0j])
    q = Quaternion.from_complex(z)
    z_reconstructed = q.to_complex()
    assert torch.allclose(z, z_reconstructed)
    print("✓ Complex lifting/projection")

    print("All quaternion algebra tests passed!\n")


def test_singularity_detection():
    """Test spectral-norm instability proxy on known cases."""
    print("Testing singularity detection...")

    dim = 4
    batch = 2
    escape = QuaternionicEscape(dim, condition_threshold=10.0, learnable=False)

    # Test 1: Small spectral norm (should NOT trigger)
    H_good = torch.eye(dim).unsqueeze(0).repeat(batch, 1, 1) * 0.01

    mask, sigma = escape.detect_singularity(H_good)
    assert not mask.any(), "Low-norm matrix triggered escape"
    print(f"✓ Low-norm: ||H||_2={sigma.max().item():.4f} < 10")

    # Test 2: Large spectral norm (should trigger for one sample)
    H_bad = torch.zeros(batch, dim, dim)
    H_bad[0, 0, 0] = 1e6
    H_bad[1, 0, 0] = 0.01

    mask, sigma = escape.detect_singularity(H_bad)
    assert mask[0].item() is True
    assert mask[1].item() is False
    print(f"  High-norm: ||H||_2={sigma[0].item():.1e} -> triggered")
    print(f"  Low-norm:  ||H||_2={sigma[1].item():.4f} -> not triggered")

    print("Singularity detection tests passed!\n")


def test_attenuation_preserves_norm():
    """Test that imaginary attenuation + renorm preserves per-token norm."""
    print("Testing norm preservation under attenuation...")

    batch, seq, dim = 2, 8, 16
    psi = torch.randn(batch, seq, dim, dtype=torch.complex64)
    psi = psi / torch.linalg.norm(psi, dim=-1, keepdim=True)  # Normalize

    escape = QuaternionicEscape(dim, condition_threshold=10.0, learnable=False)

    # Low risk: gate ~ 0 -> output ~ psi
    H_low = torch.eye(dim).unsqueeze(0).repeat(batch, 1, 1) * 0.01
    psi_low, mask_low = escape(psi, H_low)
    assert not mask_low.any()

    # High risk: gate ~ 1 -> output ~ attenuated+renorm(psi)
    H_high = torch.zeros(batch, dim, dim)
    H_high[:, 0, 0] = 1e6
    psi_high, mask_high = escape(psi, H_high)
    assert mask_high.all()

    norm_before = torch.linalg.norm(psi, dim=-1)
    norm_low = torch.linalg.norm(psi_low, dim=-1)
    norm_high = torch.linalg.norm(psi_high, dim=-1)

    assert torch.allclose(norm_low, norm_before, rtol=1e-5, atol=1e-6)
    assert torch.allclose(norm_high, norm_before, rtol=1e-5, atol=1e-6)
    print("✓ Norm preserved for gate~0 and gate~1 cases\n")


if __name__ == "__main__":
    test_quaternion_algebra()
    test_singularity_detection()
    test_attenuation_preserves_norm()
    print("All quaternionic escape tests passed! ✓")
