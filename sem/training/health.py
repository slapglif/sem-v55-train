"""SEM-specific health monitoring during training.

Tracks architecture invariants that standard loss curves cannot reveal:
- Unitarity of the Cayley propagator
- Phase coherence (mode collapse detection)
- Information density
- CG solver convergence
- SDR sparsity
"""
import torch
import warnings
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..utils.metrics import (
    unitarity_deviation,
    information_density,
    phase_coherence,
    sparsity_ratio,
)


@dataclass
class HealthReport:
    """Results from a single health check."""
    step: int
    unitarity_dev: float = 0.0
    phase_coherence_val: float = 0.0
    info_density: float = 0.0
    sdr_sparsity: float = 0.0
    cg_residual: float = 0.0
    grad_norm: float = 0.0
    has_warning: bool = False
    has_error: bool = False
    messages: List[str] = field(default_factory=list)


class HealthMonitor:
    """Monitors SEM-specific health metrics during training.

    Args:
        unitarity_warn: Warning threshold for unitarity deviation
        unitarity_error: Error threshold for unitarity deviation
        coherence_collapse: Phase coherence above this = mode collapse
        density_warn: Info density below this = representation collapse
        cg_residual_warn: CG residual above this = poor convergence
        grad_norm_warn: Gradient norm above this
    """

    def __init__(
        self,
        unitarity_warn: float = 1e-4,
        unitarity_error: float = 1e-2,
        coherence_collapse: float = 0.95,
        density_warn: float = 0.1,
        cg_residual_warn: float = 1e-4,
        cg_residual_error: float = 1e-2,
        grad_norm_warn: float = 10.0,
        grad_norm_error: float = 100.0,
    ):
        self.unitarity_warn = unitarity_warn
        self.unitarity_error = unitarity_error
        self.coherence_collapse = coherence_collapse
        self.density_warn = density_warn
        self.cg_residual_warn = cg_residual_warn
        self.cg_residual_error = cg_residual_error
        self.grad_norm_warn = grad_norm_warn
        self.grad_norm_error = grad_norm_error

        self.history: List[HealthReport] = []

    @torch.no_grad()
    def check(self, model, sample_input: Tensor, step: int,
              grad_norm: float = 0.0) -> HealthReport:
        """Run full health check on model with a sample input.

        Args:
            model: SEMModel instance
            sample_input: [B, S] token IDs for probe forward pass
            step: Current training step
            grad_norm: Gradient norm from last backward pass
        Returns:
            HealthReport with all metrics and any warnings/errors
        """
        report = HealthReport(step=step, grad_norm=grad_norm)
        was_training = model.training
        model.eval()

        try:
            # Run probe forward pass capturing intermediates
            psi_encoder = model.encoder(sample_input)

            # SDR sparsity
            report.sdr_sparsity = sparsity_ratio(psi_encoder).item()
            if report.sdr_sparsity < 0.5:
                report.has_error = True
                report.messages.append(
                    f"SDR sparsity collapsed: {report.sdr_sparsity:.3f} < 0.5"
                )

            # Information density after encoder
            report.info_density = information_density(psi_encoder).item()
            if report.info_density < self.density_warn:
                report.has_warning = True
                report.messages.append(
                    f"Low information density: {report.info_density:.4f}"
                )

            # Run through Mamba layers
            psi = psi_encoder
            for mamba_layer in model.mamba_layers:
                psi = mamba_layer(psi)

            # Phase coherence after context layers
            report.phase_coherence_val = phase_coherence(psi).item()
            if report.phase_coherence_val > self.coherence_collapse:
                report.has_error = True
                report.messages.append(
                    f"Phase collapse detected: coherence={report.phase_coherence_val:.4f}"
                )

            # Unitarity through propagator
            psi_before_prop = psi.clone()
            psi_after_prop = model.propagator(psi)
            report.unitarity_dev = unitarity_deviation(
                psi_before_prop, psi_after_prop
            ).item()

            if report.unitarity_dev > self.unitarity_error:
                report.has_error = True
                report.messages.append(
                    f"Unitarity ERROR: deviation={report.unitarity_dev:.2e}"
                )
            elif report.unitarity_dev > self.unitarity_warn:
                report.has_warning = True
                report.messages.append(
                    f"Unitarity warning: deviation={report.unitarity_dev:.2e}"
                )

            # CG residual (post-hoc check on propagator output)
            # Check ||A @ x - b|| / ||b|| for the last Cayley layer
            report.cg_residual = self._check_cg_residual(model, psi_before_prop)
            if report.cg_residual > self.cg_residual_error:
                report.has_error = True
                report.messages.append(
                    f"CG convergence ERROR: residual={report.cg_residual:.2e}"
                )
            elif report.cg_residual > self.cg_residual_warn:
                report.has_warning = True
                report.messages.append(
                    f"CG convergence warning: residual={report.cg_residual:.2e}"
                )

            # Gradient norm checks
            if grad_norm > self.grad_norm_error:
                report.has_error = True
                report.messages.append(
                    f"Gradient explosion: norm={grad_norm:.2f}"
                )
            elif grad_norm > self.grad_norm_warn:
                report.has_warning = True
                report.messages.append(
                    f"High gradient norm: {grad_norm:.2f}"
                )

        finally:
            if was_training:
                model.train()

        self.history.append(report)
        return report

    def _check_cg_residual(self, model, psi: Tensor) -> float:
        """Compute CG residual for the last Cayley propagator layer.

        Computes ||A_minus @ x - rhs|| / ||rhs|| where:
        - A_minus = I + i*dt/2*H (system matrix)
        - rhs = A_plus @ psi (right-hand side)
        - x = propagator output
        """
        try:
            last_layer = model.propagator.layers[-1]
            H = last_layer.hamiltonian.get_hamiltonian_dense()
            D = last_layer.dim
            I = torch.eye(D, dtype=H.dtype, device=H.device)
            half_dt = last_layer.dt / 2

            A_minus = I + 1j * half_dt * H
            A_plus = I - 1j * half_dt * H

            # Compute what the RHS should be
            # Apply nonlinear phase rotation first
            intensity = psi.real * psi.real + psi.imag * psi.imag
            phase_shift = last_layer.alpha * intensity
            psi_rot = psi * torch.exp(1j * phase_shift)
            rhs = torch.matmul(psi_rot, A_plus.T)

            # Get actual output
            x = last_layer(psi)

            # Compute residual
            Ax = torch.matmul(x, A_minus.T)
            residual = (Ax - rhs).norm() / (rhs.norm() + 1e-12)
            return residual.item()
        except Exception:
            return 0.0  # Can't compute, skip

    def get_metrics_dict(self) -> Dict[str, float]:
        """Get latest health metrics as a flat dict for logging."""
        if not self.history:
            return {}
        report = self.history[-1]
        return {
            "health/unitarity_deviation": report.unitarity_dev,
            "health/phase_coherence": report.phase_coherence_val,
            "health/info_density": report.info_density,
            "health/sdr_sparsity": report.sdr_sparsity,
            "health/cg_residual": report.cg_residual,
            "health/grad_norm": report.grad_norm,
        }

    def state_dict(self) -> dict:
        """Serialize for checkpointing (keep last 100 reports)."""
        return {
            "history": [
                {
                    "step": r.step,
                    "unitarity_dev": r.unitarity_dev,
                    "phase_coherence_val": r.phase_coherence_val,
                    "info_density": r.info_density,
                    "sdr_sparsity": r.sdr_sparsity,
                    "cg_residual": r.cg_residual,
                    "grad_norm": r.grad_norm,
                    "has_warning": r.has_warning,
                    "has_error": r.has_error,
                    "messages": r.messages,
                }
                for r in self.history[-100:]
            ]
        }

    def load_state_dict(self, state: dict):
        self.history = [
            HealthReport(**r) for r in state.get("history", [])
        ]
