#!/usr/bin/env python3
"""Diagnose n_eff for SEM LogSinkhorn using *real* encoder costs.

Goal:
- Capture the actual cost matrix C produced by MESHEncoder (shape [B,S,K])
- Capture the Sinkhorn transport plan T (shape [B,S,K])
- Compute per-row effective support n_eff = exp(H(row))

Notes:
- SEM LogSinkhorn uses uniform marginals by default, so each row of T sums to 1/S
  (not 1). For Shannon entropy per row, we normalize rows before computing H.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sem.config import SEMConfig
from sem.encoder.mesh_sdr import MESHEncoder


@dataclass
class RunStats:
    source: str  # "real" or "synthetic"
    eps_mode: str  # "fixed" or "auto"
    eps_value: float  # fixed eps, or mean auto eps used
    auto_scale: Optional[float]
    cost_std: float
    cost_median: float
    sigma_over_eps: float
    row_sum_mean: float
    n_eff_mean: float
    n_eff_p50: float
    n_eff_p10: float
    n_eff_p90: float
    n_eff_raw_mean: float  # entropy computed on unnormalized rows (for comparison)


def _percentile(x: torch.Tensor, q: float) -> float:
    """Percentile for 1D tensor (q in [0,1])."""
    if x.numel() == 0:
        return float("nan")
    x_sorted, _ = torch.sort(x)
    idx = int(round((x_sorted.numel() - 1) * q))
    idx = max(0, min(idx, x_sorted.numel() - 1))
    return x_sorted[idx].item()


def n_eff_from_transport(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (n_eff_normalized, n_eff_raw) as [B,S] tensors.

    n_eff_normalized uses row-normalized probabilities.
    n_eff_raw matches the (incorrect) form used in validate_theorems.py.
    """
    # Raw entropy on unnormalized rows (kept for comparison)
    T_safe = T.clamp_min(1e-30)
    H_raw = -(T * T_safe.log()).sum(dim=-1)  # [B,S]
    n_eff_raw = H_raw.exp()

    # Shannon entropy requires a probability distribution (row sums 1)
    row_sum = T.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    P = T / row_sum
    P_safe = P.clamp_min(1e-30)
    H = -(P * P_safe.log()).sum(dim=-1)  # [B,S]
    n_eff = H.exp()
    return n_eff, n_eff_raw


def summarize(
    cost: torch.Tensor,
    T: torch.Tensor,
    *,
    source: str,
    eps_mode: str,
    eps_value: float,
    auto_scale: Optional[float],
) -> RunStats:
    cost_std = cost.std().item()
    cost_median = cost.median().item()
    sigma_over_eps = cost_std / max(eps_value, 1e-30)

    row_sum = T.sum(dim=-1)  # [B,S]
    row_sum_mean = row_sum.mean().item()

    n_eff, n_eff_raw = n_eff_from_transport(T)
    n_eff_flat = n_eff.reshape(-1).detach().cpu()
    n_eff_raw_mean = n_eff_raw.mean().item()

    return RunStats(
        source=source,
        eps_mode=eps_mode,
        eps_value=eps_value,
        auto_scale=auto_scale,
        cost_std=cost_std,
        cost_median=cost_median,
        sigma_over_eps=sigma_over_eps,
        row_sum_mean=row_sum_mean,
        n_eff_mean=n_eff.mean().item(),
        n_eff_p50=_percentile(n_eff_flat, 0.50),
        n_eff_p10=_percentile(n_eff_flat, 0.10),
        n_eff_p90=_percentile(n_eff_flat, 0.90),
        n_eff_raw_mean=n_eff_raw_mean,
    )


class SinkhornCapture:
    def __init__(self):
        self.cost_in: Optional[torch.Tensor] = None
        self.transport_out: Optional[torch.Tensor] = None

    def pre_hook(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]):
        # inputs[0] is `cost`
        self.cost_in = inputs[0].detach()

    def fwd_hook(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        self.transport_out = output.detach()


def run_encoder_once(
    encoder: MESHEncoder, token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run encoder forward and capture (cost_in_to_sinkhorn, transport_plan)."""
    capture = SinkhornCapture()
    h_pre = encoder.sinkhorn.register_forward_pre_hook(capture.pre_hook)
    h_fwd = encoder.sinkhorn.register_forward_hook(capture.fwd_hook)
    try:
        with torch.no_grad():
            _ = encoder(token_ids)
        if capture.cost_in is None or capture.transport_out is None:
            raise RuntimeError(
                "Failed to capture cost/transport from encoder.sinkhorn hooks"
            )
        return capture.cost_in, capture.transport_out
    finally:
        h_pre.remove()
        h_fwd.remove()


def compute_auto_eps(cost: torch.Tensor, scale: float) -> torch.Tensor:
    """Auto-epsilon rule implemented in LogSinkhorn: eps = scale * median(cost_flat)."""
    B = cost.shape[0]
    cost_flat = cost.reshape(B, -1)
    median_cost = cost_flat.median(dim=-1).values.clamp(min=1e-6)  # [B]
    return scale * median_cost  # [B]


def print_table(rows: list[RunStats]) -> None:
    headers = [
        "source",
        "eps_mode",
        "scale",
        "eps",
        "std(C)",
        "median(C)",
        "std/eps",
        "row_sum",
        "n_eff_mean",
        "n_eff_p10",
        "n_eff_p50",
        "n_eff_p90",
        "n_eff_raw_mean",
    ]

    def fmt(r: RunStats) -> list[str]:
        scale_s = "-" if r.auto_scale is None else f"{r.auto_scale:.2f}"
        return [
            f"{r.source}",
            f"{r.eps_mode}",
            scale_s,
            f"{r.eps_value:.4g}",
            f"{r.cost_std:.4g}",
            f"{r.cost_median:.4g}",
            f"{r.sigma_over_eps:.3g}",
            f"{r.row_sum_mean:.4g}",
            f"{r.n_eff_mean:.3g}",
            f"{r.n_eff_p10:.3g}",
            f"{r.n_eff_p50:.3g}",
            f"{r.n_eff_p90:.3g}",
            f"{r.n_eff_raw_mean:.3g}",
        ]

    table = [headers] + [fmt(r) for r in rows]
    widths = [
        max(len(table[i][j]) for i in range(len(table))) for j in range(len(headers))
    ]

    def line(parts: list[str]) -> str:
        return "  ".join(p.ljust(widths[i]) for i, p in enumerate(parts))

    print(line(table[0]))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(fmt(r)))


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)
    torch.manual_seed(0)

    cfg = SEMConfig()  # defaults
    if cfg.encoder.simple_mode:
        raise RuntimeError("Expected cfg.encoder.simple_mode=False so Sinkhorn is used")

    B, S = 2, 64
    K = cfg.encoder.sdr_candidates

    token_ids = torch.randint(0, cfg.model.vocab_size, (B, S), dtype=torch.long)

    encoder = MESHEncoder(
        vocab_size=cfg.model.vocab_size,
        hidden_dim=cfg.model.hidden_dim,
        sdr_sparsity=cfg.encoder.sdr_sparsity,
        sdr_candidates=cfg.encoder.sdr_candidates,
        sinkhorn_epsilon=cfg.encoder.sinkhorn_epsilon,
        sinkhorn_max_iter=cfg.encoder.sinkhorn_max_iter,
        sinkhorn_tol=cfg.encoder.sinkhorn_tol,
        sinkhorn_auto_epsilon=cfg.encoder.sinkhorn_auto_epsilon,
        sinkhorn_auto_epsilon_scale=cfg.encoder.sinkhorn_auto_epsilon_scale,
        max_seq_length=cfg.model.max_seq_length,
        low_vram_mode=cfg.training.low_vram_mode,
        soft_sparse=cfg.encoder.soft_sparse,
        soft_sparse_temp=cfg.encoder.soft_sparse_temp,
        simple_mode=False,
    )
    encoder.eval()

    # Synthetic costs (matches validate_theorems.py experimental block)
    torch.manual_seed(42)
    synth_cost = torch.rand(B, S, K) * 0.5 + 0.1

    rows: list[RunStats] = []

    # Run 1: fixed epsilon
    encoder.sinkhorn.auto_epsilon = False
    encoder.sinkhorn.epsilon = 0.05
    cost_real, T_real = run_encoder_once(encoder, token_ids)
    rows.append(
        summarize(
            cost_real,
            T_real,
            source="real",
            eps_mode="fixed",
            eps_value=0.05,
            auto_scale=None,
        )
    )
    T_synth = encoder.sinkhorn(synth_cost)
    rows.append(
        summarize(
            synth_cost,
            T_synth,
            source="synthetic",
            eps_mode="fixed",
            eps_value=0.05,
            auto_scale=None,
        )
    )

    # Run 2+: auto epsilon with scales
    for scale in [0.05, 0.10, 0.20, 0.50]:
        encoder.sinkhorn.auto_epsilon = True
        encoder.sinkhorn.auto_epsilon_scale = scale

        cost_real_i, T_real_i = run_encoder_once(encoder, token_ids)
        eps_real = compute_auto_eps(cost_real_i, scale)
        rows.append(
            summarize(
                cost_real_i,
                T_real_i,
                source="real",
                eps_mode="auto",
                eps_value=eps_real.mean().item(),
                auto_scale=scale,
            )
        )

        T_synth_i = encoder.sinkhorn(synth_cost)
        eps_synth = compute_auto_eps(synth_cost, scale)
        rows.append(
            summarize(
                synth_cost,
                T_synth_i,
                source="synthetic",
                eps_mode="auto",
                eps_value=eps_synth.mean().item(),
                auto_scale=scale,
            )
        )

    print("SEM LogSinkhorn n_eff diagnostic")
    print(f"B={B}, S={S}, K={K}, default_sdr_sparsity={cfg.encoder.sdr_sparsity}")
    print(f"Expected Sinkhorn row sum (uniform source marginal) = 1/S = {1.0 / S:.6f}")
    print()
    print_table(rows)

    # Quick interpretation aid
    print("\nInterpretation notes")
    print("- n_eff_mean uses row-normalized entropy (Shannon).")
    print(
        "- n_eff_raw_mean matches the unnormalized formula used in validate_theorems.py; it is not comparable to K."
    )
    print("- If std(C)/eps >> 1, expect near-deterministic (low-entropy) transport.")


if __name__ == "__main__":
    main()
