#!/usr/bin/env python3
"""Fast Neural Architecture Search for SEM V5.5 using Optuna.

Searches model architecture (hidden_dim, num_layers, state_dim, mimo_groups)
and V8 feature combinations using convergence speed as proxy metric.

Usage:
    uv run python sweep/nas_search.py --n-trials 100 --n-steps 100
    uv run python sweep/nas_search.py --n-trials 200 --n-steps 200 --device cuda
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import optuna
import torch
import torch.nn.functional as F

from sem.config import (
    SEMConfig,
    ModelConfig,
    EncoderConfig,
    SpinorConfig,
    PropagatorConfig,
    SamplerConfig,
    TrainingConfig,
    V8Config,
)
from sem.model import SEMModel


def _round_sig(x: float, sig: int = 2) -> float:
    """Round to `sig` significant figures. 0.000384729 → 0.00038."""
    if x == 0:
        return 0.0
    import math

    d = math.ceil(math.log10(abs(x)))
    power = sig - d
    factor = 10**power
    return round(x * factor) / factor


def make_config(trial: optuna.Trial) -> SEMConfig:
    # Architecture
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 8, step=2)
    vocab_size = 1000
    max_seq_length = 128

    # Encoder
    sdr_sparsity = trial.suggest_int("sdr_sparsity", 8, 32, step=8)
    sdr_candidates = trial.suggest_categorical("sdr_candidates", [32, 64, 128])
    sinkhorn_epsilon = _round_sig(
        trial.suggest_float("sinkhorn_epsilon", 0.01, 0.2, log=True)
    )
    sinkhorn_max_iter = trial.suggest_int("sinkhorn_max_iter", 20, 100, step=10)
    sinkhorn_auto_epsilon = trial.suggest_categorical(
        "sinkhorn_auto_epsilon", [True, False]
    )
    if sinkhorn_auto_epsilon:
        sinkhorn_auto_epsilon_scale = _round_sig(
            trial.suggest_float("sinkhorn_auto_epsilon_scale", 0.01, 0.2, log=True)
        )
    else:
        sinkhorn_auto_epsilon_scale = EncoderConfig().sinkhorn_auto_epsilon_scale

    soft_sparse = trial.suggest_categorical("soft_sparse", [True, False])
    if soft_sparse:
        soft_sparse_temp = _round_sig(
            trial.suggest_float("soft_sparse_temp", 0.01, 1.0, log=True)
        )
    else:
        soft_sparse_temp = EncoderConfig().soft_sparse_temp

    # Spinor
    block_size = trial.suggest_categorical("block_size", [4, 8, 16])
    state_dim = trial.suggest_categorical("state_dim", [16, 32, 64])
    mimo_groups = trial.suggest_categorical("mimo_groups", [2, 4, 8])
    d_conv = trial.suggest_categorical("d_conv", [2, 4])
    memory_horizon_ratio = _round_sig(
        trial.suggest_float("memory_horizon_ratio", 0.0, 0.5)
    )

    # Propagator
    cayley_dt = _round_sig(trial.suggest_float("cayley_dt", 0.01, 0.5, log=True))
    nonlinear_alpha = _round_sig(
        trial.suggest_float("nonlinear_alpha", 0.01, 0.5, log=True)
    )
    laplacian_sparsity = trial.suggest_int("laplacian_sparsity", 3, 7)
    pit_gamma = _round_sig(trial.suggest_float("pit_gamma", 0.01, 2.0))
    # KPM always enabled — replaces CG solver entirely (O(D) vs O(D³))
    use_chebyshev_kpm = True
    chebyshev_degree = trial.suggest_categorical(
        "chebyshev_degree", [6, 8, 12, 16, 20, 24]
    )
    direct_solve = False

    # Sampler
    temperature = _round_sig(trial.suggest_float("temperature", 0.5, 2.0))
    top_k = trial.suggest_int("top_k", 10, 100, step=10)
    top_p = _round_sig(trial.suggest_float("top_p", 0.8, 1.0))
    min_p = _round_sig(trial.suggest_float("min_p", 0.0, 0.2))
    typical_p = _round_sig(trial.suggest_float("typical_p", 0.8, 1.0))
    repetition_penalty = _round_sig(trial.suggest_float("repetition_penalty", 1.0, 1.5))
    temperature_last = trial.suggest_categorical("temperature_last", [True, False])

    # Training
    learning_rate = _round_sig(
        trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    )
    weight_decay = _round_sig(trial.suggest_float("weight_decay", 1e-4, 0.1, log=True))
    encoder_lr_scale = _round_sig(
        trial.suggest_float("encoder_lr_scale", 0.001, 0.1, log=True)
    )
    gradient_clip = _round_sig(trial.suggest_float("gradient_clip", 1.0, 20.0))
    unitary_lambda = _round_sig(
        trial.suggest_float("unitary_lambda", 0.001, 0.1, log=True)
    )
    unitary_clamp_min = _round_sig(
        trial.suggest_float("unitary_clamp_min", 1e-4, 1e-2, log=True)
    )
    unitary_clamp_max = _round_sig(
        trial.suggest_float("unitary_clamp_max", 1.0, 100.0, log=True)
    )
    label_smoothing = _round_sig(trial.suggest_float("label_smoothing", 0.0, 0.15))

    # V8
    use_lindblad = trial.suggest_categorical("use_lindblad", [True, False])
    use_hybrid = trial.suggest_categorical("use_hybrid_automata", [True, False])
    use_quat = trial.suggest_categorical("use_quaternionic", [True, False])
    use_mhc = trial.suggest_categorical("use_mhc", [True, False])

    if use_mhc:
        mhc_streams = trial.suggest_categorical("mhc_streams", [2, 4, 8])
        mhc_num_iters = trial.suggest_categorical("mhc_num_iters", [5, 10, 20])
        mhc_tau = _round_sig(trial.suggest_float("mhc_tau", 0.01, 0.2, log=True))
    else:
        mhc_streams = V8Config().mhc_streams
        mhc_num_iters = V8Config().mhc_num_iters
        mhc_tau = V8Config().mhc_tau

    if use_lindblad:
        lindblad_gamma = _round_sig(
            trial.suggest_float("lindblad_gamma", 0.001, 0.1, log=True)
        )
        num_lindblad_ops = trial.suggest_int("num_lindblad_ops", 2, 8)
    else:
        lindblad_gamma = V8Config().lindblad_gamma
        num_lindblad_ops = V8Config().num_lindblad_ops

    if use_hybrid:
        curvature_threshold = _round_sig(
            trial.suggest_float("curvature_threshold", 0.01, 1.0, log=True)
        )
    else:
        curvature_threshold = V8Config().curvature_threshold

    if use_quat:
        condition_threshold = _round_sig(
            trial.suggest_float("condition_threshold", 10.0, 1000.0, log=True)
        )
    else:
        condition_threshold = V8Config().condition_threshold

    return SEMConfig(
        model=ModelConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
        ),
        encoder=EncoderConfig(
            sdr_sparsity=sdr_sparsity,
            sdr_candidates=sdr_candidates,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_max_iter=sinkhorn_max_iter,
            sinkhorn_auto_epsilon=sinkhorn_auto_epsilon,
            sinkhorn_auto_epsilon_scale=sinkhorn_auto_epsilon_scale,
            soft_sparse=soft_sparse,
            soft_sparse_temp=soft_sparse_temp,
        ),
        spinor=SpinorConfig(
            block_size=block_size,
            num_blocks=hidden_dim // block_size,
            state_dim=state_dim,
            mimo_groups=mimo_groups,
            d_conv=d_conv,
            memory_horizon_ratio=memory_horizon_ratio,
        ),
        propagator=PropagatorConfig(
            cayley_dt=cayley_dt,
            cg_max_iter=10,
            nonlinear_alpha=nonlinear_alpha,
            laplacian_sparsity=laplacian_sparsity,
            direct_solve=direct_solve,
            pit_gamma=pit_gamma,
            use_chebyshev_kpm=use_chebyshev_kpm,
            chebyshev_degree=chebyshev_degree,
        ),
        sampler=SamplerConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            repetition_penalty=repetition_penalty,
            temperature_last=temperature_last,
        ),
        training=TrainingConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            encoder_lr_scale=encoder_lr_scale,
            gradient_clip=gradient_clip,
            warmup_steps=0,
            unitary_lambda=unitary_lambda,
            unitary_clamp_min=unitary_clamp_min,
            unitary_clamp_max=unitary_clamp_max,
            label_smoothing=label_smoothing,
        ),
        v8=V8Config(
            use_lindblad=use_lindblad,
            use_hybrid_automata=use_hybrid,
            use_quaternionic=use_quat,
            use_mhc=use_mhc,
            mhc_streams=mhc_streams,
            mhc_num_iters=mhc_num_iters,
            mhc_tau=mhc_tau,
            lindblad_gamma=lindblad_gamma,
            num_lindblad_ops=num_lindblad_ops,
            curvature_threshold=curvature_threshold,
            condition_threshold=condition_threshold,
        ),
    )


def train_steps(
    model: SEMModel,
    config: SEMConfig,
    n_steps: int,
    batch_size: int,
    seq_len: int,
    device: str,
) -> dict[str, Any]:
    model.train()
    model.propagator_enabled = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        tokens = torch.randint(
            1, config.model.vocab_size, (batch_size, seq_len), device=device
        )
        output = model(tokens, targets=tokens)
        loss = output["loss"]

        if not torch.isfinite(loss):
            return {
                "final_loss": float("inf"),
                "losses": losses,
                "nan_step": step,
                "param_count": sum(p.numel() for p in model.parameters()),
            }

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip
        )
        optimizer.step()

        losses.append(loss.item())

    elapsed = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters())
    return {
        "final_loss": losses[-1] if losses else float("inf"),
        "initial_loss": losses[0] if losses else float("inf"),
        "loss_reduction": (losses[0] - losses[-1]) / (losses[0] + 1e-8)
        if losses
        else 0.0,
        "losses": losses,
        "elapsed_s": elapsed,
        "steps_per_sec": n_steps / elapsed,
        "param_count": param_count,
    }


def objective(
    trial: optuna.Trial, n_steps: int, device: str, batch_size: int, seq_len: int
) -> float:
    config = make_config(trial)

    torch.manual_seed(42)
    try:
        model = SEMModel(config).to(device)
    except Exception as e:
        trial.set_user_attr("error", str(e))
        return float("inf")

    param_count = sum(p.numel() for p in model.parameters())
    trial.set_user_attr("param_count", param_count)

    result = train_steps(model, config, n_steps, batch_size, seq_len, device)

    trial.set_user_attr("initial_loss", result.get("initial_loss", float("inf")))
    trial.set_user_attr("loss_reduction", result.get("loss_reduction", 0.0))
    trial.set_user_attr("steps_per_sec", result.get("steps_per_sec", 0.0))
    if "nan_step" in result:
        trial.set_user_attr("nan_step", result["nan_step"])

    final_loss = result["final_loss"]

    # Efficiency-adjusted score: penalize very large models slightly
    # Scale: 1M params → 0% penalty, 10M → 5%, 100M → 10%
    import math

    size_penalty = 0.05 * math.log10(max(param_count, 1) / 1e6)
    adjusted_loss = final_loss * (1.0 + max(0, size_penalty))

    trial.set_user_attr("adjusted_loss", adjusted_loss)
    return adjusted_loss


def main():
    parser = argparse.ArgumentParser(description="SEM V5.5 Neural Architecture Search")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--study-name", type=str, default="sem-v55-nas")
    parser.add_argument("--db", type=str, default="sqlite:///sweep/nas.db")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    study.optimize(
        lambda trial: objective(
            trial, args.n_steps, args.device, args.batch_size, args.seq_len
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
        catch=(Exception,),
    )

    print("\n" + "=" * 60)
    print("BEST ARCHITECTURE")
    print("=" * 60)
    best = study.best_trial
    print(f"  Adjusted Loss: {best.value:.4f}")
    print(f"  Parameters: {best.user_attrs.get('param_count', 'N/A'):,}")
    print(f"  Loss reduction: {best.user_attrs.get('loss_reduction', 'N/A'):.1%}")
    print(f"  Steps/sec: {best.user_attrs.get('steps_per_sec', 'N/A'):.1f}")

    hidden_dim = best.params.get("hidden_dim", 0)
    block_size = best.params.get("block_size", 1)
    num_blocks = hidden_dim // block_size if isinstance(hidden_dim, int) else "?"

    direct_solve = False

    print(f"\n  Architecture:")
    for k in ["hidden_dim", "num_layers"]:
        print(f"    {k}: {best.params.get(k, '?')}")
    print("    vocab_size: 1000 (fixed)")
    print("    max_seq_length: 128 (fixed)")

    print(f"\n  Encoder:")
    for k in [
        "sdr_sparsity",
        "sdr_candidates",
        "sinkhorn_epsilon",
        "sinkhorn_max_iter",
        "sinkhorn_auto_epsilon",
        "sinkhorn_auto_epsilon_scale",
        "soft_sparse",
        "soft_sparse_temp",
    ]:
        print(f"    {k}: {best.params.get(k, '?')}")

    print(f"\n  Spinor:")
    for k in [
        "block_size",
        "state_dim",
        "mimo_groups",
        "d_conv",
        "memory_horizon_ratio",
    ]:
        print(f"    {k}: {best.params.get(k, '?')}")
    print(f"    num_blocks: {num_blocks}")

    print(f"\n  Propagator:")
    for k in [
        "cayley_dt",
        "nonlinear_alpha",
        "laplacian_sparsity",
        "pit_gamma",
        "chebyshev_degree",
    ]:
        print(f"    {k}: {best.params.get(k, '?')}")
    print(f"    use_chebyshev_kpm: True (fixed)")
    print(f"    direct_solve: {direct_solve}")

    print(f"\n  Sampler:")
    for k in [
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "typical_p",
        "repetition_penalty",
        "temperature_last",
    ]:
        print(f"    {k}: {best.params.get(k, '?')}")

    print(f"\n  V8:")
    for k in [
        "use_lindblad",
        "lindblad_gamma",
        "num_lindblad_ops",
        "use_hybrid_automata",
        "curvature_threshold",
        "use_quaternionic",
        "condition_threshold",
        "use_mhc",
        "mhc_streams",
        "mhc_num_iters",
        "mhc_tau",
    ]:
        print(f"    {k}: {best.params.get(k, '?')}")

    print(f"\n  Training:")
    for k in [
        "learning_rate",
        "weight_decay",
        "encoder_lr_scale",
        "gradient_clip",
        "unitary_lambda",
        "unitary_clamp_min",
        "unitary_clamp_max",
        "label_smoothing",
    ]:
        print(f"    {k}: {best.params.get(k, '?')}")

    top_5 = sorted(
        study.trials, key=lambda t: t.value if t.value is not None else float("inf")
    )[:5]
    print(f"\nTop 5 architectures:")
    for i, t in enumerate(top_5):
        hd = t.params.get("hidden_dim", "?")
        nl = t.params.get("num_layers", "?")
        pc = t.user_attrs.get("param_count", "?")
        sps = t.user_attrs.get("steps_per_sec", 0)
        print(
            f"  #{i + 1}: loss={t.value:.4f}, dim={hd}, layers={nl}, params={pc:,}, speed={sps:.1f} it/s"
        )


if __name__ == "__main__":
    main()
