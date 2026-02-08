#!/usr/bin/env python3
"""Fast hyperparameter sweep for SEM V5.5 using Optuna.

Searches learning rate, gradient clip, V8 gammas/thresholds, and training
params using convergence speed (loss after N steps) as the proxy metric.

Usage:
    uv run python sweep/hyperparam_sweep.py --n-trials 50 --n-steps 100
    uv run python sweep/hyperparam_sweep.py --n-trials 100 --n-steps 200 --device cuda
"""

# pyright: reportMissingTypeArgument=false

import argparse
import time
from pathlib import Path

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


SMALL_MODEL = dict(
    hidden_dim=128,
    num_layers=2,
    vocab_size=1000,
    max_seq_length=128,
)


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
    # Round all floats to 2 sig figs — the model can't distinguish more

    # -----------------------------
    # TrainingConfig (search)
    # -----------------------------
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

    # -----------------------------
    # EncoderConfig (search)
    # -----------------------------
    sdr_sparsity = trial.suggest_categorical("sdr_sparsity", [8, 16, 24, 32])
    sdr_candidates = trial.suggest_categorical("sdr_candidates", [32, 64, 128])
    sinkhorn_epsilon = _round_sig(
        trial.suggest_float("sinkhorn_epsilon", 0.01, 0.2, log=True)
    )
    sinkhorn_max_iter = trial.suggest_int("sinkhorn_max_iter", 20, 100, step=10)
    sinkhorn_auto_epsilon = trial.suggest_categorical(
        "sinkhorn_auto_epsilon", [True, False]
    )
    sinkhorn_auto_epsilon_scale = 0.05
    if sinkhorn_auto_epsilon:
        sinkhorn_auto_epsilon_scale = _round_sig(
            trial.suggest_float("sinkhorn_auto_epsilon_scale", 0.01, 0.2, log=True)
        )

    soft_sparse = trial.suggest_categorical("soft_sparse", [True, False])
    soft_sparse_temp = 0.1
    if soft_sparse:
        soft_sparse_temp = _round_sig(
            trial.suggest_float("soft_sparse_temp", 0.01, 1.0, log=True)
        )

    # -----------------------------
    # SpinorConfig (search)
    # -----------------------------
    block_size = 8  # fixed (architecture)
    hidden_dim = SMALL_MODEL["hidden_dim"]
    num_blocks = hidden_dim // block_size

    memory_horizon_ratio = _round_sig(
        trial.suggest_float("memory_horizon_ratio", 0.0, 0.5)
    )

    # -----------------------------
    # PropagatorConfig (search)
    # -----------------------------
    cayley_dt = _round_sig(trial.suggest_float("cayley_dt", 0.01, 0.5, log=True))
    nonlinear_alpha = _round_sig(
        trial.suggest_float("nonlinear_alpha", 0.01, 0.5, log=True)
    )
    laplacian_sparsity = trial.suggest_int("laplacian_sparsity", 3, 7)
    pit_gamma = _round_sig(trial.suggest_float("pit_gamma", 0.01, 2.0))
    chebyshev_degree = trial.suggest_categorical(
        "chebyshev_degree", [6, 8, 12, 16, 20, 24]
    )

    # -----------------------------
    # SamplerConfig (search)
    # -----------------------------
    temperature = _round_sig(trial.suggest_float("temperature", 0.5, 2.0))
    top_k = trial.suggest_int("top_k", 10, 100, step=10)
    top_p = _round_sig(trial.suggest_float("top_p", 0.8, 1.0))
    min_p = _round_sig(trial.suggest_float("min_p", 0.0, 0.2))
    typical_p = _round_sig(trial.suggest_float("typical_p", 0.8, 1.0))
    repetition_penalty = _round_sig(trial.suggest_float("repetition_penalty", 1.0, 1.5))
    temperature_last = trial.suggest_categorical("temperature_last", [True, False])

    # -----------------------------
    # V8Config (all features ON; search parameters)
    # -----------------------------
    mhc_streams = trial.suggest_categorical("mhc_streams", [2, 4, 8])
    mhc_num_iters = trial.suggest_categorical("mhc_num_iters", [5, 10, 20])
    mhc_tau = _round_sig(trial.suggest_float("mhc_tau", 0.01, 0.2, log=True))

    lindblad_gamma = _round_sig(
        trial.suggest_float("lindblad_gamma", 0.001, 0.1, log=True)
    )
    num_lindblad_ops = trial.suggest_int("num_lindblad_ops", 2, 8)
    curvature_threshold = _round_sig(
        trial.suggest_float("curvature_threshold", 0.01, 1.0, log=True)
    )
    condition_threshold = _round_sig(
        trial.suggest_float("condition_threshold", 10.0, 1000.0, log=True)
    )

    return SEMConfig(
        model=ModelConfig(
            hidden_dim=SMALL_MODEL["hidden_dim"],
            num_layers=SMALL_MODEL["num_layers"],
            vocab_size=SMALL_MODEL["vocab_size"],
            max_seq_length=SMALL_MODEL["max_seq_length"],
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
            num_blocks=num_blocks,
            state_dim=64,
            mimo_groups=8,
            d_conv=4,
            memory_horizon_ratio=memory_horizon_ratio,
        ),
        propagator=PropagatorConfig(
            cayley_dt=cayley_dt,
            cg_max_iter=5,
            nonlinear_alpha=nonlinear_alpha,
            laplacian_sparsity=laplacian_sparsity,
            pit_gamma=pit_gamma,
            use_chebyshev_kpm=True,
            chebyshev_degree=chebyshev_degree,
            direct_solve=False,
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
            gradient_clip=gradient_clip,
            weight_decay=weight_decay,
            unitary_lambda=unitary_lambda,
            unitary_clamp_min=unitary_clamp_min,
            unitary_clamp_max=unitary_clamp_max,
            encoder_lr_scale=encoder_lr_scale,
            warmup_steps=0,
            label_smoothing=label_smoothing,
        ),
        v8=V8Config(
            use_lindblad=True,
            use_hybrid_automata=True,
            use_quaternionic=True,
            use_mhc=True,
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
) -> dict:
    model.train()
    model.propagator_enabled = True

    encoder_params = set(id(p) for p in model.encoder.parameters())
    param_groups = [
        {
            "params": [
                p
                for p in model.parameters()
                if id(p) not in encoder_params and p.requires_grad
            ],
            "lr": config.training.learning_rate,
        },
        {
            "params": [p for p in model.encoder.parameters() if p.requires_grad],
            "lr": config.training.learning_rate * config.training.encoder_lr_scale,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
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
            return {"final_loss": float("inf"), "losses": losses, "nan_step": step}

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip
        )
        optimizer.step()

        losses.append(loss.item())

    elapsed = time.time() - t0
    return {
        "final_loss": losses[-1] if losses else float("inf"),
        "initial_loss": losses[0] if losses else float("inf"),
        "loss_reduction": (losses[0] - losses[-1]) / (losses[0] + 1e-8)
        if losses
        else 0.0,
        "losses": losses,
        "elapsed_s": elapsed,
        "steps_per_sec": n_steps / elapsed,
    }


def objective(
    trial: optuna.Trial, n_steps: int, device: str, batch_size: int, seq_len: int
) -> float:
    config = make_config(trial)

    torch.manual_seed(42)
    model = SEMModel(config).to(device)

    result = train_steps(model, config, n_steps, batch_size, seq_len, device)

    trial.set_user_attr("initial_loss", result.get("initial_loss", float("inf")))
    trial.set_user_attr("loss_reduction", result.get("loss_reduction", 0.0))
    trial.set_user_attr("steps_per_sec", result.get("steps_per_sec", 0.0))
    if "nan_step" in result:
        trial.set_user_attr("nan_step", result["nan_step"])

    return result["final_loss"]


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(description="SEM V5.5 Hyperparameter Sweep")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--study-name", type=str, default="sem-v55-hyperparam")
    parser.add_argument("--db", type=str, default="sqlite:///sweep/hyperparam.db")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
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
    print("BEST TRIAL")
    print("=" * 60)
    best = study.best_trial
    print(f"  Loss: {best.value:.4f}")
    print(f"  Loss reduction: {best.user_attrs.get('loss_reduction', 'N/A'):.1%}")
    print(f"  Steps/sec: {best.user_attrs.get('steps_per_sec', 'N/A'):.1f}")
    print(f"  Params:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"    {k}: {_round_sig(v)}")
        else:
            print(f"    {k}: {v}")

    top_5 = sorted(
        study.trials, key=lambda t: t.value if t.value is not None else float("inf")
    )[:5]
    print(f"\nTop 5 trials:")
    for i, t in enumerate(top_5):
        lr = t.params.get("learning_rate", "?")
        gc = t.params.get("gradient_clip", "?")
        print(f"  #{i + 1}: loss={t.value:.4f}, lr={lr:.2e}, clip={gc:.1f}")
        params_str = ", ".join(
            f"{k}={_round_sig(v) if isinstance(v, float) else v}"
            for k, v in sorted(t.params.items())
        )
        print(f"    {params_str}")


if __name__ == "__main__":
    main()
