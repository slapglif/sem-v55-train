#!/usr/bin/env python3
"""Fast hyperparameter sweep for SEM V5.5 using Optuna.

Searches learning rate, gradient clip, V8 gammas/thresholds, and training
params using convergence speed (loss after N steps) as the proxy metric.

Usage:
    uv run python sweep/hyperparam_sweep.py --n-trials 50 --n-steps 100
    uv run python sweep/hyperparam_sweep.py --n-trials 100 --n-steps 200 --device cuda
"""

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


def make_config(trial: optuna.Trial) -> SEMConfig:
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gradient_clip = trial.suggest_float("gradient_clip", 1.0, 20.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    unitary_lambda = trial.suggest_float("unitary_lambda", 0.001, 0.1, log=True)
    encoder_lr_scale = trial.suggest_float("encoder_lr_scale", 0.001, 0.1, log=True)

    lindblad_gamma = trial.suggest_float("lindblad_gamma", 0.001, 0.1, log=True)
    num_lindblad_ops = trial.suggest_int("num_lindblad_ops", 2, 8)
    curvature_threshold = trial.suggest_float(
        "curvature_threshold", 0.01, 1.0, log=True
    )
    condition_threshold = trial.suggest_float(
        "condition_threshold", 10.0, 1000.0, log=True
    )

    cayley_dt = trial.suggest_float("cayley_dt", 0.01, 0.5, log=True)
    pit_gamma = trial.suggest_float("pit_gamma", 0.01, 2.0)

    sdr_sparsity = trial.suggest_int("sdr_sparsity", 8, 32, step=8)

    return SEMConfig(
        model=ModelConfig(**SMALL_MODEL),
        encoder=EncoderConfig(sdr_sparsity=sdr_sparsity, sdr_candidates=64),
        spinor=SpinorConfig(
            block_size=8, num_blocks=16, state_dim=32, mimo_groups=4, d_conv=4
        ),
        propagator=PropagatorConfig(
            cayley_dt=cayley_dt,
            cg_max_iter=10,
            laplacian_sparsity=5,
            direct_solve=True,
            pit_gamma=pit_gamma,
        ),
        sampler=SamplerConfig(),
        training=TrainingConfig(
            learning_rate=lr,
            gradient_clip=gradient_clip,
            weight_decay=weight_decay,
            unitary_lambda=unitary_lambda,
            encoder_lr_scale=encoder_lr_scale,
            warmup_steps=0,
        ),
        v8=V8Config(
            use_lindblad=True,
            use_hybrid_automata=True,
            use_quaternionic=True,
            use_mhc=True,
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
        print(f"    {k}: {v}")

    top_5 = sorted(
        study.trials, key=lambda t: t.value if t.value is not None else float("inf")
    )[:5]
    print(f"\nTop 5 trials:")
    for i, t in enumerate(top_5):
        lr = t.params.get("learning_rate", "?")
        gc = t.params.get("gradient_clip", "?")
        print(f"  #{i + 1}: loss={t.value:.4f}, lr={lr:.2e}, clip={gc:.1f}")


if __name__ == "__main__":
    main()
