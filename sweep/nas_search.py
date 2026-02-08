#!/usr/bin/env python3
"""Fast Neural Architecture Search for SEM V5.5 using Optuna.

Searches model architecture (hidden_dim, num_layers, state_dim, mimo_groups)
and V8 feature combinations using convergence speed as proxy metric.

Usage:
    uv run python sweep/nas_search.py --n-trials 100 --n-steps 100
    uv run python sweep/nas_search.py --n-trials 200 --n-steps 200 --device cuda
"""

import argparse
import time

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
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 8, step=2)
    state_dim = trial.suggest_categorical("state_dim", [16, 32, 64])
    mimo_groups = trial.suggest_categorical("mimo_groups", [2, 4, 8])
    block_size = trial.suggest_categorical("block_size", [4, 8, 16])
    d_conv = trial.suggest_int("d_conv", 2, 4)

    sdr_sparsity = trial.suggest_int("sdr_sparsity", 8, 32, step=8)
    sdr_candidates = trial.suggest_categorical("sdr_candidates", [32, 64, 128])

    use_lindblad = trial.suggest_categorical("use_lindblad", [True, False])
    use_hybrid = trial.suggest_categorical("use_hybrid_automata", [True, False])
    use_quat = trial.suggest_categorical("use_quaternionic", [True, False])
    use_mhc = trial.suggest_categorical("use_mhc", [True, False])

    cayley_dt = _round_sig(trial.suggest_float("cayley_dt", 0.01, 0.5, log=True))
    laplacian_sparsity = trial.suggest_int("laplacian_sparsity", 3, 7)

    return SEMConfig(
        model=ModelConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            vocab_size=1000,
            max_seq_length=128,
        ),
        encoder=EncoderConfig(sdr_sparsity=sdr_sparsity, sdr_candidates=sdr_candidates),
        spinor=SpinorConfig(
            block_size=block_size,
            num_blocks=hidden_dim // block_size,
            state_dim=state_dim,
            mimo_groups=mimo_groups,
            d_conv=d_conv,
        ),
        propagator=PropagatorConfig(
            cayley_dt=cayley_dt,
            cg_max_iter=10,
            laplacian_sparsity=laplacian_sparsity,
            direct_solve=True,
            pit_gamma=0.1,
        ),
        sampler=SamplerConfig(),
        training=TrainingConfig(
            learning_rate=3e-4,
            gradient_clip=5.0,
            weight_decay=0.01,
            warmup_steps=0,
        ),
        v8=V8Config(
            use_lindblad=use_lindblad,
            use_hybrid_automata=use_hybrid,
            use_quaternionic=use_quat,
            use_mhc=use_mhc,
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
    )

    print("\n" + "=" * 60)
    print("BEST ARCHITECTURE")
    print("=" * 60)
    best = study.best_trial
    print(f"  Adjusted Loss: {best.value:.4f}")
    print(f"  Parameters: {best.user_attrs.get('param_count', 'N/A'):,}")
    print(f"  Loss reduction: {best.user_attrs.get('loss_reduction', 'N/A'):.1%}")
    print(f"  Steps/sec: {best.user_attrs.get('steps_per_sec', 'N/A'):.1f}")
    print(f"\n  Architecture:")
    for k in [
        "hidden_dim",
        "num_layers",
        "state_dim",
        "mimo_groups",
        "block_size",
        "d_conv",
    ]:
        print(f"    {k}: {best.params.get(k, '?')}")
    print(f"\n  V8 Features:")
    for k in ["use_lindblad", "use_hybrid_automata", "use_quaternionic", "use_mhc"]:
        print(f"    {k}: {best.params.get(k, '?')}")
    print(f"\n  Encoder:")
    for k in ["sdr_sparsity", "sdr_candidates"]:
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
