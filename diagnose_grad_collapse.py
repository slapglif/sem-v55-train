"""Diagnose gradient collapse in SEM V5.5 — per-layer gradient norm analysis.

Uses synthetic data to isolate architectural gradient flow issues.
"""

import torch
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from sem.config import SEMConfig
from sem.model import SEMModel


def analyze_gradients(model):
    """Group parameters by component and report gradient norms."""
    groups = defaultdict(lambda: {"params": 0, "grad_norm_sq": 0.0, "max_grad": 0.0})

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # Classify by component
        if "encoder" in name or "embedding" in name:
            group = "encoder"
        elif "sampler" in name:
            group = "sampler"
        elif "ssm_output_scale" in name:
            # Extract layer index
            for part in name.split("."):
                if part.isdigit():
                    group = f"L{part}.ssm_scale"
                    break
            else:
                group = "ssm_scale"
        elif "residual_scale" in name:
            for part in name.split("."):
                if part.isdigit():
                    group = f"L{part}.res_scale"
                    break
            else:
                group = "res_scale"
        elif "A_log_mag" in name or "A_phase" in name:
            for part in name.split("."):
                if part.isdigit():
                    group = f"L{part}.A_param"
                    break
            else:
                group = "A_param"
        elif "B_proj" in name or "C_proj" in name:
            for part in name.split("."):
                if part.isdigit():
                    group = f"L{part}.BC"
                    break
            else:
                group = "BC"
        elif "in_proj" in name or "out_proj" in name:
            for part in name.split("."):
                if part.isdigit():
                    group = f"L{part}.proj"
                    break
            else:
                group = "proj"
        elif "dt_proj" in name:
            for part in name.split("."):
                if part.isdigit():
                    group = f"L{part}.dt"
                    break
            else:
                group = "dt"
        elif "quantizer" in name:
            group = "quantizer"
        else:
            group = "other"

        grad = param.grad
        if grad.is_complex():
            grad_real = torch.view_as_real(grad)
            norm = grad_real.norm().item()
            max_val = grad_real.abs().max().item()
        else:
            norm = grad.norm().item()
            max_val = grad.abs().max().item()

        groups[group]["params"] += param.numel()
        groups[group]["grad_norm_sq"] += norm ** 2
        groups[group]["max_grad"] = max(groups[group]["max_grad"], max_val)

    return {k: {
        "params": v["params"],
        "grad_norm": v["grad_norm_sq"] ** 0.5,
        "max_grad": v["max_grad"],
    } for k, v in groups.items()}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = SEMConfig.from_yaml("configs/rtx3060_max.yaml")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    logger.info("Building model...")
    model = SEMModel(config).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total:,}")

    # Simple optimizer
    from sem.utils.complex_adamw import ComplexAdamW
    encoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name or "embedding" in name:
            encoder_params.append(param)
        else:
            other_params.append(param)

    lr = config.training.learning_rate  # 7e-3
    enc_lr = lr * config.training.encoder_lr_scale  # 0.3 * 7e-3 = 2.1e-3
    optimizer = ComplexAdamW([
        {"params": encoder_params, "lr": enc_lr},
        {"params": other_params, "lr": lr},
    ], weight_decay=config.training.weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss()
    vocab_size = config.model.vocab_size
    seq_len = config.model.max_seq_length
    batch_size = config.training.batch_size

    logger.info(f"\n{'='*70}")
    logger.info(f"Per-Layer Gradient Analysis (synthetic data, full LR={lr})")
    logger.info(f"{'='*70}")

    for step in range(15):
        # Synthetic random tokens
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()
        output = model(input_ids, targets=input_ids)  # targets = input_ids for self-supervised
        loss = output["loss"]
        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip).item()

        grad_info = analyze_gradients(model)

        if step in [0, 4, 9, 14]:
            logger.info(f"\n--- Step {step} | loss={loss.item():.4f} | total_grad_norm={total_norm:.4f} ---")
            # Sort by layer
            for group_name in sorted(grad_info.keys()):
                info = grad_info[group_name]
                logger.info(f"  {group_name:25s} | norm={info['grad_norm']:10.6f} | max={info['max_grad']:10.6f} | params={info['params']:>10,}")

        optimizer.step()

    # Check parameter values after training
    logger.info(f"\n{'='*70}")
    logger.info("Parameter Values After 15 Steps")
    logger.info(f"{'='*70}")

    for name, param in model.named_parameters():
        if any(k in name for k in ["A_log_mag", "ssm_output_scale", "residual_scale", "D"]):
            if param.is_complex():
                val = torch.view_as_real(param).detach()
            else:
                val = param.detach()
            if val.numel() <= 10:
                logger.info(f"  {name}: {val.cpu().numpy()}")
            else:
                logger.info(f"  {name}: mean={val.mean().item():.4f}, std={val.std().item():.4f}")

    # Check |A| distribution
    logger.info(f"\n{'='*70}")
    logger.info("|A| Analysis (SSM recurrence decay)")
    logger.info(f"{'='*70}")
    for name, param in model.named_parameters():
        if "A_log_mag" in name:
            A_mag = param.detach().exp()
            logger.info(f"  {name}: |A| mean={A_mag.mean().item():.6f}, min={A_mag.min().item():.6f}, max={A_mag.max().item():.6f}")
            # Effective memory length = 1/(1-|A|)
            mem_len = (1.0 / (1.0 - A_mag.clamp(max=0.9999))).mean().item()
            logger.info(f"    Effective memory length: {mem_len:.0f} tokens")


if __name__ == "__main__":
    main()
