"""Diagnose where the wavefunction magnitude is exploding."""

import torch
import logging
from sem.config import SEMConfig
from sem.model import SEMModel
from sem.data.tokenizer import SEMTokenizer
from sem.utils.complex_adamw import ComplexAdamW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def diagnose_magnitude_explosion(config_path: str):
    """Track wavefunction magnitude through network during training."""

    # Load config
    config = SEMConfig.from_yaml(config_path)

    # Create small model for testing
    config.model.hidden_dim = 64
    config.model.num_layers = 2
    config.model.max_seq_length = 128
    config.training.micro_batch_size = 2
    config.training.learning_rate = 1e-3

    model = SEMModel(config)
    optimizer = ComplexAdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        temperature=1e-5,
    )

    # Create a simple batch
    batch_size = 2
    seq_len = 64
    token_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))

    logger.info("=" * 80)
    logger.info("MAGNITUDE EXPLOSION DIAGNOSTIC")
    logger.info("=" * 80)

    model.train()

    for step in range(5):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"STEP {step}")
        logger.info(f"{'=' * 80}")

        optimizer.zero_grad()

        # Manually trace through network
        logger.info("\nFORWARD PASS MAGNITUDES:")

        # Encoder
        psi = model.encoder(token_ids)
        mag = torch.abs(psi).mean().item()
        max_mag = torch.abs(psi).max().item()
        logger.info(f"  After Encoder:       mean={mag:.6f}, max={max_mag:.6f}")

        # Mamba layers
        for i, mamba_layer in enumerate(model.mamba_layers):
            psi_before = psi.clone()
            psi = mamba_layer(psi)
            mag = torch.abs(psi).mean().item()
            max_mag = torch.abs(psi).max().item()
            delta = torch.abs(psi - psi_before).mean().item()
            logger.info(f"  After Mamba {i}:       mean={mag:.6f}, max={max_mag:.6f}, delta={delta:.6f}")

        # Propagator
        psi_before = psi.clone()
        psi = model.propagator(psi)
        mag = torch.abs(psi).mean().item()
        max_mag = torch.abs(psi).max().item()
        delta = torch.abs(psi - psi_before).mean().item()
        logger.info(f"  After Propagator:    mean={mag:.6f}, max={max_mag:.6f}, delta={delta:.6f}")

        # Final norm
        psi_before = psi.clone()
        psi = model.final_norm(psi)
        mag = torch.abs(psi).mean().item()
        max_mag = torch.abs(psi).max().item()
        delta = torch.abs(psi - psi_before).mean().item()
        logger.info(f"  After Final Norm:    mean={mag:.6f}, max={max_mag:.6f}, delta={delta:.6f}")

        # Born collapse
        output = model.sampler(psi, sample=False)
        amp_sq_sum = output["amp_sq_sum_raw"][0, 0].item()
        logger.info(f"  Born collapse sum:   {amp_sq_sum:.6f} (should be ~1.0 after normalization)")

        # Full forward pass
        output = model(token_ids, targets=token_ids)
        loss = output["loss"]
        unitary_div = output.get("unitary_divergence", 0.0)

        logger.info(f"\nLOSS: {loss.item():.4f}, UNITARY DIV: {unitary_div:.4f}")

        # Backward pass
        loss.backward()

        # Check gradient norms
        logger.info("\nGRADIENT NORMS (Top 5):")
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm

        sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, norm in sorted_grads:
            logger.info(f"  {name}: {norm:.6f}")

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()

        # Check parameter magnitude growth
        logger.info("\nPARAMETER MAGNITUDES (Top 5):")
        param_mags = {}
        for name, param in model.named_parameters():
            mag = torch.abs(param.data).mean().item()
            param_mags[name] = mag

        sorted_mags = sorted(param_mags.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, mag in sorted_mags:
            logger.info(f"  {name}: {mag:.6f}")

    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diagnose_magnitude_explosion.py <config_path>")
        sys.exit(1)

    diagnose_magnitude_explosion(sys.argv[1])
