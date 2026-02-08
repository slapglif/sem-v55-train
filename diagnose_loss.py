"""Diagnose loss calculation issues in SEM model.

This script investigates why the loss keeps increasing during training.
Checks for:
1. Numerical stability in Born collapse
2. Loss normalization issues
3. Gradient flow problems
4. Activation magnitudes
"""

import torch
import logging
from sem.config import SEMConfig
from sem.model import SEMModel
from sem.data.tokenizer import SEMTokenizer
from sem.data.lightning_datamodule import SEMDataModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def diagnose_model_numerics(config_path: str):
    """Run diagnostic checks on model numerics."""

    # Load config
    config = SEMConfig.from_yaml(config_path)

    # Create small model for testing
    config.model.hidden_dim = 64
    config.model.num_layers = 2
    config.model.max_seq_length = 128
    config.training.micro_batch_size = 2

    model = SEMModel(config)
    tokenizer = SEMTokenizer(config.training.tokenizer_path)

    # Create a simple batch
    batch_size = 2
    seq_len = 32
    token_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))

    logger.info("=" * 80)
    logger.info("LOSS CALCULATION DIAGNOSTIC")
    logger.info("=" * 80)

    # Forward pass
    model.train()
    with torch.no_grad():
        output = model(token_ids, targets=token_ids)

    logger.info(f"\n1. LOSS VALUE CHECK")
    logger.info(f"   Loss: {output['loss'].item():.6f}")
    logger.info(f"   Unitary divergence: {output.get('unitary_divergence', 0.0):.6f}")

    # Check intermediate values
    logger.info(f"\n2. INTERMEDIATE VALUES CHECK")

    # Encode
    psi = model.encoder(token_ids)
    logger.info(f"   Encoder output shape: {psi.shape}")
    logger.info(f"   Encoder output magnitude: {torch.abs(psi).mean().item():.6f}")
    logger.info(f"   Encoder output max: {torch.abs(psi).max().item():.6f}")

    # After Mamba layers
    for i, mamba_layer in enumerate(model.mamba_layers):
        psi = mamba_layer(psi)
        logger.info(f"   After Mamba {i}: mag={torch.abs(psi).mean().item():.6f}, max={torch.abs(psi).max().item():.6f}")

    # After propagator
    psi = model.propagator(psi)
    logger.info(f"   After Propagator: mag={torch.abs(psi).mean().item():.6f}, max={torch.abs(psi).max().item():.6f}")

    # After final norm
    psi = model.final_norm(psi)
    logger.info(f"   After Final Norm: mag={torch.abs(psi).mean().item():.6f}, max={torch.abs(psi).max().item():.6f}")

    # Born collapse
    logger.info(f"\n3. BORN COLLAPSE CHECK")
    sampler_output = model.sampler(psi, sample=False)

    logits = sampler_output["logits"]
    log_probs = sampler_output["log_probs"]
    amp_sq = sampler_output["amp_sq"]
    amp_sq_sum_raw = sampler_output["amp_sq_sum_raw"]

    logger.info(f"   Logits shape: {logits.shape}")
    logger.info(f"   Logits mean: {logits.mean().item():.6f}")
    logger.info(f"   Logits std: {logits.std().item():.6f}")
    logger.info(f"   Logits min: {logits.min().item():.6f}")
    logger.info(f"   Logits max: {logits.max().item():.6f}")

    logger.info(f"\n   amp_sq (normalized probs) shape: {amp_sq.shape}")
    logger.info(f"   amp_sq sum across vocab: {amp_sq[0, 0, :].sum().item():.6f} (should be ~1.0)")
    logger.info(f"   amp_sq mean: {amp_sq.mean().item():.6e}")
    logger.info(f"   amp_sq min: {amp_sq.min().item():.6e}")
    logger.info(f"   amp_sq max: {amp_sq.max().item():.6e}")

    logger.info(f"\n   amp_sq_sum_raw: {amp_sq_sum_raw[0, 0].item():.6f} (raw sum before normalization)")
    logger.info(f"   amp_sq_sum_raw mean: {amp_sq_sum_raw.mean().item():.6f}")
    logger.info(f"   amp_sq_sum_raw std: {amp_sq_sum_raw.std().item():.6f}")

    # Check loss computation manually
    logger.info(f"\n4. MANUAL LOSS COMPUTATION")

    # Shift for next-token prediction
    amp_sq_shifted = amp_sq[:, :-1, :].contiguous()
    amp_sq_sum_raw_shifted = amp_sq_sum_raw[:, :-1].contiguous()
    target_ids = token_ids[:, 1:].contiguous()

    # Get target probabilities
    target_amp_sq = torch.gather(amp_sq_shifted, -1, target_ids.unsqueeze(-1)).squeeze(-1)
    logger.info(f"   Target amp_sq mean: {target_amp_sq.mean().item():.6e}")
    logger.info(f"   Target amp_sq min: {target_amp_sq.min().item():.6e}")
    logger.info(f"   Target amp_sq max: {target_amp_sq.max().item():.6e}")

    # NLL term
    nll_term = -torch.log(target_amp_sq + 1e-12).mean()
    logger.info(f"   NLL term: {nll_term.item():.6f}")

    # Unitary term
    unitary_lambda = config.training.unitary_lambda
    unitary_divergence = (torch.log(amp_sq_sum_raw_shifted + 1e-12) ** 2).mean()
    unitary_term = unitary_lambda * unitary_divergence
    logger.info(f"   Unitary divergence: {unitary_divergence.item():.6f}")
    logger.info(f"   Unitary term (λ={unitary_lambda}): {unitary_term.item():.6f}")

    total_loss = nll_term + unitary_term
    logger.info(f"   Total loss: {total_loss.item():.6f}")

    # Check gradient flow
    logger.info(f"\n5. GRADIENT FLOW CHECK")
    model.zero_grad()
    output = model(token_ids, targets=token_ids)
    loss = output["loss"]
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                logger.error(f"   NaN/Inf gradient in {name}")

    # Show largest gradients
    sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"   Top 5 gradient norms:")
    for name, norm in sorted_grads:
        logger.info(f"      {name}: {norm:.6f}")

    # Show smallest gradients (potential vanishing gradient)
    sorted_grads_small = sorted(grad_norms.items(), key=lambda x: x[1])[:5]
    logger.info(f"   Bottom 5 gradient norms (potential vanishing):")
    for name, norm in sorted_grads_small:
        logger.info(f"      {name}: {norm:.6e}")

    # Check for numerical issues
    logger.info(f"\n6. NUMERICAL STABILITY CHECK")

    # Re-run forward pass and check for NaN/Inf
    issues = []
    for i in range(5):
        output = model(token_ids, targets=token_ids)
        loss = output["loss"]
        if torch.isnan(loss) or torch.isinf(loss):
            issues.append(f"   Step {i}: Loss is NaN/Inf: {loss.item()}")
        else:
            logger.info(f"   Step {i}: Loss = {loss.item():.6f}")

    if issues:
        logger.error("STABILITY ISSUES FOUND:")
        for issue in issues:
            logger.error(issue)
    else:
        logger.info("   No NaN/Inf detected in 5 forward passes")

    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diagnose_loss.py <config_path>")
        sys.exit(1)

    diagnose_model_numerics(sys.argv[1])
