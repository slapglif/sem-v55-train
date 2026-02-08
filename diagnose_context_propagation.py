"""Diagnose whether the SSM propagates context from previous tokens.

If the model output at position t is identical regardless of what tokens
came before it, the SSM is not propagating context — the model is a
unigram (bag-of-tokens) model.
"""
import torch
import logging
from sem.config import SEMConfig
from sem.model import SEMModel

logging.basicConfig(level=logging.INFO, format="%(message)s")

config = SEMConfig.from_yaml("configs/test.yaml")
model = SEMModel(config).cuda()
model.eval()

B, S = 1, 64
V = config.model.vocab_size

# Create two sequences that share the LAST 32 tokens but differ in the FIRST 32
shared_suffix = torch.randint(0, V, (1, 32), device="cuda")
prefix_a = torch.randint(0, V, (1, 32), device="cuda")
prefix_b = torch.randint(0, V, (1, 32), device="cuda")

# Make sure prefixes are actually different
while (prefix_a == prefix_b).all():
    prefix_b = torch.randint(0, V, (1, 32), device="cuda")

seq_a = torch.cat([prefix_a, shared_suffix], dim=1)  # [1, 64]
seq_b = torch.cat([prefix_b, shared_suffix], dim=1)  # [1, 64]

print("=" * 70)
print("CONTEXT PROPAGATION DIAGNOSTIC")
print("=" * 70)
print(f"Sequence A prefix tokens: {prefix_a[0, :5].tolist()}...")
print(f"Sequence B prefix tokens: {prefix_b[0, :5].tolist()}...")
print(f"Shared suffix tokens: {shared_suffix[0, :5].tolist()}...")

with torch.no_grad():
    # Get outputs for both sequences
    out_a = model(seq_a)
    out_b = model(seq_b)

    logits_a = out_a["logits"]  # [1, 64, V]
    logits_b = out_b["logits"]  # [1, 64, V]

    # Compare logits at shared suffix positions (positions 32-63)
    suffix_logits_a = logits_a[:, 32:, :]
    suffix_logits_b = logits_b[:, 32:, :]

    # L2 distance between logits at each suffix position
    diff = (suffix_logits_a - suffix_logits_b).norm(dim=-1)  # [1, 32]
    logits_mag = suffix_logits_a.norm(dim=-1)  # [1, 32] for normalization

    print(f"\nLogits comparison at shared suffix positions (32-63):")
    print(f"  Mean L2 diff:     {diff.mean().item():.6f}")
    print(f"  Max L2 diff:      {diff.max().item():.6f}")
    print(f"  Min L2 diff:      {diff.min().item():.6f}")
    print(f"  Mean logits mag:  {logits_mag.mean().item():.6f}")
    print(f"  Relative diff:    {(diff / (logits_mag + 1e-12)).mean().item():.6f}")

    # Also check at the representation level (before sampler)
    # Re-run with hook to capture psi before sampler
    psi_outputs = {}

    def hook_fn(name):
        def fn(module, input, output):
            psi_outputs[name] = output.clone()
        return fn

    handle = model.final_norm.register_forward_hook(hook_fn("psi"))

    _ = model(seq_a)
    psi_a = psi_outputs["psi"]  # [1, 64, D] complex

    _ = model(seq_b)
    psi_b = psi_outputs["psi"]  # [1, 64, D] complex

    handle.remove()

    # Compare psi at suffix positions
    psi_diff = (psi_a[:, 32:, :] - psi_b[:, 32:, :]).abs()
    psi_mag = psi_a[:, 32:, :].abs()

    print(f"\nRepresentation (psi) comparison at shared suffix positions:")
    print(f"  Mean |diff| real:  {psi_diff.real.mean().item():.6f}")
    print(f"  Mean |diff| imag:  {psi_diff.imag.mean().item():.6f}")
    print(f"  Mean |psi| real:   {psi_mag.real.mean().item():.6f}")
    print(f"  Mean |psi| imag:   {psi_mag.imag.mean().item():.6f}")
    print(f"  Relative diff Re:  {(psi_diff.real / (psi_mag.real + 1e-12)).mean().item():.6f}")
    print(f"  Relative diff Im:  {(psi_diff.imag / (psi_mag.imag + 1e-12)).mean().item():.6f}")

    # Per-position breakdown
    rel_diff = (diff / (logits_mag + 1e-12))[0]
    print(f"\nPer-position relative logit diff (first 8 suffix positions):")
    for i in range(min(8, rel_diff.shape[0])):
        print(f"  Position {32+i}: {rel_diff[i].item():.6f} (L2={diff[0,i].item():.4f})")

    if diff.mean().item() < 1e-4:
        print(f"\n*** DIAGNOSIS: SSM NOT PROPAGATING CONTEXT ***")
        print(f"    Logit diff ≈ 0 → model output is independent of prefix tokens")
        print(f"    The SSM recurrence is not accumulating useful state")
    elif (diff / (logits_mag + 1e-12)).mean().item() < 0.01:
        print(f"\n*** DIAGNOSIS: SSM WEAKLY PROPAGATING CONTEXT ***")
        print(f"    Logit diff < 1% of magnitude → SSM provides tiny perturbation")
        print(f"    Context signal exists but is overwhelmed by residual")
    else:
        print(f"\n*** DIAGNOSIS: SSM IS PROPAGATING CONTEXT ***")
        print(f"    Logits depend significantly on prefix tokens")
        print(f"    The model CAN use context but may need more training")
