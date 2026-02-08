"""Diagnose gradient flow through the SEM model to find where learning breaks."""
import torch
import logging
from sem.config import SEMConfig
from sem.model import SEMModel

logging.basicConfig(level=logging.INFO, format="%(message)s")

config = SEMConfig.from_yaml("configs/test.yaml")
model = SEMModel(config).cuda()
model.train()

# Synthetic batch
B, S = 4, 256
token_ids = torch.randint(0, config.model.vocab_size, (B, S), device="cuda")

# Forward + backward
output = model(token_ids, targets=token_ids)
loss = output["loss"]
loss.backward()

print(f"\n{'='*70}")
print(f"Loss: {loss.item():.4f}")
print(f"{'='*70}\n")

# Collect gradient info per module
groups = {}
for name, param in model.named_parameters():
    if param.grad is None:
        groups.setdefault("NO_GRAD", []).append((name, param.numel(), 0.0))
        continue

    grad_norm = param.grad.norm().item()
    # Group by top-level module
    parts = name.split(".")
    group = parts[0]
    if group == "mamba_layers":
        layer_idx = parts[1]
        submodule = parts[2] if len(parts) > 2 else "?"
        group = f"mamba[{layer_idx}].{submodule}"

    groups.setdefault(group, []).append((name, param.numel(), grad_norm))

print(f"{'Module':<40} {'Params':>10} {'Grad Norm':>12} {'Grad/Param':>12}")
print("-" * 76)

for group_name in sorted(groups.keys()):
    params_list = groups[group_name]
    total_params = sum(p[1] for p in params_list)
    max_grad = max(p[2] for p in params_list)
    avg_grad = sum(p[2] for p in params_list) / len(params_list)

    # Flag issues
    flag = ""
    if max_grad == 0:
        flag = " *** DEAD ***"
    elif max_grad < 1e-6:
        flag = " ** VANISHING **"
    elif max_grad > 100:
        flag = " * EXPLODING *"

    print(f"{group_name:<40} {total_params:>10,} {max_grad:>12.6f} {avg_grad:>12.6f}{flag}")

# Check Re vs Im gradient magnitude in final hidden state
print(f"\n{'='*70}")
print("Re vs Im gradient analysis (last hidden state before sampler):")

# Run forward again with hooks
re_grads = []
im_grads = []

def hook_fn(module, grad_input, grad_output):
    if grad_output[0] is not None:
        g = grad_output[0]
        if g.is_complex():
            re_grads.append(g.real.norm().item())
            im_grads.append(g.imag.norm().item())

model.zero_grad()
handle = model.final_norm.register_full_backward_hook(hook_fn)
output = model(token_ids, targets=token_ids)
output["loss"].backward()
handle.remove()

if re_grads:
    print(f"  Gradient to Re(h): {re_grads[0]:.6f}")
    print(f"  Gradient to Im(h): {im_grads[0]:.6f}")
    print(f"  Re/Im ratio: {re_grads[0]/(im_grads[0]+1e-12):.2f}")

# Check what the model is actually predicting
print(f"\n{'='*70}")
print("Prediction analysis:")
with torch.no_grad():
    output = model(token_ids)
    logits = output["logits"]
    probs = torch.softmax(logits[:, -1, :], dim=-1)  # Last position

    # Entropy
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    print(f"  Output entropy: {entropy:.4f} (uniform={torch.log(torch.tensor(float(config.model.vocab_size))):.4f})")

    # Top-k concentration
    top1 = probs.max(dim=-1).values.mean()
    top10 = probs.topk(10, dim=-1).values.sum(dim=-1).mean()
    print(f"  Top-1 prob: {top1:.6f}")
    print(f"  Top-10 prob: {top10:.6f}")

    # Check if bias dominates
    bias = model.sampler.output_bias
    bias_probs = torch.softmax(bias, dim=-1)
    bias_entropy = -(bias_probs * (bias_probs + 1e-10).log()).sum()
    print(f"  Bias entropy: {bias_entropy:.4f}")
    print(f"  Bias top-1: {bias_probs.max():.6f}")
