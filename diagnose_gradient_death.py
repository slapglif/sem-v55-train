"""SEOP Gradient Flow Diagnostic: Trace gradient magnitude through every layer.

Runs a single forward+backward pass and reports per-parameter gradient norms.
Identifies exactly where gradients are dying in the pipeline.
"""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, "/home/mikeb/work/sem/sem-v55-train")

from sem.config import SEMConfig
from sem.model import SEMModel


def main():
    config = SEMConfig.from_yaml("configs/rtx3060_max.yaml")
    device = "cuda"

    model = SEMModel(config).to(device)
    model.train()

    B, S = 4, 256
    token_ids = torch.randint(0, config.model.vocab_size, (B, S), device=device)
    targets = torch.randint(0, config.model.vocab_size, (B, S), device=device)

    # Forward pass
    output = model(token_ids, targets=targets)
    loss = output["loss"]
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Collect gradient stats per parameter group
    print("\n" + "="*80)
    print("GRADIENT FLOW DIAGNOSTIC")
    print("="*80)

    groups = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            groups.setdefault("NO_GRAD", []).append((name, 0.0, param.numel()))
            continue
        grad_norm = param.grad.norm().item()
        grad_max = param.grad.abs().max().item()
        grad_mean = param.grad.abs().mean().item()

        # Group by module
        parts = name.split(".")
        if parts[0] == "mamba_layers":
            layer_idx = parts[1]
            submodule = ".".join(parts[2:4]) if len(parts) > 3 else parts[2]
            group_key = f"mamba_layers.{layer_idx}.{submodule}"
        else:
            group_key = parts[0]

        groups.setdefault(group_key, []).append((name, grad_norm, grad_max, grad_mean, param.numel()))

    # Print sorted by gradient norm (lowest first = potential bottleneck)
    print(f"\n{'Module':<50} {'GradNorm':>10} {'GradMax':>10} {'GradMean':>12} {'Params':>8}")
    print("-"*95)

    all_entries = []
    for group_key, params in sorted(groups.items()):
        if group_key == "NO_GRAD":
            for name, _, numel in params:
                print(f"  {'[NO GRAD] ' + name:<50} {'---':>10} {'---':>10} {'---':>12} {numel:>8}")
            continue

        total_norm = sum(p[1]**2 for p in params)**0.5
        total_params = sum(p[4] for p in params)
        max_grad = max(p[2] for p in params)
        mean_grad = sum(p[3] * p[4] for p in params) / max(total_params, 1)

        all_entries.append((group_key, total_norm, max_grad, mean_grad, total_params))

    # Sort by norm ascending to find bottlenecks
    for group_key, total_norm, max_grad, mean_grad, total_params in sorted(all_entries, key=lambda x: x[1]):
        flag = " <<< LOW" if total_norm < 0.01 else (" << MEDIUM" if total_norm < 0.1 else "")
        print(f"  {group_key:<48} {total_norm:>10.6f} {max_grad:>10.6f} {mean_grad:>12.8f} {total_params:>8}{flag}")

    # Detailed per-layer Mamba analysis
    print("\n" + "="*80)
    print("MAMBA LAYER-BY-LAYER GRADIENT FLOW")
    print("="*80)

    mamba_norms = {}
    for name, param in model.named_parameters():
        if param.grad is None or "mamba_layers" not in name:
            continue
        parts = name.split(".")
        layer_idx = int(parts[1])
        param_name = ".".join(parts[2:])
        grad_norm = param.grad.norm().item()
        mamba_norms.setdefault(param_name, {})[layer_idx] = grad_norm

    print(f"\n{'Parameter':<35} " + " ".join(f"{'L'+str(i):>8}" for i in range(config.model.num_layers)))
    print("-"*35 + "-" * (9 * config.model.num_layers))

    for param_name in sorted(mamba_norms.keys()):
        norms = mamba_norms[param_name]
        vals = " ".join(f"{norms.get(i, 0):>8.4f}" for i in range(config.model.num_layers))
        # Check for gradient death pattern (decreasing toward layer 0)
        layer_norms = [norms.get(i, 0) for i in range(config.model.num_layers)]
        if len(layer_norms) > 1 and layer_norms[0] < layer_norms[-1] * 0.1:
            flag = " <<< DEATH"
        else:
            flag = ""
        print(f"  {param_name:<33} {vals}{flag}")

    # Activation magnitude through Mamba layers (forward hooks)
    print("\n" + "="*80)
    print("ACTIVATION MAGNITUDES (|psi| at each stage)")
    print("="*80)

    model.zero_grad()
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                mag = output.abs().mean().item()
                max_mag = output.abs().max().item()
                activations[name] = (mag, max_mag)
        return hook

    hooks = []
    hooks.append(model.encoder.register_forward_hook(make_hook("encoder_out")))
    for i, layer in enumerate(model.mamba_layers):
        hooks.append(layer.register_forward_hook(make_hook(f"mamba_{i}_out")))
    hooks.append(model.final_norm.register_forward_hook(make_hook("final_norm_out")))

    with torch.no_grad():
        output = model(token_ids, targets=targets)

    for h in hooks:
        h.remove()

    print(f"\n{'Stage':<20} {'Mean |psi|':>12} {'Max |psi|':>12} {'Ratio':>10}")
    print("-"*58)

    prev_mean = None
    for name in ["encoder_out"] + [f"mamba_{i}_out" for i in range(config.model.num_layers)] + ["final_norm_out"]:
        if name in activations:
            mean_mag, max_mag = activations[name]
            ratio = mean_mag / prev_mean if prev_mean and prev_mean > 0 else 1.0
            flag = " <<< SHRINK" if ratio < 0.5 else (" <<< GROW" if ratio > 2.0 else "")
            print(f"  {name:<18} {mean_mag:>12.6f} {max_mag:>12.6f} {ratio:>10.4f}{flag}")
            prev_mean = mean_mag

    # SSM state |A| analysis
    print("\n" + "="*80)
    print("|A| DECAY ANALYSIS (per layer)")
    print("="*80)

    for i, layer in enumerate(model.mamba_layers):
        log_A_mag = layer.log_A_mag.data
        A_mag = torch.exp(-F.softplus(log_A_mag))
        A_mean = A_mag.mean().item()
        A_min = A_mag.min().item()
        A_max = A_mag.max().item()
        A_256 = (A_mag ** 256).mean().item()
        print(f"  Layer {i}: |A| mean={A_mean:.6f} min={A_min:.6f} max={A_max:.6f} | |A|^256={A_256:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
