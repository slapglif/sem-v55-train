"""Analyze the loss components to find why loss is increasing."""
import torch
import torch.nn.functional as F

print("=" * 80)
print("LOSS COMPUTATION ANALYSIS")
print("=" * 80)

# Simulate what happens in model.py lines 173-206
print("\nModel.py loss computation (lines 173-206):")
print("-" * 80)

# Create dummy data
B, S_minus_1, V = 2, 511, 50262
amp_sq = torch.softmax(torch.randn(B, S_minus_1, V), dim=-1)  # Should sum to 1.0
amp_sq_sum_normalized = torch.ones(B, S_minus_1)  # Ideal case: sum=1.0 per position
target_ids = torch.randint(0, V, (B, S_minus_1))

print(f"Shapes:")
print(f"  amp_sq: {amp_sq.shape} (Born probabilities, sum to 1.0 across vocab)")
print(f"  amp_sq_sum_normalized: {amp_sq_sum_normalized.shape} (should be ~1.0)")
print(f"  target_ids: {target_ids.shape}")

# Line 178-180: Gather target probabilities
print(f"\nLine 178-180: Gather target token probabilities")
target_amp_sq = torch.gather(amp_sq, -1, target_ids.unsqueeze(-1)).squeeze(-1)
print(f"  target_amp_sq shape: {target_amp_sq.shape}")
print(f"  target_amp_sq stats: min={target_amp_sq.min():.6f}, max={target_amp_sq.max():.6f}, mean={target_amp_sq.mean():.6f}")

# Line 181: Compute NLL
print(f"\nLine 181: NLL term")
nll_term = -torch.log(target_amp_sq + 1e-12).mean()
print(f"  nll_term = -log(target_amp_sq + 1e-12).mean()")
print(f"  nll_term = {nll_term.item():.6f}")

# Line 187: Unitary divergence
print(f"\nLine 187: Unitary divergence")
unitary_lambda = 0.1
unitary_divergence = (torch.log(amp_sq_sum_normalized + 1e-12) ** 2).mean()
print(f"  unitary_divergence = (log(amp_sq_sum_normalized + 1e-12) ** 2).mean()")
print(f"  unitary_divergence = {unitary_divergence.item():.6f}")
unitary_term = unitary_lambda * unitary_divergence.detach()
print(f"  unitary_term = {unitary_lambda} * {unitary_divergence.item():.6f} = {unitary_term.item():.6f}")

# Line 197-200: Weight penalty
print(f"\nLine 197-200: Weight penalty")
proj_real_weight = torch.randn(V, 64)  # Typical hidden_dim
proj_imag_weight = torch.randn(V, 64)
weight_penalty = (proj_real_weight.norm() ** 2 + proj_imag_weight.norm() ** 2) * 1e-3
print(f"  weight_penalty = (proj_real.weight.norm()^2 + proj_imag.weight.norm()^2) * 1e-3")
print(f"  weight_penalty = {weight_penalty.item():.6f}")

# Total loss
total_loss = nll_term + unitary_term + weight_penalty
print(f"\nTotal loss: {total_loss.item():.6f}")
print(f"  NLL contribution: {nll_term.item():.6f} ({nll_term.item() / total_loss.item() * 100:.1f}%)")
print(f"  Unitary contribution: {unitary_term.item():.6f} ({unitary_term.item() / total_loss.item() * 100:.1f}%)")
print(f"  Weight penalty contribution: {weight_penalty.item():.6f} ({weight_penalty.item() / total_loss.item() * 100:.1f}%)")

print("\n" + "=" * 80)
print("POTENTIAL ISSUES TO CHECK")
print("=" * 80)

print("\n1. ARE TARGET PROBABILITIES DECREASING?")
print("   If the model predicts uniform distributions, target_amp_sq -> 1/V -> loss increases")
print("   For V=50262: uniform probability = 1/50262 = 0.0000199")
print("   NLL for uniform = -log(0.0000199) = 10.82")
print()
print("   If model is learning, target probabilities should INCREASE over time:")
print("   - Step 0: ~uniform -> target_amp_sq ~ 0.00002 -> NLL ~ 10.8")
print("   - Step 100: learning -> target_amp_sq ~ 0.001 -> NLL ~ 6.9")
print("   - Step 1000: better -> target_amp_sq ~ 0.01 -> NLL ~ 4.6")
print()

print("2. IS THE MODEL PREDICTING UNIFORM DISTRIBUTIONS?")
print("   If Born collapse projection weights are initialized poorly:")
print("   - All tokens get similar probabilities")
print("   - target_amp_sq stays low")
print("   - NLL stays high or increases")
print()

print("3. IS UNITARY DIVERGENCE EXPLODING?")
print("   If amp_sq_sum_normalized >> 1.0:")
print("   - log(1000)^2 = 47.6 (HUGE)")
print("   - Even with lambda=0.1, unitary_term = 4.76")
print("   - This would dominate the loss")
print()
print("   Check in logs: unitary_divergence should be close to 0.0")
print("   If it's > 10, the projections are not maintaining unit norm")
print()

print("4. ARE GRADIENTS FLOWING?")
print("   Line 191: unitary_term uses .detach()")
print("   This prevents unitary loss from affecting gradients")
print("   But if NLL term has no gradient flow, loss will increase")
print()

print("5. IS WEIGHT PENALTY CAUSING ISSUES?")
print("   If projection weights are growing unbounded:")
print("   - weight.norm() increases every step")
print("   - weight_penalty increases")
print("   - Total loss increases")
print()

print("=" * 80)
print("WHAT TO CHECK IN ACTUAL TRAINING")
print("=" * 80)
print("\n1. Log individual loss components:")
print("   self.log('train/nll', nll_term)")
print("   self.log('train/unitary_div', unitary_divergence)")
print("   self.log('train/weight_penalty', weight_penalty)")
print()
print("2. Log target probabilities:")
print("   self.log('train/target_prob_mean', target_amp_sq.mean())")
print("   self.log('train/target_prob_min', target_amp_sq.min())")
print()
print("3. Log Born collapse sum:")
print("   self.log('train/amp_sq_sum', amp_sq_sum_normalized.mean())")
print()
print("4. Check if predictions are improving:")
print("   If target_amp_sq is decreasing -> model is getting WORSE")
print("   If target_amp_sq is increasing -> model is learning but total loss still increases")
print("     -> Check which component is increasing")
