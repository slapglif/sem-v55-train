import torch
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sem.config import SEMConfig
from sem.model import SEMModel

config = SEMConfig.from_yaml("configs/a100_optimized.yaml")
config.model.max_seq_length = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")

model = SEMModel(config).to(device)
model.train()

B, S = 2, 128
tokens = torch.randint(0, 32768, (B, S), device=device)
freqs = torch.ones(32768, device=device) / 32768

with torch.no_grad():
    out = model(tokens, targets=tokens, token_freqs=freqs)

amp_sq = out["amp_sq"]
print(f"loss: {out['loss'].item():.4f}")
print(f"unitary_divergence: {out['unitary_divergence'].item():.4f}")
print(
    f"amp_sq: min={amp_sq.min().item():.6e} max={amp_sq.max().item():.6e} mean={amp_sq.mean().item():.6e}"
)
amp_sq_shifted = amp_sq[:, :-1, :]
amp_sq_sum = amp_sq_shifted.sum(dim=-1)
print(
    f"amp_sq_sum: mean={amp_sq_sum.mean().item():.4f} min={amp_sq_sum.min().item():.4f} max={amp_sq_sum.max().item():.4f}"
)
log_sq = torch.log(amp_sq_sum + 1e-12) ** 2
print(
    f"log(sum)^2: mean={log_sq.mean().item():.4f} min={log_sq.min().item():.4f} max={log_sq.max().item():.4f}"
)
print(f"amp_sq dtype: {amp_sq.dtype}")
print(f"amp_sq_sum dtype: {amp_sq_sum.dtype}")
