"""Debug script to check data pipeline and loss computation."""
import torch
from pathlib import Path
import sys

# Add sem to path
sys.path.insert(0, str(Path(__file__).parent))

from sem.data.tokenizer import SEMTokenizer
from sem.data.streaming import PackedStreamingDataset

print("=" * 80)
print("DEBUGGING DATA PIPELINE FOR LOSS INCREASE ISSUE")
print("=" * 80)

# Load tokenizer
tokenizer = SEMTokenizer("tokenizer")
print(f"\nTokenizer loaded:")
print(f"  vocab_size: {tokenizer.vocab_size}")
print(f"  pad_id: {tokenizer.pad_id}")
print(f"  eos_id: {tokenizer.eos_id}")
print(f"  bos_id: {tokenizer.bos_id}")
print(f"  unk_id: {tokenizer.unk_id}")
print(f"  doc_boundary_id: {tokenizer.doc_boundary_id}")

# Create dataset
print("\nCreating streaming dataset (will take a moment)...")
dataset = PackedStreamingDataset(
    tokenizer=tokenizer,
    seq_len=512,
    min_score=2,
    shuffle_buffer=100,
    dataset_name="HuggingFaceFW/fineweb-edu",
)

# Get one batch
print("\nFetching first batch from stream...")
iterator = iter(dataset)
token_ids, token_freqs = next(iterator)

print(f"\nBatch received:")
print(f"  token_ids shape: {token_ids.shape}")
print(f"  token_freqs shape: {token_freqs.shape}")
print(f"  token_ids dtype: {token_ids.dtype}")

# Check for special tokens
print(f"\nSpecial token counts in batch:")
print(f"  <pad> (id={tokenizer.pad_id}): {(token_ids == tokenizer.pad_id).sum().item()}")
print(f"  <eos> (id={tokenizer.eos_id}): {(token_ids == tokenizer.eos_id).sum().item()}")
print(f"  <bos> (id={tokenizer.bos_id}): {(token_ids == tokenizer.bos_id).sum().item()}")
print(f"  <doc_boundary> (id={tokenizer.doc_boundary_id}): {(token_ids == tokenizer.doc_boundary_id).sum().item()}")

# Check first 50 tokens
print(f"\nFirst 50 tokens:")
print(token_ids[:50].tolist())

# Simulate what the model does
print("\n" + "=" * 80)
print("SIMULATING MODEL FORWARD PASS")
print("=" * 80)

# Add batch dimension
token_ids_batch = token_ids.unsqueeze(0)  # [1, 512]
targets = token_ids_batch  # Same as in lightning_module.py:106,116

print(f"\nInput to model.forward():")
print(f"  token_ids shape: {token_ids_batch.shape}")
print(f"  targets shape: {targets.shape}")
print(f"  Are they the same? {torch.equal(token_ids_batch, targets)}")

# Simulate the shifting that happens in model.py:173-176
print(f"\nModel.py line 173-176 shifting:")
print(f"  amp_sq computed from positions [:, :-1] = positions 0-510")
print(f"  target_ids = targets[:, 1:] = positions 1-511")

amp_sq_positions = token_ids_batch[:, :-1]
target_ids = targets[:, 1:]

print(f"\n  amp_sq_positions shape: {amp_sq_positions.shape}")
print(f"  target_ids shape: {target_ids.shape}")
print(f"  First 10 amp_sq tokens: {amp_sq_positions[0, :10].tolist()}")
print(f"  First 10 target tokens: {target_ids[0, :10].tolist()}")

print(f"\nAlignment check:")
print(f"  Position 0 predicts: {target_ids[0, 0].item()} (should be token at position 1)")
print(f"  Position 1 predicts: {target_ids[0, 1].item()} (should be token at position 2)")
print(f"  Position 2 predicts: {target_ids[0, 2].item()} (should be token at position 3)")

# Check if doc_boundary tokens are being predicted
doc_boundary_positions = (target_ids == tokenizer.doc_boundary_id).nonzero(as_tuple=True)
if len(doc_boundary_positions[0]) > 0:
    print(f"\n⚠️  WARNING: Model is being asked to predict <doc_boundary> tokens!")
    print(f"  Found {len(doc_boundary_positions[0])} doc_boundary targets")
    print(f"  These are UNNATURAL predictions (boundary tokens should be special)")
    print(f"  Positions where doc_boundary appears as target: {doc_boundary_positions[1][:10].tolist()}")

# Check the actual token sequence around boundaries
print(f"\nChecking for document boundary patterns:")
for i in range(min(50, len(token_ids) - 5)):
    if token_ids[i] == tokenizer.doc_boundary_id:
        print(f"  Position {i}: <doc_boundary>")
        print(f"    Before: {token_ids[max(0, i-3):i].tolist()}")
        print(f"    After: {token_ids[i+1:i+4].tolist()}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
