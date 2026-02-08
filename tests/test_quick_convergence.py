"""Quick convergence test - verify model learns on real FineWeb-Edu data.

Run directly:  PYTHONUNBUFFERED=1 uv run python -u tests/test_quick_convergence.py
Run via pytest: uv run pytest tests/test_quick_convergence.py -v -s

Streams real FineWeb-Edu data, tokenizes with the project BPE tokenizer,
packs sequences, and trains for 500 steps to verify the model CAN learn
on real-world data.

Pass criteria:
- Loss decreases by at least 30% from post-warmup peak to final
- Loss is trending down in the second half
- No NaN in loss or gradients
- Gradients flow (non-zero norms)
"""

import os
import sys
import time
import logging

import torch

from sem.model import SEMModel
from sem.config import (
    SEMConfig,
    ModelConfig,
    EncoderConfig,
    SpinorConfig,
    PropagatorConfig,
    QuantizerConfig,
    SamplerConfig,
    TrainingConfig,
    DistillationConfig,
    V8Config,
)
from sem.utils.complex_adamw import ComplexAdamW
from sem.training.scheduler import WSDScheduler
from sem.data.tokenizer import SEMTokenizer
from sem.data.streaming import FineWebEduStream, SequencePacker

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer")

NUM_STEPS = 500
SEQ_LEN = 128
BATCH_SIZE = 4
HIDDEN_DIM = 128
NUM_LAYERS = 2
LR = 3e-4
WARMUP_STEPS = 30
DECAY_STEPS = 470
GRAD_CLIP = 1.0


def make_config(vocab_size: int) -> SEMConfig:
    """Build a small test config matched to the real tokenizer's vocab."""
    return SEMConfig(
        model=ModelConfig(
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            vocab_size=vocab_size,
            max_seq_length=SEQ_LEN,
        ),
        encoder=EncoderConfig(
            simple_mode=True,
            sdr_sparsity=16,
            sdr_candidates=64,
            sinkhorn_epsilon=0.05,
            sinkhorn_max_iter=20,
            sinkhorn_tol=1e-3,
        ),
        spinor=SpinorConfig(
            block_size=8,
            num_blocks=16,
            state_dim=32,
            mimo_groups=4,
            d_conv=4,
        ),
        propagator=PropagatorConfig(
            cayley_dt=0.1,
            cg_max_iter=10,
            cg_tol=1e-4,
            nonlinear_alpha=0.1,
            laplacian_sparsity=5,
            lazy_cg=False,
            lazy_cg_tol=1e-3,
            direct_solve=True,
            pit_gamma=0.1,
        ),
        quantizer=QuantizerConfig(
            codebook_size=256,
            group_size=2,
            fisher_ema_decay=0.99,
            outlier_percentile=0.01,
            dead_code_threshold=100,
        ),
        sampler=SamplerConfig(
            temperature=1.0,
            top_k=50,
            top_p=0.95,
        ),
        training=TrainingConfig(
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            weight_decay=0.0,
            warmup_steps=WARMUP_STEPS,
            max_steps=NUM_STEPS,
            gradient_clip=GRAD_CLIP,
            dtype="complex64",
            low_vram_mode=False,
            unitary_lambda=0.01,
            decay_steps=DECAY_STEPS,
            lr_min_ratio=0.1,
        ),
        distillation=DistillationConfig(enabled=False),
        v8=V8Config(
            use_lindblad=True,
            use_hybrid_automata=True,
            use_quaternionic=True,
            use_mhc=True,
        ),
    )


def _make_batch_iterator(tokenizer: SEMTokenizer, seq_len: int, batch_size: int):
    """Stream FineWeb-Edu → tokenize → pack → batch.

    Yields (token_ids, token_freqs) where:
        token_ids:  [batch_size, seq_len]  LongTensor
        token_freqs: [vocab_size]          FloatTensor (EMA unigram freqs)
    """
    stream = FineWebEduStream(
        min_score=2,
        shuffle_buffer=1000,
        dataset_name="HuggingFaceFW/fineweb-edu",
    )
    packer = SequencePacker(tokenizer, seq_len)

    batch_ids = []
    last_freqs = None

    for token_ids, token_freqs in packer.pack(iter(stream)):
        batch_ids.append(token_ids)
        last_freqs = token_freqs

        if len(batch_ids) == batch_size:
            yield torch.stack(batch_ids, dim=0), last_freqs
            batch_ids = []


def run_convergence_test() -> bool:
    """Train on real FineWeb-Edu data and verify convergence. Returns True on pass."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer from {TOKENIZER_PATH}...", flush=True)
    tokenizer = SEMTokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer loaded: vocab_size={vocab_size}", flush=True)

    config = make_config(vocab_size)
    model = SEMModel(config).to(device)
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created: {param_count:,} parameters", flush=True)

    optimizer = ComplexAdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=0.0,
        temperature=1e-5,
    )
    scheduler = WSDScheduler(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.decay_steps,
        min_lr_ratio=config.training.lr_min_ratio,
    )

    losses: list[float] = []
    grad_norms: list[float] = []

    print(f"\n{'=' * 70}", flush=True)
    print("QUICK CONVERGENCE TEST — Real FineWeb-Edu Data", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(
        f"Device: {device}  |  Vocab: {vocab_size}  |  LR: {LR}  |  Steps: {NUM_STEPS}",
        flush=True,
    )
    print(
        f"Batch: {BATCH_SIZE}  |  SeqLen: {SEQ_LEN}  |  Grad clip: {GRAD_CLIP}  |  "
        f"Warmup: {WARMUP_STEPS}",
        flush=True,
    )
    print(f"{'=' * 70}\n", flush=True)

    print("Streaming FineWeb-Edu data...", flush=True)
    t_start = time.perf_counter()
    batch_iter = _make_batch_iterator(tokenizer, SEQ_LEN, BATCH_SIZE)

    print("Pre-fetching batches...", flush=True)
    all_batches = []
    t_fetch = time.perf_counter()
    for token_ids, token_freqs in batch_iter:
        all_batches.append((token_ids, token_freqs))
        if len(all_batches) % 50 == 0:
            print(f"  Fetched {len(all_batches)} batches...", flush=True)
        if len(all_batches) >= NUM_STEPS:
            break

    t_fetch_done = time.perf_counter()
    n_fetched = len(all_batches)
    print(
        f"Fetched {n_fetched} batches in {t_fetch_done - t_fetch:.1f}s "
        f"({n_fetched * BATCH_SIZE * SEQ_LEN:,} tokens)",
        flush=True,
    )

    if n_fetched == 0:
        print("FAIL: No batches fetched from FineWeb-Edu stream", flush=True)
        return False

    if n_fetched < NUM_STEPS:
        print(
            f"WARNING: Only fetched {n_fetched} batches, will cycle through them",
            flush=True,
        )

    print(f"\nStarting training ({NUM_STEPS} steps)...\n", flush=True)
    t_train = time.perf_counter()

    for step in range(NUM_STEPS):
        token_ids, token_freqs = all_batches[step % n_fetched]
        token_ids = token_ids.to(device)

        optimizer.zero_grad()
        output = model(token_ids, targets=token_ids)
        loss = output["loss"]

        if torch.isnan(loss):
            print(f"FAIL: NaN loss at step {step}", flush=True)
            return False

        loss.backward()

        total_norm_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(
                        f"FAIL: NaN/Inf gradient in {name} at step {step}", flush=True
                    )
                    return False
                total_norm_sq += param.grad.norm().item() ** 2

        total_grad_norm = total_norm_sq**0.5
        grad_norms.append(total_grad_norm)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip
        )
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (step + 1) % 50 == 0 or step == 0:
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t_train
            steps_per_sec = (step + 1) / elapsed
            print(
                f"  Step {step + 1:3d}/{NUM_STEPS}  |  Loss: {loss.item():.4f}  |  "
                f"Grad: {total_grad_norm:.4f}  |  LR: {lr_now:.2e}  |  "
                f"{steps_per_sec:.1f} steps/s",
                flush=True,
            )

    t_train_done = time.perf_counter()
    train_time = t_train_done - t_train
    total_time = t_train_done - t_start

    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    peak_loss = max(losses[: min(80, len(losses))])
    peak_to_final = (peak_loss - final_loss) / peak_loss
    trending_down = losses[-1] < losses[NUM_STEPS // 2]

    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Initial loss:     {initial_loss:.4f}", flush=True)
    print(f"  Peak loss:        {peak_loss:.4f}  (warmup transient)", flush=True)
    print(f"  Final loss:       {final_loss:.4f}", flush=True)
    print(f"  Min loss:         {min_loss:.4f}", flush=True)
    print(f"  Peak->Final:      {peak_to_final * 100:.1f}%", flush=True)
    print(f"  Avg grad norm:    {sum(grad_norms) / len(grad_norms):.4f}", flush=True)
    print(
        f"  Training time:    {train_time:.1f}s ({NUM_STEPS / train_time:.1f} steps/s)",
        flush=True,
    )
    print(f"  Total time:       {total_time:.1f}s (incl. data fetch)", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    passed = True
    if peak_to_final < 0.3:
        print(
            f"FAIL: Peak-to-final reduction {peak_to_final * 100:.1f}% < 30%",
            flush=True,
        )
        passed = False
    if not trending_down:
        print(
            f"FAIL: Loss not trending down "
            f"(step {NUM_STEPS // 2}: {losses[NUM_STEPS // 2]:.4f} -> "
            f"final: {final_loss:.4f})",
            flush=True,
        )
        passed = False
    if not all(g > 0 for g in grad_norms):
        print("FAIL: Some gradient norms are zero (no gradient flow)", flush=True)
        passed = False

    if passed:
        print(
            f"PASS  |  Peak->Final: {peak_to_final * 100:.1f}%  |  "
            f"Min: {min_loss:.4f}  |  Final: {final_loss:.4f}",
            flush=True,
        )
    return passed


def test_quick_convergence():
    assert run_convergence_test(), "Convergence test failed"


if __name__ == "__main__":
    ok = run_convergence_test()
    sys.exit(0 if ok else 1)
