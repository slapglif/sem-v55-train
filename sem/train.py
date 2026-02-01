"""Training entry point for SEM V5.5 'Lean Crystal'.

Usage:
    python -m sem.train --config configs/training.yaml --device cuda
    python -m sem.train --config configs/training.yaml --resume runs/checkpoint_step5000.pt
    python -m sem.train --tokenizer-train --config configs/training.yaml
    python -m sem.train --config configs/training.yaml --dry-run
"""

import argparse
import logging
import os
import sys

import torch

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None

from .config import SEMConfig
from .training.trainer import SEMTrainer

# Reduce spam: only show INFO for main modules, WARNING for everything else
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Enable INFO for training metrics and important events
logging.getLogger("__main__").setLevel(logging.INFO)
logging.getLogger("sem.training.trainer").setLevel(logging.INFO)
logging.getLogger("sem.training.callbacks").setLevel(logging.INFO)
logging.getLogger("sem.training.health").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def _configure_cpu_for_intel():
    """Optimize CPU and XPU settings for Intel hardware."""
    num_cores = os.cpu_count() or 8

    # Match threads to physical cores (avoid oversubscription on P+E hybrid)
    torch.set_num_threads(num_cores)

    # Enable oneDNN (MKL-DNN) fusion for supported ops
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True

    # Set OpenMP environment if not already configured
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(num_cores)

    # Let Intel Thread Director handle P-core/E-core scheduling (no hard pinning)
    if "KMP_AFFINITY" not in os.environ:
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

    # Allocator optimization: use jemalloc-style allocation if available
    if "MALLOC_CONF" not in os.environ:
        os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

    # XPU Native handling
    if ipex is not None or (hasattr(torch, "xpu") and torch.xpu.is_available()):
        os.environ.setdefault("IPEX_XPU_BACKEND", "1")
        logger.info("Intel XPU environment detected and configured")

    return num_cores


def apply_max_aggression(config: SEMConfig, device_type: str):
    """
    Tuning for Max Aggression: 16GB RAM / 8GB VRAM.
    Maximizes throughput by pushing batch sizes and worker counts to hardware limits.
    """
    logger.info(
        f"Applying 'Max Aggression' RAM tuning for {device_type} (16GB RAM / 8GB VRAM)"
    )

    # 16GB RAM Tuning (CPU side)
    config.training.num_workers = 8  # Increased from 4
    config.training.shuffle_buffer_size = 25000  # More aggressive buffering

    # VRAM Tuning (8GB)
    if device_type in ["cuda", "xpu"]:
        # For a ~100M param model, 16 is aggressive but usually fits in 8GB with gradient checkpointing
        config.training.micro_batch_size = 16
        # Target a large effective batch size for stability
        config.training.batch_size = 256
        config.training.gradient_checkpointing = True
    else:
        # CPU only "Aggression"
        config.training.micro_batch_size = 8
        config.training.batch_size = 64
        config.training.gradient_checkpointing = True

    logger.info(f"  -> micro_batch_size: {config.training.micro_batch_size}")
    logger.info(f"  -> batch_size: {config.training.batch_size}")
    logger.info(f"  -> num_workers: {config.training.num_workers}")


def train_tokenizer(config: SEMConfig):
    """Train a BPE tokenizer on FineWeb-Edu sample."""
    from .data.tokenizer import SEMTokenizer
    from .data.streaming import FineWebEduStream

    logger.info(f"Training tokenizer with vocab_size={config.model.vocab_size}")
    logger.info(f"Streaming from {config.training.dataset_name}")

    stream = FineWebEduStream(
        min_score=2,
        dataset_name=config.training.dataset_name,
    )

    # Collect texts (first ~5M documents)
    max_docs = 5_000_000

    def text_iter():
        for i, text in enumerate(stream):
            if i >= max_docs:
                break
            yield text
            if (i + 1) % 100_000 == 0:
                logger.info(f"  Processed {i + 1:,} documents...")

    tokenizer = SEMTokenizer.train(
        output_path=config.training.tokenizer_path,
        texts_iterator=text_iter(),
        vocab_size=config.model.vocab_size,
    )
    logger.info(f"Tokenizer saved to {config.training.tokenizer_path}")
    logger.info(f"Vocab size: {tokenizer.vocab_size}")


def main():
    parser = argparse.ArgumentParser(description="Train SEM V5.5 'Lean Crystal'")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu/cuda/cuda:0). Default: auto-detect",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--wandb", action="store_true", default=None, help="Enable wandb logging"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 10 steps with synthetic data for validation",
    )
    parser.add_argument(
        "--tokenizer-train",
        action="store_true",
        help="Train tokenizer instead of model",
    )
    parser.add_argument(
        "--stage", type=int, default=None, help="Force start at curriculum stage N"
    )
    parser.add_argument(
        "--max-aggression",
        action="store_true",
        help="Enable Max Aggression RAM tuning (16GB RAM / 8GB VRAM)",
    )
    args = parser.parse_args()

    # Optimize CPU for Intel hardware
    num_cores = _configure_cpu_for_intel()
    logger.info(f"CPU optimized: {num_cores} threads, oneDNN enabled")

    # Load config
    config = SEMConfig.from_yaml(args.config)

    # CLI overrides
    if args.wandb is True:
        config.training.wandb_enabled = True
    if args.no_wandb:
        config.training.wandb_enabled = False

    # Auto-detect device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"

    # Max Aggression tuning
    if args.max_aggression:
        device_type = "cpu"
        if ":" in device:
            device_type = device.split(":")[0]
        else:
            device_type = device
        apply_max_aggression(config, device_type)

    # Tokenizer training mode
    if args.tokenizer_train:
        train_tokenizer(config)
        return

    # Dry run
    if args.dry_run:
        config.training.max_steps = 10
        config.training.wandb_enabled = False
        config.curriculum.enabled = False
        config.distillation.enabled = False
        logger.info("DRY RUN: 10 steps, no wandb, no curriculum, no distillation")

    # Force stage
    if args.stage is not None and config.curriculum.enabled:
        logger.info(f"Forcing curriculum stage {args.stage}")

    # Create trainer and run
    trainer = SEMTrainer(
        config=config,
        device=device,
        resume_from=args.resume,
        dry_run=args.dry_run,
    )

    # Force stage override
    if args.stage is not None and trainer.curriculum is not None:
        trainer.curriculum.current_stage = args.stage
        trainer.curriculum.stage_start_step = trainer.global_step

    trainer.train()


if __name__ == "__main__":
    main()
