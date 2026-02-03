"""HF Jobs training launcher for SEM V5.5.

Usage (local):
    python hf_train.py --config configs/a100_optimized.yaml --dry-run

Usage (HF Jobs via launch_job.py):
    python launch_job.py --config configs/a100_optimized.yaml --dry-run --no-wandb
    python launch_job.py --config configs/a100_optimized.yaml --push-to-hub icarus112/sem-v55-lean-crystal
"""

import argparse
import logging
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("__main__").setLevel(logging.INFO)
logging.getLogger("sem.training.trainer").setLevel(logging.INFO)
logging.getLogger("sem.training.callbacks").setLevel(logging.INFO)
logging.getLogger("sem.training.health").setLevel(logging.INFO)
logging.getLogger("sem.data.streaming").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def detect_device_and_tune(config):
    """Auto-detect GPU and tune micro_batch_size to maximize throughput."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        _props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = (
            getattr(_props, "total_memory", None) or getattr(_props, "total_mem", 0)
        ) / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem_gb:.1f}GB)")

        if gpu_mem_gb >= 70:
            config.training.micro_batch_size = 32
            config.training.num_workers = 8
        elif gpu_mem_gb >= 40:
            config.training.micro_batch_size = 16
            config.training.num_workers = 4
        elif gpu_mem_gb >= 20:
            config.training.micro_batch_size = 8
            config.training.num_workers = 4
        else:
            config.training.micro_batch_size = 4
            config.training.num_workers = 2
        return device
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
        gpu_name = torch.xpu.get_device_name(0)
        gpu_mem_gb = torch.xpu.get_device_properties(0).total_memory / 1024**3
        logger.info(f"XPU: {gpu_name} ({gpu_mem_gb:.1f}GB)")

        # XPU-specific tuning (similar to CUDA but conservative)
        if gpu_mem_gb >= 16:
            config.training.micro_batch_size = 8
            config.training.num_workers = 4
        else:
            config.training.micro_batch_size = 4
            config.training.num_workers = 2
        return device
    return "cpu"


def print_system_info():
    logger.info("=" * 60)
    logger.info("SEM V5.5 Training - System Info")
    logger.info("=" * 60)
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        logger.info(f"VRAM: {vram / 1024**3:.1f}GB")
        logger.info(f"SM count: {props.multi_processor_count}")
        logger.info(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        logger.info(f"XPU available: True")
        logger.info(f"GPU: {torch.xpu.get_device_name(0)}")
        props = torch.xpu.get_device_properties(0)
        logger.info(f"VRAM: {props.total_memory / 1024**3:.1f}GB")
    logger.info(f"CPU cores: {os.cpu_count()}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SEM V5.5 Training (HF Jobs)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-aggression", action="store_true")
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (eager mode only)",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: 'default' — fast compile, good perf)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Push checkpoints to HF Hub repo (e.g. username/sem-v55)",
    )
    args = parser.parse_args()

    print_system_info()

    from sem.config import SEMConfig
    from sem.training.trainer import SEMTrainer
    from sem.train import apply_max_aggression

    config = SEMConfig.from_yaml(args.config)

    if args.no_wandb:
        config.training.wandb_enabled = False

    if args.max_steps:
        config.training.max_steps = args.max_steps

    device = args.device or detect_device_and_tune(config)
    if args.max_aggression:
        apply_max_aggression(config, device)

    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(
        f"micro_batch={config.training.micro_batch_size}, "
        f"batch={config.training.batch_size}, "
        f"accum={config.training.batch_size // config.training.micro_batch_size}"
    )

    if args.dry_run:
        config.training.max_steps = 20
        config.training.batch_size = 32
        config.training.wandb_enabled = False
        config.curriculum.enabled = False
        config.distillation.enabled = False
        config.training.timing_enabled = True
        config.training.timing_log_interval = 5
        config.training.log_interval = 1
        if config.training.micro_batch_size > 4:
            config.training.micro_batch_size = 4
        config.model.max_seq_length = 512
        logger.info(
            "DRY RUN: 20 steps, batch=32, seq=512, real streaming data, full timing"
        )

    if args.no_compile:
        config.training.no_compile = True
    if not hasattr(config.training, "no_compile"):
        config.training.no_compile = False
    config.training.compile_mode = args.compile_mode
    config.training.no_amp = getattr(args, "no_amp", False)
    if args.micro_batch:
        config.training.micro_batch_size = args.micro_batch

    t_start = time.perf_counter()
    trainer = SEMTrainer(
        config=config,
        device=device,
        resume_from=args.resume,
        dry_run=args.dry_run,
    )
    logger.info(f"Model build time: {time.perf_counter() - t_start:.1f}s")

    t_train = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t_train

    total_tokens = (
        config.training.max_steps
        * config.training.batch_size
        * (
            config.model.max_seq_length
            if not config.curriculum.enabled
            else config.curriculum.stages[0].get("seq_len", config.model.max_seq_length)
            if isinstance(config.curriculum.stages[0], dict)
            else config.model.max_seq_length
        )
    )
    logger.info("=" * 60)
    logger.info(f"Training complete in {train_time:.1f}s")
    logger.info(f"Avg throughput: {total_tokens / train_time:.0f} tok/s")
    logger.info(f"Avg step time: {train_time / config.training.max_steps * 1000:.0f}ms")
    logger.info("=" * 60)

    if args.push_to_hub:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.upload_folder(
                folder_path="runs",
                repo_id=args.push_to_hub,
                repo_type="model",
            )
            logger.info(f"Pushed checkpoints to {args.push_to_hub}")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")


if __name__ == "__main__":
    main()
