import argparse
import logging
import os
import sys
from contextlib import suppress
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    Timer,
    RichModelSummary,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.profilers import SimpleProfiler

from sem.config import SEMConfig
from sem.data.lightning_datamodule import SEMDataModule
from sem.data.tokenizer import SEMTokenizer
from sem.training.lightning_module import SEMLightningModule
from sem.training.callbacks import (
    SEMCurriculumCallback,
    SEMHealthCallback,
    SEMConsoleLogger,
)
from sem.training.xpu_accelerator import XPUAccelerator

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class _HttpLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if record.levelno <= logging.INFO and "HTTP Request:" in msg:
            return False
        return True


def main():
    parser = argparse.ArgumentParser(description="SEM V5.5 Training (Lightning)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--push-to-hub", type=str, default=None)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32-true", "bf16-mixed", "16-mixed"],
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=-1,
        help="Number of GPUs (-1 = auto-detect all available)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=[
            "auto",
            "ddp",
            "ddp_find_unused_parameters_true",
            "fsdp",
            "deepspeed_stage_2",
        ],
    )
    parser.add_argument(
        "--overfit-batches",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()

    config = SEMConfig.from_yaml(args.config)
    tokenizer = SEMTokenizer(config.training.tokenizer_path)

    # Filter HTTP logs from the root logger (catches libs that don't respect setLevel)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(_HttpLogFilter())

    # Also add filter to our own logger's handlers just in case
    for handler in logger.handlers:
        handler.addFilter(_HttpLogFilter())

    with suppress(Exception):
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)
        logging.getLogger("fsspec").setLevel(logging.WARNING)

    with suppress(Exception):
        from datasets.utils.logging import set_verbosity_warning

        set_verbosity_warning()

    with suppress(Exception):
        from huggingface_hub.utils.logging import (
            set_verbosity_warning as hf_set_verbosity_warning,
        )

        hf_set_verbosity_warning()

    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)

    # Auto-detect accelerator
    accelerator = "auto"
    device_type = "cpu"
    strategy = None
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        logger.info("XPU detected via torch.xpu.is_available()")
        device_type = "xpu"
        accelerator = XPUAccelerator()
        from pytorch_lightning.strategies import SingleDeviceStrategy

        strategy = SingleDeviceStrategy(device=torch.device("xpu", 0))
    elif torch.cuda.is_available():
        logger.info("CUDA detected via torch.cuda.is_available()")
        device_type = "cuda"
        n_gpus = torch.cuda.device_count()
        logger.info(f"CUDA devices available: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)"
            )

    if device_type == "xpu":
        try:
            props = torch.xpu.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            if vram_gb <= 8:
                config.training.micro_batch_size = 1
                config.training.batch_size = min(config.training.batch_size, 8)
                config.training.low_vram_mode = True
                config.training.born_chunk_size = 2048
                config.distillation.enabled = False
                config.model.hidden_dim = min(config.model.hidden_dim, 128)
                config.model.num_layers = min(config.model.num_layers, 4)
                config.model.max_seq_length = min(config.model.max_seq_length, 1024)
                config.encoder.sdr_candidates = min(config.encoder.sdr_candidates, 32)
                config.encoder.sdr_sparsity = min(config.encoder.sdr_sparsity, 8)
                config.encoder.sinkhorn_max_iter = min(
                    config.encoder.sinkhorn_max_iter, 10
                )
                config.encoder.sinkhorn_tol = max(config.encoder.sinkhorn_tol, 1e-2)
                config.encoder.sinkhorn_epsilon = max(
                    config.encoder.sinkhorn_epsilon, 0.08
                )
                config.spinor.state_dim = min(config.spinor.state_dim, 64)
                config.spinor.mimo_groups = min(config.spinor.mimo_groups, 8)
                if config.curriculum.enabled and config.curriculum.stages:
                    config.curriculum.stages[0]["seq_len"] = min(
                        config.curriculum.stages[0]["seq_len"], 128
                    )
        except Exception as exc:
            logger.warning(f"XPU VRAM probe failed: {exc}")

    model = SEMLightningModule(config)

    if device_type == "cuda" and not args.no_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("torch.compile enabled (reduce-overhead)")
        except Exception as e:
            logger.warning(f"torch.compile failed, falling back to eager: {e}")
    elif args.no_compile:
        logger.info("torch.compile disabled (--no-compile)")

    datamodule = SEMDataModule(config, tokenizer)

    # IPEX optimization for XPU (Intel Arc GPUs)
    if device_type == "xpu":
        try:
            import importlib

            ipex = importlib.import_module("intel_extension_for_pytorch")
            logger.info("Applying IPEX optimizations for XPU training...")
            dtype = torch.float32
            model = ipex.optimize(model, dtype=dtype)
            logger.info(f"IPEX optimization applied (dtype={dtype})")
        except ImportError:
            logger.warning(
                "intel_extension_for_pytorch not installed. "
                "XPU training will proceed without IPEX optimizations. "
                "Install with: pip install intel-extension-for-pytorch"
            )

    plugins = []
    logger.info("SEOP Fix 31: Running in full fp32 mode (bf16 disabled for stability)")

    loggers = []
    if config.training.wandb_enabled and os.environ.get("WANDB_API_KEY"):
        loggers.append(
            WandbLogger(project=config.training.wandb_project, config=config)
        )
    else:
        if config.training.wandb_enabled:
            logger.warning(
                "wandb enabled but WANDB_API_KEY not found. Using CSVLogger."
            )
        loggers.append(CSVLogger("logs"))

    callbacks = [
        ModelCheckpoint(
            dirpath="runs",
            filename="sem-{step}-{train/loss:.2f}",
            every_n_train_steps=config.training.checkpoint_interval,
            save_top_k=-1,
            monitor="train/loss",
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(leave=True),
        RichModelSummary(max_depth=2),
        Timer(),
        SEMCurriculumCallback(config),
        SEMHealthCallback(check_interval=config.training.health_check_interval),
        SEMConsoleLogger(log_interval=config.training.log_interval),
    ]

    profiler = (
        SimpleProfiler(filename="profile_report")
        if config.training.timing_enabled
        else None
    )

    # --- Resolve devices / strategy / precision per accelerator ---
    if device_type == "xpu":
        # XPU: single-device only, fp32, no DDP
        effective_devices = 1
        effective_strategy = strategy  # SingleDeviceStrategy set above
        effective_precision = None if plugins else "32-true"
    else:
        # CUDA / CPU: honour CLI args
        effective_devices = args.devices  # -1 = all GPUs
        effective_strategy = strategy if strategy is not None else args.strategy
        effective_precision = args.precision if not plugins else None

    # Gradient accumulation: global_batch = micro_batch * devices * accum
    num_devices = (
        effective_devices
        if effective_devices > 0
        else (torch.cuda.device_count() if device_type == "cuda" else 1)
    )
    accumulate = max(
        1,
        config.training.batch_size // (config.training.micro_batch_size * num_devices),
    )
    logger.info(
        f"Trainer config: devices={effective_devices}, strategy={effective_strategy}, "
        f"precision={effective_precision}, accumulate_grad_batches={accumulate}"
    )

    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        accelerator=accelerator,
        devices=effective_devices,
        strategy=effective_strategy if effective_strategy is not None else "auto",
        precision=effective_precision,
        plugins=plugins if plugins else None,
        overfit_batches=int(args.overfit_batches)
        if args.overfit_batches >= 1
        else args.overfit_batches,
        accumulate_grad_batches=accumulate,
        gradient_clip_val=config.training.gradient_clip,
        logger=loggers,
        callbacks=callbacks,
        profiler=profiler,
        benchmark=True,
        log_every_n_steps=config.training.log_interval,
    )

    logger.info("Launching SEM V5.5 Training with PyTorch Lightning")
    trainer.fit(model, datamodule=datamodule)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path="runs", repo_id=args.push_to_hub, repo_type="model"
        )


if __name__ == "__main__":
    main()
