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
from sem.training.precision_plugin import SEMPrecisionPlugin
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
        default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed", "bf16-true"],
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
                    config.encoder.sinkhorn_epsilon, 0.1
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
    datamodule = SEMDataModule(config, tokenizer)

    # IPEX optimization for XPU (Intel Arc GPUs)
    if device_type == "xpu":
        try:
            import intel_extension_for_pytorch as ipex

            logger.info("Applying IPEX optimizations for XPU training...")
            dtype = torch.float32
            if args.precision in ["16-mixed", "bf16-mixed", "bf16-true"]:
                dtype = torch.bfloat16
            model = ipex.optimize(model, dtype=dtype)
            logger.info(f"IPEX optimization applied (dtype={dtype})")
        except ImportError:
            logger.warning(
                "intel_extension_for_pytorch not installed. "
                "XPU training will proceed without IPEX optimizations. "
                "Install with: pip install intel-extension-for-pytorch"
            )

    # Use custom precision plugin to prevent autocast poisoning in Cayley propagator
    plugins = []
    if args.precision in ["bf16-mixed", "16-mixed"]:
        precision_plugin = SEMPrecisionPlugin(
            precision=args.precision, device=device_type
        )
        precision_plugin.setup_module(model)
        plugins.append(precision_plugin)
        logger.info(
            "Using SEMPrecisionPlugin to exclude CayleySolitonPropagator from "
            f"{args.precision} autocast (device={device_type})"
        )

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

    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        accelerator=accelerator,
        devices=1,
        strategy=strategy if strategy is not None else "auto",
        precision=args.precision if not plugins else None,
        plugins=plugins if plugins else None,
        accumulate_grad_batches=config.training.batch_size
        // config.training.micro_batch_size,
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
