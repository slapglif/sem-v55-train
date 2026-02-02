import argparse
import logging
import os
import sys
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    DeviceStatsMonitor,
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SEM V5.5 Training (Lightning)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--push-to-hub", type=str, default=None)
    args = parser.parse_args()

    config = SEMConfig.from_yaml(args.config)
    tokenizer = SEMTokenizer(config.training.tokenizer_path)

    torch.set_float32_matmul_precision("high")
    L.seed_everything(42)

    model = SEMLightningModule(config)
    datamodule = SEMDataModule(config, tokenizer)

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
        DeviceStatsMonitor(),
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

    precision = (
        "bf16-mixed"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "16-mixed"
    )

    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        accelerator="auto",
        devices=1,
        precision=precision,
        accumulate_grad_batches=config.training.batch_size
        // config.training.micro_batch_size,
        gradient_clip_val=config.training.gradient_clip,
        logger=loggers,
        callbacks=callbacks,
        profiler=profiler,
        benchmark=True,
        log_every_n_steps=config.training.log_interval,
    )

    logger.info("🚀 Launching SEM V5.5 Training with PyTorch Lightning")
    trainer.fit(model, datamodule=datamodule)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path="runs", repo_id=args.push_to_hub, repo_type="model"
        )


if __name__ == "__main__":
    main()
