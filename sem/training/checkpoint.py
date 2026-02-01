"""Complex-aware checkpoint save/resume for SEM V5.5.

Handles saving and loading of:
- Model state (complex64 parameters)
- Optimizer state (ComplexAdamW momentum in real-view format)
- Scheduler state
- EMA teacher state
- Curriculum stage
- RNG states for reproducibility
- Health monitor history
- Fisher tracker state
- wandb run ID for resumption
- Hub upload (optional, for cloud training)
"""

import os
import torch
import random
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint saving, loading, and rotation.

    Args:
        checkpoint_dir: Directory for checkpoint files
        keep_checkpoints: Number of regular checkpoints to keep
        hub_repo_id: Optional HF Hub repo to upload checkpoints to
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_checkpoints: int = 3,
        hub_repo_id: Optional[str] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_checkpoints = keep_checkpoints
        self._regular_checkpoints = []  # Paths, oldest first
        self.hub_repo_id = hub_repo_id
        self._hub_api = None
        if hub_repo_id:
            try:
                from huggingface_hub import HfApi

                self._hub_api = HfApi()
                logger.info(f"Hub upload enabled: {hub_repo_id}")
            except ImportError:
                logger.warning("huggingface_hub not installed, Hub upload disabled")

    def save(
        self,
        model,
        optimizer,
        scheduler,
        global_step: int,
        epoch: int = 0,
        ema_teacher=None,
        curriculum_manager=None,
        health_monitor=None,
        quantizer=None,
        wandb_run_id: Optional[str] = None,
        is_stage_transition: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a full training checkpoint.

        Args:
            model: SEMModel
            optimizer: ComplexAdamW
            scheduler: WSDScheduler
            global_step: Current step
            epoch: Current epoch
            ema_teacher: Optional EMATeacher
            curriculum_manager: Optional CurriculumManager
            health_monitor: Optional HealthMonitor
            quantizer: Optional HASVQ
            wandb_run_id: wandb run ID for resumption
            is_stage_transition: If True, keep this checkpoint permanently
            extra: Additional data to save
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
        }

        if ema_teacher is not None:
            checkpoint["ema_state_dict"] = ema_teacher.teacher.state_dict()
            checkpoint["ema_decay"] = ema_teacher.decay

        if curriculum_manager is not None:
            checkpoint["curriculum_state"] = curriculum_manager.state_dict()

        if health_monitor is not None:
            checkpoint["health_state"] = health_monitor.state_dict()

        if quantizer is not None and hasattr(quantizer, "fisher"):
            checkpoint["fisher_state"] = quantizer.fisher.state_dict()

        if wandb_run_id is not None:
            checkpoint["wandb_run_id"] = wandb_run_id

        if extra:
            checkpoint.update(extra)

        # Determine filename
        if is_stage_transition:
            stage = curriculum_manager.current_stage if curriculum_manager else 0
            filename = f"checkpoint_stage{stage}_step{global_step}.pt"
        else:
            filename = f"checkpoint_step{global_step}.pt"

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        # Upload to Hub if configured
        if self._hub_api and self.hub_repo_id:
            self._upload_to_hub(path, filename)

        # Rotate regular (non-stage-transition) checkpoints
        if not is_stage_transition:
            self._regular_checkpoints.append(path)
            while len(self._regular_checkpoints) > self.keep_checkpoints:
                old = self._regular_checkpoints.pop(0)
                if old.exists():
                    old.unlink()
                    logger.info(f"Removed old checkpoint: {old}")

        return path

    def _upload_to_hub(self, local_path: Path, filename: str):
        try:
            self._hub_api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"checkpoints/{filename}",
                repo_id=self.hub_repo_id,
                commit_message=f"Checkpoint: {filename}",
            )
            logger.info(f"Uploaded {filename} to Hub: {self.hub_repo_id}")
        except Exception as e:
            logger.warning(f"Hub upload failed for {filename}: {e}")

    def load(
        self,
        path: str,
        model,
        optimizer=None,
        scheduler=None,
        ema_teacher=None,
        curriculum_manager=None,
        health_monitor=None,
        quantizer=None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load a checkpoint and restore all state.

        Args:
            path: Path to checkpoint file
            model: SEMModel to restore
            optimizer: Optional ComplexAdamW to restore
            scheduler: Optional WSDScheduler to restore
            ema_teacher: Optional EMATeacher to restore
            curriculum_manager: Optional CurriculumManager to restore
            health_monitor: Optional HealthMonitor to restore
            quantizer: Optional HASVQ to restore Fisher state
            device: Device to map tensors to
        Returns:
            Dict with remaining checkpoint data (global_step, epoch, etc.)
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Model
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Restored model state")

        # Optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Restored optimizer state")

        # Scheduler
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Restored scheduler state")

        # EMA teacher
        if ema_teacher is not None and "ema_state_dict" in checkpoint:
            ema_teacher.teacher.load_state_dict(checkpoint["ema_state_dict"])
            if "ema_decay" in checkpoint:
                ema_teacher.decay = checkpoint["ema_decay"]
            logger.info("Restored EMA teacher state")

        # Curriculum
        if curriculum_manager is not None and "curriculum_state" in checkpoint:
            curriculum_manager.load_state_dict(checkpoint["curriculum_state"])
            logger.info("Restored curriculum state")

        # Health monitor
        if health_monitor is not None and "health_state" in checkpoint:
            health_monitor.load_state_dict(checkpoint["health_state"])
            logger.info("Restored health monitor state")

        # Fisher tracker
        if quantizer is not None and "fisher_state" in checkpoint:
            quantizer.fisher.load_state_dict(checkpoint["fisher_state"])
            logger.info("Restored Fisher tracker state")

        # RNG states
        if "rng_states" in checkpoint:
            rng = checkpoint["rng_states"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.random.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            logger.info("Restored RNG states")

        return {
            "global_step": checkpoint.get("global_step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "wandb_run_id": checkpoint.get("wandb_run_id"),
        }

    def find_latest(self) -> Optional[Path]:
        """Find the latest checkpoint in the directory."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        return checkpoints[-1] if checkpoints else None
