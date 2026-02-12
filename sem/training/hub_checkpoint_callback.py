"""Lightning callback to upload checkpoints to HuggingFace Hub.

Prevents weight loss when cloud training containers are destroyed (e.g., SIGTERM).
Uploads checkpoint + config + training state every N steps on rank 0.

Checkpoint format is Lightning-compatible (state_dict + optimizer_states +
lr_schedulers + global_step + epoch) so `trainer.fit(ckpt_path=...)` can
resume training seamlessly across multiple cloud job runs.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as L
import torch

logger = logging.getLogger(__name__)


class HubCheckpointCallback(L.Callback):
    """Upload checkpoints to HuggingFace Hub every N training steps.

    Args:
        repo_id: HF Hub repo (e.g., 'icarus112/sem-v55-lean-crystal')
        every_n_steps: Upload frequency in global steps
        keep_last_k: Number of checkpoints to keep on Hub (older ones deleted)
        local_dir: Local temp directory for checkpoint staging
        config_path: Path to training config YAML to upload alongside
    """

    def __init__(
        self,
        repo_id: str,
        every_n_steps: int = 500,
        keep_last_k: int = 3,
        local_dir: str = "hub_checkpoints",
        config_path: Optional[str] = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.every_n_steps = every_n_steps
        self.keep_last_k = keep_last_k
        self.local_dir = Path(local_dir)
        self.config_path = config_path
        self._api = None
        self._uploaded_steps: list[int] = []
        self._last_uploaded_step = -1

    def _get_api(self):
        """Lazy-init HfApi (avoids import at callback creation time)."""
        if self._api is None:
            try:
                from huggingface_hub import HfApi

                self._api = HfApi()
            except ImportError:
                logger.error("huggingface_hub not installed — Hub upload disabled")
        return self._api

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        if step == self._last_uploaded_step:
            return
        if trainer.global_rank == 0:
            self._save_and_upload(trainer, pl_module, step)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        step = trainer.global_step
        if step != self._last_uploaded_step:
            self._save_and_upload(trainer, pl_module, step, is_final=True)

    def _save_and_upload(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        step: int,
        is_final: bool = False,
    ) -> None:
        api = self._get_api()
        if api is None:
            return

        tag = "final" if is_final else f"step_{step}"
        self.local_dir.mkdir(parents=True, exist_ok=True)

        ckpt_name = f"{tag}.ckpt"
        ckpt_path = self.local_dir / ckpt_name
        try:
            # Build Lightning-compatible checkpoint dict manually.
            # We CANNOT call trainer.save_checkpoint() — it triggers DDP
            # BROADCAST collectives that desync ranks when called from rank 0 only.
            ckpt_dict = {
                "state_dict": pl_module.state_dict(),
                "global_step": step,
                "epoch": trainer.current_epoch,
                "pytorch-lightning_version": L.__version__,
            }

            # Optimizer states (needed to resume without LR spike / momentum loss)
            if trainer.optimizers:
                ckpt_dict["optimizer_states"] = [
                    opt.state_dict() for opt in trainer.optimizers
                ]

            # LR scheduler states (needed to resume decay phase correctly)
            if trainer.lr_scheduler_configs:
                ckpt_dict["lr_schedulers"] = [
                    cfg.scheduler.state_dict() for cfg in trainer.lr_scheduler_configs
                ]

            # Callback states (curriculum stage, etc.)
            ckpt_dict["callbacks"] = {}
            for cb in trainer.callbacks:
                cb_key = type(cb).__qualname__
                if hasattr(cb, "state_dict"):
                    cb_state = cb.state_dict()
                    if cb_state:
                        ckpt_dict["callbacks"][cb_key] = cb_state

            torch.save(ckpt_dict, str(ckpt_path))
            logger.info(
                f"[HubCkpt] Saved local checkpoint: {ckpt_path} "
                f"({ckpt_path.stat().st_size / 1e6:.1f} MB)"
            )
        except Exception as e:
            logger.error(f"[HubCkpt] Failed to save local checkpoint: {e}")
            return

        # 2. Build training state metadata
        loss_metric = trainer.callback_metrics.get("train/loss")
        loss_val = None
        if loss_metric is not None:
            loss_val = (
                loss_metric.item()
                if isinstance(loss_metric, torch.Tensor)
                else float(loss_metric)
            )

        state = {
            "global_step": step,
            "epoch": trainer.current_epoch,
            "loss": loss_val,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "is_final": is_final,
        }
        state_path = self.local_dir / "training_state.json"
        state_path.write_text(json.dumps(state, indent=2))

        # 3. Upload files to Hub
        files_to_upload = [
            (str(ckpt_path), f"checkpoints/{ckpt_name}"),
            (str(state_path), "checkpoints/training_state.json"),
        ]
        if self.config_path and os.path.exists(self.config_path):
            files_to_upload.append((self.config_path, "checkpoints/config.yaml"))

        for local, remote in files_to_upload:
            try:
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=remote,
                    repo_id=self.repo_id,
                    commit_message=f"Checkpoint {tag} (loss={loss_val:.4f})"
                    if loss_val is not None
                    else f"Checkpoint {tag}",
                )
            except Exception as e:
                logger.warning(f"[HubCkpt] Upload failed for {remote}: {e}")
                # Don't return — try other files

        logger.info(f"[HubCkpt] Uploaded checkpoint {tag} to {self.repo_id}")
        self._last_uploaded_step = step
        self._uploaded_steps.append(step)

        # 4. Clean up old checkpoints on Hub (keep last K)
        self._cleanup_old_checkpoints(api)

        # 5. Clean up local file (save disk space in container)
        try:
            ckpt_path.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def resume_from_hub(
        repo_id: str,
        local_dir: str = "hub_checkpoints",
    ) -> Optional[str]:
        """Download the latest checkpoint from HF Hub for resume.

        Returns the local path to the checkpoint file, or None if no
        checkpoint exists on the Hub.  Intended to be called on rank 0
        before ``trainer.fit()`` so the path can be passed as ``ckpt_path``.
        """
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            logger.warning("[HubCkpt] huggingface_hub not installed — cannot resume")
            return None

        api = HfApi()
        try:
            files = api.list_repo_files(repo_id)
        except Exception as e:
            logger.warning(f"[HubCkpt] Could not list repo files: {e}")
            return None

        ckpt_files = [
            f for f in files if f.startswith("checkpoints/") and f.endswith(".ckpt")
        ]
        if not ckpt_files:
            logger.info("[HubCkpt] No checkpoints found on Hub — starting fresh")
            return None

        # Find the highest step checkpoint (or 'final')
        best_file = None
        best_step = -1
        for f in ckpt_files:
            if "final" in f:
                best_file = f
                best_step = float("inf")
            else:
                match = re.search(r"step_(\d+)", f)
                if match:
                    s = int(match.group(1))
                    if s > best_step:
                        best_step = s
                        best_file = f

        if best_file is None:
            logger.info("[HubCkpt] No valid checkpoint files found on Hub")
            return None

        logger.info(f"[HubCkpt] Downloading checkpoint: {best_file}")
        try:
            local_path = hf_hub_download(repo_id, best_file, local_dir=local_dir)
            logger.info(
                f"[HubCkpt] Downloaded to {local_path} "
                f"({os.path.getsize(local_path) / 1e6:.1f} MB)"
            )
            return local_path
        except Exception as e:
            logger.error(f"[HubCkpt] Download failed: {e}")
            return None

    def _cleanup_old_checkpoints(self, api: Any) -> None:
        """Delete old checkpoint files from Hub, keeping only the last K."""
        if len(self._uploaded_steps) <= self.keep_last_k:
            return

        steps_to_remove = self._uploaded_steps[: -self.keep_last_k]
        self._uploaded_steps = self._uploaded_steps[-self.keep_last_k :]

        for old_step in steps_to_remove:
            remote_path = f"checkpoints/step_{old_step}.ckpt"
            try:
                api.delete_file(
                    path_in_repo=remote_path,
                    repo_id=self.repo_id,
                    commit_message=f"Remove old checkpoint step_{old_step}",
                )
                logger.info(f"[HubCkpt] Deleted old checkpoint: {remote_path}")
            except Exception as e:
                logger.debug(f"[HubCkpt] Could not delete {remote_path}: {e}")
