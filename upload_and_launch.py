"""Upload current code to HF Hub and launch training job."""

import os
import subprocess
from huggingface_hub import HfApi, create_repo, upload_folder, get_token


def main():
    # Get token
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("ERROR: Set HF_TOKEN environment variable")
        return 1

    repo_id = "GAInTech/sem-v55-lean-crystal"

    # Upload current code to Hub
    print(f"Uploading code to {repo_id}...")
    api = HfApi(token=token)

    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"Repo exists or error: {e}")

    # Upload files
    upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="model",
        token=token,
        ignore_patterns=["__pycache__", "*.pyc", ".venv", ".venv/**", ".git", ".git/**", "*.log", ".claude", ".claude/**", ".letta", ".letta/**", "runs", "runs/**", "*.pt", "*.ckpt"],
    )
    print("Upload complete!")

    # Launch job
    print("Launching HF job...")
    cmd = r"""set -e
pytime() { python3 -c "import time; print(f'{time.time():.3f}')"; }
pdiff() { python3 -c "print(f'{float(\"$2\")-float(\"$1\"):.1f}')"; }
banner() { echo ""; echo "=============================="; echo "$1"; echo "=============================="; }

banner "[STAGE 1/4] Installing UV..."
S=$(pytime)
pip install -q uv 2>&1 | tail -1
E=$(pytime); echo "[STAGE 1/4] Done in $(pdiff $S $E)s"

banner "[STAGE 2/4] Installing dependencies via UV..."
S=$(pytime)
uv pip install --system datasets tokenizers pyyaml scipy einops wandb pytorch-lightning rich 'huggingface_hub[hf_xet]'
E=$(pytime); echo "[STAGE 2/4] Done in $(pdiff $S $E)s"

banner "[STAGE 3/4] Downloading model repo..."
S=$(pytime)
REPO_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('GAInTech/sem-v55-lean-crystal', repo_type='model'))")
E=$(pytime); echo "[STAGE 3/4] Repo downloaded to $REPO_PATH in $(pdiff $S $E)s"

banner "[STAGE 4/4] Starting training..."
cd $REPO_PATH
export PYTHONPATH=$REPO_PATH:$PYTHONPATH
export PYTHONUNBUFFERED=1
python3 hf_train_lightning.py --config configs/h100_max.yaml --push-to-hub GAInTech/sem-v55-lean-crystal"""

    job = api.run_job(
        image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        command=["bash", "-c", cmd],
        flavor="a100-large",
        timeout=14400,  # 4 hours
        secrets={"HF_TOKEN": token},
        namespace="GAInTech",  # Bill to organization, not personal account
    )

    print(f"\n{'=' * 60}")
    print(f"Job ID: {job.id}")
    print(f"URL: {job.url}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    exit(main())
