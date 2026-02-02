"""Clean and push SEM code to HF Hub model repo."""

import os
import shutil
import tempfile
from huggingface_hub import HfApi

api = HfApi()
repo_id = "icarus112/sem-v55-lean-crystal"

# 1. DELETE GARBAGE FROM HUB FIRST (The nuclear option)
print("Cleaning garbage from Hub repo...")
garbage_folders = [".venv", ".pytest_cache", ".ruff_cache", "runs", "checkpoints"]
for folder in garbage_folders:
    try:
        api.delete_folder(repo_id=repo_id, path_in_repo=folder, repo_type="model")
        print(f"Deleted {folder} from Hub")
    except Exception as e:
        # Folder might not exist, that's fine
        pass

# 2. TARGETED UPLOAD
include_files = [
    "hf_train.py",
    "hf_train_lightning.py",
    "launch_job.py",
    "pyproject.toml",
    "monitor_job.py",
    "check_hf_logs.py",
    "test_xpu_local.py",
    "AGENTS.md",
    "DIAGNOSIS_NAN_LOSS.md",
    "PROFILE_REPORT.md",
    "CLAUDE.md",
    "TODO.md",
]
include_dirs = [
    "sem",
    "configs",
    "tokenizer",
]

with tempfile.TemporaryDirectory() as tmpdir:
    print(f"Preparing clean upload in {tmpdir}...")

    # Copy files
    for f in include_files:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(tmpdir, f))

    # Copy directories
    for d in include_dirs:
        if os.path.exists(d):
            # Strict copy, skip all garbage
            shutil.copytree(
                d,
                os.path.join(tmpdir, d),
                ignore=shutil.ignore_patterns("__pycache__", "*.pt", "*.bin", "*.pyc"),
            )

    print("Uploading clean codebase to Hugging Face...")
    api.upload_folder(
        folder_path=tmpdir,
        repo_id=repo_id,
        repo_type="model",
        delete_patterns=["*"],  # Delete anything in Hub that is NOT in this upload
    )

print("Push complete! Repository is now lean.")
