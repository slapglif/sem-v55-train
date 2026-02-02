"""Push SEM code to HF Hub model repo."""

from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path=".",
    repo_id="icarus112/sem-v55-lean-crystal",
    repo_type="model",
    allow_patterns=[
        "sem/**/*.py",
        "configs/*.yaml",
        "hf_train.py",
        "launch_job.py",
        "pyproject.toml",
        "tokenizer/*",
    ],
    ignore_patterns=[
        "**/__pycache__/**",
        "*.egg-info/**",
    ],
)
print("Push complete!")
