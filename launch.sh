#!/bin/bash
set -e
pip install -q datasets tokenizers pyyaml scipy einops 'huggingface_hub[hf_xet]'
REPO_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('icarus112/sem-v55-lean-crystal', repo_type='model'))")
cd "$REPO_PATH"
export PYTHONPATH="$REPO_PATH:$PYTHONPATH"
python hf_train.py "$@"
