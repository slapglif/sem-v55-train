#!/bin/bash
set -euo pipefail

echo "=== SEM V5.5 Cloud Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"

pip install -q huggingface_hub[hf_xet] datasets tokenizers transformers tqdm einops scipy pyyaml numpy

echo "=== Downloading code from Hub ==="
huggingface-cli download icarus112/sem-v55-lean-crystal --local-dir /workspace/code --repo-type model

cd /workspace/code
pip install -e ".[training]" -q

export HF_HUB_REPO_ID=icarus112/sem-v55-lean-crystal

echo "=== Starting Training ==="
python -m sem.train --config configs/cloud_a10g.yaml --device cuda

echo "=== Training Complete ==="
