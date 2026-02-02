#!/bin/bash
# Pull latest SEOP-fixed code from HF Hub before training
set -e

echo "Downloading latest SEOP-fixed code..."
huggingface-cli download icarus112/sem-v55-lean-crystal sem/propagator/hamiltonian.py --local-dir /app --quiet

echo "Starting training..."
python -m sem.train --config configs/cloud_a10g.yaml --device cuda
