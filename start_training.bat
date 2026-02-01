@echo off
echo Starting SEM training with HF_TOKEN set
echo Make sure HF_TOKEN is set in your environment!
.venv\Scripts\python -m sem.train --config configs\max_aggression.yaml --device xpu --max-aggression > training_output.log 2>&1
echo Training completed
