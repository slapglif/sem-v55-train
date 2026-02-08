"""Launch HP + NAS sweeps on HuggingFace Jobs with 4 parallel workers (one per GPU).

Usage:
    uv run python launch_sweep.py --org gaintech
    uv run python launch_sweep.py --org gaintech --flavor a100x4 --timeout 3600
"""

import os
import argparse
import shlex

from huggingface_hub import HfApi, get_token

parser = argparse.ArgumentParser(description="Launch SEM sweep on HF Jobs")
parser.add_argument("--org", type=str, required=True)
parser.add_argument("--flavor", type=str, default="a100x4")
parser.add_argument("--timeout", type=int, default=3600)
parser.add_argument("--hp-trials", type=int, default=50)
parser.add_argument("--nas-trials", type=int, default=100)
parser.add_argument("--n-steps", type=int, default=200)
args = parser.parse_args()

cmd = rf"""set -e
pytime() {{ python3 -c "import time; print(f'{{time.time():.3f}}')"; }}
banner() {{ echo ""; echo "=============================="; echo "$1"; echo "=============================="; }}

banner "[1/4] Installing UV + deps..."
pip install -q uv 2>&1 | tail -1
uv pip install --system datasets tokenizers pyyaml scipy einops optuna 'huggingface_hub[hf_xet]' rich

banner "[2/4] Downloading model repo..."
REPO_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('icarus112/sem-v55-lean-crystal', repo_type='model', force_download=True))")
cd $REPO_PATH
export PYTHONPATH=$REPO_PATH:$PYTHONPATH
export PYTHONUNBUFFERED=1

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Found $NUM_GPUS GPUs"

banner "[2.5/4] Diagnostic forward pass..."
python3 -u -c "
import torch, sys
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print(f'PyTorch {{torch.__version__}}, CUDA {{torch.version.cuda}}')
print(f'GPU: {{torch.cuda.get_device_name(0)}}')
print(f'Compute cap: {{torch.cuda.get_device_capability(0)}}')

from sem.config import SEMConfig, ModelConfig, EncoderConfig, SpinorConfig, PropagatorConfig, SamplerConfig, TrainingConfig, V8Config
config = SEMConfig(
    model=ModelConfig(hidden_dim=128, num_layers=2, vocab_size=1000, max_seq_length=64),
    encoder=EncoderConfig(sdr_sparsity=16, sdr_candidates=64),
    spinor=SpinorConfig(block_size=64, num_blocks=2, state_dim=64, mimo_groups=8, d_conv=4),
    propagator=PropagatorConfig(cayley_dt=0.05, cg_max_iter=5, use_chebyshev_kpm=True, chebyshev_degree=16, direct_solve=False),
    sampler=SamplerConfig(),
    training=TrainingConfig(learning_rate=1e-3, gradient_clip=1.0, weight_decay=0.01, unitary_lambda=0.001, warmup_steps=0),
    v8=V8Config(use_lindblad=True, use_hybrid_automata=True, use_quaternionic=True, use_mhc=True),
)

torch.manual_seed(42)
from sem.model import SEMModel
model = SEMModel(config).cuda()
tokens = torch.randint(1, 1000, (4, 32), device='cuda')

# Layer-by-layer diagnostic
with torch.no_grad():
    psi = model.encoder(tokens)
    print(f'encoder: nan={{torch.isnan(psi).any()}}, inf={{torch.isinf(psi).any()}}, range=[{{psi.real.min():.4f}}, {{psi.real.max():.4f}}]')
    for i, layer in enumerate(model.mamba_layers):
        psi = layer(psi)
        print(f'mamba[{{i}}]: nan={{torch.isnan(psi).any()}}, inf={{torch.isinf(psi).any()}}, range=[{{psi.real.min():.4f}}, {{psi.real.max():.4f}}]')
    psi_out = model.propagator(psi)
    print(f'propagator: nan={{torch.isnan(psi_out).any()}}, inf={{torch.isinf(psi_out).any()}}, range=[{{psi_out.real.min():.4f}}, {{psi_out.real.max():.4f}}]')
    psi_normed = model.final_norm(psi_out)
    print(f'final_norm: nan={{torch.isnan(psi_normed).any()}}, inf={{torch.isinf(psi_normed).any()}}')

# Full forward + loss
output = model(tokens, targets=tokens)
loss = output['loss']
print(f'DIAGNOSTIC LOSS: {{loss.item()}} (finite={{torch.isfinite(loss).item()}})')
if not torch.isfinite(loss):
    # Try without propagator
    model.propagator_enabled = False
    output2 = model(tokens, targets=tokens)
    loss2 = output2['loss']
    print(f'LOSS WITHOUT PROPAGATOR: {{loss2.item()}} (finite={{torch.isfinite(loss2).item()}})')
    # Try without V8
    for attr in ['lindblad_layer', 'hybrid_automata', 'quaternionic_gate', 'mhc_residual']:
        if hasattr(model, attr):
            setattr(model, attr, None)
    model.propagator_enabled = True
    output3 = model(tokens, targets=tokens)
    loss3 = output3['loss']
    print(f'LOSS WITHOUT V8: {{loss3.item()}} (finite={{torch.isfinite(loss3).item()}})')
    sys.exit(1)
print('Diagnostic PASSED')
"

banner "[3/4] HP Sweep ({args.hp_trials} trials, $NUM_GPUS workers)..."
HP_DB="sqlite:///sweep/hyperparam_cloud.db"
python3 -c "import optuna; optuna.create_study(study_name='sem-v55-hp-cloud', storage='$HP_DB', direction='minimize', load_if_exists=True)"
for i in $(seq 0 $(($NUM_GPUS - 1))); do
    sleep $i
    CUDA_VISIBLE_DEVICES=$i python3 -u sweep/hyperparam_sweep.py \
        --n-trials {args.hp_trials // 4} --n-steps {args.n_steps} \
        --device cuda --db $HP_DB \
        --study-name sem-v55-hp-cloud &
done
wait
echo "HP sweep done"
python3 -c "
import optuna
s = optuna.load_study('sem-v55-hp-cloud', '$HP_DB')
print(f'HP Best: {{s.best_value:.4f}}')
for k, v in s.best_params.items():
    print(f'  {{k}}: {{v}}')
"

banner "[4/4] NAS Sweep ({args.nas_trials} trials, $NUM_GPUS workers)..."
NAS_DB="sqlite:///sweep/nas_cloud.db"
python3 -c "import optuna; optuna.create_study(study_name='sem-v55-nas-cloud', storage='$NAS_DB', direction='minimize', load_if_exists=True)"
for i in $(seq 0 $(($NUM_GPUS - 1))); do
    sleep $i
    CUDA_VISIBLE_DEVICES=$i python3 -u sweep/nas_search.py \
        --n-trials {args.nas_trials // 4} --n-steps {args.n_steps} \
        --device cuda --db $NAS_DB \
        --study-name sem-v55-nas-cloud &
done
wait
echo "NAS sweep done"
python3 -c "
import optuna
s = optuna.load_study('sem-v55-nas-cloud', '$NAS_DB')
print(f'NAS Best: {{s.best_value:.4f}}')
for k, v in s.best_params.items():
    print(f'  {{k}}: {{v}}')
"

banner "UPLOADING RESULTS..."
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
import glob
for db in glob.glob('sweep/*_cloud.db'):
    api.upload_file(path_or_fileobj=db, path_in_repo=db, repo_id='icarus112/sem-v55-lean-crystal', repo_type='model')
    print(f'Uploaded {{db}}')
"
echo "ALL SWEEPS COMPLETE"
"""

token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    raise ValueError("No HF_TOKEN found.")

api = HfApi(token=token)
job = api.run_job(
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    command=["bash", "-c", cmd],
    flavor=args.flavor,
    timeout=args.timeout,
    secrets={"HF_TOKEN": token},
    namespace=args.org,
)
print(f"Job ID: {job.id}")
print(f"URL:    {job.url}")
