import os
from huggingface_hub import HfApi, get_token
import argparse
import shlex

parser = argparse.ArgumentParser()
parser.add_argument("--timeout", type=int, default=21600)
parser.add_argument("--flavor", type=str, default="a100x4")
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--push-to-hub", type=str, default=None)
parser.add_argument(
    "--org", type=str, default=None, help="Organization namespace to run the job under"
)
args, extra = parser.parse_known_args()

extra_args_list = extra.copy()
if args.config:
    extra_args_list.extend(["--config", args.config])
if args.push_to_hub:
    extra_args_list.extend(["--push-to-hub", args.push_to_hub])

extra_args = " ".join(shlex.quote(a) for a in extra_args_list)

cmd = (
    r"""set -e
pytime() { python3 -c "import time; print(f'{time.time():.3f}')"; }
pdiff() { python3 -c "print(f'{float(\"$2\")-float(\"$1\"):.1f}')"; }
banner() { echo ""; echo "=============================="; echo "$1"; echo "=============================="; }

banner "[STAGE 1/5] Installing UV..."
S=$(pytime)
pip install -q uv 2>&1 | tail -1
E=$(pytime); echo "[STAGE 1/5] Done in $(pdiff $S $E)s"

banner "[STAGE 2/5] Installing dependencies via UV..."
S=$(pytime)
uv pip install --system datasets tokenizers pyyaml scipy einops wandb 'huggingface_hub[hf_xet]' lightning rich mamba-ssm causal-conv1d
E=$(pytime); echo "[STAGE 2/5] Done in $(pdiff $S $E)s"

banner "[STAGE 3/5] Downloading model repo..."
S=$(pytime)
REPO_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('icarus112/sem-v55-lean-crystal', repo_type='model'))")
E=$(pytime); echo "[STAGE 3/5] Repo downloaded to $REPO_PATH in $(pdiff $S $E)s"

banner "[STAGE 3.5/5] Pre-downloading FineWeb-Edu subset..."
S=$(pytime)
# datasets/Arrow aborts on thread cleanup (exit code 134); data is fine, tolerate the crash
python3 -c "
import os, json, sys
from datasets import load_dataset
from pathlib import Path

cache_dir = Path('/tmp/fineweb_cache')
cache_dir.mkdir(parents=True, exist_ok=True)
cache_file = cache_dir / 'fineweb_edu_score2.jsonl'

if cache_file.exists() and cache_file.stat().st_size > 100_000_000:
    print(f'Cache already exists: {cache_file.stat().st_size / 1e9:.2f} GB')
    sys.exit(0)

print('Downloading FineWeb-Edu subset (500K docs)...')
ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True, token=os.environ.get('HF_TOKEN'))
ds = ds.filter(lambda x: x.get('score', 0) >= 2)
count = 0
with open(cache_file, 'w') as f:
    for doc in ds:
        text = doc.get('text', '')
        if text.strip():
            f.write(json.dumps({'text': text}) + '\n')
            count += 1
            if count % 10000 == 0:
                print(f'  Downloaded {count} docs...')
            if count >= 500000:
                break
    f.flush()
    os.fsync(f.fileno())
print(f'Downloaded {count} docs to {cache_file} ({cache_file.stat().st_size / 1e9:.2f} GB)')
os._exit(0)
" || true
python3 -c "
import os
f = '/tmp/fineweb_cache/fineweb_edu_score2.jsonl'
if os.path.exists(f) and os.path.getsize(f) > 100_000_000:
    sz = os.path.getsize(f) / 1e9
    lines = sum(1 for _ in open(f))
    print(f'[VERIFIED] Cache file OK: {lines} docs, {sz:.2f} GB')
else:
    print('[ERROR] Cache file missing or too small, falling back to streaming')
"
E=$(pytime); echo "[STAGE 3.5/5] Done in $(pdiff $S $E)s"

banner "[STAGE 4/5] Starting training..."
cd $REPO_PATH
export PYTHONPATH=$REPO_PATH:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 hf_train_lightning.py --no-compile --devices -1 --strategy ddp_find_unused_parameters_true --precision bf16-mixed """
    + extra_args
)

token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    raise ValueError("No HF_TOKEN found in environment or local login.")

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
print(f"URL: {job.url}")
