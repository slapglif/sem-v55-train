import os
from huggingface_hub import HfApi, get_token
import argparse
import shlex

parser = argparse.ArgumentParser()
parser.add_argument("--timeout", type=int, default=1800)
parser.add_argument("--flavor", type=str, default="l40sx1")
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--push-to-hub", type=str, default=None)
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

banner "[STAGE 1/4] Installing UV..."
S=$(pytime)
pip install -q uv 2>&1 | tail -1
E=$(pytime); echo "[STAGE 1/4] Done in $(pdiff $S $E)s"

banner "[STAGE 2/4] Installing dependencies via UV..."
S=$(pytime)
uv pip install --system datasets tokenizers pyyaml scipy einops wandb 'huggingface_hub[hf_xet]'
E=$(pytime); echo "[STAGE 2/4] Done in $(pdiff $S $E)s"

banner "[STAGE 3/4] Downloading model repo..."
S=$(pytime)
REPO_PATH=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('icarus112/sem-v55-lean-crystal', repo_type='model'))")
E=$(pytime); echo "[STAGE 3/4] Repo downloaded to $REPO_PATH in $(pdiff $S $E)s"

banner "[STAGE 4/4] Starting training..."
cd $REPO_PATH
export PYTHONPATH=$REPO_PATH:$PYTHONPATH
export PYTHONUNBUFFERED=1
python3 hf_train.py """
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
)
print(f"Job ID: {job.id}")
print(f"URL: {job.url}")
