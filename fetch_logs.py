"""Fetch logs from HF job."""

import os
from huggingface_hub import HfApi
import sys

job_id = "6980078357c5f7d79b72a666"

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
print(f"Fetching logs for job {job_id}...")
try:
    lines = []
    for chunk in api.fetch_job_logs(job_id=job_id):
        lines.append(chunk)

    full_log = "".join(lines)
    if not full_log:
        print("(no logs returned yet)")
    else:
        with open("hf_job.log", "w", encoding="utf-8") as handle:
            handle.write(full_log)
        print("=" * 40)
        tail = full_log[-4000:]
        print(tail.encode("utf-8", "replace").decode("utf-8"))
        print("=" * 40)
except Exception as e:
    with open("hf_job_error.txt", "w", encoding="utf-8") as handle:
        handle.write(repr(e))
    print("Error: see hf_job_error.txt")
