import os
import sys
from huggingface_hub import HfApi

api = HfApi()
job_id = "69804ce057c5f7d79b72a66a"

print(f"Checking Job: {job_id}")
try:
    # Get last 100 lines of logs
    logs = api.fetch_job_logs(job_id=job_id)
    lines = []
    for i, line in enumerate(logs):
        lines.append(line)
        if i > 1000:
            break  # Guard

    # Print last 50 lines to see if it failed or is progressing
    print("".join(lines[-50:]))
except Exception as e:
    print(f"Error: {e}")
