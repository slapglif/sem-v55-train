from huggingface_hub import HfApi
import sys

api = HfApi()
job_id = "698059426f80a03a5692fd97"

try:
    logs = api.fetch_job_logs(job_id=job_id)
    # Just grab the last line available
    last_line = ""
    for i, line in enumerate(logs):
        if line.strip():
            last_line = line
        if i > 1000:
            break
    print(f"Last Log: {last_line}")
except Exception as e:
    print(f"Error: {e}")
