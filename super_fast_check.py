from huggingface_hub import HfApi
import sys

api = HfApi()
job_id = "69804ce057c5f7d79b72a66a"

print(f"Checking Job: {job_id}")
try:
    logs = api.fetch_job_logs(job_id=job_id)
    count = 0
    lines = []
    for line in logs:
        if line.strip():
            lines.append(line)
        if len(lines) > 200:
            break

    if lines:
        print("".join(lines[-20:]))
    else:
        print("No logs yet.")
except Exception as e:
    print(f"Error: {e}")
