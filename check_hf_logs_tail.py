from huggingface_hub import HfApi
import sys

api = HfApi()
job_id = "698055906f80a03a5692fd85"

print(f"Checking Job: {job_id}")
try:
    logs = api.fetch_job_logs(job_id=job_id)
    lines = []
    for line in logs:
        if line.strip():
            lines.append(line)
        if len(lines) > 5000:  # Increase limit to find metrics
            break

    if lines:
        print("".join(lines[-50:]))
    else:
        print("No logs yet.")
except Exception as e:
    print(f"Error: {e}")
