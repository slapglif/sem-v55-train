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
        if len(lines) > 20000:  # Increase limit
            break

    if lines:
        # Print last 100 non-empty lines
        for l in lines[-100:]:
            sys.stdout.write(l)
    else:
        print("No logs yet.")
except Exception as e:
    print(f"Error: {e}")
