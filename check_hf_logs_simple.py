from huggingface_hub import HfApi
import sys

api = HfApi()
job_id = "698059426f80a03a5692fd97"

print(f"Checking Job: {job_id}")
try:
    logs = api.fetch_job_logs(job_id=job_id)
    count = 0
    for line in logs:
        if line.strip():
            sys.stdout.buffer.write(line.encode("utf-8"))
            sys.stdout.flush()
            count += 1
        if count > 100:
            break
except Exception as e:
    print(f"Error: {e}")
