from huggingface_hub import HfApi
import os
import sys

api = HfApi()
job_id = "698053be57c5f7d79b72a670"

print(f"Inspecting job: {job_id}")
try:
    job = api.inspect_job(job_id=job_id)
    print(f"Status: {job.status}")

    print("\nFetching logs...")
    # fetch_job_logs returns a generator of lines
    logs = api.fetch_job_logs(job_id=job_id)
    count = 0
    for line in logs:
        try:
            sys.stdout.buffer.write(line.encode("utf-8"))
            sys.stdout.flush()
        except:
            pass
        count += 1
        if count > 500:  # Limit output
            break
except Exception as e:
    print(f"\nError: {e}")
