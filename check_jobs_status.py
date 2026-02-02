from huggingface_hub import HfApi
import os

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)

jobs_to_check = ["697fa74d6f80a03a5692fa78", "697faa0b57c5f7d79b72a628"]

print(f"Checking {len(jobs_to_check)} jobs...")
try:
    # list_jobs yields all jobs. We filter client-side.
    all_jobs = api.list_jobs(author="icarus112")
    for job in all_jobs:
        if job.id in jobs_to_check:
            print(f"Job {job.id}: {job.run_state} (Hardware: {job.hardware})")

except Exception as e:
    print(f"Error: {e}")
