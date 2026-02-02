from huggingface_hub import HfApi
import os

token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
job_id = "697faa0b57c5f7d79b72a628"

print(f"Listing jobs to find {job_id}...")
try:
    # Try listing all jobs for the user
    jobs = api.list_jobs(author="icarus112")
    found = False
    for job in jobs:
        if job.id == job_id:
            print(f"Job Found: {job.id}")
            print(f"Status: {job.state}")
            print(f"Hardware: {job.hardware}")
            print(f"Created: {job.created_at}")
            found = True
            break
    if not found:
        print("Job not found in list")
except Exception as e:
    print(f"Error: {e}")
