from huggingface_hub import HfApi
import os
import time

token = os.environ.get("HF_TOKEN")
if not token:
    print("No token found")
    exit(1)

api = HfApi(token=token)
job_id = "697fa7f76f80a03a5692fa7e"
repo_id = "icarus112/sem-v55-lean-crystal"

print(f"Checking status for job {job_id}...")
try:
    # HfApi.run_job returns a job object, but to get status we need to fetch it
    # Since there's no direct get_job_status, we infer from logs or list_jobs
    # Let's try to just fetch logs loop until we see something or it fails

    # Actually, let's use list_jobs and filter
    jobs = api.list_jobs(repo_id=repo_id)
    for job in jobs:
        if job.id == job_id:
            print(f"Status: {job.status}")
            print(f"Hardware: {job.hardware}")
            break
    else:
        print("Job not found in list")

except Exception as e:
    print(f"Error checking status: {e}")
