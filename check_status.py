from huggingface_hub import HfApi
import os

token = os.environ.get("HF_TOKEN")
if not token:
    print("No token found")
    exit(1)

api = HfApi(token=token)
# Job ID from previous turn
job_id = "697feefd57c5f7d79b72a661"
try:
    # Use hf api to get job status directly if possible, or list jobs
    jobs = api.list_jobs()
    found = False
    for j in jobs:
        if j.id == job_id:
            print(f"Job Status: {j.status}")
            print(f"Hardware: {j.compute}")
            found = True
            break
    if not found:
        print(f"Job {job_id} not found in recent list")
except Exception as e:
    print(f"Error checking job: {e}")
