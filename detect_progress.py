from huggingface_hub import HfApi
import sys

api = HfApi()
job_id = "69804ce057c5f7d79b72a66a"

print(f"Checking Job: {job_id}")
try:
    logs = api.fetch_job_logs(job_id=job_id)
    for line in logs:
        if "STAGE 4/4" in line:
            print("Reached STAGE 4/4!")
        if "Launching SEM V5.5 Training" in line:
            print("Training loop started!")
        if "loss=" in line:
            print(f"Found loss: {line.strip()}")
            break
except Exception as e:
    print(f"Error: {e}")
