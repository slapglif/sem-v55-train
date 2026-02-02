"""Monitor HF job status and fetch logs."""

import os
import sys
import time
import argparse
from huggingface_hub import HfApi, get_token


def monitor_job(job_id, interval=30):
    """Monitor job until completion."""
    token = os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)

    print(f"Monitoring job: {job_id}")
    print(f"URL: https://huggingface.co/jobs/icarus112/{job_id}")
    print("=" * 60)

    last_status = None

    while True:
        try:
            # Get job info
            job = api.get_job(job_id)
            status = job.status

            if status != last_status:
                print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")
                last_status = status

                if status == "completed":
                    print("=" * 60)
                    print("Job completed successfully!")
                    return 0
                elif status == "failed":
                    print("=" * 60)
                    print("Job failed!")
                    # Try to fetch logs
                    try:
                        logs = api.get_job_logs(job_id)
                        print("\nLast 50 lines of logs:")
                        print("\n".join(logs.split("\n")[-50:]))
                    except:
                        pass
                    return 1
                elif status == "cancelled":
                    print("=" * 60)
                    print("Job was cancelled!")
                    return 1

            # Still running
            if status == "running":
                sys.stdout.write(".")
                sys.stdout.flush()

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            return 0
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="698013276f80a03a5692fc67")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    exit(monitor_job(args.job_id, args.interval))
