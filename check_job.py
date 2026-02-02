"""Check HF job status and fetch logs with timeout."""

import sys
import os
import threading

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from huggingface_hub import HfApi

job_id = sys.argv[1] if len(sys.argv) > 1 else "697f8bf36f80a03a5692f9fc"
api = HfApi()

job = api.inspect_job(job_id=job_id)
print(f"Status: {job.status}")
print(f"Created: {job.created_at}")
print(f"Flavor: {job.flavor}")

lines = []
done = threading.Event()


def collect_logs():
    try:
        for chunk in api.fetch_job_logs(job_id=job_id):
            lines.append(chunk)
            if done.is_set() or len(lines) > 1000:
                break
    except Exception:
        pass


t = threading.Thread(target=collect_logs, daemon=True)
t.start()
t.join(timeout=15)
done.set()

text = "".join(lines)
if text:
    print(f"\n--- LOGS ({len(lines)} chunks) ---")
    if len(text) > 8000:
        print(text[-8000:])
    else:
        print(text)
else:
    print("\n(no logs available yet)")
