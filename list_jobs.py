import subprocess
import os


def list_jobs():
    env = os.environ.copy()
    env["CI"] = "true"
    try:
        result = subprocess.run(
            ["hf", "jobs", "ps"], capture_output=True, text=True, env=env
        )
        print(result.stdout)
        print(result.stderr)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    list_jobs()
