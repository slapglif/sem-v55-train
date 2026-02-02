import subprocess
import sys


def get_jobs():
    try:
        result = subprocess.run(
            ["hf", "jobs", "status", "--all"], capture_output=True, text=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    get_jobs()
