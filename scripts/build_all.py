"""Run the full offline data pipeline in order. Each step caches its outputs,
so re-running after a partial failure resumes cheaply.
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    "fetch_country_list.py",
    "fetch_city_list.py",
    "fetch_structured_stats.py",
    "build_profiles.py",
    "build_embeddings.py",
]


def main():
    for step in STEPS:
        path = os.path.join(SCRIPT_DIR, step)
        print(f"\n=== Running {step} ===")
        result = subprocess.run([sys.executable, path])
        if result.returncode != 0:
            print(f"FAILED at {step} (exit code {result.returncode})")
            sys.exit(result.returncode)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
