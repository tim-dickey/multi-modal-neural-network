import os
import requests
import sys

owner = "tim-dickey"
repo = "multi-modal-neural-network"
run_id = 19767186334
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

s = requests.Session()
s.headers.update(headers)

jobs_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
print("Requesting jobs for run", run_id)
r = s.get(jobs_url, timeout=30)
print("Status", r.status_code)
if r.status_code != 200:
    print("Response:", r.text[:1000])
    sys.exit(1)
jobs = r.json().get("jobs", [])
print("Found", len(jobs), "jobs")
for job in jobs:
    print("---")
    print("Job id:", job.get("id"))
    print("Name:", job.get("name"))
    print("Status:", job.get("status"), "Conclusion:", job.get("conclusion"))
    steps = job.get("steps", [])
    for step in steps:
        name = step.get("name")
        status = step.get("status")
        conclusion = step.get("conclusion")
        number = step.get("number")
        print(
            f"  Step {number}: {name} -> status={status} conclusion={conclusion}"
        )
        # Use local variables to keep the conditional line short
        if conclusion == "failure" or (
            status == "completed" and conclusion != "success"
        ):
            print(
                "    Failure details (if available):",
                step.get("number"),
                step.get("status"),
                step.get("conclusion"),
            )

print("\nDone")
