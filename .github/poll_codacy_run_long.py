import os
import time
import requests
import sys
import zipfile

owner = "tim-dickey"
repo = "multi-modal-neural-network"
branch = "fix/upgrade-pytorch-2.6"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

s = requests.Session()
s.headers.update(headers)

runs_url = (
    f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    f"?branch={branch}&per_page=10"
)
run_info = None
print("Polling for completed workflow run (timeout 30 minutes)...")
for i in range(120):
    try:
        r = s.get(runs_url, timeout=30)
        r.raise_for_status()
        data = r.json()
        runs = data.get("workflow_runs", [])
        if not runs:
            print("No runs found; sleeping...")
        else:
            run = runs[0]
            print(
                "Attempt",
                i + 1,
                ": run id=",
                run.get("id"),
                "status=",
                run.get("status"),
                "conclusion=",
                run.get("conclusion"),
            )
            if run.get("status") == "completed":
                run_info = run
                break
    except Exception as exc:
        print("Error querying runs:", exc)
    time.sleep(15)

if run_info is None:
    print("Timeout waiting for run completion. Exiting with code 2.")
    sys.exit(2)

run_id = run_info.get("id")
print("Run completed:", run_id, run_info.get("conclusion"))

# Fetch jobs and steps
jobs_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
r = s.get(jobs_url, timeout=30)
r.raise_for_status()
jobs = r.json().get("jobs", [])
print("\nJobs and steps:")
for job in jobs:
    print("---")
    print("Job id:", job.get("id"))
    print("Name:", job.get("name"))
    print("Status:", job.get("status"), "Conclusion:", job.get("conclusion"))
    for step in job.get("steps", []):
        print(
            " Step:",
            step.get("number"),
            step.get("name"),
            "->",
            step.get("status"),
            step.get("conclusion"),
        )

# Try to list and download artifacts
artifacts_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
print("\nListing artifacts...")
r = s.get(artifacts_url, timeout=30)
if r.status_code != 200:
    print("Failed to list artifacts:", r.status_code, r.text[:500])
else:
    arts = r.json().get("artifacts", [])
    print("Found", len(arts), "artifacts")
    os.makedirs("test_logs/codacy", exist_ok=True)
    for art in arts:
        name = art.get("name")
        url = art.get("archive_download_url")
        print("Downloading", name)
        rr = s.get(url, stream=True, timeout=60)
        if rr.status_code == 200:
            zpath = f"test_logs/codacy/{name}.zip"
            open(zpath, "wb").write(rr.content)
            try:
                with zipfile.ZipFile(zpath) as z:
                    z.extractall("test_logs/codacy")
                print("Extracted", zpath)
            except Exception as exc:
                print("Extract failed", exc)
        else:
            print("Download failed with", rr.status_code, rr.text[:500])

print("\nDone")
