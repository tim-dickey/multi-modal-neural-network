import os
import time
import requests
import zipfile
import sys

owner = "tim-dickey"
repo = "multi-modal-neural-network"
branch = "fix/upgrade-pytorch-2.6"

GITHUB_TOKEN = (
    os.environ.get("GITHUB_TOKEN")
    or os.environ.get("GH_TOKEN")
    or os.environ.get("CODACY_PROJECT_TOKEN")
)
headers = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

session = requests.Session()
session.headers.update(headers)

print("Querying GitHub Actions for workflow runs...")

runs_url = (
    f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    f"?branch={branch}&per_page=50"
)

run_info = None
for attempt in range(1, 13):
    try:
        r = session.get(runs_url, timeout=30)
        r.raise_for_status()
        data = r.json()
        runs = data.get("workflow_runs", [])
        if not runs:
            print(f"No runs found (attempt {attempt}).")
        else:
            # pick the most recent run
            run = runs[0]
            run_id = run.get("id")
            status = run.get("status")
            conclusion = run.get("conclusion")
            created = run.get("created_at")
            print(
                "Found run",
                run_id,
                "status=", status,
                "conclusion=", conclusion,
                "created_at=", created,
            )
            if status == "completed":
                run_info = run
                break
            else:
                print("Run not completed yet; waiting...")
    except Exception as exc:
        print("Error querying runs:", exc)
    time.sleep(15)

if run_info is None:
    print("No completed workflow run found within timeout. Exiting.")
    sys.exit(2)

run_id = run_info.get("id")
print("Using run_id=", run_id)

artifacts_url = (
    f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
)
try:
    r = session.get(artifacts_url, timeout=30)
    r.raise_for_status()
    artifacts = r.json().get("artifacts", [])
    print(f"Found {len(artifacts)} artifacts")
except Exception as exc:
    print("Failed to list artifacts:", exc)
    sys.exit(3)

os.makedirs("test_logs/codacy", exist_ok=True)

for art in artifacts:
    name = art.get("name")
    aid = art.get("id")
    url = art.get("archive_download_url")
    print(f"Downloading artifact {name} (id={aid})...")
    try:
        resp = session.get(url, stream=True, timeout=60)
        # resp may be a redirect to storage; requests follows redirects
        resp.raise_for_status()
        zdata = resp.content
        zip_path = f"test_logs/codacy/{name or aid}.zip"
        with open(zip_path, "wb") as f:
            f.write(zdata)
        print("Saved", zip_path)
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall("test_logs/codacy")
            print("Extracted", zip_path)
        except zipfile.BadZipFile:
            print("Artifact is not a zip or failed to extract:", zip_path)
    except Exception as exc:
        print("Failed to download artifact:", exc)

print("\nContents of test_logs/codacy:")
for root, dirs, files in os.walk("test_logs/codacy"):
    for fname in files:
        print(os.path.join(root, fname))
print("\nDone.")
