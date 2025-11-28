import os
import sys
import requests
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
    f"?branch={branch}&per_page=5"
)
print("Listing recent runs...")
r = s.get(runs_url, timeout=30)
r.raise_for_status()
data = r.json()
runs = data.get("workflow_runs", [])
if not runs:
    print("No runs found")
    sys.exit(1)
run = runs[0]
print("Latest run:", run.get("id"), run.get("status"), run.get("conclusion"))
run_id = run.get("id")

logs_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
print("Requesting logs archive...")
resp = s.get(logs_url, stream=True, timeout=60)
if resp.status_code == 200:
    # sometimes returns zip directly
    zdata = resp.content
    path = f"test_logs/codacy/workflow_{run_id}_logs.zip"
    open(path, "wb").write(zdata)
    print("Saved logs to", path)
    try:
        with zipfile.ZipFile(path) as z:
            z.extractall(f"test_logs/codacy/workflow_{run_id}_logs")
        print("Extracted logs")
    except Exception as exc:
        print("Failed to extract logs zip:", exc)
else:
    print("Logs request status:", resp.status_code)
    print("Response headers:", resp.headers)
    text = resp.text
    print("Response text (truncated):", text[:1000])
    sys.exit(2)

# Search for failure hints
errors = []
target_dir = f"test_logs/codacy/workflow_{run_id}_logs"
for root, dirs, files in os.walk(target_dir):
    for fn in files:
        if fn.endswith((".txt", ".log", ".out", ".err")):
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                lc = content.lower()
                if (
                    "error" in lc
                    or "traceback" in lc
                    or "exit code" in lc
                    or "failed" in lc
                ):
                    tail = content.splitlines()[-20:] if content.count("\n") else []
                    errors.append((fp, tail))
            except Exception:
                pass

print("\nFound", len(errors), "files with likely errors (listed):")
for fp, tail in errors:
    print("---", fp)
    for line in tail:
        print(line)

print("\nDone")
