import os
import requests

owner = "tim-dickey"
repo = "multi-modal-neural-network"
branch = "fix/ci-python-matrix"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

s = requests.Session()
s.headers.update(headers)
url = (
    f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    f"?branch={branch}&per_page=10"
)
print("Querying runs for", branch)
r = s.get(url, timeout=30)
print("Status", r.status_code)
if r.status_code != 200:
    print(r.text)
else:
    data = r.json()
    runs = data.get("workflow_runs", [])
    for run in runs:
        print(
            run.get("id"),
            run.get("name"),
            run.get("status"),
            run.get("conclusion"),
            run.get("created_at"),
        )
