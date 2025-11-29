## PyTorch Upgrade Plan (branch: fix/upgrade-pytorch-2.8)

Goal
----
Prepare and validate an upgrade of the project's PyTorch runtime while minimizing risk to CI and runtime behavior.

Approach
--------
- Perform the upgrade in a dedicated branch `fix/upgrade-pytorch-2.8`.
- Do not merge until the upgrade passes targeted smoke tests and full test-suite runs on CI.
- Keep non-breaking library upgrades in separate PRs (this branch focuses on runtime upgrade verification).

Suggested steps
---------------
1. Select target versions:
   - Choose a specific `torch` and `torchvision` patch/minor pair compatible with our environment (e.g. `torch>=2.8.0` and matching `torchvision>=0.23.0`).
   - Prefer CPU wheels for CI smoke tests; schedule GPU matrix separately if needed.

2. Add CI matrix job(s):
   - Add a GitHub Actions job `pytorch-upgrade-smoke` that runs on `ubuntu-latest` with Python versions used by the project and installs the chosen `torch`/`torchvision` binaries via pip wheel URLs or `pip install torch==X torchvision==Y --extra-index-url` as appropriate.
   - Limit the run to a fast smoke suite: `pytest -m 'not slow and smoke'` (add a small marker set if needed), plus a quick one-batch trainer smoke run.

3. Update `pyproject.toml`/`requirements.txt` in the branch to pin the target `torch`/`torchvision` versions.

4. Run CI and iterate on failures:
   - If tests fail with numeric differences or API changes, create targeted code fixes or roll back changes.
   - For GPU-specific issues, coordinate a separate GPU test job or run manually on a GPU runner.

5. After CI green on smoke tests, run full test suite in CI (or merge to an integration branch that runs nightly full tests) before merging to `main`.

Notes & risks
---------------
- PyTorch upgrades can introduce ABI, behavior, or numerical differences. Tests may pass but runtime behavior can still differ for training.
- Do not attempt to upgrade PyTorch in the low-risk dependency PRs — keep it isolated.

Checklist (to complete in this branch)
- [ ] Choose exact `torch`/`torchvision` versions to test and pin them.
- [ ] Add CI matrix job for smoke tests (CPU-only first).
- [ ] Pin versions and run CI.
- [ ] Triage failures and apply fixes.
- [ ] Run GPU verification (separate job or manual run).
- [ ] Merge when satisfied.

If you want, I can start by pinning a candidate pair (I will look up recent 2.x patch releases and propose a specific `torch`/`torchvision` pair for CPU-only testing).
