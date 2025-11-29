## PyTorch Upgrade: `torch>=2.8.0` / `torchvision>=0.20.0`

**Summary**
- This PR proposes upgrading PyTorch to `>=2.8.0` and `torchvision` to `>=0.20.0` (candidate tested in smoke workflows).

**What I changed**
- Pins candidate PyTorch pair in dependency manifests on the upgrade branch.
- Adds GPU smoke workflow to download & inspect wheels for native libraries.

**CI / Verification**
- CPU smoke tests (matrix: Python 3.10, 3.11) run in `pytorch-upgrade.yml`.
- GPU smoke job runs on a self-hosted runner labeled `gpu` and downloads the wheels for inspection.
- Please add the `CODACY_PROJECT_TOKEN` repo secret to allow Codacy/Trivy runs and SARIF upload.

**Manual checks performed**
- Local CPU smoke test: `pytest tests/test_integration.py::TestEndToEndTraining::test_training_epoch` (passed locally).

**Risk & Rollback plan**
- This PR is isolated to a candidate pin and CI checks. If tests fail on CI or GPU artifacts indicate a problem, the PR can be reverted.

**Next steps (after merge)**
- Run full CI + GPU verification on a GPU runner.
- If GPU run passes, coordinate a staged rollout and monitor releases for any security advisories.

**Notes / Attachments**
- See `docs/UPGRADE_PYTORCH.md` for the full upgrade plan and wheel inspection guidance.
