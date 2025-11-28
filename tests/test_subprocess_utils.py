import subprocess

import pytest

from src.utils.subprocess_utils import _safe_subprocess_run


def test_safe_subprocess_run_missing_executable(monkeypatch):
    # Simulate the executable not being on PATH
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    res = _safe_subprocess_run(["nonexistent-cmd"], timeout=1)
    assert res is None


def test_safe_subprocess_run_success(monkeypatch):
    # Simulate a successful completed process
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/fakecmd")

    completed = subprocess.CompletedProcess(args=["/usr/bin/fakecmd"], returncode=0, stdout="ok\n")

    def fake_run(cmd, capture_output=True, text=True, timeout=1, check=False, cwd=None, **kwargs):
        return completed

    monkeypatch.setattr("subprocess.run", fake_run)

    res = _safe_subprocess_run(["fakecmd", "--version"], timeout=1)
    assert res is completed
    assert res.returncode == 0
    assert res.stdout.strip() == "ok"


def test_safe_subprocess_run_timeout(monkeypatch):
    # Simulate a timeout being raised by subprocess.run
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/fakecmd")

    def fake_run(cmd, capture_output=True, text=True, timeout=1, check=False, cwd=None, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired):
        _safe_subprocess_run(["fakecmd"], timeout=1)
