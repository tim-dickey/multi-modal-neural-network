"""Tests for subprocess_utils module."""

import subprocess
import tempfile

import pytest

from src.utils import subprocess_utils
from src.utils.subprocess_utils import _safe_subprocess_run


class TestSafeSubprocessRun:
    """Tests for _safe_subprocess_run function."""

    def test_safe_subprocess_run_no_exe(self, monkeypatch):
        """When executable not found, return None."""
        monkeypatch.setattr(subprocess_utils.shutil, "which", lambda x: None)
        res = _safe_subprocess_run(["nonexistent"], timeout=1)
        assert res is None

    def test_safe_subprocess_run_success(self, monkeypatch):
        """Test subprocess.run success case.

        When executable exists and subprocess.run returns a
        CompletedProcess, return it.
        """
        monkeypatch.setattr(subprocess_utils.shutil, "which", lambda x: "/bin/true")

        class CP:
            def __init__(self):
                self.returncode = 0
                self.stdout = "ok"

        def fake_run(cmd, **kwargs):
            return CP()

        monkeypatch.setattr(subprocess, "run", fake_run)

        res = _safe_subprocess_run(["true"], timeout=1)
        assert res is not None
        assert getattr(res, "stdout", None) == "ok"

    def test_safe_subprocess_run_missing_executable(self, monkeypatch):
        """Simulate the executable not being on PATH."""
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        res = _safe_subprocess_run(["nonexistent-cmd"], timeout=1)
        assert res is None

    def test_safe_subprocess_run_timeout(self, monkeypatch):
        """Simulate a timeout being raised by subprocess.run."""
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/fakecmd")

        def fake_run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
            cwd=None,
            **kwargs,
        ):
            raise subprocess.TimeoutExpired(
                "fakecmd", timeout
            )  # nosec B603 - test mock

        monkeypatch.setattr("subprocess.run", fake_run)

        with pytest.raises(subprocess.TimeoutExpired):
            _safe_subprocess_run(["fakecmd"], timeout=1)

    def test_safe_subprocess_run_empty_cmd(self):
        """Test with empty command list returns None."""
        res = _safe_subprocess_run([], timeout=1)
        assert res is None

    def test_safe_subprocess_run_file_not_found(self, monkeypatch):
        """Test FileNotFoundError is caught and returns None."""
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/fakecmd")

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("No such file")

        monkeypatch.setattr("subprocess.run", fake_run)

        res = _safe_subprocess_run(["fakecmd"], timeout=1)
        assert res is None

    def test_safe_subprocess_run_os_error(self, monkeypatch):
        """Test OSError is caught and returns None."""
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/fakecmd")

        def fake_run(cmd, **kwargs):
            raise OSError("Permission denied")

        monkeypatch.setattr("subprocess.run", fake_run)

        res = _safe_subprocess_run(["fakecmd"], timeout=1)
        assert res is None

    def test_safe_subprocess_run_with_cwd(self, monkeypatch):
        """Test with cwd parameter."""
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/fakecmd")

        cwd_used = None

        def fake_run(_cmd, cwd=None, **_kwargs):
            nonlocal cwd_used
            cwd_used = cwd
            return subprocess.CompletedProcess(
                args=["fakecmd"], returncode=0, stdout="ok"
            )  # nosec B603 - test mock

        monkeypatch.setattr("subprocess.run", fake_run)

        tmp_dir = tempfile.gettempdir()
        res = _safe_subprocess_run(["fakecmd"], timeout=1, cwd=tmp_dir)
        assert res is not None
        assert cwd_used == tmp_dir

    def test_safe_subprocess_run_with_args(self, monkeypatch):
        """Test command with arguments."""
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/cmd")

        cmd_used = None

        def fake_run(cmd, **kwargs):
            nonlocal cmd_used
            cmd_used = cmd
            return subprocess.CompletedProcess(
                args=["cmd"], returncode=0, stdout="ok"
            )  # nosec B603 - test mock

        monkeypatch.setattr("subprocess.run", fake_run)

        res = _safe_subprocess_run(["cmd", "--version", "-v"], timeout=1)
        assert res is not None
        assert cmd_used == ["/usr/bin/cmd", "--version", "-v"]
