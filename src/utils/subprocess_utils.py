"""Safe subprocess helpers for probe commands.

Centralize safe, audited subprocess invocation used by hardware probe
utilities. The helper resolves executables via `shutil.which`, runs
commands as an argument list (no shell interpolation), enforces sensible
timeouts and capture options, and returns ``None`` when the command
cannot be executed on the host.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)


def _safe_subprocess_run(
    cmd: List[str],
    timeout: int = 5,
    *,
    capture_output: bool = True,
    text: bool = True,
    cwd: Optional[str] = None,
) -> Optional[subprocess.CompletedProcess]:
    """Run a command safely for probe-style use.

    - `cmd` must be a non-empty list (executable + args).
    - Returns a ``subprocess.CompletedProcess`` on success, ``None`` if the
      executable is not available or another non-fatal error occurs.
    - Raises ``subprocess.TimeoutExpired`` when the command times out so
      callers can decide how to handle timeouts explicitly.
    """
    if not cmd:
        return None

    exe = shutil.which(cmd[0])
    if not exe:
        return None

    cmd = [exe, *cmd[1:]]

    try:
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        # Propagate timeouts so callers can handle them
        raise
    except FileNotFoundError:
        logger.debug("_safe_subprocess_run: executable not found: %s", cmd[0])
        return None
    except OSError:
        # OS-level errors (permission, resource limits, etc.) — fail safely
        logger.debug("_safe_subprocess_run: OSError running %s", cmd, exc_info=True)
        return None
