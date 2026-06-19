"""Package logging is quiet on import and configurable via the CLI / env."""

from __future__ import annotations

import logging
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_import_is_quiet_and_has_nullhandler():
    import flash._logging as ctlog

    root = logging.getLogger("flash")
    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)
    # get_logger namespaces under "flash".
    assert ctlog.get_logger("providers.runpod.train").name == "flash.providers.runpod.train"
    assert ctlog.get_logger(__name__).name.startswith("flash.")


def test_configure_logging_sets_level():
    import flash._logging as ctlog

    ctlog.configure_logging(verbosity=2)
    assert logging.getLogger("flash").level == logging.DEBUG
    ctlog.configure_logging(verbosity=1)
    assert logging.getLogger("flash").level == logging.INFO
    ctlog.configure_logging(verbosity=0)
    assert logging.getLogger("flash").level == logging.WARNING
    # configure twice -> exactly one console handler (no stacking).
    ctlog.configure_logging(verbosity=0)
    consoles = [
        h for h in logging.getLogger("flash").handlers if getattr(h, "_flash_console", False)
    ]
    assert len(consoles) == 1


def test_importing_flash_emits_no_stderr():
    # A fresh interpreter importing slm must produce no output.
    proc = subprocess.run(
        [sys.executable, "-c", "import flash, flash.runner, flash.providers.runpod.train"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == ""
