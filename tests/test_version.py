"""`flash version` / `flash --version` surface the package version."""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from flash import __version__


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "flash.cli.main", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )


def test_version_flag():
    proc = _run(["--version"])
    assert proc.returncode == 0, proc.stdout
    assert __version__ in proc.stdout


def test_version_subcommand():
    proc = _run(["version"])
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.strip() == f"flash {__version__}"
