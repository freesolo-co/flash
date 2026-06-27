"""`flash version` / `flash --version` surface the package version."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from flash import __version__


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "flash.cli", *args],
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


def test_version_matches_installed_metadata():
    # regression for the 0.2.20 desync: the wheel metadata said 0.2.20 while flash/__init__.py
    # still hard-coded __version__ = "0.2.19", so flash nagged to upgrade forever while uv reported
    # nothing to upgrade. __version__ must now be read from the installed distribution metadata
    # (single-sourced from pyproject), never a hand-maintained literal that can drift.
    try:
        installed = importlib.metadata.version("freesolo-flash")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("freesolo-flash is not installed; __version__ falls back to a dev sentinel")
    assert __version__ == installed
