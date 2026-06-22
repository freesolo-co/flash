"""Regression test: the documented `flash env install <owner/name>` flow.

DEFECT (since fixed): `flash env install primeintellect/hendrycks-math` ran
`pip install primeintellect/hendrycks-math` (a local path) with no package index, so it
always failed. The fix derives the bare wheel name from the `owner/name` id, adds the
extra index URL, and records the index in the manifest so the GPU worker can install it too.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


class _FakeProc:
    returncode = 0


class _FailedProc:
    returncode = 7
    stdout = "stdout detail"
    stderr = "pip could not find the package"


def test_env_install_prime_hub_slug(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import flash.envs.registry as registry
        from flash.cli import main as cli

        monkeypatch.setattr(registry, "INSTALLED_MANIFEST", Path(tmp) / "envs.json")

        recorded = {}
        monkeypatch.setattr(
            "subprocess.run", lambda cmd, *a, **k: recorded.update(cmd=cmd) or _FakeProc()
        )
        # No `prime` CLI; `uv` present -> pip path via uv.
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        args = argparse.Namespace(env_id="primeintellect/hendrycks-math")
        rc = cli.cmd_env_install(args)
        assert rc == 0

        cmd = recorded["cmd"]
        # installs the BARE wheel name, not the owner/name slug (which pip treats as a path)
        assert "hendrycks-math" in cmd
        assert "primeintellect/hendrycks-math" not in cmd
        # carries the package index
        assert "--extra-index-url" in cmd
        assert any("hub.primeintellect.ai" in str(c) for c in cmd)

        # manifest records package + index so the worker can reinstall it
        with open(os.path.join(tmp, "envs.json")) as f:
            manifest = json.load(f)
        entry = manifest["primeintellect/hendrycks-math"]
        assert entry["package"] == "hendrycks-math"
        assert "hub.primeintellect.ai" in entry["extra_index_url"]


def test_env_install_failure_surfaces_installer_output(monkeypatch, capsys):
    import flash.envs.registry as registry
    from flash.cli import main as cli

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(registry, "INSTALLED_MANIFEST", Path(tmp) / "envs.json")
        monkeypatch.setattr("subprocess.run", lambda *a, **k: _FailedProc())
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        rc = cli.cmd_env_install(argparse.Namespace(env_id="owner/missing-env"))

    assert rc == 7
    err = capsys.readouterr().err
    assert "install failed" in err
    assert "pip could not find the package" in err
    assert "stdout detail" in err


def test_env_install_rejects_bare_id(monkeypatch, capsys):
    # The managed service requires full `owner/name` ids. A bare id can't be resolved,
    # so `cmd_env_install` must reject it (return 1) WITHOUT shelling out to prime/pip.
    from flash.cli import main as cli

    called = {"n": 0}
    monkeypatch.setattr("subprocess.run", lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    rc = cli.cmd_env_install(argparse.Namespace(env_id="gsm8k"))
    assert rc == 1
    assert called["n"] == 0  # no install attempted
    err = capsys.readouterr().err
    assert "owner/name" in err

    # Malformed slugs (trailing slash, extra segment) are rejected too.
    assert cli.cmd_env_install(argparse.Namespace(env_id="owner/")) == 1
    assert cli.cmd_env_install(argparse.Namespace(env_id="a/b/c")) == 1
    assert called["n"] == 0
