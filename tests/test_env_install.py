"""Regression test: the documented `slm env install <prime-hub-env>` flow.

DEFECT (fixed in this PR): `slm env install primeintellect/hendrycks-math` ran
`pip install primeintellect/hendrycks-math` (a local path) with no Prime Hub index, so it
always failed. The fix derives the bare wheel name from the `owner/name` slug, defaults
Hub slugs to the Prime index via `--extra-index-url`, and records the index in the manifest
so the GPU worker can install it too.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class _FakeProc:
    returncode = 0


def test_env_install_prime_hub_slug(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("AUTOSLM_ENVS_MANIFEST", os.path.join(tmp, "envs.json"))
        import autoslm.envs.registry as registry
        from autoslm.cli import main as cli

        importlib.reload(registry)

        recorded = {}
        monkeypatch.setattr(
            "subprocess.run", lambda cmd, *a, **k: recorded.update(cmd=cmd) or _FakeProc()
        )
        # No `prime` CLI; `uv` present -> pip path via uv.
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        args = argparse.Namespace(
            env_id="primeintellect/hendrycks-math", package=None, extra_index_url=None
        )
        rc = cli.cmd_env_install(args)
        assert rc == 0

        cmd = recorded["cmd"]
        # installs the BARE wheel name, not the owner/name slug (which pip treats as a path)
        assert "hendrycks-math" in cmd
        assert "primeintellect/hendrycks-math" not in cmd
        # carries the Prime Hub index
        assert "--extra-index-url" in cmd
        assert any("hub.primeintellect.ai" in str(c) for c in cmd)

        # manifest records package + index so the worker can reinstall it
        with open(os.path.join(tmp, "envs.json")) as f:
            manifest = json.load(f)
        entry = manifest["primeintellect/hendrycks-math"]
        assert entry["package"] == "hendrycks-math"
        assert "hub.primeintellect.ai" in entry["extra_index_url"]

        monkeypatch.delenv("AUTOSLM_ENVS_MANIFEST", raising=False)
        importlib.reload(registry)


def test_env_install_respects_explicit_package_and_index(monkeypatch):
    """A user-supplied --package/--extra-index-url must be honored (not overridden)."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("AUTOSLM_ENVS_MANIFEST", os.path.join(tmp, "envs.json"))
        import autoslm.envs.registry as registry
        from autoslm.cli import main as cli

        importlib.reload(registry)

        recorded = {}
        monkeypatch.setattr(
            "subprocess.run", lambda cmd, *a, **k: recorded.update(cmd=cmd) or _FakeProc()
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

        args = argparse.Namespace(
            env_id="owner/custom",
            package="my-wheel==1.2",
            extra_index_url="https://example.com/simple/",
        )
        rc = cli.cmd_env_install(args)
        assert rc == 0
        cmd = recorded["cmd"]
        assert "my-wheel==1.2" in cmd
        assert "https://example.com/simple/" in cmd

        with open(os.path.join(tmp, "envs.json")) as f:
            manifest = json.load(f)
        assert manifest["owner/custom"]["package"] == "my-wheel==1.2"
        assert manifest["owner/custom"]["extra_index_url"] == "https://example.com/simple/"

        monkeypatch.delenv("AUTOSLM_ENVS_MANIFEST", raising=False)
        importlib.reload(registry)
