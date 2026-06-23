"""`flash env install` records Freesolo environment ids."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def test_env_install_github_ref(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import flash.envs.registry as registry
        from flash.cli import main as cli

        manifest_path = Path(tmp) / "envs.json"
        monkeypatch.setattr(registry, "INSTALLED_MANIFEST", manifest_path)
        called = {"n": 0}
        monkeypatch.setattr("subprocess.run", lambda *a, **k: called.__setitem__("n", called["n"] + 1))

        env_id = "github:owner/repo@main:envs/math/environment.py"
        rc = cli.cmd_env_install(argparse.Namespace(env_id=env_id))
        assert rc == 0
        assert called["n"] == 0

        manifest = json.loads(manifest_path.read_text())
        assert manifest[env_id]["package"] == "freesolo"


def test_env_install_rejects_non_github_ref(monkeypatch, capsys):
    from flash.cli import main as cli

    called = {"n": 0}
    monkeypatch.setattr("subprocess.run", lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    rc = cli.cmd_env_install(argparse.Namespace(env_id="gsm8k"))
    assert rc == 1
    assert called["n"] == 0
    err = capsys.readouterr().err
    assert "Freesolo environment id" in err

    assert cli.cmd_env_install(argparse.Namespace(env_id="owner/name")) == 1
    assert called["n"] == 0
