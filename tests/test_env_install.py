"""`flash env install` records Freesolo environment ids."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def test_env_install_slug(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as tmp:
        import flash.cli as cli
        import flash.envs.registry as registry

        manifest_path = Path(tmp) / "envs.json"
        monkeypatch.setattr(registry, "INSTALLED_MANIFEST", manifest_path)
        called = {"n": 0}
        monkeypatch.setattr(
            "subprocess.run", lambda *a, **k: called.__setitem__("n", called["n"] + 1)
        )

        env_id = "owner/name"
        rc = cli.cmd_env_install(argparse.Namespace(env_id=env_id))
        assert rc == 0
        assert called["n"] == 0

        manifest = json.loads(manifest_path.read_text())
        assert manifest[env_id]["package"] == "freesolo"
        assert f"recorded in {manifest_path}" in capsys.readouterr().out


def test_env_install_rejects_bad_id(monkeypatch, capsys):
    import flash.cli as cli

    called = {"n": 0}
    monkeypatch.setattr("subprocess.run", lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    rc = cli.cmd_env_install(argparse.Namespace(env_id="gsm8k"))
    assert rc == 1
    assert called["n"] == 0
    err = capsys.readouterr().err
    assert "Freesolo environment id" in err
