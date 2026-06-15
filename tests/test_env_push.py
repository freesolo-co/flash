"""`slm env push` packages a bare verifiers module and publishes it to the Prime Hub.

The freesolo training agent emits a single `environment.py`, while `prime env push` requires an
environment directory with a `pyproject.toml`. `slm env push` bridges that: pointed at a `.py`
module it wraps it in a Prime-compatible package, pointed at a real env dir it pushes as-is, and
it climbs past version conflicts on re-publish.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoslm.cli import main as cli


class _Proc:
    def __init__(self, rc: int, out: str = "", err: str = "") -> None:
        self.returncode = rc
        self.stdout = out
        self.stderr = err


def _push_dir(cmd: list[str]) -> Path:
    # base command is: prime env push --plain --path <dir> [--name ...] [--visibility ...]
    i = cmd.index("--plain") + 1
    if i < len(cmd) and cmd[i] == "--path":
        i += 1
    return Path(cmd[i])


def _fake_prime(*, slug="acme/env", fail_times=0, conflict=False):
    calls: list[list[str]] = []
    state = {"left": fail_times}

    def run(cmd, capture_output=True, text=True, env=None):
        calls.append(list(cmd))
        if state["left"] > 0:
            state["left"] -= 1
            return _Proc(1, err=("version already exists" if conflict else "boom"))
        owner, name = slug.split("/", 1)
        prime_dir = _push_dir(cmd) / ".prime"
        prime_dir.mkdir(parents=True, exist_ok=True)
        (prime_dir / ".env-metadata.json").write_text(json.dumps({"owner": owner, "name": name}))
        return _Proc(0, out=f"Successfully pushed {slug}\n")

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_push_single_py_module_is_packaged(monkeypatch, tmp_path, capsys):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    captured: dict = {}

    fake = _fake_prime(slug="acme/environment")

    def run(cmd, capture_output=True, text=True, env=None):
        captured["pkg"] = _push_dir(cmd)
        # The packaged dir must hold a pyproject + an importable module before prime sees it.
        assert (captured["pkg"] / "pyproject.toml").is_file()
        modules = [p for p in captured["pkg"].iterdir() if p.is_dir()]
        assert modules
        assert (modules[0] / "__init__.py").is_file()
        return fake(cmd, capture_output=capture_output, text=text)

    monkeypatch.setattr("subprocess.run", run)
    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file), name=None, visibility=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert "published acme/environment" in out
    # First push of a fresh package does not force a version bump.
    assert "--auto-bump" not in fake.calls[0]


def test_push_dir_with_pyproject_is_passthrough(monkeypatch, tmp_path):
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text('[project]\nname = "my-env"\nversion = "0.1.0"\n')
    (env_dir / "my_env.py").write_text("def load_environment(**k):\n    return None\n")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    fake = _fake_prime(slug="acme/my-env")
    monkeypatch.setattr("subprocess.run", fake)

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_dir), name=None, visibility=None))
    assert rc == 0
    # Pushed the directory as-is (no temp packaging dir).
    assert _push_dir(fake.calls[0]) == env_dir


def test_push_requires_prime_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("shutil.which", lambda name: None)
    rc = cli.cmd_env_push(argparse.Namespace(path=str(tmp_path), name=None, visibility=None))
    assert rc == 1
    assert "`prime` CLI is required" in capsys.readouterr().out


def test_push_climbs_past_version_conflict(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    fake = _fake_prime(slug="acme/environment", fail_times=1, conflict=True)
    monkeypatch.setattr("subprocess.run", fake)

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file), name=None, visibility=None))
    assert rc == 0
    assert len(fake.calls) == 2
    assert "--auto-bump" in fake.calls[1]


def test_push_nonexistent_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    rc = cli.cmd_env_push(
        argparse.Namespace(path=str(tmp_path / "nope.py"), name=None, visibility=None)
    )
    assert rc == 1
    assert "no such path" in capsys.readouterr().err
