"""`slm env push` packages a bare verifiers module and publishes it to the Prime Hub.

The freesolo training agent emits a single `environment.py`, while `prime env push` requires an
environment directory with a `pyproject.toml`. `slm env push` bridges that: pointed at a `.py`
module it wraps it in a Prime-compatible package, pointed at a real env dir it pushes as-is, and
it climbs past version conflicts on re-publish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flash.cli import main as cli


class _Proc:
    def __init__(self, rc: int, out: str = "", err: str = "") -> None:
        self.returncode = rc
        self.stdout = out
        self.stderr = err


def _push_dir(cmd: list[str]) -> Path:
    # base command is: prime env push --plain --path <dir> --visibility PRIVATE [--auto-bump]
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
    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "published acme/environment" in out
    # First push of a fresh package does not force a version bump.
    assert "--auto-bump" not in fake.calls[0]
    # Published environments are always private.
    assert fake.calls[0][fake.calls[0].index("--visibility") + 1] == "PRIVATE"


def test_push_dir_with_pyproject_is_passthrough(monkeypatch, tmp_path):
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text('[project]\nname = "my-env"\nversion = "0.1.0"\n')
    (env_dir / "my_env.py").write_text("def load_environment(**k):\n    return None\n")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    fake = _fake_prime(slug="acme/my-env")
    monkeypatch.setattr("subprocess.run", fake)

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_dir)))
    assert rc == 0
    # Pushed the directory as-is (no temp packaging dir).
    assert _push_dir(fake.calls[0]) == env_dir
    # A first publish does not force --auto-bump (the conflict retry enables it only on collision).
    assert "--auto-bump" not in fake.calls[0]


def test_push_single_py_ships_sibling_datasets(monkeypatch, tmp_path):
    # A committed `datasets/` sibling of the env file must ship inside the package so a
    # `__file__`-relative data read resolves on the worker.
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text('{"x": 1}\n')
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    captured: dict = {}
    fake = _fake_prime(slug="acme/environment")

    def run(cmd, capture_output=True, text=True, env=None):
        pkg = _push_dir(cmd)
        modules = [p for p in pkg.iterdir() if p.is_dir() and p.name != ".prime"]
        captured["data_files"] = sorted(
            str(p.relative_to(pkg)) for p in (modules[0] / "datasets").glob("*")
        )
        return fake(cmd, capture_output=capture_output, text=text)

    monkeypatch.setattr("subprocess.run", run)
    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    # The data landed under <module>/datasets/ (ships with the package dir).
    assert captured["data_files"] == ["environment/datasets/train.jsonl"]


def test_push_single_py_uses_sibling_config_id_name(monkeypatch, tmp_path):
    # A bare environment.py with a sibling flash_grpo.toml whose [environment] id is
    # "owner/myenv" must re-publish to that SAME env: prime is called with `--name myenv`
    # (the id's name part), so an edited env mints a new version of the existing env.
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "flash_grpo.toml").write_text(
        'model = "m"\nalgorithm = "grpo"\n[environment]\nid = "owner/myenv"\n'
    )
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    fake = _fake_prime(slug="owner/myenv")
    monkeypatch.setattr("subprocess.run", fake)

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    cmd = fake.calls[0]
    assert "--name" in cmd
    assert cmd[cmd.index("--name") + 1] == "myenv"
    # Re-publish of an EXISTING env auto-bumps from the first attempt (not is_new), so it
    # doesn't restart at 0.1.0 and climb through version conflicts.
    assert "--auto-bump" in cmd


def test_push_single_py_no_sibling_config_uses_file_stem(monkeypatch, tmp_path):
    # Without a sibling autoslm.toml/id, the published name falls back to the file stem.
    env_file = tmp_path / "my_task.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    fake = _fake_prime(slug="acme/my-task")
    monkeypatch.setattr("subprocess.run", fake)

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    cmd = fake.calls[0]
    assert cmd[cmd.index("--name") + 1] == "my-task"
    # A brand-new env (no sibling id) does NOT auto-bump on the first attempt.
    assert "--auto-bump" not in cmd


def test_push_requires_prime_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("shutil.which", lambda name: None)
    rc = cli.cmd_env_push(argparse.Namespace(path=str(tmp_path)))
    assert rc == 1
    assert "`prime` CLI is required" in capsys.readouterr().out


def test_push_climbs_past_version_conflict(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    fake = _fake_prime(slug="acme/environment", fail_times=1, conflict=True)
    monkeypatch.setattr("subprocess.run", fake)

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    assert len(fake.calls) == 2
    assert "--auto-bump" in fake.calls[1]


def test_push_nonexistent_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/prime")
    rc = cli.cmd_env_push(argparse.Namespace(path=str(tmp_path / "nope.py")))
    assert rc == 1
    assert "no such path" in capsys.readouterr().err
