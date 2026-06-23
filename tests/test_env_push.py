"""`flash env push` packages local Freesolo envs and uploads them through the server."""

from __future__ import annotations

import argparse
import base64
import io
import tarfile

from flash.cli import main as cli


def _fake_client(capture: dict, *, slug: str = "acme/environment"):
    """A stand-in ApiClient that records the publish_env call and returns an env id."""

    class _C:
        def publish_env(self, *, name, package_b64):
            capture.update(name=name, package_b64=package_b64)
            return {"id": slug}

    return lambda: _C()


def _members(package_b64: str) -> dict[str, str]:
    raw = base64.b64decode(package_b64)
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile():
                out[m.name] = tar.extractfile(m).read().decode()
    return out


def _args(path, *, name: str = "my-env"):
    return argparse.Namespace(path=str(path), name=name)


def test_push_single_py_module_is_packaged(monkeypatch, tmp_path, capsys):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr(
        "flash.client.client_from_config", _fake_client(cap, slug="freesolo-co/u-environment")
    )

    rc = cli.cmd_env_push(_args(env_file, name="math-env"))
    assert rc == 0
    files = _members(cap["package_b64"])
    assert "environment.py" in files
    assert "pyproject.toml" not in files
    assert not any(name.startswith("freesolo/") for name in files)
    assert cap["name"] == "math-env"
    assert "published freesolo-co/u-environment" in capsys.readouterr().out


def test_push_dir_with_pyproject_uses_explicit_name(monkeypatch, tmp_path):
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text('[project]\nname = "my-env"\nversion = "0.1.0"\n')
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "source").mkdir()
    (env_dir / "source" / "index.ts").write_text("console.log('agent workspace')\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    rc = cli.cmd_env_push(_args(env_dir, name="explicit-env"))
    assert rc == 0
    files = _members(cap["package_b64"])
    assert "environment.py" in files
    assert "pyproject.toml" not in files
    assert not any(name.startswith("source/") for name in files)
    assert cap["name"] == "explicit-env"


def test_push_dir_prefers_environment_py_and_ships_helpers(monkeypatch, tmp_path):
    env_dir = tmp_path / "math"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text(
        "import helper\n\ndef load_environment(**k):\n    return helper.build()\n"
    )
    (env_dir / "helper.py").write_text("def build():\n    return None\n")
    (env_dir / "datasets").mkdir()
    (env_dir / "datasets" / "train.jsonl").write_text('{"input":"2+2","output":"4"}\n')
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir, name="math")) == 0
    files = _members(cap["package_b64"])
    assert "environment.py" in files
    assert "helper.py" in files
    assert "datasets/train.jsonl" in files
    assert cap["name"] == "math"


def test_push_single_py_ships_sibling_datasets(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "train.jsonl").write_text('{"x": 1}\n')
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    assert "datasets/train.jsonl" in _members(cap["package_b64"])


def test_push_single_py_ships_sibling_database(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "state.sqlite").write_text("sqlite bytes")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    assert "state.sqlite" in _members(cap["package_b64"])


def test_push_single_py_ships_sibling_helper_modules(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("import helper\n\ndef load_environment(**k):\n    return None\n")
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    assert "helper.py" in _members(cap["package_b64"])


def test_push_requires_explicit_name(tmp_path, capsys):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file))) == 1
    assert "--name" in capsys.readouterr().err


def test_push_sibling_config_does_not_override_explicit_name(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "grpo.toml").write_text(
        'model = "m"\nalgorithm = "grpo"\n[environment]\nid = "user/old-name"\n'
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file, name="new-name")) == 0
    names = set(_members(cap["package_b64"]))
    assert cap["name"] == "new-name"
    assert "environment.py" in names


def test_push_needs_no_local_github_credentials(monkeypatch, tmp_path):
    # The client does not need GitHub credentials; publishing is server-side.
    monkeypatch.setattr("shutil.which", lambda name: None)
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_file)) == 0
    assert cap["package_b64"]


def test_push_reports_server_error(monkeypatch, tmp_path, capsys):
    from flash.client import ClientError

    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")

    class _C:
        def publish_env(self, **k):
            raise ClientError("not logged in — run `flash login`")

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())
    assert cli.cmd_env_push(_args(env_file)) == 1
    assert "not logged in" in capsys.readouterr().err


def test_push_nonexistent_path(tmp_path, capsys):
    rc = cli.cmd_env_push(_args(tmp_path / "nope.py"))
    assert rc == 1
    assert "no such path" in capsys.readouterr().err


def test_push_excludes_metadata_and_cache_dirs(monkeypatch, tmp_path):
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text('[project]\nname = "my-env"\nversion = "0.1.0"\n')
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / ".prime").mkdir()
    (env_dir / ".prime" / ".env-metadata.json").write_text('{"owner": "someone-else"}')
    (env_dir / "__pycache__").mkdir()
    (env_dir / "__pycache__" / "x.pyc").write_text("junk")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(_args(env_dir, name="clean-env")) == 0
    names = set(_members(cap["package_b64"]))
    assert "environment.py" in names
    assert not any(n.startswith(".prime") for n in names)
    assert not any("__pycache__" in n for n in names)
