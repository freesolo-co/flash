"""`flash env push` packages a local verifiers env and uploads it to the MANAGED Environments Hub.

The control plane publishes it under FreeSolo's Prime account, so the user needs no Prime
account or local `prime` CLI. The freesolo training agent emits a single `environment.py` while
the Hub requires an environment directory with a `pyproject.toml`; `flash env push` bridges that:
pointed at a `.py` module it wraps it in a Prime-compatible package, pointed at a real env dir it
uploads as-is, and the server climbs past version conflicts on re-publish. These tests cover the
client side — packaging + upload. The server-side publish (prime, per-identity namespacing,
conflict-climbing) lives in test_env_publish.py.
"""

from __future__ import annotations

import argparse
import base64
import io
import tarfile

from flash.cli import main as cli


def _fake_client(capture: dict, *, slug: str = "freesolo-co/acme-env"):
    """A stand-in ApiClient that records the publish_env call and returns a slug."""

    class _C:
        def publish_env(self, *, name, is_new, package_b64):
            capture.update(name=name, is_new=is_new, package_b64=package_b64)
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


def test_push_single_py_module_is_packaged(monkeypatch, tmp_path, capsys):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr(
        "flash.client.client_from_config", _fake_client(cap, slug="freesolo-co/u-environment")
    )

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    files = _members(cap["package_b64"])
    # The uploaded package holds a pyproject + an importable module.
    assert "pyproject.toml" in files
    assert "environment/__init__.py" in files
    assert cap["name"] == "environment"
    assert cap["is_new"] is True
    assert "published freesolo-co/u-environment" in capsys.readouterr().out


def test_push_dir_with_pyproject_is_passthrough(monkeypatch, tmp_path):
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text('[project]\nname = "my-env"\nversion = "0.1.0"\n')
    (env_dir / "my_env.py").write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_dir)))
    assert rc == 0
    files = _members(cap["package_b64"])
    assert "pyproject.toml" in files
    assert "my_env.py" in files  # uploaded as-is
    assert cap["name"] == "my-env"  # from the pyproject [project] name
    assert cap["is_new"] is True


def test_push_single_py_ships_sibling_datasets(monkeypatch, tmp_path):
    # A committed `datasets/` sibling must ship inside the package so a `__file__`-relative read
    # resolves on the worker.
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "train.jsonl").write_text('{"x": 1}\n')
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    assert "environment/datasets/train.jsonl" in _members(cap["package_b64"])


def test_push_single_py_uses_sibling_config_id_name(monkeypatch, tmp_path):
    # A bare environment.py with a sibling flash_grpo.toml whose [environment] id is "owner/myenv"
    # re-publishes to that SAME env: name=myenv, is_new=False (the server then auto-bumps).
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "flash_grpo.toml").write_text(
        'model = "m"\nalgorithm = "grpo"\n[environment]\nid = "owner/myenv"\n'
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    assert cap["name"] == "myenv"
    assert cap["is_new"] is False


def test_push_single_py_no_sibling_config_uses_file_stem(monkeypatch, tmp_path):
    env_file = tmp_path / "my_task.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    assert cap["name"] == "my-task"
    assert cap["is_new"] is True


def test_push_needs_no_local_prime(monkeypatch, tmp_path):
    # The client no longer requires the `prime` CLI — publishing is server-side.
    monkeypatch.setattr("shutil.which", lambda name: None)
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file))) == 0
    assert cap["package_b64"]


def test_push_reports_server_error(monkeypatch, tmp_path, capsys):
    from flash.client import ClientError

    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")

    class _C:
        def publish_env(self, **k):
            raise ClientError("not logged in — run `slm login`")

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())
    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file))) == 1
    assert "not logged in" in capsys.readouterr().err


def test_push_nonexistent_path(tmp_path, capsys):
    rc = cli.cmd_env_push(argparse.Namespace(path=str(tmp_path / "nope.py")))
    assert rc == 1
    assert "no such path" in capsys.readouterr().err


def test_push_excludes_prime_and_cache_dirs(monkeypatch, tmp_path):
    # A `.prime/` dir (Prime CLI metadata from a prior local push) and tool caches must NOT be
    # shipped in the upload: they aren't env source, bloat the package, and stale `.prime/`
    # metadata could confuse server-side slug discovery.
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text('[project]\nname = "my-env"\nversion = "0.1.0"\n')
    (env_dir / "my_env.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / ".prime").mkdir()
    (env_dir / ".prime" / ".env-metadata.json").write_text('{"owner": "someone-else"}')
    (env_dir / "__pycache__").mkdir()
    (env_dir / "__pycache__" / "x.pyc").write_text("junk")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(argparse.Namespace(path=str(env_dir))) == 0
    names = set(_members(cap["package_b64"]))
    assert "pyproject.toml" in names
    assert "my_env.py" in names
    assert not any(n.startswith(".prime") for n in names)  # metadata stripped
    assert not any("__pycache__" in n for n in names)
