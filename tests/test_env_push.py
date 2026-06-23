"""`flash env push` packages local Freesolo envs and uploads them through the server."""

from __future__ import annotations

import argparse
import base64
import io
import tarfile

from flash.cli import main as cli


def _fake_client(
    capture: dict, *, slug: str = "github:freesolo-co/envs@main:acme/freesolo/environment.py"
):
    """A stand-in ApiClient that records the publish_env call and returns a GitHub ref."""

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
    # The uploaded package holds a pyproject + the canonical Freesolo environment module.
    assert "pyproject.toml" in files
    assert "freesolo/__init__.py" in files
    assert "freesolo/environment.py" in files
    assert cap["name"] == "environment"
    assert cap["is_new"] is True
    assert "published freesolo-co/u-environment" in capsys.readouterr().out


def test_push_dir_with_pyproject_is_passthrough(monkeypatch, tmp_path):
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text('[project]\nname = "my-env"\nversion = "0.1.0"\n')
    (env_dir / "freesolo").mkdir()
    (env_dir / "freesolo" / "__init__.py").write_text("")
    (env_dir / "freesolo" / "environment.py").write_text(
        "def load_environment(**k):\n    return None\n"
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_dir)))
    assert rc == 0
    files = _members(cap["package_b64"])
    assert "pyproject.toml" in files
    assert "freesolo/environment.py" in files  # uploaded as-is
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
    assert "freesolo/datasets/train.jsonl" in _members(cap["package_b64"])


def test_push_single_py_uses_sibling_config_id_name(monkeypatch, tmp_path):
    # A bare environment.py with a sibling flash_grpo.toml whose [environment] id points at
    # a training-style <namespace>/<project>/<publish-id>/freesolo/environment.py ref
    # re-publishes to that SAME logical project.
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "flash_grpo.toml").write_text(
        'model = "m"\nalgorithm = "grpo"\n[environment]\n'
        'id = "github:owner/repo@main:user/myenv/12345678-1234-4321-abcd-123456789abc/freesolo/environment.py"\n'
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    assert cap["name"] == "myenv"
    assert cap["is_new"] is False


def test_push_single_py_uses_legacy_environments_config_id_name(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "flash_grpo.toml").write_text(
        'model = "m"\nalgorithm = "grpo"\n[environment]\n'
        'id = "github:owner/repo@main:environments/user/myenv/freesolo/environment.py"\n'
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    rc = cli.cmd_env_push(argparse.Namespace(path=str(env_file)))
    assert rc == 0
    assert cap["name"] == "myenv"
    assert cap["is_new"] is False


def test_push_sibling_config_id_with_dot_yields_valid_module(monkeypatch, tmp_path):
    # A sibling config env path may include dots. The packaged Python layout stays canonical
    # regardless of the logical environment name.
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "flash_grpo.toml").write_text(
        'model = "m"\nalgorithm = "grpo"\n[environment]\n'
        'id = "github:owner/repo@main:user/my.weird.env/12345678-1234-4321-abcd-123456789abc/freesolo/environment.py"\n'
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file))) == 0
    names = set(_members(cap["package_b64"]))
    assert cap["name"] == "my.weird.env"
    assert "freesolo/environment.py" in names
    assert not any("my.weird" in n for n in names)


def test_push_sibling_config_repo_root_refs_do_not_override_name(monkeypatch, tmp_path):
    for env_id in ("github:owner/repo", "github:owner/repo@main", "https://github.com/owner/repo"):
        env_file = tmp_path / "my_task.py"
        env_file.write_text("def load_environment(**k):\n    return None\n")
        (tmp_path / "flash_grpo.toml").write_text(
            f'model = "m"\nalgorithm = "grpo"\n[environment]\nid = "{env_id}"\n'
        )
        cap: dict = {}
        monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

        assert cli.cmd_env_push(argparse.Namespace(path=str(env_file))) == 0
        assert cap["name"] == "my-task"
        assert cap["is_new"] is True


def test_push_sibling_config_github_url_path_derives_name(monkeypatch, tmp_path):
    env_file = tmp_path / "environment.py"
    env_file.write_text("def load_environment(**k):\n    return None\n")
    (tmp_path / "flash_grpo.toml").write_text(
        'model = "m"\nalgorithm = "grpo"\n[environment]\n'
        'id = "https://github.com/owner/repo/blob/main/envs/urlenv/freesolo/environment.py"\n'
    )
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file))) == 0
    assert cap["name"] == "urlenv"
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


def test_push_needs_no_local_github_credentials(monkeypatch, tmp_path):
    # The client does not need GitHub credentials — publishing is server-side.
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
            raise ClientError("not logged in — run `flash login`")

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())
    assert cli.cmd_env_push(argparse.Namespace(path=str(env_file))) == 1
    assert "not logged in" in capsys.readouterr().err


def test_push_nonexistent_path(tmp_path, capsys):
    rc = cli.cmd_env_push(argparse.Namespace(path=str(tmp_path / "nope.py")))
    assert rc == 1
    assert "no such path" in capsys.readouterr().err


def test_push_excludes_metadata_and_cache_dirs(monkeypatch, tmp_path):
    # Tool metadata and caches must NOT be shipped in the upload: they aren't env source and
    # bloat the package.
    env_dir = tmp_path / "my-env"
    env_dir.mkdir()
    (env_dir / "pyproject.toml").write_text('[project]\nname = "my-env"\nversion = "0.1.0"\n')
    (env_dir / "freesolo").mkdir()
    (env_dir / "freesolo" / "__init__.py").write_text("")
    (env_dir / "freesolo" / "environment.py").write_text(
        "def load_environment(**k):\n    return None\n"
    )
    (env_dir / ".prime").mkdir()
    (env_dir / ".prime" / ".env-metadata.json").write_text('{"owner": "someone-else"}')
    (env_dir / "__pycache__").mkdir()
    (env_dir / "__pycache__" / "x.pyc").write_text("junk")
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))

    assert cli.cmd_env_push(argparse.Namespace(path=str(env_dir))) == 0
    names = set(_members(cap["package_b64"]))
    assert "pyproject.toml" in names
    assert "freesolo/environment.py" in names
    assert not any(n.startswith(".prime") for n in names)  # metadata stripped
    assert not any("__pycache__" in n for n in names)
