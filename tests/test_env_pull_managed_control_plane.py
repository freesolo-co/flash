"""Managed `flash env pull` through the Flash control-plane package endpoint."""

from __future__ import annotations

import io
import tarfile
from argparse import Namespace

import pytest

from flash.cli.envpush import cmd_env_pull
from flash.envs import loader as adapter
from flash.envs.pull import (
    download_environment_file_from_archive,
    pull_environment_package_from_archive,
)


def _package_tarball(entries: dict[str, bytes] | None = None) -> bytes:
    entries = entries or {
        "environment.py": b"# env\n",
        "datasets/train.jsonl": b'{"a":1}\n',
        "README.md": b"# Read me\n",
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _args(**kw) -> Namespace:
    base = {"env_id": "david-freesolo-co/stuff", "path": None, "output": None, "force": False}
    base.update(kw)
    return Namespace(**base)


def _client_factory(package: bytes, seen: dict[str, str] | None = None):
    class _Client:
        def download_env_package(self, env_id: str) -> bytes:
            if seen is not None:
                seen["env_id"] = env_id
            return package

    return lambda: _Client()


def test_pull_environment_package_from_archive_copies_flat_package(tmp_path):
    dest = tmp_path / "out"

    out = pull_environment_package_from_archive(_package_tarball(), dest)

    assert out == dest
    assert (dest / "environment.py").read_text() == "# env\n"
    assert (dest / "datasets" / "train.jsonl").read_bytes() == b'{"a":1}\n'


def test_download_environment_file_from_archive_reads_one_file():
    data = download_environment_file_from_archive(_package_tarball(), "datasets/train.jsonl")

    assert data == b'{"a":1}\n'


def test_package_archive_rejects_unsafe_path(tmp_path):
    with pytest.raises(RuntimeError, match="unsafe path"):
        pull_environment_package_from_archive(
            _package_tarball({"../secret.txt": b"nope\n", "environment.py": b"# env\n"}),
            tmp_path / "out",
        )


def test_cmd_env_pull_managed_whole_env_uses_authenticated_package(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    seen: dict[str, str] = {}

    def fail_github(ref):
        raise AssertionError("managed env pull should use the Flash control plane")

    monkeypatch.setattr(
        "flash.client.client_from_config", _client_factory(_package_tarball(), seen)
    )
    monkeypatch.setattr(adapter, "_download_github_tarball", fail_github)
    dest = tmp_path / "stuff"

    rc = cmd_env_pull(_args(output=str(dest)))

    assert rc == 0
    assert seen == {"env_id": "david-freesolo-co/stuff"}
    assert (dest / "environment.py").is_file()
    assert (dest / "datasets" / "train.jsonl").is_file()


def test_cmd_env_pull_managed_whole_env_strips_slug_before_request(monkeypatch, tmp_path):
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        "flash.client.client_from_config", _client_factory(_package_tarball(), seen)
    )

    rc = cmd_env_pull(_args(env_id="  david-freesolo-co/stuff  ", output=str(tmp_path / "stuff")))

    assert rc == 0
    assert seen == {"env_id": "david-freesolo-co/stuff"}


def test_cmd_env_pull_managed_rejects_mixed_case_before_request(monkeypatch, tmp_path):
    calls = {"n": 0}

    class _Client:
        def download_env_package(self, env_id: str) -> bytes:
            calls["n"] += 1
            return _package_tarball()

    monkeypatch.setattr("flash.client.client_from_config", lambda: _Client())

    rc = cmd_env_pull(_args(env_id="David-Freesolo-Co/stuff", output=str(tmp_path / "stuff")))

    assert rc == 1
    assert calls["n"] == 0


def test_cmd_env_pull_managed_refuses_existing_dir_before_download(monkeypatch, tmp_path):
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        "flash.client.client_from_config", _client_factory(_package_tarball(), seen)
    )
    dest = tmp_path / "stuff"
    dest.mkdir()
    (dest / "keep.txt").write_text("existing")

    rc = cmd_env_pull(_args(output=str(dest)))

    assert rc == 1
    assert seen == {}
    assert (dest / "keep.txt").read_text() == "existing"


def test_cmd_env_pull_managed_single_file_uses_authenticated_package(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        "flash.client.client_from_config", _client_factory(_package_tarball(), seen)
    )
    out = tmp_path / "train.jsonl"

    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(out)))

    assert rc == 0
    assert seen == {"env_id": "david-freesolo-co/stuff"}
    assert out.read_bytes() == b'{"a":1}\n'


def test_cmd_env_pull_managed_auth_error(monkeypatch, tmp_path, capsys):
    from flash.client import ClientError

    def fail_client():
        raise ClientError("not logged in")

    monkeypatch.setattr("flash.client.client_from_config", fail_client)

    rc = cmd_env_pull(_args(output=str(tmp_path / "out")))

    assert rc == 1
    assert "not logged in" in capsys.readouterr().err
