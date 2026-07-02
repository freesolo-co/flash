"""Tests for `flash env pull`."""

from __future__ import annotations

import io
import json
import stat
import tarfile
import urllib.parse
import urllib.request
from argparse import Namespace
from pathlib import Path

import pytest

from flash.cli.envpush import cmd_env_pull
from flash.envs import loader as adapter
from flash.envs.pull import environment_local_dirname

_TOP = "freesolo-co-environment-hub-deadbeef"


def _tarball(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(f"{_TOP}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _top_level_file_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"not a directory\n"
        info = tarfile.TarInfo(_TOP)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _package_tarball(entries: dict[str, bytes]) -> bytes:
    """A flat managed-environment package tarball (files at the archive root)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _FakeClient:
    def __init__(self, package: bytes):
        self._package = package

    def download_env_package(self, env_id: str) -> bytes:
        return self._package


def _patch_client(monkeypatch, package: bytes) -> None:
    monkeypatch.setattr("flash.client.client_from_config", lambda: _FakeClient(package))


def _margs(**kw) -> Namespace:
    base = {
        "env_id": "david-freesolo-co/stuff",
        "path": None,
        "output": None,
        "force": False,
    }
    base.update(kw)
    return Namespace(**base)


# --- managed-only `flash env pull` CLI -------------------------------------------------


def test_cmd_env_pull_single_file(monkeypatch, tmp_path):
    _patch_client(
        monkeypatch,
        _package_tarball(
            {"environment.py": b"# env\n", "datasets/train.jsonl": b"line1\nline2\n"}
        ),
    )
    out = tmp_path / "train.jsonl"

    rc = cmd_env_pull(_margs(path="datasets/train.jsonl", output=str(out)))

    assert rc == 0
    assert out.read_bytes() == b"line1\nline2\n"


def test_cmd_env_pull_rejects_non_managed_ref(capsys):
    rc = cmd_env_pull(_margs(env_id="github:owner/repo@main:env/environment.py"))

    assert rc == 1
    assert "managed Freesolo hub slug" in capsys.readouterr().err


def test_cmd_env_pull_rejects_noncanonical_slug(capsys):
    rc = cmd_env_pull(_margs(env_id="David-Freesolo-Co/Stuff"))

    assert rc == 1
    assert "lowercase" in capsys.readouterr().err


def test_cmd_env_pull_refuses_existing_file_without_force(monkeypatch, tmp_path):
    _patch_client(
        monkeypatch,
        _package_tarball({"environment.py": b"# env\n", "datasets/train.jsonl": b"x"}),
    )
    out = tmp_path / "train.jsonl"
    out.write_text("keep")

    rc = cmd_env_pull(_margs(path="datasets/train.jsonl", output=str(out)))

    assert rc == 1
    assert out.read_text() == "keep"


def test_cmd_env_pull_single_file_replaces_symlink_with_force(monkeypatch, tmp_path):
    _patch_client(
        monkeypatch,
        _package_tarball({"environment.py": b"# env\n", "datasets/train.jsonl": b"new\n"}),
    )
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("keep me")
    out = tmp_path / "train.jsonl"
    out.symlink_to(target)

    rc = cmd_env_pull(_margs(path="datasets/train.jsonl", output=str(out), force=True))

    assert rc == 0
    assert not out.is_symlink()
    assert out.read_bytes() == b"new\n"
    assert target.read_text() == "keep me"


def test_cmd_env_pull_single_file_refuses_real_dir(monkeypatch, tmp_path):
    _patch_client(
        monkeypatch,
        _package_tarball({"environment.py": b"# env\n", "datasets/train.jsonl": b"x"}),
    )
    out = tmp_path / "adir"
    out.mkdir()

    rc = cmd_env_pull(_margs(path="datasets/train.jsonl", output=str(out), force=True))

    assert rc == 1


def test_cmd_env_pull_whole_env(monkeypatch, tmp_path):
    _patch_client(
        monkeypatch,
        _package_tarball(
            {"environment.py": b"# env\n", "datasets/train.jsonl": b'{"a":1}\n' * 1000}
        ),
    )
    dest = tmp_path / "stuff"

    rc = cmd_env_pull(_margs(output=str(dest)))

    assert rc == 0
    assert (dest / "environment.py").is_file()
    assert (dest / "datasets" / "train.jsonl").is_file()


def test_cmd_env_pull_whole_env_refuses_nonempty_dest_without_force(monkeypatch, tmp_path):
    _patch_client(
        monkeypatch,
        _package_tarball({"environment.py": b"# env\n"}),
    )
    dest = tmp_path / "stuff"
    dest.mkdir()
    (dest / "keep.txt").write_text("existing")

    rc = cmd_env_pull(_margs(output=str(dest)))

    assert rc == 1
    assert (dest / "keep.txt").read_text() == "existing"


def test_environment_local_dirname():
    assert environment_local_dirname("david-freesolo-co/stuff") == "stuff"
    with pytest.raises(ValueError, match="managed Freesolo environment slug"):
        environment_local_dirname("github:freesolo-co/environment-hub@main:a/b/environment.py")


# --- loader GitHub env-resolution machinery (used by training/serving) -----------------


def test_safe_extract_archive_bounds_total_members_scanned(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_MAX_ARCHIVE_SCAN_MEMBERS", 4)
    entries = {f"unrelated/f{i}.txt": b"x" for i in range(20)}

    with pytest.raises(RuntimeError, match="too many entries to scan"):
        adapter._safe_extract_archive(_tarball(entries), tmp_path, subdir="wanted")


def test_safe_extract_archive_rejects_top_level_file(tmp_path):
    with pytest.raises(RuntimeError, match="unexpected layout"):
        adapter._safe_extract_archive(_top_level_file_tarball(), tmp_path)


def test_download_github_tarball_uses_whole_repo_ceiling(monkeypatch):
    assert adapter._MAX_TARBALL_BYTES > adapter._MAX_ARCHIVE_BYTES
    big = b"x" * (adapter._MAX_ARCHIVE_BYTES + 10)

    def fake_urlopen(req, timeout=None, max_bytes=None, out=None):
        assert max_bytes == adapter._MAX_TARBALL_BYTES
        out.write(big)
        return b""

    monkeypatch.setattr(adapter, "_urlopen", fake_urlopen)
    monkeypatch.setattr(adapter, "_github_token", lambda: None)
    ref = adapter._parse_github_environment_ref(
        adapter.managed_slug_to_github_ref("david-freesolo-co/stuff")
    )

    tarball = adapter._download_github_tarball(ref)
    try:
        assert tarball.read_bytes() == big
    finally:
        tarball.unlink()


def test_resolve_managed_hub_env_downloads_only_requested_package(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)

    def fail_tarball(ref):
        raise AssertionError("managed hub refs should not download the whole repo tarball")

    monkeypatch.setattr(adapter, "_download_github_tarball", fail_tarball)

    trees = {
        ("a" * 40, False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "david-freesolo-co", "sha": "namespace-sha"}],
        },
        ("namespace-sha", False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "stuff", "sha": "package-sha"}],
        },
        ("package-sha", True): {
            "truncated": False,
            "tree": [
                {
                    "type": "blob",
                    "path": "environment.py",
                    "mode": "100644",
                    "size": len(b"# env\n"),
                },
                {"type": "tree", "path": "datasets", "sha": "datasets-sha"},
                {
                    "type": "blob",
                    "path": "datasets/train.jsonl",
                    "mode": "100644",
                    "size": len(b'{"a":1}\n'),
                },
                {"type": "tree", "path": "bin", "sha": "bin-sha"},
                {
                    "type": "blob",
                    "path": "bin/run-helper",
                    "mode": "100755",
                    "size": len(b"#!/bin/sh\n"),
                },
            ],
        },
    }
    files = {
        "david-freesolo-co/stuff/environment.py": b"# env\n",
        "david-freesolo-co/stuff/datasets/train.jsonl": b'{"a":1}\n',
        "david-freesolo-co/stuff/bin/run-helper": b"#!/bin/sh\n",
    }
    seen_urls: list[str] = []

    def fake_urlopen(req, timeout=None, max_bytes=None, out=None):
        seen_urls.append(req.full_url)
        accept = req.headers.get("Accept")
        if "/git/trees/" in req.full_url:
            assert accept == "application/vnd.github+json"
            treeish = urllib.parse.unquote(req.full_url.split("/git/trees/", 1)[1].split("?", 1)[0])
            return json.dumps(trees[(treeish, "recursive=1" in req.full_url)]).encode()
        path = urllib.parse.unquote(req.full_url.split("/contents/", 1)[1].split("?", 1)[0])
        if accept == "application/vnd.github+json":
            raise AssertionError("managed hub directory listings should use the Git trees API")
        if accept == "application/vnd.github.raw":
            payload = files[path]
            if out is not None:
                out.write(payload)
                return b""
            return payload
        raise AssertionError(f"unexpected Accept header: {accept!r}")

    monkeypatch.setattr(adapter, "_urlopen", fake_urlopen)

    env_file = Path(adapter._resolve_environment_reference("david-freesolo-co/stuff"))

    assert env_file.read_bytes() == b"# env\n"
    assert (env_file.parent / "datasets" / "train.jsonl").read_bytes() == b'{"a":1}\n'
    assert stat.S_IMODE((env_file.parent / "bin" / "run-helper").stat().st_mode) == 0o755
    assert not (env_file.parents[2] / "other-org").exists()
    assert all("other-org" not in url for url in seen_urls)


def test_explicit_environment_hub_github_ref_downloads_only_requested_package(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)

    def fail_tarball(ref):
        raise AssertionError("environment-hub refs should not download the whole repo tarball")

    monkeypatch.setattr(adapter, "_download_github_tarball", fail_tarball)
    trees = {
        ("a" * 40, False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "david-freesolo-co", "sha": "namespace-sha"}],
        },
        ("namespace-sha", False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "stuff", "sha": "package-sha"}],
        },
        ("package-sha", True): {
            "truncated": False,
            "tree": [
                {
                    "type": "blob",
                    "path": "environment.py",
                    "mode": "100644",
                    "size": len(b"# env\n"),
                },
                {
                    "type": "blob",
                    "path": "datasets/train.jsonl",
                    "mode": "100644",
                    "size": len(b'{"a":1}\n'),
                },
            ],
        },
    }
    files = {
        "david-freesolo-co/stuff/environment.py": b"# env\n",
        "david-freesolo-co/stuff/datasets/train.jsonl": b'{"a":1}\n',
    }

    def fake_urlopen(req, timeout=None, max_bytes=None, out=None):
        accept = req.headers.get("Accept")
        if "/git/trees/" in req.full_url:
            assert accept == "application/vnd.github+json"
            treeish = urllib.parse.unquote(req.full_url.split("/git/trees/", 1)[1].split("?", 1)[0])
            return json.dumps(trees[(treeish, "recursive=1" in req.full_url)]).encode()
        assert accept == "application/vnd.github.raw"
        path = urllib.parse.unquote(req.full_url.split("/contents/", 1)[1].split("?", 1)[0])
        payload = files[path]
        if out is not None:
            out.write(payload)
            return b""
        return payload

    monkeypatch.setattr(adapter, "_urlopen", fake_urlopen)

    env_file = Path(
        adapter._resolve_environment_reference(
            "github:freesolo-co/environment-hub@main:david-freesolo-co/stuff/environment.py"
        )
    )

    assert env_file.read_bytes() == b"# env\n"
    assert (env_file.parent / "datasets" / "train.jsonl").read_bytes() == b'{"a":1}\n'
    assert not (env_file.parents[2] / "shared").exists()


def test_environment_hub_github_ref_requires_package_path(monkeypatch):
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)

    with pytest.raises(ValueError, match="namespace/name"):
        adapter._resolve_environment_reference("github:freesolo-co/environment-hub@main")


def test_github_tree_url_encodes_treeish_path_segment():
    ref = adapter.GitHubEnvironmentRef(
        "freesolo-co",
        "environment-hub",
        "a" * 40,
        "david-freesolo-co/stuff/environment.py",
    )

    url = adapter._github_tree_url(ref, "a" * 40 + ":david-freesolo-co/stuff", recursive=True)

    assert url.endswith("a" * 40 + "%3Adavid-freesolo-co%2Fstuff?recursive=1")
    assert "/stuff" not in url.split("/git/trees/", 1)[1]


def test_download_github_directory_handles_large_tree_listing(monkeypatch, tmp_path):
    ref = adapter.GitHubEnvironmentRef(
        "freesolo-co",
        "environment-hub",
        "b" * 40,
        "david-freesolo-co/big/environment.py",
    )
    shard_count = 1001
    trees = {
        ("b" * 40, False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "david-freesolo-co", "sha": "namespace-sha"}],
        },
        ("namespace-sha", False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "big", "sha": "package-sha"}],
        },
        ("package-sha", True): {
            "truncated": False,
            "tree": [
                {
                    "type": "blob",
                    "path": "environment.py",
                    "mode": "100644",
                    "size": len(b"# env\n"),
                },
                *[
                    {
                        "type": "blob",
                        "path": f"shard-{idx}.jsonl",
                        "mode": "100644",
                        "size": 1,
                    }
                    for idx in range(shard_count)
                ],
            ],
        },
    }

    def fake_urlopen(req, timeout=None, max_bytes=None, out=None):
        accept = req.headers.get("Accept")
        if "/git/trees/" in req.full_url:
            assert accept == "application/vnd.github+json"
            treeish = urllib.parse.unquote(req.full_url.split("/git/trees/", 1)[1].split("?", 1)[0])
            return json.dumps(trees[(treeish, "recursive=1" in req.full_url)]).encode()
        assert accept == "application/vnd.github.raw"
        path = urllib.parse.unquote(req.full_url.split("/contents/", 1)[1].split("?", 1)[0])
        payload = b"# env\n" if path.endswith("/environment.py") else b"x"
        if out is not None:
            out.write(payload)
            return b""
        return payload

    monkeypatch.setattr(adapter, "_urlopen", fake_urlopen)

    repo_root = adapter._download_github_directory(ref, "david-freesolo-co/big", tmp_path)

    assert (repo_root / "david-freesolo-co/big/environment.py").read_bytes() == b"# env\n"
    assert (repo_root / "david-freesolo-co/big/shard-1000.jsonl").read_bytes() == b"x"


def test_download_github_directory_surfaces_tree_error_message(monkeypatch, tmp_path):
    ref = adapter.GitHubEnvironmentRef(
        "freesolo-co",
        "environment-hub",
        "c" * 40,
        "david-freesolo-co/missing/environment.py",
    )

    def fake_urlopen(req, timeout=None, max_bytes=None, out=None):
        return json.dumps({"message": "Not Found"}).encode()

    monkeypatch.setattr(adapter, "_urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Not Found"):
        adapter._download_github_directory(ref, "david-freesolo-co/missing", tmp_path)


def test_urlopen_streams_and_aborts_over_max_bytes(monkeypatch):
    served = {"n": 0}

    class _Resp:
        def __init__(self, data):
            self._buf = io.BytesIO(data)

        def read(self, size=-1):
            chunk = self._buf.read(size)
            served["n"] += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(adapter, "_DOWNLOAD_CHUNK_BYTES", 4)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(b"x" * 64))
    req = urllib.request.Request("https://api.github.com/x")

    with pytest.raises(RuntimeError, match="exceeded the maximum allowed size"):
        adapter._urlopen(req, max_bytes=16)
    assert served["n"] <= 16 + adapter._DOWNLOAD_CHUNK_BYTES


def test_urlopen_returns_bytes_when_capped(monkeypatch):
    class _Resp:
        def __init__(self, data):
            self._buf = io.BytesIO(data)

        def read(self, size=-1):
            return self._buf.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(b"ok"))
    req = urllib.request.Request("https://api.github.com/x")

    data = adapter._urlopen(req, max_bytes=16)

    assert isinstance(data, bytes)
    assert data == b"ok"


def test_urlopen_streams_capped_response_to_output(monkeypatch):
    class _Resp:
        def __init__(self, data):
            self._buf = io.BytesIO(data)

        def read(self, size=-1):
            return self._buf.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(b"ok"))
    req = urllib.request.Request("https://api.github.com/x")
    out = io.BytesIO()

    data = adapter._urlopen(req, max_bytes=16, out=out)

    assert data == b""
    assert out.getvalue() == b"ok"


def test_resolve_github_env_extracts_repo_level_siblings(monkeypatch, tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in {
            "repo-sha/envs/e/environment.py": b"# env\n",
            "repo-sha/envs/datasets/train.jsonl": b'{"a":1}\n',
        }.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: buf.getvalue())
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")

    env_file = adapter._resolve_github_environment_file(
        "github:owner/repo@main:envs/e/environment.py"
    )

    assert env_file.is_file()
    assert (env_file.parents[1] / "datasets" / "train.jsonl").is_file()
