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
from flash.envs import adapter
from flash.envs import pull as env_pull
from flash.envs.pull import (
    download_environment_file,
    environment_local_dirname,
    pull_environment_package,
)

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


def _hub_tarball() -> bytes:
    return _tarball(
        {
            "david-freesolo-co/stuff/environment.py": b"# env\n",
            "david-freesolo-co/stuff/datasets/train.jsonl": b'{"a":1}\n' * 1000,
            "other-org/other/environment.py": b"# other\n",
        }
    )


def _custom_entrypoint_tarball() -> bytes:
    return _tarball(
        {
            "david-freesolo-co/custom-env/custom.py": b"# custom\n",
            "david-freesolo-co/custom-env/datasets/train.jsonl": b'{"a":1}\n',
        }
    )


def _args(**kw) -> Namespace:
    base = {"env_id": "david-freesolo-co/stuff", "path": None, "output": None, "force": False}
    base.update(kw)
    return Namespace(**base)


def test_download_environment_file_uses_raw_media_type(monkeypatch):
    captured: dict[str, object] = {}
    payload = b'{"prompt":"x","answer":"y"}\n' * 100_000

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["accept"] = req.headers.get("Accept")
        return _Resp(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    data = download_environment_file("david-freesolo-co/stuff", "datasets/train.jsonl")

    assert data == payload
    assert captured["accept"] == "application/vnd.github.raw"
    assert (
        "/repos/freesolo-co/environment-hub/contents/"
        "david-freesolo-co/stuff/datasets/train.jsonl" in str(captured["url"])
    )
    assert "ref=main" in str(captured["url"])


def test_download_environment_file_sends_token(monkeypatch):
    captured: dict[str, object] = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout):
        captured["auth"] = req.headers.get("Authorization")
        return _Resp(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")

    assert download_environment_file("david-freesolo-co/stuff", "datasets/train.jsonl") == b"ok"
    assert captured["auth"] == "Bearer ghp_secret"


@pytest.mark.parametrize("bad", ["../secrets", "/etc/passwd", "a/../../b", "", "  "])
def test_download_environment_file_rejects_unsafe_path(bad):
    with pytest.raises(ValueError, match="invalid environment file path"):
        download_environment_file("david-freesolo-co/stuff", bad)


def test_download_environment_file_with_custom_entrypoint_ref(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None, max_bytes=None):
        seen["url"] = req.full_url
        return b"data"

    monkeypatch.setattr(adapter, "_urlopen", fake_urlopen)
    download_environment_file("github:owner/repo@main:envs/e/custom.py", "datasets/train.jsonl")
    assert "envs/e/datasets/train.jsonl" in seen["url"]
    assert "custom.py" not in seen["url"]


def test_pull_environment_package_copies_only_requested_env(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    dest = tmp_path / "stuff"

    out = pull_environment_package("david-freesolo-co/stuff", dest)

    assert out == dest
    assert (dest / "environment.py").is_file()
    assert (dest / "datasets" / "train.jsonl").read_bytes() == b'{"a":1}\n' * 1000
    assert not (dest / "other-org").exists()


def test_pull_environment_package_refuses_nonempty_dest_without_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    dest = tmp_path / "stuff"
    dest.mkdir()
    (dest / "keep.txt").write_text("existing")

    with pytest.raises(FileExistsError):
        pull_environment_package("david-freesolo-co/stuff", dest)
    assert (dest / "keep.txt").read_text() == "existing"


def test_pull_environment_package_overwrite_replaces_occupied_dest(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    dest = tmp_path / "stuff"
    dest.mkdir()
    (dest / "stale.txt").write_text("old")

    pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)

    assert not (dest / "stale.txt").exists()
    assert (dest / "environment.py").is_file()


def test_pull_environment_package_populates_empty_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    dest = tmp_path / "stuff"
    dest.mkdir()
    dest.chmod(0o700)

    pull_environment_package("david-freesolo-co/stuff", dest)

    assert (dest / "environment.py").is_file()
    assert dest.stat().st_mode & 0o777 == 0o700


def test_pull_environment_package_populates_empty_dir_with_internal_staging(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    real_mkdtemp = env_pull.tempfile.mkdtemp
    staging_dirs: list[Path] = []

    def fake_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        if kwargs.get("prefix") == ".flash-env-pull-":
            staging_dirs.append(Path(kwargs["dir"]))
        return path

    monkeypatch.setattr(env_pull.tempfile, "mkdtemp", fake_mkdtemp)
    dest = tmp_path / "stuff"
    dest.mkdir()

    pull_environment_package("david-freesolo-co/stuff", dest)

    assert dest in staging_dirs
    assert (dest / "environment.py").is_file()


def test_pull_environment_package_replaces_file_with_force(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    dest = tmp_path / "stuff"
    dest.write_text("file")

    with pytest.raises(FileExistsError):
        pull_environment_package("david-freesolo-co/stuff", dest)

    pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)
    assert dest.is_dir()
    assert (dest / "environment.py").is_file()


def test_pull_environment_package_creates_nested_parent_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    dest = tmp_path / "a" / "b" / "stuff"

    pull_environment_package("david-freesolo-co/stuff", dest)

    assert (dest / "environment.py").is_file()


def test_pull_environment_package_preserves_dest_when_staging_copy_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    dest = tmp_path / "stuff"
    dest.mkdir()
    (dest / "keep.txt").write_text("precious")
    real_copytree = env_pull.shutil.copytree

    def boom_copytree(src, dst, *args, **kwargs):
        if ".flash-env-pull-" in str(dst):
            raise OSError("disk full")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(env_pull.shutil, "copytree", boom_copytree)

    with pytest.raises(OSError, match="disk full"):
        pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)
    assert (dest / "keep.txt").read_text() == "precious"


def test_pull_environment_package_restores_dest_when_final_swap_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    dest = tmp_path / "stuff"
    dest.mkdir()
    (dest / "keep.txt").write_text("precious")
    real_replace = env_pull.os.replace

    def fail_final_replace(src, dst):
        if Path(src).name == dest.name and Path(dst) == dest:
            raise OSError("swap failed")
        return real_replace(src, dst)

    monkeypatch.setattr(env_pull.os, "replace", fail_final_replace)

    with pytest.raises(OSError, match="swap failed"):
        pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)
    assert (dest / "keep.txt").read_text() == "precious"


def test_pull_environment_package_replaces_symlink_without_touching_target(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    real_target = tmp_path / "real"
    real_target.mkdir()
    (real_target / "precious.txt").write_text("keep me")
    dest = tmp_path / "stuff"
    dest.symlink_to(real_target)

    with pytest.raises(FileExistsError):
        pull_environment_package("david-freesolo-co/stuff", dest)

    pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)
    assert not dest.is_symlink()
    assert (dest / "environment.py").is_file()
    assert (real_target / "precious.txt").read_text() == "keep me"


def test_pull_environment_package_replaces_symlink_to_cwd_without_touching_target(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    real_target = tmp_path / "real"
    real_target.mkdir()
    (real_target / "precious.txt").write_text("keep me")
    dest = tmp_path / "stuff"
    dest.symlink_to(real_target, target_is_directory=True)
    monkeypatch.chdir(real_target)

    pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)

    assert not dest.is_symlink()
    assert (dest / "environment.py").is_file()
    assert (real_target / "precious.txt").read_text() == "keep me"
    assert not (real_target / "environment.py").exists()


def test_pull_environment_package_refuses_whole_hub_root(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    with pytest.raises(ValueError, match="whole shared environment hub"):
        pull_environment_package(
            "github:Freesolo-Co/Environment-Hub@main:environment.py", tmp_path / "x"
        )


def test_pull_environment_package_accepts_directory_ref(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    dest = tmp_path / "out"

    pull_environment_package(
        "github:freesolo-co/environment-hub@main:david-freesolo-co/stuff", dest
    )

    assert (dest / "environment.py").is_file()
    assert (dest / "datasets" / "train.jsonl").is_file()
    assert not (dest / "other-org").exists()


def test_pull_environment_package_rejects_dir_without_entrypoint(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    with pytest.raises(FileNotFoundError, match="entrypoint"):
        pull_environment_package(
            "github:freesolo-co/environment-hub@main:david-freesolo-co/stuff/datasets",
            tmp_path / "out",
        )


def test_pull_environment_package_custom_entrypoint_ref(monkeypatch, tmp_path):
    monkeypatch.setattr(
        adapter, "_download_github_tarball", lambda ref: _custom_entrypoint_tarball()
    )
    dest = tmp_path / "out"

    pull_environment_package(
        "github:freesolo-co/environment-hub@main:david-freesolo-co/custom-env/custom.py", dest
    )

    assert (dest / "custom.py").is_file()
    assert (dest / "datasets" / "train.jsonl").is_file()


def test_pull_into_empty_cwd_populates_in_place(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    pull_environment_package("david-freesolo-co/stuff", ".")

    assert (work / "environment.py").is_file()
    assert (work / "datasets" / "train.jsonl").is_file()


def test_pull_into_occupied_cwd_is_refused_without_touching_children(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    external = tmp_path / "external"
    external.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    (work / "datasets").symlink_to(external)
    monkeypatch.chdir(work)

    with pytest.raises(RuntimeError, match="current working directory"):
        pull_environment_package("david-freesolo-co/stuff", ".", overwrite=True)

    assert (work / "datasets").is_symlink()
    assert not (external / "train.jsonl").exists()
    assert not (work / "environment.py").exists()


def test_pull_environment_package_filters_hub_to_env_under_member_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_MAX_ARCHIVE_MEMBERS", 2)
    entries = {
        "david-freesolo-co/stuff/environment.py": b"# env\n",
        "david-freesolo-co/stuff/datasets/train.jsonl": b"data\n",
    }
    entries.update({f"other/env{i}/file.txt": b"x" for i in range(10)})
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _tarball(entries))

    pull_environment_package("david-freesolo-co/stuff", tmp_path / "out")

    assert (tmp_path / "out" / "datasets" / "train.jsonl").read_text() == "data\n"


def test_safe_extract_archive_bounds_total_members_scanned(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_MAX_ARCHIVE_SCAN_MEMBERS", 4)
    entries = {f"unrelated/f{i}.txt": b"x" for i in range(20)}

    with pytest.raises(RuntimeError, match="too many entries to scan"):
        adapter._safe_extract_archive(_tarball(entries), tmp_path, subdir="wanted")


def test_safe_extract_archive_rejects_top_level_file(tmp_path):
    with pytest.raises(RuntimeError, match="unexpected layout"):
        adapter._safe_extract_archive(_top_level_file_tarball(), tmp_path)


def test_pull_rechecks_destination_after_download(monkeypatch, tmp_path):
    dest = tmp_path / "out"

    def racing_download(ref):
        dest.write_text("created during download")
        return _hub_tarball()

    monkeypatch.setattr(adapter, "_download_github_tarball", racing_download)
    with pytest.raises(FileExistsError):
        pull_environment_package("david-freesolo-co/stuff", dest)
    assert dest.read_text() == "created during download"


def test_environment_local_dirname():
    assert environment_local_dirname("david-freesolo-co/stuff") == "stuff"
    assert (
        environment_local_dirname("github:freesolo-co/environment-hub@main:a/b/environment.py")
        == "b"
    )


def test_coerce_environment_github_ref_rejects_garbage():
    with pytest.raises(ValueError, match="not a Freesolo or GitHub"):
        env_pull._coerce_environment_github_ref("not a valid env id !!!")


def test_cmd_env_pull_single_file(monkeypatch, tmp_path):
    monkeypatch.setattr(env_pull, "download_environment_file", lambda env, path: b"line1\nline2\n")
    out = tmp_path / "train.jsonl"

    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(out)))

    assert rc == 0
    assert out.read_bytes() == b"line1\nline2\n"


def test_cmd_env_pull_refuses_existing_file_without_force(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_download(env, path):
        calls["n"] += 1
        return b"x"

    monkeypatch.setattr(env_pull, "download_environment_file", fake_download)
    out = tmp_path / "train.jsonl"
    out.write_text("keep")

    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(out)))

    assert rc == 1
    assert out.read_text() == "keep"
    assert calls["n"] == 0


def test_cmd_env_pull_single_file_replaces_symlink_with_force(monkeypatch, tmp_path):
    monkeypatch.setattr(env_pull, "download_environment_file", lambda env, path: b"new\n")
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("keep me")
    out = tmp_path / "train.jsonl"
    out.symlink_to(target)

    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(out), force=True))

    assert rc == 0
    assert not out.is_symlink()
    assert out.read_bytes() == b"new\n"
    assert target.read_text() == "keep me"


def test_cmd_env_pull_single_file_refuses_real_dir(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_download(env, path):
        calls["n"] += 1
        return b"x"

    monkeypatch.setattr(env_pull, "download_environment_file", fake_download)
    out = tmp_path / "adir"
    out.mkdir()

    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(out), force=True))

    assert rc == 1
    assert calls["n"] == 0


def test_cmd_env_pull_whole_env(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _hub_tarball())
    dest = tmp_path / "stuff"

    rc = cmd_env_pull(_args(output=str(dest)))

    assert rc == 0
    assert (dest / "datasets" / "train.jsonl").is_file()


def test_cmd_env_pull_token_hint_for_github_request_failure(monkeypatch, tmp_path, capsys):
    def fail(env, path):
        raise RuntimeError("GitHub environment request failed (404): no")

    monkeypatch.setattr(env_pull, "download_environment_file", fail)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(tmp_path / "x")))

    assert rc == 1
    assert "set GITHUB_TOKEN" in capsys.readouterr().err


def test_download_github_tarball_uses_whole_repo_ceiling(monkeypatch):
    assert adapter._MAX_TARBALL_BYTES > adapter._MAX_ARCHIVE_BYTES
    big = b"x" * (adapter._MAX_ARCHIVE_BYTES + 10)

    def fake_urlopen(req, timeout=None, max_bytes=None, out=None):
        assert max_bytes == adapter._MAX_TARBALL_BYTES
        out.write(big)
        return b""

    monkeypatch.setattr(adapter, "_urlopen", fake_urlopen)
    monkeypatch.setattr(adapter, "_github_token", lambda: None)
    ref = env_pull._coerce_environment_github_ref("david-freesolo-co/stuff")

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
        ("a" * 40 + ":david-freesolo-co/stuff", True): {
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


def test_explicit_environment_hub_github_ref_keeps_tarball_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)

    def fail_directory(ref, repo_dir, dest):
        raise AssertionError("explicit GitHub refs should keep the whole-repo tarball path")

    monkeypatch.setattr(adapter, "_download_github_directory", fail_directory)
    monkeypatch.setattr(
        adapter,
        "_download_github_tarball",
        lambda ref: _tarball(
            {
                "david-freesolo-co/stuff/environment.py": b"# env\n",
                "shared/helper.py": b"# helper\n",
            }
        ),
    )

    env_file = Path(
        adapter._resolve_environment_reference(
            "github:freesolo-co/environment-hub@main:david-freesolo-co/stuff/environment.py"
        )
    )

    assert env_file.read_bytes() == b"# env\n"
    assert (env_file.parents[2] / "shared" / "helper.py").read_bytes() == b"# helper\n"


def test_download_github_directory_handles_large_tree_listing(monkeypatch, tmp_path):
    ref = adapter.GitHubEnvironmentRef(
        "freesolo-co",
        "environment-hub",
        "b" * 40,
        "david-freesolo-co/big/environment.py",
    )
    shard_count = 1001
    trees = {
        ("b" * 40 + ":david-freesolo-co/big", True): {
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
