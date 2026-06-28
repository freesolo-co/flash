"""Tests for `flash env pull` and the size-robust environment download helpers.

These exercise the fix for the GitHub "contents" API returning an empty body for blobs over
1 MB: single files are pulled via the raw media type, whole environments via the tarball.
"""

from __future__ import annotations

import io
import tarfile
import urllib.error
import urllib.request
from argparse import Namespace

import pytest

from flash.cli.envpush import cmd_env_pull
from flash.envs import adapter
from flash.envs.adapter import (
    download_environment_file,
    environment_local_dirname,
    pull_environment_package,
)

_TOP = "freesolo-co-environment-hub-deadbeef"


def _make_hub_tarball() -> bytes:
    """A tiny stand-in for the environment-hub tarball: one top dir, two orgs' envs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:

        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(f"{_TOP}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add("david-freesolo-co/stuff/environment.py", b"# env\n")
        add("david-freesolo-co/stuff/datasets/train.jsonl", b'{"a":1}\n' * 1000)
        # a sibling env that must NOT be copied into the target output
        add("other-org/other/environment.py", b"# other\n")
    return buf.getvalue()


# --- download_environment_file (raw media type, the E2 fix) ------------------


def test_download_environment_file_uses_raw_media_type_and_returns_full_body(monkeypatch):
    captured: dict[str, object] = {}
    payload = b'{"prompt":"x","answer":"y"}\n' * 100_000  # ~2.7 MB, well over the 1 MB contents cap

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["accept"] = req.headers.get("Accept")
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    data = download_environment_file("david-freesolo-co/stuff", "datasets/train.jsonl")

    assert data == payload  # full body, not an empty/truncated contents response
    assert captured["accept"] == "application/vnd.github.raw"
    assert (
        "/repos/freesolo-co/environment-hub/contents/"
        "david-freesolo-co/stuff/datasets/train.jsonl" in str(captured["url"])
    )
    assert "ref=main" in str(captured["url"])


def test_download_environment_file_sends_token_when_set(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured["auth"] = req.headers.get("Authorization")
        return io.BytesIO(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")

    download_environment_file("david-freesolo-co/stuff", "datasets/train.jsonl")
    assert captured["auth"] == "Bearer ghp_secret"


@pytest.mark.parametrize("bad", ["../secrets", "/etc/passwd", "a/../../b", "", "  "])
def test_download_environment_file_rejects_unsafe_path(bad):
    with pytest.raises(ValueError, match="invalid environment file path"):
        download_environment_file("david-freesolo-co/stuff", bad)


# --- pull_environment_package (tarball) -------------------------------------


def test_pull_environment_package_copies_only_the_env_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "stuff"

    out = pull_environment_package("david-freesolo-co/stuff", dest)

    assert out == dest
    assert (dest / "environment.py").is_file()
    assert (dest / "datasets" / "train.jsonl").read_bytes() == b'{"a":1}\n' * 1000
    # the sibling org's env must not leak into the output
    assert not (dest / "other-org").exists()
    assert not (dest / "environment-hub").exists()


def test_pull_environment_package_refuses_nonempty_dest_without_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "stuff"
    dest.mkdir()
    (dest / "keep.txt").write_text("existing")

    with pytest.raises(FileExistsError):
        pull_environment_package("david-freesolo-co/stuff", dest)
    # the pre-existing content is untouched
    assert (dest / "keep.txt").read_text() == "existing"


def test_pull_environment_package_overwrite_replaces(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "stuff"
    dest.mkdir()
    (dest / "stale.txt").write_text("old")

    pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)
    assert not (dest / "stale.txt").exists()
    assert (dest / "environment.py").is_file()


def test_pull_environment_package_populates_existing_empty_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "stuff"
    dest.mkdir()  # exists but empty -> allowed without overwrite
    pull_environment_package("david-freesolo-co/stuff", dest)
    assert (dest / "environment.py").is_file()


def test_pull_environment_package_dest_is_a_file(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "stuff"
    dest.write_text("i am a file")  # occupied non-dir path

    with pytest.raises(FileExistsError):
        pull_environment_package("david-freesolo-co/stuff", dest)
    # overwrite replaces the file with the env directory
    pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)
    assert dest.is_dir()
    assert (dest / "environment.py").is_file()


def test_pull_environment_package_creates_nested_parent_dirs(monkeypatch, tmp_path):
    # A nested output path whose parents don't exist must be created (mirrors the single-file
    # download), not fail in copytree.
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "a" / "b" / "c" / "stuff"
    out = pull_environment_package("david-freesolo-co/stuff", dest)
    assert out == dest
    assert (dest / "environment.py").is_file()


def test_pull_environment_package_preserves_dest_when_copy_fails(monkeypatch, tmp_path):
    # Overwriting must stage the new tree and swap it in: a mid-copy failure must NOT destroy the
    # existing destination (no remove-before-copy).
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "stuff"
    dest.mkdir()
    (dest / "keep.txt").write_text("precious")

    real_copytree = adapter.shutil.copytree

    def boom_copytree(src, dst, *a, **k):
        if ".flash-env-pull-" in str(dst):  # the staging copy
            raise OSError("disk full")
        return real_copytree(src, dst, *a, **k)

    monkeypatch.setattr(adapter.shutil, "copytree", boom_copytree)
    with pytest.raises(OSError, match="disk full"):
        pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)
    # the original is intact — it was never removed before the copy succeeded
    assert (dest / "keep.txt").read_text() == "precious"


def test_pull_environment_package_replaces_symlink_without_touching_target(monkeypatch, tmp_path):
    # A symlinked destination is unlinked (not rmtree'd, which shutil refuses; not followed), so the
    # symlink TARGET is left untouched and a real directory is written in its place.
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    real_target = tmp_path / "real"
    real_target.mkdir()
    (real_target / "precious.txt").write_text("keep me")
    dest = tmp_path / "stuff"
    dest.symlink_to(real_target)

    pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)
    assert not dest.is_symlink()
    assert (dest / "environment.py").is_file()
    assert (real_target / "precious.txt").read_text() == "keep me"  # target NOT deleted


def test_pull_environment_package_refuses_whole_hub_root(monkeypatch, tmp_path):
    # An explicit github ref to the shared hub's ROOT environment.py would copy every namespace;
    # refuse it (managed namespace/name ids always resolve to a subdir and are unaffected).
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    with pytest.raises(ValueError, match="whole shared environment hub"):
        pull_environment_package(
            "github:freesolo-co/environment-hub@main:environment.py", tmp_path / "x"
        )


def test_pull_environment_package_into_cwd_merges_without_nuking(monkeypatch, tmp_path):
    # Pulling into '.' (the cwd) must merge in place, never rmtree the directory itself — that would
    # delete unrelated files before the copy (shutil.rmtree('.') removes contents then raises).
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "unrelated.txt").write_text("keep me")
    pull_environment_package("david-freesolo-co/stuff", ".", overwrite=True)
    assert (tmp_path / "unrelated.txt").read_text() == "keep me"  # not nuked
    assert (tmp_path / "environment.py").is_file()  # env merged in


def test_pull_environment_package_accepts_directory_ref(monkeypatch, tmp_path):
    # A github ref naming the environment DIRECTORY (no trailing environment.py) resolves to that
    # dir, not its parent.
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "out"
    pull_environment_package("github:freesolo-co/environment-hub@main:david-freesolo-co/stuff", dest)
    assert (dest / "environment.py").is_file()
    assert (dest / "datasets" / "train.jsonl").is_file()
    assert not (dest / "other-org").exists()  # the env's own dir, not the parent


def test_pull_environment_package_rejects_dir_without_entrypoint(monkeypatch, tmp_path):
    # A dir that exists but lacks environment.py must error, not report a successful pull.
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    with pytest.raises(FileNotFoundError, match="entrypoint"):
        pull_environment_package(
            "github:freesolo-co/environment-hub@main:david-freesolo-co/stuff/datasets",
            tmp_path / "out",
        )


def _make_custom_entrypoint_tarball() -> bytes:
    """A hub tarball where an env's entrypoint file is NOT named ``environment.py``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:

        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(f"{_TOP}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add("david-freesolo-co/custom-env/custom.py", b"# custom entry\n")
        add("david-freesolo-co/custom-env/datasets/train.jsonl", b'{"a":1}\n')
    return buf.getvalue()


def test_pull_environment_package_refuses_whole_hub_root_case_insensitively(monkeypatch, tmp_path):
    # GitHub owner/repo are case-insensitive, so a differently-cased ref to the hub root must STILL
    # be refused — otherwise it would extract the entire shared hub.
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    with pytest.raises(ValueError, match="whole shared environment hub"):
        pull_environment_package(
            "github:Freesolo-Co/Environment-Hub@main:environment.py", tmp_path / "x"
        )


def test_pull_environment_package_custom_entrypoint_ref(monkeypatch, tmp_path):
    # A ref naming a custom entrypoint file (not environment.py) resolves to that file's DIRECTORY,
    # so the whole env (entrypoint + sidecars) is pulled, not a path under ``custom.py/``.
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_custom_entrypoint_tarball())
    dest = tmp_path / "out"
    pull_environment_package(
        "github:freesolo-co/environment-hub@main:david-freesolo-co/custom-env/custom.py", dest
    )
    assert (dest / "custom.py").is_file()
    assert (dest / "datasets" / "train.jsonl").is_file()


def test_download_environment_file_with_custom_entrypoint_ref(monkeypatch):
    # A sidecar path is relative to the env DIR, even when the ref names a custom entrypoint file.
    seen = {}

    def fake_urlopen(req, timeout=None, max_bytes=None):
        seen["url"] = req.full_url
        return b"data"

    monkeypatch.setattr(adapter, "_urlopen", fake_urlopen)
    download_environment_file(
        "github:owner/repo@main:envs/e/custom.py", "datasets/train.jsonl"
    )
    assert "envs/e/datasets/train.jsonl" in seen["url"]
    assert "custom.py" not in seen["url"]


def test_pull_environment_package_refuses_symlink_dest_without_overwrite(monkeypatch, tmp_path):
    # A symlink destination is "occupied": replacing a link needs explicit consent (overwrite=True).
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    real_target = tmp_path / "real"
    real_target.mkdir()
    dest = tmp_path / "link"
    dest.symlink_to(real_target)
    with pytest.raises(FileExistsError):
        pull_environment_package("david-freesolo-co/stuff", dest)  # no overwrite
    assert dest.is_symlink()  # untouched


def test_pull_into_cwd_does_not_follow_child_symlink(monkeypatch, tmp_path):
    # If the cwd already has e.g. ``datasets -> /external``, an in-place --force merge must NOT write
    # through that link into the external target; it replaces the link with a real directory.
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    external = tmp_path / "external"
    external.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    (work / "datasets").symlink_to(external)
    monkeypatch.chdir(work)
    pull_environment_package("david-freesolo-co/stuff", ".", overwrite=True)
    assert not (work / "datasets").is_symlink()  # link replaced by a real dir
    assert (work / "datasets" / "train.jsonl").is_file()  # written inside the output tree
    assert not (external / "train.jsonl").exists()  # NOT written through the link


# --- helpers ----------------------------------------------------------------


def test_environment_local_dirname():
    assert environment_local_dirname("david-freesolo-co/stuff") == "stuff"
    assert (
        environment_local_dirname("github:freesolo-co/environment-hub@main:a/b/environment.py")
        == "b"
    )


def test_coerce_environment_github_ref_rejects_garbage():
    with pytest.raises(ValueError, match="not a Freesolo or GitHub"):
        adapter._coerce_environment_github_ref("not a valid env id !!!")


# --- CLI --------------------------------------------------------------------


def _args(**kw) -> Namespace:
    base = {"env_id": "david-freesolo-co/stuff", "path": None, "output": None, "force": False}
    base.update(kw)
    return Namespace(**base)


def test_cmd_env_pull_single_file(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "download_environment_file", lambda env, path: b"line1\nline2\n")
    out = tmp_path / "train.jsonl"
    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(out)))
    assert rc == 0
    assert out.read_bytes() == b"line1\nline2\n"


def test_cmd_env_pull_refuses_overwrite_of_existing_file(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_download(env, path):
        calls["n"] += 1
        return b"x"

    monkeypatch.setattr(adapter, "download_environment_file", fake_download)
    out = tmp_path / "train.jsonl"
    out.write_text("keep")

    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(out)))
    assert rc == 1
    assert out.read_text() == "keep"  # not clobbered
    assert calls["n"] == 0  # refused before any download


def test_cmd_env_pull_whole_env(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "stuff"
    rc = cmd_env_pull(_args(output=str(dest)))
    assert rc == 0
    assert (dest / "datasets" / "train.jsonl").is_file()


def test_cmd_env_pull_single_file_refuses_dangling_symlink_without_force(monkeypatch, tmp_path):
    # A dangling symlink output (exists()==False) must still be treated as occupied, so a pull does
    # not proceed and write THROUGH the link to its target outside the requested path.
    calls = {"n": 0}

    def fake_download(env, path):
        calls["n"] += 1
        return b"x"

    monkeypatch.setattr(adapter, "download_environment_file", fake_download)
    target = tmp_path / "secret" / "train.jsonl"
    out = tmp_path / "train.jsonl"
    out.symlink_to(target)  # dangling: target does not exist
    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(out)))
    assert rc == 1
    assert calls["n"] == 0  # refused before any download
    assert not target.exists()  # nothing written through the link


def test_cmd_env_pull_single_file_force_replaces_symlink_not_target(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "download_environment_file", lambda env, path: b"new\n")
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("keep me")
    out = tmp_path / "train.jsonl"
    out.symlink_to(target)
    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(out), force=True))
    assert rc == 0
    assert not out.is_symlink()  # link replaced by a real file
    assert out.read_bytes() == b"new\n"
    assert target.read_text() == "keep me"  # the link target was NOT written through


def test_pull_environment_package_filters_hub_to_env_under_member_limit(monkeypatch, tmp_path):
    # A small env pull from the shared hub must not fail just because UNRELATED sibling envs push the
    # whole repo over the member limit — only the requested subtree is extracted and counted.
    monkeypatch.setattr(adapter, "_MAX_ARCHIVE_MEMBERS", 2)

    def _big_hub_tarball() -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            def add(name: str, data: bytes) -> None:
                info = tarfile.TarInfo(f"{_TOP}/{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            add("david-freesolo-co/stuff/environment.py", b"# env\n")  # the wanted env: 1 member
            for i in range(10):  # unrelated siblings that blow the (patched) member limit
                add(f"other-org/env{i}/environment.py", b"# other\n")
        return buf.getvalue()

    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _big_hub_tarball())
    dest = tmp_path / "out"
    pull_environment_package("david-freesolo-co/stuff", dest)
    assert (dest / "environment.py").is_file()
    assert not (dest / "env0").exists()  # unrelated siblings not extracted


def test_cmd_env_pull_rejects_bad_env_id(capsys):
    rc = cmd_env_pull(_args(env_id="!!!not-an-id!!!", path="x"))
    assert rc == 1
    assert "env id must be" in capsys.readouterr().err


def test_cmd_env_pull_no_token_hint_for_non_auth_error(monkeypatch, tmp_path, capsys):
    # A non-auth RuntimeError (e.g. "too large") must NOT print the misleading GITHUB_TOKEN hint.
    def boom(*a, **k):
        raise RuntimeError("environment archive is too large (999 bytes; limit 1 bytes)")

    monkeypatch.setattr(adapter, "pull_environment_package", boom)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    rc = cmd_env_pull(_args(output=str(tmp_path / "out")))
    assert rc == 1
    err = capsys.readouterr().err
    assert "too large" in err
    assert "GITHUB_TOKEN" not in err


def test_cmd_env_pull_token_hint_for_github_request_failure(monkeypatch, tmp_path, capsys):
    # A GitHub request failure (the auth-relevant case) DOES surface the token hint.
    def boom(*a, **k):
        raise RuntimeError("GitHub environment request failed (404): Not Found")

    monkeypatch.setattr(adapter, "pull_environment_package", boom)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    rc = cmd_env_pull(_args(output=str(tmp_path / "out")))
    assert rc == 1
    assert "GITHUB_TOKEN" in capsys.readouterr().err


# --- additional coverage (from adversarial review) --------------------------


def test_download_environment_file_rejects_oversized_body(monkeypatch):
    monkeypatch.setattr(adapter, "_MAX_ARCHIVE_BYTES", 16)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: io.BytesIO(b"x" * 100))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    # Now caught by the streaming cap (aborts mid-download) rather than the post-read length check;
    # either way it's a RuntimeError that rejects the oversized body before returning it.
    with pytest.raises(RuntimeError, match=r"exceeded the maximum allowed size|too large"):
        download_environment_file("david-freesolo-co/stuff", "datasets/train.jsonl")


def test_download_environment_file_url_encodes_special_chars(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        return io.BytesIO(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    download_environment_file("david-freesolo-co/stuff", "datasets/my data #1.jsonl")
    url = str(captured["url"])
    assert "my%20data%20%231.jsonl" in url  # space -> %20, '#' -> %23
    assert " " not in url


def test_download_environment_file_propagates_network_error(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        download_environment_file("david-freesolo-co/stuff", "datasets/train.jsonl")


def test_cmd_env_pull_single_file_refuses_directory_dest(monkeypatch, tmp_path):
    """A single-file pull must not crash (IsADirectoryError) when -o points at a directory."""
    calls = {"n": 0}

    def fake_download(env, path):
        calls["n"] += 1
        return b"data"

    monkeypatch.setattr(adapter, "download_environment_file", fake_download)
    dest = tmp_path / "train.jsonl"
    dest.mkdir()  # a directory where a file output is expected
    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(dest), force=True))
    assert rc == 1
    assert dest.is_dir()  # untouched
    assert calls["n"] == 0  # refused before any download, no traceback


def test_cmd_env_pull_whole_env_refuses_overwrite_without_force(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "stuff"
    dest.mkdir()
    (dest / "keep.txt").write_text("existing")
    rc = cmd_env_pull(_args(output=str(dest)))
    assert rc == 1
    assert (dest / "keep.txt").read_text() == "existing"  # untouched


def test_cmd_env_pull_whole_env_dir_not_found(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    rc = cmd_env_pull(
        _args(env_id="david-freesolo-co/does-not-exist", output=str(tmp_path / "out"))
    )
    assert rc == 1
    assert "environment directory" in capsys.readouterr().err


def test_cmd_env_pull_runtime_error_shows_token_hint(monkeypatch, tmp_path, capsys):
    def boom(env, path):
        raise RuntimeError("GitHub environment request failed (404)")

    monkeypatch.setattr(adapter, "download_environment_file", boom)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(tmp_path / "x.jsonl")))
    assert rc == 1
    err = capsys.readouterr().err
    assert "env pull failed" in err
    assert "GITHUB_TOKEN" in err  # hint shown for GitHub errors when no token


def test_cmd_env_pull_single_file_default_output_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(adapter, "download_environment_file", lambda env, path: b"D")
    rc = cmd_env_pull(_args(path="datasets/train.jsonl"))  # no --output
    assert rc == 0
    assert (tmp_path / "train.jsonl").read_bytes() == b"D"  # basename of the requested path


def test_cmd_env_pull_whole_env_default_output_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    rc = cmd_env_pull(_args())  # no path, no output -> dir named after the env
    assert rc == 0
    assert (tmp_path / "stuff" / "environment.py").is_file()


def test_pull_overwrite_preserves_old_package_when_swap_fails(monkeypatch, tmp_path):
    # If the final swap-in fails, the previous package must be restored intact (never half-removed).
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    dest = tmp_path / "env"
    dest.mkdir()
    (dest / "old.txt").write_text("precious")

    real_replace = adapter.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        # let the move-aside succeed; fail the staging->dest swap, then allow the restore
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated swap failure")
        return real_replace(src, dst)

    monkeypatch.setattr(adapter.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated swap failure"):
        pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)
    assert (dest / "old.txt").read_text() == "precious"  # original restored intact
    assert not (dest / "environment.py").exists()  # the new tree was not installed


def test_merge_into_dir_replaces_file_atomically(monkeypatch, tmp_path):
    # A failed copy during an in-place merge must not truncate the existing same-named file.
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "environment.py").write_text("ORIGINAL")  # collides with the env's environment.py

    def boom_copy(src, tmp):
        raise OSError("disk full")

    monkeypatch.setattr(adapter.shutil, "copy2", boom_copy)
    with pytest.raises(OSError, match="disk full"):
        pull_environment_package("david-freesolo-co/stuff", ".", overwrite=True)
    assert (tmp_path / "environment.py").read_text() == "ORIGINAL"  # not truncated/partially written


def test_download_github_tarball_uses_whole_repo_ceiling(monkeypatch):
    # The whole-repo tarball download is bounded by the larger _MAX_TARBALL_BYTES, not the per-env
    # _MAX_ARCHIVE_BYTES — so a small env pull from a big shared hub isn't rejected at download.
    assert adapter._MAX_TARBALL_BYTES > adapter._MAX_ARCHIVE_BYTES
    big = b"x" * (adapter._MAX_ARCHIVE_BYTES + 10)  # over the per-env cap, under the tarball ceiling
    monkeypatch.setattr(adapter, "_urlopen", lambda req, timeout=None, max_bytes=None: big)
    monkeypatch.setattr(adapter, "_github_token", lambda: None)
    ref = adapter._coerce_environment_github_ref("david-freesolo-co/stuff")
    assert adapter._download_github_tarball(ref) == big  # not rejected
    too_big = b"x" * (adapter._MAX_TARBALL_BYTES + 10)
    monkeypatch.setattr(adapter, "_urlopen", lambda req, timeout=None, max_bytes=None: too_big)
    with pytest.raises(RuntimeError, match="tarball is too large"):
        adapter._download_github_tarball(ref)


def test_download_github_tarball_passes_streaming_cap(monkeypatch):
    # _download_github_tarball must hand _urlopen its _MAX_TARBALL_BYTES ceiling so the body is
    # stream-capped (aborts early) rather than buffered whole and only then length-checked.
    seen = {}

    def fake_urlopen(req, timeout=None, max_bytes=None):
        seen["max_bytes"] = max_bytes
        return b"ok"

    monkeypatch.setattr(adapter, "_urlopen", fake_urlopen)
    monkeypatch.setattr(adapter, "_github_token", lambda: None)
    ref = adapter._coerce_environment_github_ref("david-freesolo-co/stuff")
    adapter._download_github_tarball(ref)
    assert seen["max_bytes"] == adapter._MAX_TARBALL_BYTES


def test_urlopen_streams_and_aborts_over_max_bytes(monkeypatch):
    # A body larger than max_bytes aborts mid-stream (early-abort) instead of being read whole.
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
    payload = b"x" * 64
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(payload))
    req = urllib.request.Request("https://api.github.com/x")
    with pytest.raises(RuntimeError, match="exceeded the maximum allowed size"):
        adapter._urlopen(req, max_bytes=16)
    # early abort: stopped reading not long after crossing the limit, not the whole 64-byte body
    assert served["n"] <= 16 + adapter._DOWNLOAD_CHUNK_BYTES


def test_urlopen_streams_full_body_under_cap(monkeypatch):
    # Under the cap the streamed body is returned intact (chunk reassembly is lossless).
    payload = b"abcdefghij" * 5
    monkeypatch.setattr(adapter, "_DOWNLOAD_CHUNK_BYTES", 7)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: io.BytesIO(payload))
    req = urllib.request.Request("https://api.github.com/x")
    assert adapter._urlopen(req, max_bytes=10_000) == payload


def test_safe_extract_archive_bounds_total_members_scanned(monkeypatch, tmp_path):
    # Pulling a tiny subdir from a tarball with many UNRELATED members must still bound the scan:
    # the total-headers cap fires even though almost nothing would be extracted.
    monkeypatch.setattr(adapter, "_MAX_ARCHIVE_SCAN_MEMBERS", 4)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for i in range(20):  # 20 unrelated members, only "wanted/" would be extracted
            info = tarfile.TarInfo(name=f"repo-sha/unrelated/f{i}.txt")
            data = b"x"
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with pytest.raises(RuntimeError, match="too many entries to scan"):
        adapter._safe_extract_archive(buf.getvalue(), tmp_path, subdir="wanted")


# --- adversarial-review hardening (round 4) ---------------------------------


def test_read_capped_returns_bytearray_no_double_copy(monkeypatch):
    # _read_capped returns a single growing bytearray (no retained chunk list + join), so a near-cap
    # body never needs a second contiguous allocation. Result is correct and bytes-comparable.
    monkeypatch.setattr(adapter, "_DOWNLOAD_CHUNK_BYTES", 4)
    out = adapter._read_capped(io.BytesIO(b"x" * 20), 100)
    assert isinstance(out, bytearray)
    assert out == b"x" * 20


def test_resolve_github_env_extracts_repo_level_siblings(monkeypatch, tmp_path):
    # The execution resolver must extract the WHOLE repo (not just the env subtree) so an env that
    # references a repo-level sidecar via a relative param/import (e.g. ../datasets) finds it in cache.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def add(name, data):
            info = tarfile.TarInfo(name=f"repo-sha/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        add("envs/e/environment.py", b"# env\n")
        add("envs/datasets/train.jsonl", b'{"a":1}\n')  # sibling one level up from the env dir

    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: buf.getvalue())
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **k: "a" * 40)
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    env_file = adapter._resolve_github_environment_file(
        "github:owner/repo@main:envs/e/environment.py"
    )
    assert env_file.is_file()
    # the repo-level sibling is present in cache next to the env dir — it would be MISSING if the
    # resolver had filtered extraction to envs/e only.
    envs_dir = env_file.parents[1]  # .../cache/<key>/repo-sha/envs/e/environment.py -> .../envs
    assert (envs_dir / "datasets" / "train.jsonl").is_file()


def test_merge_opposite_type_restores_file_on_copy_failure(monkeypatch, tmp_path):
    # An incoming DIR replacing an existing FILE must not destroy the file if the recursive copy fails.
    source = tmp_path / "src" / "data"
    source.mkdir(parents=True)
    (source.parent / "data" / "f.txt").write_text("new")
    dest = tmp_path / "dst"
    dest.mkdir()
    (dest / "data").write_text("ORIGINAL")  # existing FILE where the incoming tree has a DIR

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(adapter, "_atomic_copy2", boom)
    with pytest.raises(OSError, match="disk full"):
        adapter._merge_into_dir(tmp_path / "src", dest)
    # the user's original file survived intact (move-aside/restore, not unlink-then-copy)
    assert (dest / "data").is_file()
    assert (dest / "data").read_text() == "ORIGINAL"
    assert not any(p.name.startswith(".data") for p in dest.iterdir())  # backup cleaned up


def test_pull_preserves_symlink_dest_on_copy_failure(monkeypatch, tmp_path):
    # `--force` over a SYMLINK destination: if the staged copy fails, the user's link must survive
    # (it is not unlinked up front), and its target is left untouched.
    monkeypatch.setattr(adapter, "_download_github_tarball", lambda ref: _make_hub_tarball())
    realtarget = tmp_path / "real"
    realtarget.mkdir()
    (realtarget / "keep").write_text("x")
    dest = tmp_path / "link"
    dest.symlink_to(realtarget, target_is_directory=True)

    orig_copytree = adapter.shutil.copytree

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(adapter.shutil, "copytree", boom)
    with pytest.raises(OSError, match="disk full"):
        adapter.pull_environment_package("david-freesolo-co/stuff", dest, overwrite=True)
    monkeypatch.setattr(adapter.shutil, "copytree", orig_copytree)
    assert dest.is_symlink()  # link preserved
    assert dest.resolve() == realtarget.resolve()
    assert (realtarget / "keep").read_text() == "x"  # target untouched
