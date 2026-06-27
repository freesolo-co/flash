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


def test_cmd_env_pull_single_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(adapter, "download_environment_file", lambda env, path: b"line1\nline2\n")
    out = tmp_path / "train.jsonl"
    rc = cmd_env_pull(_args(path="datasets/train.jsonl", output=str(out)))
    assert rc == 0
    assert out.read_bytes() == b"line1\nline2\n"


def test_cmd_env_pull_refuses_overwrite_of_existing_file(monkeypatch, tmp_path, capsys):
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


def test_cmd_env_pull_rejects_bad_env_id(capsys):
    rc = cmd_env_pull(_args(env_id="!!!not-an-id!!!", path="x"))
    assert rc == 1
    assert "env id must be" in capsys.readouterr().err


# --- additional coverage (from adversarial review) --------------------------


def test_download_environment_file_rejects_oversized_body(monkeypatch):
    monkeypatch.setattr(adapter, "_MAX_ARCHIVE_BYTES", 16)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: io.BytesIO(b"x" * 100))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="too large"):
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
