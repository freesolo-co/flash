"""Tests for `flash env pull`."""

from __future__ import annotations

import io
import json
import shlex
import stat
import tarfile
import urllib.parse
import urllib.request
from argparse import Namespace
from pathlib import Path

import pytest

from flash._channel import CLI_NAME
from flash.cli import envpush
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
        _package_tarball({"environment.py": b"# env\n", "datasets/train.jsonl": b"line1\nline2\n"}),
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


def test_cmd_env_pull_positional_dir_names_the_destination_form(monkeypatch, tmp_path, capsys):
    """`flash env pull ns/env ./dir` must say how to ask for a destination directory.

    The second positional is a path inside the env, so passing a directory there reads as
    "fetch this file" and fails on the overwrite check. The error has to name the -o form.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "into-here").mkdir()

    rc = cmd_env_pull(_margs(path="into-here", output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "--output=into-here" in err
    assert "david-freesolo-co/stuff" in err


def test_cmd_env_pull_explicit_output_dir_keeps_the_single_file_diagnostic(
    monkeypatch, tmp_path, capsys
):
    """An explicit `-o dir` named the in-env path on purpose, so do not tell the user to drop it.

    `flash env pull ns/env config.json -o existing-dir` is a single-file pull whose destination is
    wrong. Following the mistaken-positional hint here would abandon config.json and turn the
    command into a whole-environment download instead of fixing the destination.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing-dir").mkdir()

    rc = cmd_env_pull(_margs(path="config.json", output="existing-dir", force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "needs -o to be a FILE path" in err
    assert "drop the second positional" not in err


def test_cmd_env_pull_positional_dir_hint_is_shell_quoted(monkeypatch, tmp_path, capsys):
    """The suggested command must survive a copy-paste out of a directory name with spaces.

    Unquoted, `-o into here` splits into two shell arguments and argparse rejects the very command
    offered as the remedy.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "into here").mkdir()

    rc = cmd_env_pull(_margs(path="into here", output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "'--output=into here'" in err


def _hint_command(err: str) -> str:
    """The suggested command from the hint line, as the user would copy it."""
    for line in err.splitlines():
        if " env pull " in line and "hint" in line:
            return line[line.index(f"{CLI_NAME} env pull ") :]
    raise AssertionError(f"no hint command in:\n{err}")


def _parse_as_cli(command: str) -> Namespace:
    """Round-trip a suggested command through a POSIX shell split and the real `env pull` parser.

    A hint is only useful if pasting it back works, so assert against the parser the user would
    actually hit rather than against the string we happened to build.
    """
    from flash.cli import _build_parser

    argv = shlex.split(command)
    assert argv[:3] == [CLI_NAME, "env", "pull"], argv
    return _build_parser().parse_args(argv[1:])


def test_cmd_env_pull_dash_prefixed_dir_hint_survives_the_parser(monkeypatch, tmp_path, capsys):
    """A destination whose basename starts with `-` must not be read back as an option.

    `-o -dest` makes argparse fail with "expected one argument", so the remedy would be rejected by
    the same parser that produced the error. Quoting does not fix it -- the attached `--output=`
    form does, because the value can no longer start its own token.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "-dest").mkdir()

    rc = cmd_env_pull(_margs(path="./-dest", output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    parsed = _parse_as_cli(_hint_command(err))
    assert parsed.output == "-dest"
    assert parsed.path is None


def test_cmd_env_pull_positional_dir_hint_round_trips_through_the_parser(
    monkeypatch, tmp_path, capsys
):
    """The suggested command must parse back to a whole-env pull aimed at that same directory."""
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "into here").mkdir()

    rc = cmd_env_pull(_margs(path="into here", output=None, force=False))

    assert rc == 1
    parsed = _parse_as_cli(_hint_command(capsys.readouterr().err))
    assert parsed.output == "into here"
    assert parsed.path is None


def test_cmd_env_pull_absolute_positional_dir_is_diagnosed(monkeypatch, tmp_path, capsys):
    """An absolute mistaken destination must be recognized before it is reduced to its basename.

    `env pull ns/env /tmp/into-here` is the same user error as the relative form, but `out` is
    already `Path("into-here")` by the time the check ran, so comparing against `out` could never
    match: the command downloaded the package and failed with an unrelated invalid-path error
    instead of the hint that explains what went wrong.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.chdir(tmp_path)
    absolute = tmp_path / "into-here"
    absolute.mkdir()

    rc = cmd_env_pull(_margs(path=str(absolute), output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "drop the second positional" in err
    # the suggested destination must be the directory the user actually named, not its basename --
    # a bare `into-here` would aim the whole-env download at a different path in the cwd.
    parsed = _parse_as_cli(_hint_command(err))
    assert parsed.output == str(absolute)
    assert parsed.path is None


def test_cmd_env_pull_hint_omits_a_command_cmd_exe_would_mangle(monkeypatch, tmp_path, capsys):
    """On Windows a destination holding cmd.exe metacharacters gets no copy-pasteable command.

    `list2cmdline` implements MS C-runtime argv quoting, not cmd.exe escaping, so `foo&bar` comes
    back unquoted and pasting the hint would run `bar` as a separate command. Emitting no command is
    strictly better than emitting one that executes unintended text.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.setattr(envpush.os, "name", "nt")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo&bar").mkdir()

    rc = cmd_env_pull(_margs(path="foo&bar", output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "drop the second positional" in err
    # the directory is still NAMED so the user knows which one is meant; what must not appear is a
    # runnable command carrying the unescaped metacharacter.
    assert "foo&bar" in err
    assert "--output=foo&bar" not in err
    assert "--output=DEST" in err


def test_cmd_env_pull_hint_still_quotes_a_windows_destination_with_a_space(
    monkeypatch, tmp_path, capsys
):
    """Refusing metacharacter destinations must not cost the ordinary Windows quoting case."""
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.setattr(envpush.os, "name", "nt")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "into here").mkdir()

    rc = cmd_env_pull(_margs(path="into here", output=None, force=False))

    assert rc == 1
    assert '"--output=into here"' in capsys.readouterr().err


def test_cmd_env_pull_basename_collision_keeps_the_single_file_diagnostic(
    monkeypatch, tmp_path, capsys
):
    """A local dir sharing the positional's BASENAME is not evidence of a mistaken destination.

    `env pull ns/env assets/config` with a local ./config/ is a genuine single-file pull whose
    output name happens to collide. Telling that user to drop the positional would abandon the file
    they asked for, and the nonempty branch would aim --force at an unrelated directory.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "keep.txt").write_text("mine")

    rc = cmd_env_pull(_margs(path="assets/config", output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "needs -o to be a FILE path" in err
    assert "drop the second positional" not in err
    assert "--force" not in err


def test_cmd_env_pull_cwd_destination_hint_does_not_offer_force(monkeypatch, tmp_path, capsys):
    """`env pull ns/env .` cannot be fixed with --force, so the hint must not suggest it.

    ensure_environment_pull_destination_available() refuses to replace any directory containing the
    cwd, even with overwrite=True, so the generic nonempty remedy would fail a second time.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing.txt").write_text("keep me")

    rc = cmd_env_pull(_margs(path=".", output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "contains the current working directory" in err
    # scoped to the SUGGESTED COMMAND, not the whole message: naming --force to say it would not
    # help here is the point of this branch. what must not happen is offering it as the remedy.
    command = _hint_command(err)
    assert "--force" not in command, command
    assert _parse_as_cli(command).path is None


def test_cmd_env_pull_positional_nonempty_dir_hint_says_how_to_replace_it(
    monkeypatch, tmp_path, capsys
):
    """A nonempty destination needs --force, so the bare command would just fail again.

    The whole-environment path runs ensure_environment_pull_destination_available(overwrite=False),
    which rejects every nonempty directory -- offering it as the remedy trades one error for the next.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "into-here").mkdir()
    (tmp_path / "into-here" / "existing.txt").write_text("keep me")

    rc = cmd_env_pull(_margs(path="into-here", output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "not empty" in err
    assert "--force" in err


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
        adapter._safe_extract_archive(_tarball(entries), tmp_path)


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
