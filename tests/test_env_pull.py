"""Tests for `flash env pull`."""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import stat
import tarfile
import urllib.parse
import urllib.request
from argparse import Namespace
from pathlib import Path

import pytest

from flash._internal.channel import CLI_NAME
from flash.cli.commands.env.ops import push as envpush
from flash.cli.commands.env.ops.push import cmd_env_pull
from flash.envs.loading import loader as adapter
from flash.envs.loading.pull import environment_local_dirname

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
        "env_id": "david-freesolo-co/my-project/stuff",
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
    rc = cmd_env_pull(_margs(env_id="David-Freesolo-Co/My-Project/Stuff"))

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
    assert "david-freesolo-co/my-project/stuff" in err


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

    Asserted as POSIX quoting because the platform is pinned, not assumed: `_quote_shell_token`
    deliberately emits `"--output=into here"` via `list2cmdline` on Windows, so an unconditional
    single-quote assertion would fail a native Windows run on correct output. The Windows shape has
    its own test.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.setattr(envpush, "_on_windows", lambda: False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "into here").mkdir()

    rc = cmd_env_pull(_margs(path="into here", output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "'--output=into here'" in err
    # the property that actually matters: it parses back to the destination the user named.
    assert _parse_as_cli(_hint_command(err)).output == "into here"


def _hint_command(err: str) -> str:
    """The suggested command from the hint line, as the user would copy it."""
    for line in err.splitlines():
        if " env pull " in line and "hint" in line:
            return line[line.index(f"{CLI_NAME} env pull ") :]
    raise AssertionError(f"no hint command in:\n{err}")


def _split_as_shell(command: str) -> list[str]:
    """Split a suggested command the way the shell it was quoted FOR would.

    `_quote_shell_token` picks its quoting per platform, so the split has to match: a native
    Windows run emits an unquoted `--output=C:\\Users\\...` whenever the path holds no space, and
    POSIX `shlex.split` reads those backslashes as escapes and deletes them -- turning a correct
    hint into `--output=C:Usersinto-here` and failing the assertion. Non-POSIX mode
    keeps them, but also keeps the surrounding quotes `list2cmdline` adds, so strip those back off.
    """
    if not envpush._on_windows():
        return shlex.split(command)
    return [token.strip('"') for token in shlex.split(command, posix=False)]


def _parse_as_cli(command: str) -> Namespace:
    """Round-trip a suggested command through a shell split and the real `env pull` parser.

    A hint is only useful if pasting it back works, so assert against the parser the user would
    actually hit rather than against the string we happened to build.
    """
    from flash.cli.parsing.main import _build_parser

    argv = _split_as_shell(command)
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


@pytest.mark.parametrize("positional", ["../into-here", "sub/../../into-here"])
def test_cmd_env_pull_parent_relative_dir_is_diagnosed(monkeypatch, tmp_path, capsys, positional):
    """An upward-traversing positional cannot name anything inside the environment.

    `_safe_repo_relative_path` rejects every component of `..`, so there is no in-env reading to
    protect and no ambiguity for the multi-component gate to resolve. Suppressing the hint here
    downloaded the package only to fail with an invalid environment path, instead of explaining
    `--output`.

    The second form matters because the rejection is on the parts, not on a leading `..`:
    `sub/../../into-here` is equally impossible as an env path.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    work = tmp_path / "work"
    (work / "sub").mkdir(parents=True)
    (tmp_path / "into-here").mkdir()
    monkeypatch.chdir(work)

    rc = cmd_env_pull(_margs(path=positional, output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "drop the second positional" in err, err
    # the destination the user named, not one resolved to somewhere else on disk.
    parsed = _parse_as_cli(_hint_command(err))
    assert parsed.output == positional
    assert parsed.path is None


@pytest.mark.parametrize(
    "dest", [r"C:\Users\runner\into-here", r"C:\Program Files\into here", "-dest", "into-here"]
)
def test_windows_quoted_token_survives_the_split_used_to_check_it(monkeypatch, dest):
    """A token quoted for Windows must come back out of `_split_as_shell` unchanged.

    Asserted on the quote/split pair rather than through `cmd_env_pull`, because the command can
    never carry a backslash on this platform: it normalizes `\\` to `/` on the way in, and a POSIX
    `Path` renders the destination with forward slashes regardless. So a hint built from a real
    Windows path is only reachable in the unit, and a round-trip test driven through the CLI passes
    identically with the split fixed and broken.

    `list2cmdline` quotes only when it has to, so an ordinary `C:\\Users\\...` comes back bare;
    reading that with POSIX rules treats each backslash as an escape and deletes it, mangling a
    correct hint into `C:Usersinto-here`. The spaced form takes the other branch --
    quoted, which POSIX mode would strip correctly but non-POSIX mode leaves in place -- so both
    halves of the split are pinned by the same assertion.
    """
    monkeypatch.setattr(envpush, "_on_windows", lambda: True)

    token = envpush._quote_shell_token(f"--output={dest}")

    assert _split_as_shell(f"{CLI_NAME} env pull ns/env {token}")[4:] == [f"--output={dest}"]


def test_cmd_env_pull_hint_omits_a_command_cmd_exe_would_mangle(monkeypatch, tmp_path, capsys):
    """On Windows a destination holding cmd.exe metacharacters gets no copy-pasteable command.

    `list2cmdline` implements MS C-runtime argv quoting, not cmd.exe escaping, so `foo&bar` comes
    back unquoted and pasting the hint would run `bar` as a separate command. Emitting no command is
    strictly better than emitting one that executes unintended text.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.setattr(envpush, "_on_windows", lambda: True)
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
    monkeypatch.setattr(envpush, "_on_windows", lambda: True)
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


def test_cmd_env_pull_whole_env_preserves_toml_configs(monkeypatch, tmp_path):
    sft_config = b"[training]\nalgorithm = 'sft'\n"
    opd_config = b"[teacher]\nthinking = true\n"
    _patch_client(
        monkeypatch,
        _package_tarball(
            {
                "environment.py": b"# env\n",
                "configs/sft.toml": sft_config,
                "configs/nested/opd.toml": opd_config,
            }
        ),
    )
    dest = tmp_path / "stuff"

    rc = cmd_env_pull(_margs(output=str(dest)))

    assert rc == 0
    assert (dest / "configs" / "sft.toml").read_bytes() == sft_config
    assert (dest / "configs" / "nested" / "opd.toml").read_bytes() == opd_config


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
    assert environment_local_dirname("david-freesolo-co/my-project/stuff") == "stuff"
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
        adapter.managed_slug_to_github_ref("david-freesolo-co/my-project/stuff")
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

    # the walk descends one non-recursive tree per slug segment, then lists the package root
    # recursively. the recursive listing is rooted at the ENVIRONMENT, so a sibling environment
    # in the same project is never enumerated -- `sibling-env` below must stay unvisited.
    trees = {
        ("a" * 40, False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "david-freesolo-co", "sha": "namespace-sha"}],
        },
        ("namespace-sha", False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "my-project", "sha": "project-sha"}],
        },
        ("project-sha", False): {
            "truncated": False,
            "tree": [
                {"type": "tree", "path": "stuff", "sha": "env-sha"},
                {"type": "tree", "path": "sibling-env", "sha": "sibling-sha"},
            ],
        },
        ("env-sha", True): {
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
        "david-freesolo-co/my-project/stuff/environment.py": b"# env\n",
        "david-freesolo-co/my-project/stuff/datasets/train.jsonl": b'{"a":1}\n',
        "david-freesolo-co/my-project/stuff/bin/run-helper": b"#!/bin/sh\n",
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

    env_file = Path(adapter._resolve_environment_reference("david-freesolo-co/my-project/stuff"))

    assert env_file.read_bytes() == b"# env\n"
    assert (env_file.parent / "datasets" / "train.jsonl").read_bytes() == b'{"a":1}\n'
    assert stat.S_IMODE((env_file.parent / "bin" / "run-helper").stat().st_mode) == 0o755
    assert not (env_file.parents[2] / "other-org").exists()
    assert all("other-org" not in url for url in seen_urls)
    # a sibling environment in the same project is neither fetched nor written: the package root
    # is the environment directory, not the project directory holding every environment.
    assert not (env_file.parent.parent / "sibling-env").exists()
    assert all("sibling" not in url for url in seen_urls)


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
            "tree": [{"type": "tree", "path": "my-project", "sha": "project-sha"}],
        },
        ("project-sha", False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "stuff", "sha": "env-sha"}],
        },
        ("env-sha", True): {
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
        "david-freesolo-co/my-project/stuff/environment.py": b"# env\n",
        "david-freesolo-co/my-project/stuff/datasets/train.jsonl": b'{"a":1}\n',
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
            "github:freesolo-co/environment-hub@main:david-freesolo-co/my-project/stuff/environment.py"
        )
    )

    assert env_file.read_bytes() == b"# env\n"
    assert (env_file.parent / "datasets" / "train.jsonl").read_bytes() == b'{"a":1}\n'
    assert not (env_file.parents[2] / "shared").exists()


def test_environment_hub_github_ref_requires_package_path(monkeypatch):
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)

    with pytest.raises(ValueError, match="namespace/project/name"):
        adapter._resolve_environment_reference("github:freesolo-co/environment-hub@main")


def test_github_tree_url_encodes_treeish_path_segment():
    ref = adapter.GitHubEnvironmentRef(
        "freesolo-co",
        "environment-hub",
        "a" * 40,
        "david-freesolo-co/my-project/stuff/environment.py",
    )

    url = adapter._github_tree_url(
        ref, "a" * 40 + ":david-freesolo-co/my-project/stuff", recursive=True
    )

    assert url.endswith("a" * 40 + "%3Adavid-freesolo-co%2Fmy-project%2Fstuff?recursive=1")
    assert "/stuff" not in url.split("/git/trees/", 1)[1]


def test_download_github_directory_handles_large_tree_listing(monkeypatch, tmp_path):
    ref = adapter.GitHubEnvironmentRef(
        "freesolo-co",
        "environment-hub",
        "b" * 40,
        "david-freesolo-co/my-project/big/environment.py",
    )
    shard_count = 1001
    trees = {
        ("b" * 40, False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "david-freesolo-co", "sha": "namespace-sha"}],
        },
        ("namespace-sha", False): {
            "truncated": False,
            "tree": [{"type": "tree", "path": "my-project", "sha": "project-sha"}],
        },
        ("project-sha", False): {
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

    repo_root = adapter._download_github_directory(
        ref, "david-freesolo-co/my-project/big", tmp_path
    )

    assert (
        repo_root / "david-freesolo-co/my-project/big/environment.py"
    ).read_bytes() == b"# env\n"
    assert (repo_root / "david-freesolo-co/my-project/big/shard-1000.jsonl").read_bytes() == b"x"


def test_download_github_directory_surfaces_tree_error_message(monkeypatch, tmp_path):
    ref = adapter.GitHubEnvironmentRef(
        "freesolo-co",
        "environment-hub",
        "c" * 40,
        "david-freesolo-co/my-project/missing/environment.py",
    )

    def fake_urlopen(req, timeout=None, max_bytes=None, out=None):
        return json.dumps({"message": "Not Found"}).encode()

    monkeypatch.setattr(adapter, "_urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Not Found"):
        adapter._download_github_directory(ref, "david-freesolo-co/my-project/missing", tmp_path)


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


def _github_env_tarball(content: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="repo-sha/environment.py")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def test_resolve_github_env_ignores_symlinked_cache_entry(monkeypatch, tmp_path):
    # a cache root that was previously group/other-writable can still hold a symlink planted
    # by another local account at this guessable cache key (a sha of the ref) even after the
    # root's own permissions are repaired; the cache-hit path must never follow it.
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)

    monkeypatch.setattr(
        adapter, "_download_github_tarball", lambda ref: _github_env_tarball(b"# original\n")
    )
    first = adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")
    assert first.read_bytes() == b"# original\n"
    cache_dir = first.parent

    # plant a symlink at the now-known cache_dir, standing in for another account's foreign
    # content (this run's own uid would never have written a symlink there).
    shutil.rmtree(cache_dir)
    evil = tmp_path / "evil"
    evil.mkdir()
    (evil / "environment.py").write_bytes(b"evil\n")
    cache_dir.symlink_to(evil, target_is_directory=True)

    monkeypatch.setattr(
        adapter, "_download_github_tarball", lambda ref: _github_env_tarball(b"# refreshed\n")
    )
    second = adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")

    assert second.read_bytes() == b"# refreshed\n"


def test_resolve_github_env_replaces_a_file_squatting_at_the_cache_key(monkeypatch, tmp_path):
    # a cache entry is a directory. a regular FILE at the key (manual corruption, an interrupted
    # write) is owned by us, so the ownership check trusts it and leaves it in place; the
    # download path's rmtree(ignore_errors=True) then swallows NotADirectoryError and copytree
    # dies with FileExistsError on this key on every run.
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)
    monkeypatch.setattr(
        adapter, "_download_github_tarball", lambda ref: _github_env_tarball(b"# original\n")
    )
    first = adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")
    cache_dir = first.parent
    shutil.rmtree(cache_dir)
    cache_dir.write_bytes(b"not a directory")

    monkeypatch.setattr(
        adapter, "_download_github_tarball", lambda ref: _github_env_tarball(b"# refreshed\n")
    )
    second = adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")

    assert second.read_bytes() == b"# refreshed\n"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only uid check")
def test_resolve_github_env_ignores_foreign_owned_cache_entry(monkeypatch, tmp_path):
    # same guessable-cache-key hazard as the symlink case, but the planted entry is a real
    # directory under a different owner rather than a symlink.
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)

    monkeypatch.setattr(
        adapter, "_download_github_tarball", lambda ref: _github_env_tarball(b"# original\n")
    )
    first = adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")
    cache_dir = first.parent
    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        result = real_lstat(path, *args, **kwargs)
        if Path(path) in (cache_dir, first):
            result = os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    result.st_uid + 1,
                    result.st_gid,
                    result.st_size,
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )
        return result

    monkeypatch.setattr(adapter.os, "lstat", fake_lstat)
    monkeypatch.setattr(
        adapter, "_download_github_tarball", lambda ref: _github_env_tarball(b"# refreshed\n")
    )

    second = adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")

    assert second.read_bytes() == b"# refreshed\n"


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only uid check")
def test_resolve_github_env_refuses_unremovable_foreign_cache_entry(monkeypatch, tmp_path):
    # the planted entry is foreign-owned AND cannot be deleted by us -- its contents are
    # readable but not writable, so rmtree fails partway. best-effort removal would swallow
    # that, download the environment anyway, and then die in copytree on the entry still
    # sitting there, every single run. refuse the key before the download, and never fall
    # back to importing what the ownership check just rejected.
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)
    monkeypatch.setattr(
        adapter, "_download_github_tarball", lambda ref: _github_env_tarball(b"# original\n")
    )
    first = adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")
    cache_dir = first.parent
    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        result = real_lstat(path, *args, **kwargs)
        if Path(path) in (cache_dir, first):
            fields = list(result)
            fields[4] = result.st_uid + 1
            return os.stat_result(tuple(fields))
        return result

    real_rmtree = shutil.rmtree

    def refuse_rmtree(path, *args, **kwargs):
        # honours ignore_errors, so the stub reproduces the pre-fix shape too: best-effort
        # removal returns quietly and leaves the entry behind, rather than reporting failure.
        if Path(path) == cache_dir:
            if kwargs.get("ignore_errors"):
                return None
            raise PermissionError(13, "Permission denied", str(cache_dir))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(adapter.os, "lstat", fake_lstat)
    monkeypatch.setattr(adapter.shutil, "rmtree", refuse_rmtree)
    downloaded = []
    monkeypatch.setattr(
        adapter,
        "_download_github_tarball",
        lambda ref: downloaded.append(ref) or _github_env_tarball(b"# refreshed\n"),
    )

    with pytest.raises(RuntimeError, match="could not be removed"):
        adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")

    assert downloaded == []
    # refused, not trusted: the rejected entry is still there and still never imported.
    assert (cache_dir / "environment.py").read_bytes() == b"# original\n"


def _report_foreign_lstat(monkeypatch, paths):
    """make os.lstat report `paths` as owned by another uid, leaving every other field alone."""
    targets = {Path(p) for p in paths}
    real_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        result = real_lstat(path, *args, **kwargs)
        if Path(path) in targets:
            fields = list(result)
            fields[4] = result.st_uid + 1
            return os.stat_result(tuple(fields))
        return result

    monkeypatch.setattr(adapter.os, "lstat", fake_lstat)


def _seed_cache_dir_without_entrypoint(monkeypatch, tmp_path):
    """resolve once to learn the cache dir, then strip the entrypoint out of it."""
    monkeypatch.setattr(adapter, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(adapter, "_resolve_ref_sha", lambda parsed, **kwargs: "a" * 40)
    monkeypatch.setattr(
        adapter, "_download_github_tarball", lambda ref: _github_env_tarball(b"# original\n")
    )
    first = adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")
    cache_dir = first.parent
    (cache_dir / "environment.py").unlink()
    return cache_dir


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only uid check")
def test_resolve_github_env_refuses_unremovable_foreign_cache_entry_without_entrypoint(
    monkeypatch, tmp_path
):
    # gating the ownership checks on the entrypoint being present left the worst case unchecked:
    # a foreign-owned, unremovable directory at this key with no environment.py inside skipped
    # trust entirely, so the resolver downloaded and then wrote INTO another account's
    # directory. an entry is vetted because it exists, not because it looks complete.
    cache_dir = _seed_cache_dir_without_entrypoint(monkeypatch, tmp_path)
    _report_foreign_lstat(monkeypatch, [cache_dir])
    real_rmtree = shutil.rmtree

    def refuse_rmtree(path, *args, **kwargs):
        if Path(path) == cache_dir:
            if kwargs.get("ignore_errors"):
                return None
            raise PermissionError(13, "Permission denied", str(cache_dir))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(adapter.shutil, "rmtree", refuse_rmtree)
    downloaded = []
    monkeypatch.setattr(
        adapter,
        "_download_github_tarball",
        lambda ref: downloaded.append(ref) or _github_env_tarball(b"# refreshed\n"),
    )

    # the same actionable error the unremovable-with-entrypoint case raises, before any download.
    with pytest.raises(RuntimeError, match="could not be removed"):
        adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")

    assert downloaded == []


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="posix-only uid check")
def test_resolve_github_env_discards_removable_foreign_cache_entry_without_entrypoint(
    monkeypatch, tmp_path
):
    # the other half of the same gap: when the foreign entry CAN be removed, the recoverable
    # path must still work -- and the removal has to happen BEFORE the download, not as the
    # best-effort rmtree that used to sit after it. that ordering is the whole point: it is what
    # turns "download into a directory another account owns" into a clean refusal or a clean
    # refetch, and it is the only observable difference here from the unvetted behavior.
    cache_dir = _seed_cache_dir_without_entrypoint(monkeypatch, tmp_path)
    (cache_dir / "planted.py").write_bytes(b"# planted\n")
    _report_foreign_lstat(monkeypatch, [cache_dir])
    existed_at_download = []
    monkeypatch.setattr(
        adapter,
        "_download_github_tarball",
        lambda ref: (
            existed_at_download.append(cache_dir.exists()) or _github_env_tarball(b"# refreshed\n")
        ),
    )

    resolved = adapter._resolve_github_environment_file("github:owner/repo@main:environment.py")

    assert existed_at_download == [False]
    assert resolved.read_bytes() == b"# refreshed\n"
    assert not (cache_dir / "planted.py").exists()


def test_cmd_env_pull_multi_component_in_env_path_is_not_a_destination(
    monkeypatch, tmp_path, capsys
):
    """`assets/config` names a path INSIDE the environment, even when it exists locally as a dir.

    Testing `is_dir()` alone treated any local directory positional as a mistaken destination, so
    this real single-file pull was refused -- and because the directory is nonempty, the hint aimed
    a whole-environment `--force` replace at a local ./assets/config the user never named. A
    destination is written as its own name or absolute; a multi-component relative path is not.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "assets" / "config").mkdir(parents=True)
    (tmp_path / "assets" / "config" / "keep.txt").write_text("mine")

    rc = cmd_env_pull(_margs(path="assets/config", output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    # it proceeds as the single-file pull it is, and fails on the real reason: the environment has
    # no such file. that is the diagnosis the user needs -- the local directory is irrelevant.
    assert "assets/config" in err
    assert "not found in package" in err
    # not diagnosed as a mistaken destination...
    assert "drop the second positional" not in err
    # ...and above all, no destructive whole-env remedy aimed at the local directory.
    assert "--force" not in err
    # the local directory the user never named is untouched.
    assert (tmp_path / "assets" / "config" / "keep.txt").read_text() == "mine"


def test_cmd_env_pull_hint_omits_a_command_powershell_would_mangle(monkeypatch, tmp_path, capsys):
    """`os.name == "nt"` does not say which shell, and powershell is the current default.

    `foo;calc` passed a cmd.exe-only metacharacter set and rendered as `--output=foo;calc`.
    Powershell splits at the semicolon and runs `calc` as its own statement, so the suggested
    remedy executes something the user never asked for.
    """
    _patch_client(monkeypatch, _package_tarball({"environment.py": b"# env\n"}))
    monkeypatch.setattr(envpush, "_on_windows", lambda: True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "foo;calc").mkdir()

    rc = cmd_env_pull(_margs(path="foo;calc", output=None, force=False))

    assert rc == 1
    err = capsys.readouterr().err
    assert "drop the second positional" in err
    # the directory is still named so the user knows which one is meant; what must not appear is a
    # runnable command carrying the unescaped separator.
    assert "foo;calc" in err
    assert "--output=foo;calc" not in err
    assert "--output=DEST" in err


def test_quote_shell_token_refuses_every_windows_shell_separator(monkeypatch):
    """The refusal set must cover both shells, since we cannot know which one receives the paste."""
    monkeypatch.setattr(envpush, "_on_windows", lambda: True)
    for token in ("foo;calc", "foo&bar", "foo|bar", "foo$(x)", "foo`x", "foo{x}", "foo,x"):
        assert envpush._quote_shell_token(f"--output={token}") == "", token
    # an ordinary destination with a space is still quoted rather than refused.
    assert envpush._quote_shell_token("--output=into here") == '"--output=into here"'
