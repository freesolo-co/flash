"""The themed `flash --help` page (render.help_page) is the styled twin of argparse's flat
default. These tests pin both sides of the TTY gate: the styled path shows the brand banner +
grouped commands, while piped/scripted `--help` falls back to argparse's plain text (so existing
greps stay byte-for-byte). They also keep the themed help catalog in lockstep with the registered
subcommands, so a newly added command can't silently go unlisted on the help page.
"""

from __future__ import annotations

import argparse

import pytest

import flash.cli as cli


def _registered_subcommands() -> set[str]:
    """The subcommand names argparse actually registers on the root parser."""
    parser = cli._build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return set(sub.choices)


def _registered_env_subcommands() -> set[str]:
    """The nested environment commands argparse registers under `flash env`."""
    parser = cli._build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    env_parser = sub.choices["env"]
    env_sub = next(a for a in env_parser._actions if isinstance(a, argparse._SubParsersAction))
    return {f"env {name}" for name in env_sub.choices}


def _catalog_commands() -> set[str]:
    """The root commands represented by the themed help grid (_HELP_GROUPS).

    Rows may show nested commands such as `env push`, but argparse only registers `env`
    at the root level.
    """
    return {cmd.split()[0] for _, rows in cli._HELP_GROUPS for cmd, _ in rows}


def _catalog_rows() -> set[str]:
    """The display rows listed in the themed help grid (_HELP_GROUPS)."""
    return {cmd for _, rows in cli._HELP_GROUPS for cmd, _ in rows}


def _catalog_env_rows() -> set[str]:
    """The nested environment rows listed in the themed help grid."""
    return {cmd for cmd in _catalog_rows() if cmd.startswith("env ")}


def test_help_catalog_matches_registered_subcommands() -> None:
    """Every real subcommand appears once in the themed help, and the help lists no command that
    isn't real — so the grouped `flash --help` can't drift from the actual CLI surface."""
    registered = _registered_subcommands()
    catalog = _catalog_commands()
    assert catalog == registered, (
        f"themed help is out of sync with the CLI:\n"
        f"  missing from help: {sorted(registered - catalog)}\n"
        f"  listed but not a command: {sorted(catalog - registered)}"
    )
    # no command listed under two groups
    flat = [cmd for _, rows in cli._HELP_GROUPS for cmd, _ in rows]
    assert len(flat) == len(set(flat)), "a command is listed under more than one help group"


def test_help_catalog_matches_registered_env_subcommands() -> None:
    assert _catalog_env_rows() == _registered_env_subcommands()


def test_help_styled_is_themed_and_exits_zero(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")  # layout kept, color dropped — assert on contiguous text

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0  # argparse's --help exit flow is preserved

    out = capsys.readouterr().out
    assert "managed LoRA post-training" in out  # themed banner (lowercase) vs argparse description
    for title in ("getting started", "catalog", "environments", "training", "serving & export"):
        assert title in out  # grouped, not a flat dump
    for cmd in _catalog_rows():
        assert cmd in out
    assert f"{cli.CLI_NAME} env setup" in out
    assert f"{cli.CLI_NAME} env push --name my-env ." in out
    assert "usage:" in out
    assert f"{cli.CLI_NAME} <command> --help" in out  # next-step hint


def test_help_plain_when_piped(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "0")  # not a TTY -> argparse fallback

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0

    out = capsys.readouterr().out
    # argparse's own section, never emitted by the themed page
    assert "positional arguments:" in out
    assert "getting started" not in out  # themed group titles absent on the plain path


def test_help_page_is_ascii_locale_safe(monkeypatch) -> None:
    """Forced-on theme must degrade (not raise UnicodeEncodeError) on an ASCII stdout."""
    from flash.cli import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    class _AsciiStdout:
        encoding = "ascii"

        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(render.sys, "stdout", _AsciiStdout())
    page = render.help_page(
        "managed LoRA post-training",
        f"{cli.CLI_NAME} [--debug] [-v] <command> [args]",
        cli._HELP_GROUPS,
        cli._HELP_OPTIONS,
        ["docs: https://freesolo.co/docs"],
    )
    page.encode("ascii")  # raises if any non-ASCII glyph slipped through
