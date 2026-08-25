"""The themed `flash --help` page (render.help_page) is the styled twin of argparse's flat
default. These tests pin both sides of the TTY gate: the styled path shows the brand banner +
grouped commands, while piped/scripted `--help` falls back to argparse's plain text (so existing
greps stay byte-for-byte). They also keep the themed help catalog in lockstep with the registered
subcommands, so a newly added command can't silently go unlisted on the help page.
"""

from __future__ import annotations

import argparse

import pytest

import flash.cli.parsing.main as cli
from flash.cli.commands.env.testing.eval import cmd_env_eval
from flash.cli.commands.ops.account import cmd_projects_create, cmd_projects_list
from flash.cli.commands.ops.catalog import cmd_models
from flash.cli.commands.ops.deploy import cmd_chat, cmd_deployments
from flash.cli.commands.ops.runs import cmd_runs, cmd_status


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


def test_grouped_command_argparse_contracts() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["models", "deployments", "--json"])
    assert args.json is True
    assert args.func is cmd_deployments

    args = parser.parse_args(["runs", "status", "run-1", "-f", "--json"])
    assert args.run_id == "run-1"
    assert args.follow is True
    assert args.json is True
    assert args.func is cmd_status

    args = parser.parse_args(["models", "chat", "run-1", "-m", "hello"])
    assert args.run_id == "run-1"
    assert args.message == "hello"
    assert args.func is cmd_chat

    args = parser.parse_args(["env", "eval", "run-1", "--suite", "math", "--concurrency", "4"])
    assert args.target == "run-1"
    assert args.suite == "math"
    assert args.concurrency == 4
    assert args.func is cmd_env_eval
    # the evaluated run names the environment, so there is no second positional to take one
    # locally. asserted here because the parser is the only place that refusal is expressed.
    with pytest.raises(SystemExit):
        parser.parse_args(["env", "eval", "run-1", "./my-env"])

    args = parser.parse_args(["projects", "list"])
    assert args.func is cmd_projects_list

    args = parser.parse_args(["projects", "create", "my project"])
    assert args.name == "my project"
    assert args.func is cmd_projects_create


def test_bare_group_commands_keep_their_pre_grouping_behavior() -> None:
    """`flash models` and `flash runs` predate the grouping and must still run bare.

    deployed agents invoke bare `flash runs`, and `flash models` was the model catalog before it
    became a group, so requiring a subcommand would break both callers.
    """
    parser = cli._build_parser()

    args = parser.parse_args(["models"])
    assert args.func is cmd_models

    args = parser.parse_args(["runs"])
    assert args.func is cmd_runs

    # the grouped forms still resolve to the same handlers.
    assert parser.parse_args(["models", "list"]).func is cmd_models
    assert parser.parse_args(["runs", "list"]).func is cmd_runs


def test_root_model_commands_are_not_registered() -> None:
    assert not {"deploy", "chat", "deployments", "undeploy", "export"} & _registered_subcommands()


def test_root_run_aliases_are_not_registered() -> None:
    assert not {"status", "log", "cancel", "checkpoints"} & _registered_subcommands()


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
    assert f"{cli.CLI_NAME} env push --project PROJECT_UUID --name my-env ." in out
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
    from flash.cli.ui import render

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
        ["docs: https://docs.freesolo.co"],
    )
    page.encode("ascii")  # raises if any non-ASCII glyph slipped through


def test_env_setup_project_help_states_the_noninteractive_requirement() -> None:
    """`--project` is optional only when there is a prompt to choose one.

    `_require_setup_project` hard-requires it with `--yes`, a redirected stdin, or any other
    noninteractive run. argparse renders it as an ordinary optional flag, so help that omits the
    condition sends scripted and CI callers into a failure the help text said would not happen.
    """
    # assert on the rendered page rather than the action object: the help text is only wrong if
    # what the user READS is wrong, and format_help is what they read.
    parser = cli._build_parser()
    setup = parser._subparsers._group_actions[0].choices["env"]
    page = setup._subparsers._group_actions[0].choices["setup"].format_help()

    assert "--project" in page
    project_help = page[page.index("--project") :]
    assert "required" in project_help
    assert "--yes" in project_help
