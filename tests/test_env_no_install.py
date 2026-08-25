"""`flash env install` (and its local manifest) was removed.

Only setup/list/push remain under `flash env`. The list reports published environments and local
sources, with no "installed" manifest section.
"""

from __future__ import annotations

import argparse

import pytest

import flash.cli.commands.env.ops.list as env_list_commands
from flash._internal.channel import CLI_NAME
from flash.cli.commands.env.ops.list import cmd_env_list
from flash.cli.parsing.main import _build_parser


def _no_published(monkeypatch):
    monkeypatch.setattr(env_list_commands, "_published_envs", lambda: ([], None))


def test_env_install_subcommand_is_gone():
    parser = _build_parser()
    with pytest.raises(SystemExit):  # argparse: invalid choice 'install'
        parser.parse_args(["env", "install", "owner/name"])


def test_remaining_env_subcommands_still_parse():
    parser = _build_parser()
    assert parser.parse_args(["env", "setup"]).env_cmd == "setup"
    assert parser.parse_args(["env", "list"]).env_cmd == "list"
    assert (
        parser.parse_args(
            [
                "env",
                "push",
                "--project",
                "11111111-1111-4111-8111-111111111111",
                "--name",
                "x",
                ".",
            ]
        ).env_cmd
        == "push"
    )


def test_install_manifest_machinery_is_removed():
    import flash.envs.loading.base as registry

    for gone in (
        "record_installed_env",
        "list_installed_environments",
        "load_installed_manifest",
        "INSTALLED_MANIFEST",
    ):
        assert not hasattr(registry, gone), f"{gone} should have been removed"


def test_env_list_reports_local_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(
        "FLASH_STYLE", "0"
    )  # force the plain renderer so substring asserts are stable
    monkeypatch.chdir(tmp_path)
    _no_published(monkeypatch)
    (tmp_path / "environment.py").write_text("# env\n")
    rc = cmd_env_list(argparse.Namespace())
    assert rc == 0
    out = capsys.readouterr().out
    assert "local env sources" in out
    assert "." in out
    assert f"{CLI_NAME} env push --project <project-uuid> --name <name> <path>" in out
    assert "installed environments" not in out  # the removed manifest section


def test_env_list_empty_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(
        "FLASH_STYLE", "0"
    )  # force the plain renderer so substring asserts are stable
    monkeypatch.chdir(tmp_path)
    _no_published(monkeypatch)
    rc = cmd_env_list(argparse.Namespace())
    assert rc == 0
    assert "no environments yet" in capsys.readouterr().out
