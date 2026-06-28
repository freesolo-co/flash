"""`flash env install` (and its local manifest) was removed.

Only setup/list/push remain under `flash env`, and `flash env list` now reports local
environment sources only (no "installed" manifest section).
"""

from __future__ import annotations

import argparse

import pytest

from flash.cli import _build_parser
from flash.cli.commands import cmd_env_list


def test_env_install_subcommand_is_gone():
    parser = _build_parser()
    with pytest.raises(SystemExit):  # argparse: invalid choice 'install'
        parser.parse_args(["env", "install", "owner/name"])


def test_remaining_env_subcommands_still_parse():
    parser = _build_parser()
    assert parser.parse_args(["env", "setup"]).env_cmd == "setup"
    assert parser.parse_args(["env", "list"]).env_cmd == "list"
    assert parser.parse_args(["env", "push", "--name", "x", "."]).env_cmd == "push"


def test_install_manifest_machinery_is_removed():
    import flash.envs.registry as registry

    for gone in (
        "record_installed_env",
        "list_installed_environments",
        "load_installed_manifest",
        "INSTALLED_MANIFEST",
    ):
        assert not hasattr(registry, gone), f"{gone} should have been removed"


def test_env_list_reports_local_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLASH_STYLE", "0")  # force the plain renderer so substring asserts are stable
    monkeypatch.chdir(tmp_path)
    (tmp_path / "environment.py").write_text("# env\n")
    rc = cmd_env_list(argparse.Namespace())
    assert rc == 0
    out = capsys.readouterr().out
    assert "local env sources" in out
    assert "." in out
    assert "installed environments" not in out  # the removed manifest section


def test_env_list_empty_hint(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLASH_STYLE", "0")  # force the plain renderer so substring asserts are stable
    monkeypatch.chdir(tmp_path)
    rc = cmd_env_list(argparse.Namespace())
    assert rc == 0
    assert "no environments yet" in capsys.readouterr().out
