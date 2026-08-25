"""An ambient FREESOLO_API_KEY must not silently redirect writes to another organization.

``~/.flash/config.json`` is shared mutable state and the env var wins over it, so an inherited or
exported key can point a command at a different org than the one the user logged into. Both keys
authenticate, so nothing fails: the run, project, or deployment just lands in the other org. The
existing 401/403 hint in ApiClient cannot cover this, because there is no auth failure to hang it
on. These tests pin the proactive warning that does.
"""

from __future__ import annotations

import argparse

import pytest

import flash.cli.commands.ops.account as cli_account
import flash.cli.parsing.main as cli
import flash.client.config as client_config
from flash.cli.commands.env.ops.setup import cmd_env_setup
from flash.cli.commands.env.testing.eval import cmd_env_eval
from flash.cli.commands.ops.account import cmd_whoami
from flash.cli.commands.ops.traces import cmd_traces_export
from flash.cli.commands.ops.train import cmd_train


def _patch_saved_key(monkeypatch, saved: str | None) -> None:
    config = {"api_key": saved} if saved else {}
    monkeypatch.setattr(client_config, "_read_config", lambda: config)


def test_shadowed_login_warns_on_genuine_mismatch(monkeypatch):
    _patch_saved_key(monkeypatch, "fslo-saved-login")
    monkeypatch.setenv("FREESOLO_API_KEY", "fslo-other-org")

    warning = client_config.shadowed_login_warning()
    assert warning is not None
    assert "FREESOLO_API_KEY" in warning
    assert "flash whoami" in warning

    # the dev channel installs the executable as `flash-dev`, so a hardcoded `flash whoami` names a
    # binary that is not installed. asserting the substring alone would not catch it: "flash whoami"
    # is a substring of "flash-dev whoami" only in the other direction, so pin the interpolation.
    monkeypatch.setattr(client_config, "CLI_NAME", "flash-dev")
    assert "flash-dev whoami" in client_config.shadowed_login_warning()


def test_shadowed_login_silent_when_key_matches_saved_login(monkeypatch):
    # `source .env` with the same key is the common case; warning on it would train users to
    # ignore the warning that matters.
    _patch_saved_key(monkeypatch, "fslo-saved-login")
    monkeypatch.setenv("FREESOLO_API_KEY", "fslo-saved-login")

    assert client_config.shadowed_login_warning() is None


def test_shadowed_login_silent_without_env_key(monkeypatch):
    _patch_saved_key(monkeypatch, "fslo-saved-login")
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)

    assert client_config.shadowed_login_warning() is None


def test_shadowed_login_silent_when_never_logged_in(monkeypatch):
    # env var only, no saved login: nothing is being shadowed, so this is the intended setup
    # (ci, containers) and must stay quiet.
    _patch_saved_key(monkeypatch, None)
    monkeypatch.setenv("FREESOLO_API_KEY", "fslo-ci-key")

    assert client_config.shadowed_login_warning() is None


def test_org_binding_command_warns(monkeypatch, capsys):
    monkeypatch.setattr(cli, "shadowed_login_warning", lambda: "shadowed!")
    args = argparse.Namespace(func=cmd_train)

    cli._warn_if_login_shadowed(args)

    assert "shadowed!" in capsys.readouterr().err


def test_train_cost_stays_quiet(monkeypatch, capsys):
    # `train --cost` is catalog-only and never contacts an org, so warning that it runs against the
    # environment key's organization would describe a request the command does not make.
    monkeypatch.setattr(cli, "shadowed_login_warning", lambda: "shadowed!")
    args = argparse.Namespace(func=cmd_train, cost=True)

    cli._warn_if_login_shadowed(args)

    assert capsys.readouterr().err == ""


def test_read_only_command_stays_quiet(monkeypatch, capsys):
    # `flash whoami` names the key source in its own output, so a second warning is noise.
    monkeypatch.setattr(cli, "shadowed_login_warning", lambda: "shadowed!")
    args = argparse.Namespace(func=cmd_whoami)

    cli._warn_if_login_shadowed(args)

    assert capsys.readouterr().err == ""


def test_every_org_binding_command_is_registered():
    # a new org-binding command must opt in explicitly; this pins the current set so an addition
    # that forgets the warning shows up as a failing test rather than a silent wrong-org write.
    names = {command.__name__ for command in cli._ORG_BINDING_COMMANDS}
    assert names == {
        "cmd_train",
        "cmd_deploy",
        "cmd_undeploy",
        "cmd_export",
        "cmd_cancel",
        "cmd_projects_create",
        "cmd_env_push",
        "cmd_env_delete",
        "cmd_traces_export",
        "cmd_env_setup",
        "cmd_env_eval",
    }


@pytest.mark.parametrize("upload", [False, True])
def test_env_eval_warns_whether_or_not_it_uploads(monkeypatch, capsys, upload):
    # gating this on whether results are recorded treats a --no-upload eval as local, but every case
    # still makes an authenticated chat_stream request and the target's accessibility is resolved
    # from the ambient key. so such an eval can report the target inaccessible, or spend the whole
    # suite against the unintended org, with no warning at all.
    monkeypatch.setattr(cli, "shadowed_login_warning", lambda: "shadowed!")

    cli._warn_if_login_shadowed(argparse.Namespace(func=cmd_env_eval, upload=upload))
    assert "shadowed!" in capsys.readouterr().err


@pytest.mark.parametrize("handler", [cmd_traces_export, cmd_env_setup])
def test_project_scaffolding_command_warns(monkeypatch, capsys, handler):
    # neither writes to the org remotely, but both resolve a project with the ambient key and land
    # it in the working tree: dataset/train.jsonl for the export, the project uuid in the generated
    # configs for setup. by the time a later `train` warns, the wrong org is already scaffolded in.
    monkeypatch.setattr(cli, "shadowed_login_warning", lambda: "shadowed!")
    args = argparse.Namespace(func=handler)

    cli._warn_if_login_shadowed(args)

    assert "shadowed!" in capsys.readouterr().err


def test_whoami_reports_the_key_source(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_account,
        "load_credentials_with_source",
        lambda: ("https://flash.freesolo.co", "fslo-key", "FREESOLO_API_KEY"),
    )

    class _FakeClient:
        def me(self):
            return {"kind": "freesolo_api_key", "key_prefix": "fslo-ke", "email": "me@x.co"}

    monkeypatch.setattr(cli_account, "client_from_config", lambda: _FakeClient())

    assert cmd_whoami(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "FREESOLO_API_KEY" in out
    assert "me@x.co" in out
