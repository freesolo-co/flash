"""An ambient FREESOLO_API_KEY must not silently redirect writes to another organization.

``~/.flash/config.json`` is shared mutable state and the env var wins over it, so an inherited or
exported key can point a command at a different org than the one the user logged into. Both keys
authenticate, so nothing fails: the run, project, or deployment just lands in the other org. The
existing 401/403 hint in ApiClient cannot cover this, because there is no auth failure to hang it
on. These tests pin the proactive warning that does.
"""

from __future__ import annotations

import argparse

import flash.cli as cli
import flash.client.config as client_config
from flash.cli.commands import cmd_train, cmd_whoami


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


def test_org_mutating_command_warns(monkeypatch, capsys):
    monkeypatch.setattr(cli, "shadowed_login_warning", lambda: "shadowed!")
    args = argparse.Namespace(func=cmd_train)

    cli._warn_if_login_shadowed(args)

    assert "shadowed!" in capsys.readouterr().err


def test_read_only_command_stays_quiet(monkeypatch, capsys):
    # `flash whoami` names the key source in its own output, so a second warning is noise.
    monkeypatch.setattr(cli, "shadowed_login_warning", lambda: "shadowed!")
    args = argparse.Namespace(func=cmd_whoami)

    cli._warn_if_login_shadowed(args)

    assert capsys.readouterr().err == ""


def test_every_org_mutating_command_is_registered():
    # a new write command must opt in explicitly; this pins the current set so an addition that
    # forgets the warning shows up as a failing test rather than a silent wrong-org write.
    names = {command.__name__ for command in cli._ORG_MUTATING_COMMANDS}
    assert names == {
        "cmd_train",
        "cmd_deploy",
        "cmd_undeploy",
        "cmd_export",
        "cmd_cancel",
        "cmd_projects_create",
        "cmd_env_push",
        "cmd_env_delete",
    }


def test_whoami_reports_the_key_source(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.commands,
        "load_credentials_with_source",
        lambda: ("https://flash.freesolo.co", "fslo-key", "FREESOLO_API_KEY"),
    )

    class _FakeClient:
        def me(self):
            return {"kind": "freesolo_api_key", "key_prefix": "fslo-ke", "email": "me@x.co"}

    monkeypatch.setattr(cli.commands, "client_from_config", lambda: _FakeClient())

    assert cmd_whoami(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "FREESOLO_API_KEY" in out
    assert "me@x.co" in out


def test_api_url_source_names_the_channel_default(monkeypatch):
    # a config holding only an api_key resolves to the channel default, which on the release
    # channel is production. the source string has to say so rather than look configured.
    monkeypatch.delenv("FLASH_API_URL", raising=False)
    _patch_saved_key(monkeypatch, "fslo-saved-login")

    source = client_config.api_url_source()
    assert "default" in source
    assert client_config.CHANNEL in source


def test_api_url_source_names_the_env_override(monkeypatch):
    monkeypatch.setenv("FLASH_API_URL", "http://127.0.0.1:8000")

    assert client_config.api_url_source() == "FLASH_API_URL"


def test_api_url_source_names_the_config_file(monkeypatch):
    monkeypatch.delenv("FLASH_API_URL", raising=False)
    monkeypatch.setattr(
        client_config, "_read_config", lambda: {"api_key": "k", "api_url": "https://custom"}
    )

    assert client_config.api_url_source() == str(client_config.CONFIG_PATH)


def test_whoami_names_the_control_plane_it_resolved(monkeypatch, capsys):
    """The destination has two halves: which key, and which control plane.

    Naming only the key still lets someone believe they are pointed at a local or dev plane
    while every command lands in production.
    """
    monkeypatch.setattr(
        cli.commands,
        "load_credentials_with_source",
        lambda: ("https://flash.freesolo.co", "fslo-key", "FREESOLO_API_KEY"),
    )
    monkeypatch.setattr(cli.commands, "api_url_source", lambda: "default for the release channel")

    class _FakeClient:
        def me(self):
            return {"kind": "freesolo_api_key", "key_prefix": "fslo-ke", "email": "me@x.co"}

    monkeypatch.setattr(cli.commands, "client_from_config", lambda: _FakeClient())

    assert cmd_whoami(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "https://flash.freesolo.co" in out
    assert "default for the release channel" in out
