"""`flash whoami` must name the control plane it resolved, and where that came from.

The destination of every command has two halves: which key authenticates, and which control plane
that key is sent to. A config holding only an api_key resolves the URL from the channel default,
which on the release channel is production. Nothing in the output said so, so someone who believed
they were pointed at a local or dev plane got no signal until a run had already been created in the
wrong place. These tests pin the resolved plane, its source, and the single-read snapshot that
keeps the reported plane the plane actually used.
"""

from __future__ import annotations

import argparse

import pytest

import flash.cli as cli
import flash.client.config as client_config
from flash.cli.commands import cmd_whoami


def _patch_saved_key(monkeypatch, saved: str | None) -> None:
    config = {"api_key": saved} if saved else {}
    monkeypatch.setattr(client_config, "_read_config", lambda: config)


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


def test_credential_snapshot_reads_the_config_once(monkeypatch):
    """One read for all four values, so a concurrent `flash login` cannot split them.

    Resolving the URL, the key, and the URL's origin through separate reads lets whoami query one
    control plane while reporting another -- exactly the confusion this output exists to remove.
    """
    monkeypatch.delenv("FLASH_API_URL", raising=False)
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)
    reads = []

    def _counted_read():
        reads.append(1)
        return {"api_key": "fslo-saved-login", "api_url": "https://custom"}

    monkeypatch.setattr(client_config, "_read_config", _counted_read)

    api_url, api_key, key_source, url_source = client_config.credential_snapshot()
    assert len(reads) == 1
    assert api_url == "https://custom"
    assert api_key == "fslo-saved-login"
    assert key_source == str(client_config.CONFIG_PATH)
    assert url_source == str(client_config.CONFIG_PATH)


def test_credential_snapshot_reports_logged_out(monkeypatch):
    monkeypatch.delenv("FLASH_API_URL", raising=False)
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)
    _patch_saved_key(monkeypatch, None)

    _api_url, api_key, key_source, url_source = client_config.credential_snapshot()
    assert api_key is None
    assert key_source is None
    # the plane is still resolvable when logged out, and still defaults to the channel's url.
    assert client_config.CHANNEL in url_source


def test_whoami_names_the_control_plane_it_resolved(monkeypatch, capsys):
    """The destination has two halves: which key, and which control plane.

    Naming only the key still lets someone believe they are pointed at a local or dev plane
    while every command lands in production.
    """
    monkeypatch.setattr(
        cli.commands,
        "credential_snapshot",
        lambda: (
            "https://flash.freesolo.co",
            "fslo-key",
            "FREESOLO_API_KEY",
            "default for the release channel",
        ),
    )

    class _FakeClient:
        def me(self):
            return {"kind": "freesolo_api_key", "key_prefix": "fslo-ke", "email": "me@x.co"}

    monkeypatch.setattr(cli.commands, "ApiClient", lambda *a, **k: _FakeClient())

    assert cmd_whoami(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "https://flash.freesolo.co" in out
    assert "default for the release channel" in out
    assert "me@x.co" in out


def test_whoami_fails_clearly_when_logged_out(monkeypatch):
    """whoami builds its own client, so it owns the logged-out message client_from_config gave."""
    from flash.client import ClientError

    monkeypatch.setattr(
        cli.commands,
        "credential_snapshot",
        lambda: ("https://flash.freesolo.co", None, None, "default for the release channel"),
    )

    with pytest.raises(ClientError, match="not logged in"):
        cmd_whoami(argparse.Namespace())
