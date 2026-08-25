"""Regression coverage for ``flash env list`` reporting published environments."""

from __future__ import annotations

import argparse

import pytest

import flash.cli.commands.env.ops.list as commands
from flash.client import ApiClient, ClientError


@pytest.fixture(autouse=True)
def plain_renderer(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_STYLE", "0")
    monkeypatch.chdir(tmp_path)


def _logged_in(monkeypatch, published, *, error: Exception | None = None):
    monkeypatch.setattr(commands, "load_credentials", lambda: ("https://plane.example", "key"))

    class Client:
        def list_envs(self):
            if error is not None:
                raise error
            return published

    monkeypatch.setattr("flash.client.client_from_config", Client)


def test_published_environment_is_not_reported_as_empty(monkeypatch, capsys):
    _logged_in(monkeypatch, ["acme/project/my-env"])

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    output = capsys.readouterr().out
    assert "acme/project/my-env" in output
    assert "no environments yet" not in output


def test_published_environment_flows_from_endpoint_to_cli(monkeypatch, capsys):
    import flash.server.domain.registry.envs as domain
    from flash.server.routes import envs as routes

    monkeypatch.setattr(domain, "list_namespace_slugs", lambda **_kwargs: ["acme/project/my-env"])
    monkeypatch.setattr(commands, "load_credentials", lambda: ("https://plane.example", "key"))
    client = ApiClient(api_url="https://plane.example", api_key="key")

    def bridge_request(_method, _path, **_kwargs):
        return routes.list_envs(key={"org_slug": "acme"})

    monkeypatch.setattr(client, "_request", bridge_request)
    monkeypatch.setattr("flash.client.client_from_config", lambda: client)

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    output = capsys.readouterr().out
    assert "acme/project/my-env" in output
    assert "no environments yet" not in output


def test_failed_lookup_is_distinct_from_an_empty_hub(monkeypatch, capsys):
    _logged_in(monkeypatch, [], error=ClientError("control plane is unreachable"))

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    output = capsys.readouterr().out
    assert "published environments unavailable: control plane is unreachable" in output
    assert "no environments yet" in output


def test_empty_hub_and_no_local_sources_shows_empty_hint(monkeypatch, capsys):
    _logged_in(monkeypatch, [])

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    assert "no environments yet" in capsys.readouterr().out


def test_local_and_published_sources_are_both_reported(monkeypatch, capsys, tmp_path):
    (tmp_path / "environment.py").write_text("# env\n")
    _logged_in(monkeypatch, ["acme/project/my-env"])

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    output = capsys.readouterr().out
    assert "acme/project/my-env" in output
    assert "local env sources" in output


def test_custom_plane_is_asked_instead_of_rejected_client_side(monkeypatch, capsys):
    monkeypatch.setattr(commands, "load_credentials", lambda: ("http://localhost:8000", "key"))

    class Client:
        def list_envs(self):
            return ["acme/project/my-env"]

    monkeypatch.setattr("flash.client.client_from_config", Client)

    commands.cmd_env_list(argparse.Namespace())

    assert "acme/project/my-env" in capsys.readouterr().out
