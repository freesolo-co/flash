"""`flash env delete` removes a published Freesolo env through the server."""

from __future__ import annotations

import argparse

import flash.cli as cli
from flash.cli.envpush import cmd_env_delete


def _fake_client(capture: dict, *, deleted: bool = True):
    """A stand-in ApiClient that records the delete_env call and returns the server payload."""

    class _C:
        def delete_env(self, env_id):
            capture["env_id"] = env_id
            return {"id": env_id, "deleted": deleted}

    return lambda: _C()


def _args(env_id: str = "acme/env", *, yes: bool = True):
    return argparse.Namespace(env_id=env_id, yes=yes)


def test_delete_calls_client_and_returns_zero(monkeypatch, capsys):
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    rc = cmd_env_delete(_args("acme/my-env"))
    assert rc == 0
    assert cap["env_id"] == "acme/my-env"
    assert "deleted acme/my-env" in capsys.readouterr().out


def test_delete_reports_absent_env_as_success(monkeypatch, capsys):
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap, deleted=False))
    rc = cmd_env_delete(_args("acme/gone"))
    assert rc == 0
    assert "not found" in capsys.readouterr().out


def test_delete_rejects_non_env_id(monkeypatch):
    called = {"n": 0}

    class _C:
        def delete_env(self, env_id):  # pragma: no cover - must not be reached
            called["n"] += 1
            return {}

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())
    # Rejected before any network call: not a slug, a github ref (can't delete from the hub), and
    # mixed-case / whitespace that the server's lowercase-only slug validator would 400 on.
    for bad in (
        "not-an-id",
        "github:owner/repo@main:environment.py",
        "MyNs/MyEnv",
        "acme/My-Env",
        "ns/bad name",
        "acme/env/extra",
    ):
        assert cmd_env_delete(_args(bad)) == 1, bad
    assert called["n"] == 0


def test_delete_strips_and_sends_canonical_id(monkeypatch, capsys):
    # Surrounding whitespace is stripped so the id sent to the server is canonical.
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    rc = cmd_env_delete(_args("  acme/my-env  "))
    assert rc == 0
    assert cap["env_id"] == "acme/my-env"


def test_delete_aborts_on_declined_confirmation(monkeypatch):
    called = {"n": 0}

    class _C:
        def delete_env(self, env_id):  # pragma: no cover - must not be reached
            called["n"] += 1
            return {"deleted": True}

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())
    monkeypatch.setattr("builtins.input", lambda *_a: "n")
    assert cmd_env_delete(_args("acme/env", yes=False)) == 1
    assert called["n"] == 0


def test_delete_proceeds_on_confirmation(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    monkeypatch.setattr("builtins.input", lambda *_a: "y")
    assert cmd_env_delete(_args("acme/env", yes=False)) == 0
    assert cap["env_id"] == "acme/env"


def test_delete_surfaces_api_error(monkeypatch, capsys):
    from flash.client import ApiError

    class _C:
        def delete_env(self, env_id):
            raise ApiError(403, "you can only delete environments in your own namespace")

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())
    assert cmd_env_delete(_args("someone-else/env")) == 1
    assert "your own namespace" in capsys.readouterr().err


def test_delete_subcommand_dispatches_to_handler():
    parser = cli._build_parser()
    args = parser.parse_args(["env", "delete", "acme/my-env"])
    assert args.func is cmd_env_delete
    assert args.env_id == "acme/my-env"
    assert args.yes is False
    args_yes = parser.parse_args(["env", "delete", "-y", "acme/my-env"])
    assert args_yes.yes is True
