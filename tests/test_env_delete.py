"""`flash env delete` removes a published Freesolo env through the server."""

from __future__ import annotations

import argparse

import flash.cli.parsing.main as cli
from flash.cli.commands.env.ops.push import cmd_env_delete


def _fake_client(capture: dict, *, deleted: bool = True):
    """A stand-in ApiClient that records the delete_env call and returns the server payload."""

    class _C:
        def delete_env(self, env_id, *, project_id):
            capture.update(env_id=env_id, project_id=project_id)
            return {"id": env_id, "deleted": deleted}

    return lambda: _C()


def _args(
    env_id: str = "acme/checkout-bot/env",
    *,
    project: str = "11111111-1111-4111-8111-111111111111",
    yes: bool = True,
):
    return argparse.Namespace(env_id=env_id, project=project, yes=yes)


def test_delete_calls_client_and_returns_zero(monkeypatch, capsys):
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    rc = cmd_env_delete(_args("acme/checkout-bot/my-env"))
    assert rc == 0
    assert cap["env_id"] == "acme/checkout-bot/my-env"
    assert cap["project_id"] == "11111111-1111-4111-8111-111111111111"
    assert "deleted acme/checkout-bot/my-env" in capsys.readouterr().out


def test_delete_reports_absent_env_as_success(monkeypatch, capsys):
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap, deleted=False))
    rc = cmd_env_delete(_args("acme/checkout-bot/gone"))
    assert rc == 0
    assert "not found" in capsys.readouterr().out


def test_delete_rejects_non_env_id(monkeypatch):
    called = {"n": 0}

    class _C:
        def delete_env(self, env_id, *, project_id):  # pragma: no cover - must not be reached
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
        "acme/checkout-bot/env/extra",
    ):
        assert cmd_env_delete(_args(bad)) == 1, bad
    assert called["n"] == 0


def test_delete_strips_and_sends_canonical_id(monkeypatch, capsys):
    # surrounding whitespace is stripped so the id sent to the server is canonical.
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    rc = cmd_env_delete(_args("  acme/checkout-bot/my-env  "))
    assert rc == 0
    assert cap["env_id"] == "acme/checkout-bot/my-env"


def test_delete_requires_project_before_network(monkeypatch, capsys):
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    assert cmd_env_delete(_args(project="")) == 1
    assert "project id is required" in capsys.readouterr().err
    assert cap == {}


def test_delete_aborts_on_declined_confirmation(monkeypatch):
    called = {"n": 0}

    class _C:
        def delete_env(self, env_id, *, project_id):  # pragma: no cover - must not be reached
            called["n"] += 1
            return {"deleted": True}

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())
    monkeypatch.setattr("builtins.input", lambda *_a: "n")
    assert cmd_env_delete(_args("acme/checkout-bot/env", yes=False)) == 1
    assert called["n"] == 0


def test_delete_proceeds_on_confirmation(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr("flash.client.client_from_config", _fake_client(cap))
    monkeypatch.setattr("builtins.input", lambda *_a: "y")
    assert cmd_env_delete(_args("acme/checkout-bot/env", yes=False)) == 0
    assert cap["env_id"] == "acme/checkout-bot/env"


def test_delete_surfaces_api_error(monkeypatch, capsys):
    from flash.client import ApiError

    class _C:
        def delete_env(self, env_id, *, project_id):
            raise ApiError(403, "you can only delete environments in your own namespace")

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())
    assert cmd_env_delete(_args("someone-else/project/env")) == 1
    assert "your own namespace" in capsys.readouterr().err


def test_delete_subcommand_dispatches_to_handler():
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "env",
            "delete",
            "--project",
            "11111111-1111-4111-8111-111111111111",
            "acme/checkout-bot/my-env",
        ]
    )
    assert args.func is cmd_env_delete
    assert args.env_id == "acme/checkout-bot/my-env"
    assert args.project == "11111111-1111-4111-8111-111111111111"
    assert args.yes is False
    args_yes = parser.parse_args(
        [
            "env",
            "delete",
            "--project",
            "11111111-1111-4111-8111-111111111111",
            "-y",
            "acme/checkout-bot/my-env",
        ]
    )
    assert args_yes.yes is True
