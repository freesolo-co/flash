"""`flash env list` reports published environments, not just local scaffold directories.

The bug this covers: with no server-side list endpoint the command enumerated local directories
only, so after a verified successful publish it printed "no environments yet". The honest reading of
that is "the publish silently failed", which invites re-pushing under new names.
"""

from __future__ import annotations

import argparse

import pytest

import flash.cli.commands as commands
from flash.client import ClientError


@pytest.fixture(autouse=True)
def plain_renderer(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_STYLE", "0")  # plain renderer keeps substring asserts stable
    monkeypatch.chdir(tmp_path)


def _logged_in(monkeypatch, published, *, error: Exception | None = None):
    """A logged-in client against a Freesolo backend, returning ``published`` (or raising)."""
    monkeypatch.setattr(
        commands, "load_credentials", lambda: ("https://api.freesolo.co", "fslo-key")
    )
    monkeypatch.setattr(commands, "has_freesolo_backend", lambda _url: True)

    class _C:
        def list_envs(self):
            if error is not None:
                raise error
            return published

    monkeypatch.setattr("flash.client.client_from_config", lambda: _C())


def test_published_environments_are_listed(monkeypatch, capsys):
    _logged_in(monkeypatch, ["acme/my-env", "acme/beta"])

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    out = capsys.readouterr().out
    assert "published environments" in out
    assert "acme/my-env" in out
    assert "acme/beta" in out


def test_a_published_environment_is_not_reported_as_no_environments(monkeypatch, capsys):
    """The regression: an org WITH a published env must never see the empty-state hint."""
    _logged_in(monkeypatch, ["acme/my-env"])

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    assert "no environments yet" not in capsys.readouterr().out


def test_empty_hint_still_shows_when_nothing_exists_anywhere(monkeypatch, capsys):
    _logged_in(monkeypatch, [])

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    assert "no environments yet" in capsys.readouterr().out


def test_local_and_published_are_both_reported(monkeypatch, capsys, tmp_path):
    (tmp_path / "environment.py").write_text("# env\n")
    _logged_in(monkeypatch, ["acme/my-env"])

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    out = capsys.readouterr().out
    assert "acme/my-env" in out
    assert "local env sources" in out


def test_an_unreachable_plane_is_reported_not_silently_empty(monkeypatch, capsys):
    """A failed lookup must say so; reporting the empty state would recreate the original bug."""
    _logged_in(monkeypatch, [], error=ClientError("control plane is unreachable"))

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    out = capsys.readouterr().out
    assert "published environments unavailable" in out
    assert "control plane is unreachable" in out


def test_a_server_refusal_is_reported_not_silently_empty(monkeypatch, capsys):
    """ApiError subclasses ClientError, so a 503 from the plane must land on the reason line too."""
    from flash.client import ApiError

    _logged_in(monkeypatch, [], error=ApiError(503, "GITHUB_TOKEN is required"))

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    out = capsys.readouterr().out
    assert "published environments unavailable" in out
    assert "GITHUB_TOKEN is required" in out
    assert "no environments yet" in out  # nothing local either, and that hint is still true


def test_logged_out_says_so_instead_of_claiming_nothing_is_published(monkeypatch, capsys):
    monkeypatch.setattr(commands, "load_credentials", lambda: ("https://api.freesolo.co", None))
    monkeypatch.setattr(
        commands,
        "has_freesolo_backend",
        lambda _url: pytest.fail("must refuse before checking the backend"),
    )
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: pytest.fail("must not call the plane while logged out"),
    )

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    out = capsys.readouterr().out
    assert "published environments unavailable" in out
    assert "not logged in" in out


def test_self_hosted_plane_reports_no_managed_hub(monkeypatch, capsys):
    """A self-hosted plane has no managed hub; say that rather than imply an empty one."""
    monkeypatch.setattr(commands, "load_credentials", lambda: ("http://localhost:8000", "key"))
    monkeypatch.setattr(commands, "has_freesolo_backend", lambda _url: False)
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: pytest.fail("a self-hosted plane has no hub to list"),
    )

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    assert "no managed environment hub" in capsys.readouterr().out


def test_styled_renderer_receives_published_and_unavailable(monkeypatch, capsys):
    seen: dict = {}
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(
        commands.render,
        "env_list",
        lambda paths, *, published, unavailable: (
            seen.update(paths=paths, published=published, unavailable=unavailable) or "styled"
        ),
    )
    _logged_in(monkeypatch, ["acme/my-env"])

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    assert capsys.readouterr().out == "styled\n"
    assert seen == {"paths": [], "published": ["acme/my-env"], "unavailable": None}


def test_styled_renderer_reports_the_failure_reason(monkeypatch):
    """The styled renderer must surface why the published list is missing, not just omit it."""
    from flash.cli.ui import render

    out = render.env_list([], published=[], unavailable="control plane is unreachable")

    assert "published environments unavailable" in out
    assert "control plane is unreachable" in out


def test_styled_renderer_lists_published_ids_above_local_sources(monkeypatch):
    from flash.cli.ui import render

    out = render.env_list(["."], published=["acme/my-env"], unavailable=None)

    assert out.index("acme/my-env") < out.index("local sources")
    assert "no environments yet" not in out
