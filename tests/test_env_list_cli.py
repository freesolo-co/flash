"""`flash env list` reports published environments, not just local scaffold directories.

The bug this covers: with no server-side list endpoint the command enumerated local directories
only, so after a verified successful publish it printed "no environments yet". The honest reading of
that is "the publish silently failed", which invites re-pushing under new names.
"""

from __future__ import annotations

import argparse
import http.client

import pytest

import flash.cli.commands.env.list as commands
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


class _Resp:
    """A urlopen response whose body read or content fails the way a real one can."""

    def __init__(self, failure):
        self._failure = failure
        self.headers = {"Content-Type": "application/json"}

    def read(self, *_a):
        if isinstance(self._failure, BaseException):
            raise self._failure
        return self._failure

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        # a reset or a server hangup mid-response: ConnectionError, NOT a urllib URLError, so the
        # URLError clause never saw these.
        (ConnectionResetError("connection reset by peer"), "was interrupted"),
        (http.client.RemoteDisconnected("remote end closed connection"), "was interrupted"),
        # IncompleteRead is an HTTPException, not a ConnectionError -- catching ConnectionError
        # alone would still let this one escape raw.
        (http.client.IncompleteRead(b"partial"), "was interrupted"),
        # a truncated or non-JSON body, and a non-UTF-8 one: JSONDecodeError and UnicodeDecodeError
        # are both ValueError, so `_decode_response` owns these two and words them identically.
        (b"{not json", "did not return JSON"),
        (b'{"environments": "\xff\xfe"}', "did not return JSON"),
    ],
    ids=["reset", "hangup", "truncated-read", "non-json", "non-utf8"],
)
def test_a_broken_response_degrades_to_the_reason_line_not_a_traceback(
    monkeypatch, capsys, failure, expected
):
    """The whole point of the reason line: a transport fault must not print a raw traceback.

    These reach `json.loads`/`resp.read()` INSIDE the client's error-translation block, so each has
    to become a ClientError for `_published_envs` to catch it and degrade cleanly.
    """
    from flash.client import ApiClient

    monkeypatch.setattr(
        commands, "load_credentials", lambda: ("https://api.freesolo.co", "fslo-key")
    )
    monkeypatch.setattr(commands, "has_freesolo_backend", lambda _url: True)
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: ApiClient("https://api.freesolo.co", "fslo-key"),
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: _Resp(failure))

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    out = capsys.readouterr().out
    assert "published environments unavailable" in out
    assert expected in out
    assert "Traceback" not in out


def test_a_truncated_error_body_degrades_instead_of_raising(monkeypatch, capsys):
    """A non-2xx whose body drops mid-read must still surface as the status, not a traceback.

    The read happens in `_detail_from_http_error`, called from inside the `except HTTPError` handler.
    Python does not offer an exception raised inside one handler to that block's sibling clauses, so
    the IncompleteRead clause added for successful responses cannot catch this one -- it has to be
    handled at the read itself.
    """
    import urllib.error

    from flash.client import ApiClient

    class _TruncatedErrorBody:
        def read(self, *_a):
            raise http.client.IncompleteRead(b"partial")

        def close(self):
            """HTTPError's tempfile teardown calls this; without it the GC raises on collection."""

    def fake_urlopen(*_a, **_k):
        raise urllib.error.HTTPError(
            url="https://api.freesolo.co/v1/envs",
            code=502,
            msg="Bad Gateway",
            hdrs={},  # type: ignore[arg-type]
            fp=_TruncatedErrorBody(),
        )

    monkeypatch.setattr(
        commands, "load_credentials", lambda: ("https://api.freesolo.co", "fslo-key")
    )
    monkeypatch.setattr(commands, "has_freesolo_backend", lambda _url: True)
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: ApiClient("https://api.freesolo.co", "fslo-key"),
    )
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert commands.cmd_env_list(argparse.Namespace()) == 0

    out = capsys.readouterr().out
    assert "published environments unavailable" in out
    assert "Traceback" not in out
    assert "502" in out, "the status is the useful part and must survive an unreadable body"


def test_list_envs_outlasts_the_servers_own_github_retry_budget():
    """The client must not time out before the plane can answer.

    The server can spend its whole retry window (socket timeouts AND backoff, twice over) before
    returning a controlled 429/502. If the client gives up first, that diagnosable status is
    replaced by a generic timeout, which is the failure mode this change exists to remove.
    """
    from flash.client import ApiClient
    from flash.client.http import (
        ENV_LIST_CLIENT_TIMEOUT_SECONDS,
        ENV_LIST_SERVER_BUDGET_SECONDS,
    )

    seen: dict = {}
    client = ApiClient("https://api.freesolo.co", "fslo-key")

    def fake_request(method, path, *_a, timeout=None, **_k):
        seen.update(method=method, path=path, timeout=timeout)
        return {"environments": []}

    client._request = fake_request
    assert client.list_envs() == []
    assert seen["method"] == "GET"
    assert seen["path"] == "/v1/envs"
    assert seen["timeout"] is not None, "must pass an explicit timeout, not the shared default"
    assert seen["timeout"] == ENV_LIST_CLIENT_TIMEOUT_SECONDS
    assert seen["timeout"] > ENV_LIST_SERVER_BUDGET_SECONDS, "client must lose the race to time out"
    assert seen["timeout"] > client.timeout, "must exceed the shared default, not fall back to it"


def test_the_client_budget_matches_the_servers_actual_list_read_budget():
    """The mirrored arithmetic must not drift from the loader it describes.

    `flash.client` cannot import the loader (it must stay importable without the server extra), so
    the client's terms are a hand-mirror of `_LIST_READ_BUDGET`. That is exactly the kind of copy
    that rots silently: if the server's budget grows and this does not, the CLI starts timing out
    before the plane answers and the 429/502 becomes unreachable again.
    """
    import flash.client.http as client_http
    from flash.envs.loader import _LIST_READ_BUDGET

    assert _LIST_READ_BUDGET["timeout"] == client_http._ENV_LIST_SOCKET_TIMEOUT_SECONDS
    # the initial attempt is not a retry, hence the +1
    expected_attempts = _LIST_READ_BUDGET["max_rate_limit_retries"] + 1
    assert expected_attempts == client_http._ENV_LIST_ATTEMPTS_PER_READ


def test_the_list_read_budget_stays_interactive():
    """A person is waiting at a terminal, so the server's ceiling must stay in minutes.

    On the batch default (6 attempts x 120s socket + 180s backoff, twice) the ceiling is ~30
    minutes. No interactive caller waits that long, so the controlled 502 the domain layer raises
    would never actually be seen -- every client would time out first and report something generic.
    """
    from flash.client.http import (
        ENV_LIST_CLIENT_TIMEOUT_SECONDS,
        ENV_LIST_SERVER_BUDGET_SECONDS,
    )

    assert ENV_LIST_SERVER_BUDGET_SECONDS < 5 * 60, ENV_LIST_SERVER_BUDGET_SECONDS
    # the client wait is what the person actually experiences, so bound it too -- it necessarily
    # exceeds the server ceiling, and only this assertion keeps that margin from growing unbounded.
    assert ENV_LIST_CLIENT_TIMEOUT_SECONDS < 6 * 60, ENV_LIST_CLIENT_TIMEOUT_SECONDS


def test_the_client_budget_counts_the_body_read_not_just_the_connect():
    """Each attempt can spend the socket timeout AND the body deadline; counting one halves it.

    urllib restarts `timeout` on every blocking read, so a large tree body streamed in chunks is
    bounded by `body_deadline`, not by `timeout`. A mirror that omits it understates the server's
    ceiling and the client goes back to timing out first.
    """
    import flash.client.http as client_http
    from flash.envs.loader import _LIST_READ_BUDGET

    assert _LIST_READ_BUDGET["body_deadline"] == client_http._ENV_LIST_BODY_DEADLINE_SECONDS
    per_attempt = _LIST_READ_BUDGET["timeout"] + _LIST_READ_BUDGET["body_deadline"]
    attempts = _LIST_READ_BUDGET["max_rate_limit_retries"] + 1
    real_ceiling = 2 * (attempts * per_attempt + client_http._ENV_LIST_MAX_BACKOFF_PER_READ_SECONDS)
    mirrored_budget = client_http.ENV_LIST_SERVER_BUDGET_SECONDS
    client_timeout = client_http.ENV_LIST_CLIENT_TIMEOUT_SECONDS
    assert mirrored_budget == real_ceiling
    assert client_timeout > real_ceiling
