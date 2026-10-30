"""Exercise dispatch responses without sending events or waiting for GitHub."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, Mock
from urllib.error import HTTPError, URLError

import pytest


@pytest.fixture
def notifier(monkeypatch):
    path = Path(__file__).resolve().parents[1] / ".github/scripts/notify_tests_repo.py"
    spec = importlib.util.spec_from_file_location("notify_tests_repo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", Mock())
    monkeypatch.setattr(module.time, "time", lambda: 1000)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100)
    monkeypatch.setenv("PAT", "fake-token")
    monkeypatch.setenv("SHA", "a" * 40)
    monkeypatch.setenv("REF", "dev")
    response = MagicMock()
    response.__enter__.return_value.status = 204
    monkeypatch.setattr(module.urllib.request, "urlopen", Mock(return_value=response))
    return module


def failure(status, message="Forbidden", headers=None):
    return HTTPError(
        "https://api.github.com/repos/freesolo-co/tests/dispatches",
        status,
        "error",
        headers or {},
        io.BytesIO(json.dumps({"message": message}).encode()),
    )


def test_dispatch_posts_exact_commit_and_ref(notifier):
    assert notifier.main() == 0
    opener = notifier.urllib.request.urlopen
    opener.assert_called_once()
    request = opener.call_args.args[0]
    assert request.full_url == notifier.DISPATCH_URL
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer fake-token"
    assert json.loads(request.data) == {
        "event_type": "flash-merged",
        "client_payload": {"sha": "a" * 40, "ref": "dev"},
    }
    assert opener.call_args.kwargs["timeout"] == 30
    notifier.time.sleep.assert_not_called()


@pytest.mark.parametrize(
    ("status", "message", "headers", "delay"),
    [
        (
            403,
            "API rate limit exceeded",
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "4600"},
            3601,
        ),
        (403, "secondary rate limit", {"Retry-After": "120"}, 120),
        (403, "secondary rate limit", {}, 60),
        (429, "Too many requests", {}, 60),
        (503, "Service unavailable", {}, 2),
    ],
)
def test_retry_then_success(notifier, status, message, headers, delay):
    opener = notifier.urllib.request.urlopen
    opener.side_effect = [failure(status, message, headers), opener.return_value]
    assert notifier.main() == 0
    assert opener.call_count == 2
    notifier.time.sleep.assert_called_once_with(delay)
    assert opener.call_args_list[0] == opener.call_args_list[1]


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_permanent_failure_is_not_retried(notifier, capsys, status):
    notifier.urllib.request.urlopen.side_effect = failure(status, "Resource not accessible")
    assert notifier.main() == 1
    notifier.urllib.request.urlopen.assert_called_once()
    notifier.time.sleep.assert_not_called()
    assert f"HTTP {status}" in capsys.readouterr().err


def test_rate_limit_retries_are_bounded(notifier, capsys):
    notifier.urllib.request.urlopen.side_effect = [
        failure(403, "secondary rate limit") for _ in range(5)
    ]
    assert notifier.main() == 1
    assert notifier.urllib.request.urlopen.call_count == 5
    assert [call.args[0] for call in notifier.time.sleep.call_args_list] == [60, 120, 240, 480]
    assert "retry budget exhausted" in capsys.readouterr().err


def test_reset_beyond_deadline_fails_without_early_retry(notifier, capsys):
    notifier.urllib.request.urlopen.side_effect = failure(
        403, "API rate limit exceeded", {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "99999"}
    )
    assert notifier.main() == 1
    notifier.time.sleep.assert_not_called()
    notifier.urllib.request.urlopen.assert_called_once()
    assert "retry budget exhausted" in capsys.readouterr().err


def test_missing_credential_never_contacts_github(notifier, monkeypatch):
    monkeypatch.delenv("PAT")
    assert notifier.main() == 1
    notifier.urllib.request.urlopen.assert_not_called()


def test_transport_failure_does_not_claim_success(notifier, capsys):
    notifier.urllib.request.urlopen.side_effect = URLError("connection refused")
    assert notifier.main() == 1
    assert "connection refused" in capsys.readouterr().err


def test_diagnostics_redact_credentials_and_escape_annotations(notifier, capsys):
    notifier.urllib.request.urlopen.side_effect = failure(403, "fake-token\n::notice::bad%")
    assert notifier.main() == 1
    output = capsys.readouterr().err
    assert "fake-token" not in output
    assert "***%0A::notice::bad%25" in output


def test_unexpected_success_status_is_rejected(notifier):
    notifier.urllib.request.urlopen.return_value.__enter__.return_value.status = 200
    assert notifier.main() == 1
