"""Tests for _urlopen retry behavior on GitHub secondary rate-limit (HTTP 403)."""

from __future__ import annotations

import io
import random
import time
import urllib.error
import urllib.request

import pytest

from flash.envs.adapter import GitHubRateLimitError, _urlopen


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.github.com/test",
        code=code,
        msg="",
        hdrs={},  # type: ignore[arg-type]
        fp=io.BytesIO(body.encode()),
    )


_RATE_LIMIT_BODY = (
    '{"message": "API rate limit exceeded for user ID 12345.", '
    '"documentation_url": "https://docs.github.com/en/rest/using-the-rest-api", '
    '"status": "403"}'
)


def test_urlopen_retries_on_rate_limit_then_succeeds(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(len(calls))
        if len(calls) < 3:
            raise _http_error(403, _RATE_LIMIT_BODY)
        return io.BytesIO(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    result = _urlopen(urllib.request.Request("https://api.github.com/test"))
    assert result == b"ok"
    assert len(calls) == 3


def test_urlopen_raises_after_max_retries(monkeypatch):
    # A PERSISTENT rate limit (every attempt 403s) is reclassified as the typed, retriable
    # GitHubRateLimitError after the in-process retries are exhausted (#209) — so the worker
    # reschedules on a fresh worker instead of hard-failing. It is still a RuntimeError subclass.
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise _http_error(403, _RATE_LIMIT_BODY)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(GitHubRateLimitError, match="rate limit exceeded"):
        _urlopen(urllib.request.Request("https://api.github.com/test"))
    # Initial attempt + 5 retries before giving up.
    assert len(calls) == 6


def test_urlopen_does_not_retry_non_rate_limit_403(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise _http_error(403, '{"message": "Forbidden"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="GitHub environment request failed \\(403\\)"):
        _urlopen(urllib.request.Request("https://api.github.com/test"))

    assert len(calls) == 1


def test_urlopen_does_not_retry_404(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise _http_error(404, "Not Found")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="GitHub environment request failed \\(404\\)"):
        _urlopen(urllib.request.Request("https://api.github.com/test"))

    assert len(calls) == 1


def test_urlopen_sleep_called_with_jitter(monkeypatch):
    sleep_calls = []
    attempt_count = []

    def fake_urlopen(req, timeout):
        attempt_count.append(1)
        if len(attempt_count) < 2:
            raise _http_error(403, _RATE_LIMIT_BODY)
        return io.BytesIO(b"data")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.2)

    _urlopen(urllib.request.Request("https://api.github.com/test"))

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(10.0 * 1 * 1.2)
