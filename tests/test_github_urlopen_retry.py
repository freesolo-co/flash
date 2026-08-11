"""Tests for _urlopen retry behavior on GitHub secondary rate-limit (HTTP 403)."""

from __future__ import annotations

import http.client
import io
import random
import time
import urllib.error
import urllib.request

import pytest

from flash.envs.loader import GitHubRateLimitError, _urlopen


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


def test_urlopen_retries_5xx_then_succeeds(monkeypatch):
    # A transient GitHub 5xx (incident/codeload blip) is infra, same class as a TCP reset — retry it.
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(503, "service unavailable")
        return io.BytesIO(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    assert _urlopen(urllib.request.Request("https://api.github.com/test")) == b"ok"
    assert len(calls) == 3


def test_urlopen_persistent_5xx_becomes_retriable_signal(monkeypatch):
    # A persistent 5xx must surface as the retriable GitHubRateLimitError (worker reschedules), NOT a
    # fatal RuntimeError that permanently fails the run — mirroring the rate-limit/TCP-transient paths.
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise _http_error(502, "bad gateway")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(GitHubRateLimitError, match="server error"):
        _urlopen(urllib.request.Request("https://api.github.com/test"))
    assert len(calls) == 6  # initial + 5 retries


def test_urlopen_retries_transient_url_error_then_succeeds(monkeypatch):
    # A connection-phase URLError (reset/DNS) is transient infra on the same cold-spawn wave as a
    # rate limit, so it must retry within the same budget rather than fail the run.
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("connection reset by peer")
        return io.BytesIO(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    assert _urlopen(urllib.request.Request("https://api.github.com/test")) == b"ok"
    assert len(calls) == 3


def test_urlopen_retries_a_truncated_body_then_succeeds(monkeypatch):
    # GitHub can drop a response after sending only part of a declared body, which surfaces at
    # `resp.read()` as http.client.IncompleteRead. That is an HTTPException: NOT a URLError,
    # TimeoutError, ConnectionError, or even an OSError -- so it matched none of the handlers here
    # and escaped the retry loop entirely. Worse, it is no RuntimeError either, so every caller's
    # translation missed it too and it surfaced as an uncaught 500 instead of a retry or a 502.
    calls = []

    class _TruncatedBody(io.BytesIO):
        def read(self, *_a):
            raise http.client.IncompleteRead(b"partial")

    def fake_urlopen(req, timeout):
        calls.append(1)
        return _TruncatedBody(b"") if len(calls) < 3 else io.BytesIO(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    assert _urlopen(urllib.request.Request("https://api.github.com/test")) == b"ok"
    assert len(calls) == 3


def test_urlopen_persistent_truncated_body_becomes_retriable_signal(monkeypatch):
    # A body that truncates on every attempt must end as the typed retriable error, so the domain
    # layer answers a controlled 502 instead of letting an HTTPException escape as a 500.
    calls = []

    class _TruncatedBody(io.BytesIO):
        def read(self, *_a):
            raise http.client.IncompleteRead(b"partial")

    def fake_urlopen(req, timeout):
        calls.append(1)
        return _TruncatedBody(b"")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(GitHubRateLimitError, match="transient network") as excinfo:
        _urlopen(urllib.request.Request("https://api.github.com/test"))
    assert len(calls) == 6  # initial + 5 retries
    # not throttling: a truncated body is an upstream fault, so it must map to 502 and not 429.
    assert excinfo.value.throttled is False


class _TruncatedErrorBody(io.BytesIO):
    """An HTTPError whose body drops mid-read, as when GitHub 429s then closes the connection."""

    def read(self, *_a):
        raise http.client.IncompleteRead(b"partial")


def test_urlopen_retries_a_5xx_whose_error_body_truncates(monkeypatch):
    # the truncation happens at `exc.read()` INSIDE the `except HTTPError` handler. Python does not
    # offer an exception raised in one handler to that block's sibling clauses, so the IncompleteRead
    # clause below cannot catch this: it escaped unretried as an uncaught 500 rather than being
    # classified. the status alone is enough to classify a 5xx, so an unreadable body must not stop
    # the retry.
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError(
                url="https://api.github.com/test",
                code=502,
                msg="",
                hdrs={},  # type: ignore[arg-type]
                fp=_TruncatedErrorBody(b""),
            )
        return io.BytesIO(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    assert _urlopen(urllib.request.Request("https://api.github.com/test")) == b"ok"
    assert len(calls) == 3


def test_urlopen_still_classifies_a_throttled_429_with_a_truncated_body(monkeypatch):
    # a 429 classifies on its status, so losing the body must not cost the `throttled` flag that
    # decides 429-vs-502 for the caller.
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=429,
            msg="",
            hdrs={},  # type: ignore[arg-type]
            fp=_TruncatedErrorBody(b""),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(GitHubRateLimitError) as excinfo:
        _urlopen(urllib.request.Request("https://api.github.com/test"))
    assert excinfo.value.throttled is True


def test_urlopen_transient_network_becomes_retriable_signal(monkeypatch):
    # A persistent transient failure (URLError every attempt, or a read-phase TimeoutError that is
    # NEITHER an HTTPError NOR a URLError) must surface as the typed retriable GitHubRateLimitError
    # — NOT a plain RuntimeError/bare TimeoutError that the worker would classify as a fatal crash.
    for exc in (urllib.error.URLError("dns failure"), TimeoutError("read timed out")):
        calls = []

        def fake_urlopen(req, timeout, _exc=exc, _calls=calls):
            _calls.append(1)
            raise _exc

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(time, "sleep", lambda _: None)
        monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

        with pytest.raises(GitHubRateLimitError, match="transient network"):
            _urlopen(urllib.request.Request("https://api.github.com/test"))
        assert len(calls) == 6  # initial + 5 retries


class _FlakyBody:
    """A response body that serves at most ``fail_after`` bytes, then raises mid-body."""

    def __init__(self, payload: bytes, fail_after: int | None = None) -> None:
        self._buf = io.BytesIO(payload)
        self._fail_after = fail_after
        self._served = 0

    def read(self, size: int = -1) -> bytes:
        if self._fail_after is not None and self._served >= self._fail_after:
            raise ConnectionResetError("connection reset by peer")
        want = size if size > 0 else None
        if self._fail_after is not None:
            budget = self._fail_after - self._served
            want = budget if want is None else min(want, budget)
        chunk = self._buf.read(want)
        self._served += len(chunk)
        return chunk

    def __enter__(self) -> _FlakyBody:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


@pytest.mark.parametrize("max_bytes", [None, 4096])
def test_urlopen_truncates_sink_before_retrying_mid_body_failure(monkeypatch, tmp_path, max_bytes):
    # A reset part-way through the body must not leave `partial_prefix + full_body_of_retry` on
    # disk, i.e. a corrupt tarball, or worse a corrupt raw file that still parses. The other retry
    # tests all fail BEFORE the first body byte, so only this one exercises a partial sink.
    payload = b"".join(f"line-{i:04d}\n".encode() for i in range(64))
    bodies = [
        _FlakyBody(payload, fail_after=37),
        _FlakyBody(payload, fail_after=11),
        _FlakyBody(payload),
    ]

    def fake_urlopen(req, timeout):
        return bodies.pop(0)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    sink = tmp_path / "artifact.bin"
    with sink.open("wb") as out:
        assert (
            _urlopen(
                urllib.request.Request("https://api.github.com/test"),
                max_bytes=max_bytes,
                out=out,
            )
            == b""
        )

    assert not bodies
    assert sink.read_bytes() == payload


def test_urlopen_byte_cap_binds_across_retried_attempts(monkeypatch, tmp_path):
    # Each attempt starts from the sink's initial offset, so the per-attempt cap in
    # _iter_capped_chunks is also the cap on the final artifact: three attempts that each write
    # just under the cap must not accumulate to 3x the cap on disk.
    payload = b"z" * 900
    bodies = [
        _FlakyBody(payload, fail_after=900),
        _FlakyBody(payload, fail_after=900),
        _FlakyBody(payload),
    ]

    def fake_urlopen(req, timeout):
        return bodies.pop(0)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    sink = tmp_path / "capped.bin"
    with sink.open("wb") as out:
        _urlopen(
            urllib.request.Request("https://api.github.com/test"),
            max_bytes=1000,
            out=out,
        )

    assert sink.stat().st_size == len(payload)


def test_urlopen_cap_still_rejects_oversized_retried_attempt(monkeypatch, tmp_path):
    bodies = [
        _FlakyBody(b"x" * 400, fail_after=120),
        _FlakyBody(b"x" * 4000),
    ]

    def fake_urlopen(req, timeout):
        return bodies.pop(0)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    sink = tmp_path / "oversized.bin"
    with (
        sink.open("wb") as out,
        pytest.raises(RuntimeError, match="exceeded the maximum allowed size"),
    ):
        _urlopen(
            urllib.request.Request("https://api.github.com/test"),
            max_bytes=1000,
            out=out,
        )

    # the partial prefix from the failed first attempt must not survive into the error state
    assert sink.stat().st_size == 0


def test_urlopen_handles_http_error_without_body(monkeypatch):
    # urllib can raise an HTTPError whose `fp` is None (no body to read); calling exc.read() then
    # blows up with AttributeError and masks the real status code.
    error = urllib.error.HTTPError(
        url="https://api.github.com/test",
        code=404,
        msg="Not Found",
        hdrs={},  # type: ignore[arg-type]
        fp=None,
    )

    def fake_urlopen(req, timeout):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="GitHub environment request failed \\(404\\)"):
        _urlopen(urllib.request.Request("https://api.github.com/test"))


def test_urlopen_classifies_bodyless_403_by_header(monkeypatch):
    # A bodyless 403 still has to be classified from X-RateLimit-Remaining alone.
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=403,
            msg="Forbidden",
            hdrs={"X-RateLimit-Remaining": "0"},  # type: ignore[arg-type]
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(GitHubRateLimitError, match="rate limit exceeded"):
        _urlopen(urllib.request.Request("https://api.github.com/test"))
    assert len(calls) == 6


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
