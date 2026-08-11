"""Tests for _urlopen retry behavior on GitHub secondary rate-limit (HTTP 403)."""

from __future__ import annotations

import http.client
import io
import random
import ssl
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


def test_urlopen_retries_a_tls_read_failure_then_succeeds(monkeypatch):
    # GitHub can tear down TLS while the body is streaming. `ssl.SSLEOFError` is an OSError but NOT
    # a ConnectionError, URLError, TimeoutError, or IncompleteRead, so naming only those let it skip
    # the retry loop entirely and escape as an uncaught 500.
    calls = []

    class _TlsCutBody(io.BytesIO):
        def read(self, *_a):
            raise ssl.SSLEOFError("EOF occurred in violation of protocol")

    def fake_urlopen(req, timeout):
        calls.append(1)
        return _TlsCutBody(b"") if len(calls) == 1 else io.BytesIO(b"recovered")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    assert _urlopen(urllib.request.Request("https://api.github.com/test")) == b"recovered"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "framing_failure",
    [
        http.client.BadStatusLine("<html>502 from an intermediary</html>"),
        http.client.LineTooLong("chunk size"),
    ],
    ids=["garbled-status-line", "overlong-chunk-line"],
)
def test_urlopen_retries_a_framing_failure_then_succeeds(monkeypatch, framing_failure):
    # GitHub or a proxy in front of it can answer with a malformed status line or a chunk-size line
    # over the limit. Both are `HTTPException` SIBLINGS of `IncompleteRead`, not subclasses, so a
    # clause naming only IncompleteRead skipped the retry loop and escaped as an uncaught 500.
    calls = []

    class _BrokenFraming(io.BytesIO):
        def read(self, *_a):
            raise framing_failure

    def fake_urlopen(req, timeout):
        calls.append(1)
        return _BrokenFraming(b"") if len(calls) == 1 else io.BytesIO(b"recovered")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    assert _urlopen(urllib.request.Request("https://api.github.com/test")) == b"recovered"
    assert len(calls) == 2


def test_urlopen_retries_a_garbled_status_line_raised_from_urlopen_itself(monkeypatch):
    # `BadStatusLine` comes out of `getresponse`, so it surfaces from `urlopen` rather than from a
    # body read -- and urllib does not wrap it in a URLError there. The retry clause has to cover it
    # at that position too, not only when a stubbed body raises it.
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise http.client.BadStatusLine("<html>502 from an intermediary</html>")
        return io.BytesIO(b"recovered")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    assert _urlopen(urllib.request.Request("https://api.github.com/test")) == b"recovered"
    assert len(calls) == 2


def test_urlopen_persistent_framing_failure_becomes_retriable_signal(monkeypatch):
    # Exhausting retries on a framing fault must end as the typed retriable error so the domain layer
    # answers a controlled 502, not a 500.
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout: (_ for _ in ()).throw(http.client.LineTooLong("chunk size")),
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(GitHubRateLimitError, match="transient network") as excinfo:
        _urlopen(urllib.request.Request("https://api.github.com/test"))
    # an upstream transport fault, not throttling: 502 rather than 429.
    assert excinfo.value.throttled is False


def test_urlopen_error_body_framing_failure_keeps_the_classifying_status(monkeypatch):
    # The error-body read runs inside the `except HTTPError` handler, and Python does not offer an
    # exception raised there to that block's sibling clauses. A framing fault while reading a 503's
    # body therefore had to be handled at the read, or it escaped past the retry as a raw traceback --
    # losing the status that classifies the failure.
    class _BrokenFramingBody:
        def read(self, *_a):
            raise http.client.LineTooLong("chunk size")

        def close(self):
            """HTTPError's tempfile teardown calls this."""

    exc = urllib.error.HTTPError(
        url="https://api.github.com/test",
        code=503,
        msg="Service Unavailable",
        hdrs={},  # type: ignore[arg-type]
        fp=_BrokenFramingBody(),
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(exc))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(GitHubRateLimitError, match="503") as excinfo:
        _urlopen(urllib.request.Request("https://api.github.com/test"))
    assert excinfo.value.throttled is False


def test_urlopen_persistent_tls_read_failure_becomes_retriable_signal(monkeypatch):
    # Exhausting retries on a TLS cut must end as the typed retriable error so the domain layer
    # answers a controlled 502, not a 500.
    class _TlsCutBody(io.BytesIO):
        def read(self, *_a):
            raise ssl.SSLEOFError("EOF occurred in violation of protocol")

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: _TlsCutBody(b""))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(GitHubRateLimitError, match="transient network") as excinfo:
        _urlopen(urllib.request.Request("https://api.github.com/test"))
    # an upstream transport fault, not throttling: 502 rather than 429.
    assert excinfo.value.throttled is False


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


def test_urlopen_asks_the_error_body_for_a_bounded_amount(monkeypatch):
    """The error-body read must pass a size, because a sizeless read is unbounded.

    `HTTPResponse.read(None)` is documented as an unbounded read and streams a chunked body until it
    ends, so a slow 429/5xx whose chunks each land inside the socket timeout held this handler open
    past the interactive list's deadline -- the caller then timed out locally instead of receiving
    the controlled 429/502 the status was about to produce.

    Asserting on the size ARGUMENT rather than on wall-clock is deliberate: the loop lives inside a
    real `HTTPResponse`, and `exc.read()` delegates one-to-one to the file object, so no in-memory
    stub can reproduce the endless stream. What is observable -- and what actually fixes it -- is
    that we never issue a sizeless read.
    """
    sizes: list[object] = []

    class _RecordingErrorBody(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            return b"x" * 512 if len(sizes) < 3 else b""

        def close(self):  # HTTPError's tempfile teardown calls this
            pass

    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=404,  # not transient: classify once, no retry, keeps the assertion unambiguous
            msg="",
            hdrs={},  # type: ignore[arg-type]
            fp=_RecordingErrorBody(b""),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(RuntimeError, match="GitHub environment request failed"):
        _urlopen(
            urllib.request.Request("https://api.github.com/test"),
            max_bytes=16 * 1024 * 1024,
            body_deadline=20.0,
        )

    assert sizes, "the error body was never read"
    # collect first, assert once: a per-iteration assertion reports only the first bad read.
    unbounded = [size for size in sizes if not isinstance(size, int) or size <= 0]
    assert unbounded == [], f"unbounded error-body reads: {unbounded!r}"


def test_urlopen_stops_reading_an_error_body_at_its_cap(monkeypatch):
    """A huge error body must not be buffered in full just to classify a status."""
    served = [0]

    class _EndlessErrorBody(io.BytesIO):
        def read(self, size=-1):
            served[0] += 1
            if served[0] > 10_000:  # a stand-in for "forever", so a regression fails loudly
                raise AssertionError("error body read was never capped")
            return b"x" * (size if isinstance(size, int) and size > 0 else 1024)

        def close(self):
            pass

    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=404,
            msg="",
            hdrs={},  # type: ignore[arg-type]
            fp=_EndlessErrorBody(b""),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(RuntimeError, match="GitHub environment request failed"):
        _urlopen(urllib.request.Request("https://api.github.com/test"), max_bytes=16 * 1024 * 1024)


def test_urlopen_body_deadline_bounds_a_slow_dripping_response(monkeypatch):
    # urllib restarts `timeout` on every blocking read, so a peer returning one chunk just inside
    # each window keeps ONE attempt alive indefinitely -- the per-read timeout is not a per-attempt
    # bound. `body_deadline` is what actually bounds the attempt, which matters for the interactive
    # list whose client timeout is sized against that ceiling.
    clock = [0.0]

    class _SlowBody(io.BytesIO):
        def read(self, *_a):
            clock[0] += 5.0  # each chunk arrives inside the socket timeout, but time still passes
            return b"x" * 1024

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: _SlowBody(b""))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(GitHubRateLimitError, match="transient network"):
        _urlopen(
            urllib.request.Request("https://api.github.com/test"),
            max_bytes=16 * 1024 * 1024,
            body_deadline=20.0,
            max_rate_limit_retries=1,
        )


def test_urlopen_body_deadline_covers_the_connect_not_just_the_body(monkeypatch):
    """The deadline has to start BEFORE `urlopen`, or the connect is outside the bound.

    DNS, the TCP connect, the TLS handshake and header parsing all happen inside `urlopen`, and a
    peer making progress inside every socket window can hold it there. A deadline that only began
    once headers had arrived left that whole stretch unbounded, so the server could still be waiting
    when the client hit its own fixed timeout -- replacing the controlled 429/502 with a local
    timeout, which is the failure the budget exists to prevent.
    """
    clock = [0.0]

    def slow_connect(req, timeout):
        clock[0] += 25.0  # headers take longer than the whole body deadline
        return io.BytesIO(b"payload")

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(urllib.request, "urlopen", slow_connect)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    # the connect alone has already outlived the 20s budget, so the first body read must fail rather
    # than treat the attempt as fresh. exhausting the one retry lands on the retriable signal.
    with pytest.raises(GitHubRateLimitError, match="transient network"):
        _urlopen(
            urllib.request.Request("https://api.github.com/test"),
            max_bytes=16 * 1024 * 1024,
            body_deadline=20.0,
            max_rate_limit_retries=1,
        )


def test_urlopen_deadline_is_enforced_after_a_slow_header_phase(monkeypatch):
    """Starting the deadline before `urlopen` is necessary but NOT sufficient; it must be enforced.

    urllib restarts its `timeout` window on every blocking socket read, so a peer that drip-feeds
    response headers -- each read landing inside the window -- walks past the attempt's deadline
    without any single read exceeding it. Recording the start time only helps if something then acts
    on it, and the body-read path cannot: `_iter_capped_chunks` inspects the deadline only once it is
    already draining, and a caller with no `max_bytes` never goes through it at all. So the attempt has
    to be failed at the point the headers land.

    Distinct from `..._covers_the_connect_not_just_the_body`: that one proves WHEN the deadline starts,
    this one proves it is CHECKED after the open. Note `max_bytes` is deliberately unset here -- that
    is the case no drain-time check can ever catch.
    """
    clock = [0.0]

    def drip_fed_headers(req, timeout):
        # no single socket read exceeded its window; the phase as a whole outlived the budget.
        clock[0] += 25.0
        return io.BytesIO(b"payload")

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(urllib.request, "urlopen", drip_fed_headers)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)

    with pytest.raises(GitHubRateLimitError, match="transient network"):
        _urlopen(
            urllib.request.Request("https://api.github.com/test"),
            body_deadline=20.0,
            max_rate_limit_retries=1,
        )


def test_urlopen_open_timeout_never_exceeds_the_remaining_deadline(monkeypatch):
    """The socket window handed to `urlopen` cannot outlast the attempt's own remaining budget.

    A no-op on the list path, where `timeout` and `body_deadline` are both 20s -- but it holds the
    invariant for any caller whose socket timeout is the larger of the two, where a single blocking
    open would otherwise be allowed to outlive the whole attempt budget in one call.
    """
    seen: list[float] = []

    def record_timeout(req, timeout):
        seen.append(timeout)
        return io.BytesIO(b"ok")

    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(urllib.request, "urlopen", record_timeout)

    assert (
        _urlopen(
            urllib.request.Request("https://api.github.com/test"),
            timeout=60.0,
            body_deadline=20.0,
        )
        == b"ok"
    )
    assert seen == [20.0], "a 60s socket window would outlive the 20s attempt budget"


def test_urlopen_without_a_body_deadline_keeps_its_full_socket_timeout(monkeypatch):
    """No deadline means no cap: the background path must keep the socket window it asked for."""
    seen: list[float] = []

    def record_timeout(req, timeout):
        seen.append(timeout)
        return io.BytesIO(b"ok")

    monkeypatch.setattr(urllib.request, "urlopen", record_timeout)

    assert _urlopen(urllib.request.Request("https://api.github.com/test"), timeout=60.0) == b"ok"
    assert seen == [60.0]


def test_urlopen_without_a_body_deadline_still_finishes_a_slow_download(monkeypatch):
    # background downloads must NOT inherit the bound: finishing a large transfer matters more than
    # a deadline there, so an unset `body_deadline` has to leave the old behaviour untouched.
    chunks = [b"x" * 1024, b"x" * 1024, b""]
    clock = [0.0]

    class _SlowBody(io.BytesIO):
        def read(self, *_a):
            clock[0] += 600.0  # far past any deadline the list path would impose
            return chunks.pop(0)

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: _SlowBody(b""))

    assert (
        _urlopen(urllib.request.Request("https://api.github.com/test"), max_bytes=16 * 1024 * 1024)
        == b"x" * 2048
    )


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
