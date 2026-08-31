"""CPU-only containment tests for Vast asynchronous result retrieval."""

from __future__ import annotations

import contextlib
import http.client
import http.server
import io
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from flash._internal import http as http_module
from flash.providers.vast.client import result as vast_result


class _Response:
    """a response exposing only ``read1``, the one bounded-read call the module is allowed to make.

    every fake here withholds ``read`` deliberately. ``read(amt)`` loops over socket receives until
    it has ``amt`` bytes, so a module that reaches for it is unbounded no matter what the deadline
    loop around it checks; withholding it turns that regression into an immediate AttributeError
    across the whole suite instead of a timing defect no fake can see.

    ``length`` mirrors the stdlib: bytes still owed on a declared body, dropping to ``0`` once the
    body is fully read. ``owed_after`` stages a peer that declares more than it sends.
    """

    def __init__(self, body: bytes, status: int = 200, *, owed_after: int = 0):
        self._stream = io.BytesIO(body)
        self._status = status
        self.length = len(body) + owed_after

    def getcode(self):
        return self._status

    def read1(self, size=-1):
        chunk = self._stream.read(size)
        self.length -= len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _fake_transport(monkeypatch, outcome):
    """replace the stdlib opener the module reaches for, recording the requests it receives."""
    seen = []

    def urlopen(request, timeout=None):
        seen.append((request, timeout))
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return seen


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://s3.amazonaws.com/x", code, "no", {}, io.BytesIO(b""))


# ---------------------------------------------------------------------------
# configured_result_origins: the operator-facing allowlist
# ---------------------------------------------------------------------------
def test_blank_or_unset_origins_fall_back_to_the_s3_default(monkeypatch):
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    assert vast_result.configured_result_origins() == ("https://s3.amazonaws.com",)
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "")
    assert vast_result.configured_result_origins() == ("https://s3.amazonaws.com",)


def test_multiple_canonical_origins_are_accepted_in_order(monkeypatch):
    monkeypatch.setenv(
        vast_result.RESULT_ORIGINS_ENV,
        "https://s3.amazonaws.com,https://logs.vast.ai",
    )
    assert vast_result.configured_result_origins() == (
        "https://s3.amazonaws.com",
        "https://logs.vast.ai",
    )


@pytest.mark.parametrize(
    "value",
    [
        "http://logs.example.com",  # not https
        "https://user:secret@logs.example.com",  # credentials
        "https://logs.example.com:8443",  # explicit port
        "https://logs.example.com/prefix",  # path
        "https://logs.example.com?a=b",  # query
        "https://logs.example.com#frag",  # fragment
        "https://*.example.com",  # wildcard
        "https://Logs.Example.com",  # non-canonical case
        "https://logs.example.com,",  # empty member
        ",https://logs.example.com",  # empty member
        "https://logs.example.com, https://b.example.com",  # space
        "https://logs.example.com\n",  # control character
        "https://logs.example.com,https://logs.example.com",  # duplicate
        "https://[2606:4700::1111]",  # ip literal authority
        "https://logs..example.com",  # empty label
        "https://-logs.example.com",  # label boundary
        "not-a-url",
    ],
)
def test_malformed_origin_configuration_is_rejected(monkeypatch, value):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, value)
    with pytest.raises(ValueError, match=vast_result.RESULT_ORIGINS_ENV) as exc_info:
        vast_result.configured_result_origins()
    detail = str(exc_info.value)
    assert vast_result.RESULT_ORIGINS_ENV in detail
    # the operator's value may be a signed url or carry credentials, so the message states the rule
    # rather than echoing what was configured.
    assert value not in detail


# ---------------------------------------------------------------------------
# fetch_result: destination admission
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "https://logs.attacker.example.com/x",  # origin not on the list
        "http://s3.amazonaws.com/x",  # downgraded scheme
        "https://s3.amazonaws.com.attacker.example/x",  # suffix confusion
        "https://attacker.example/https://s3.amazonaws.com/x",  # allowed origin in the path
        "https://user:secret@s3.amazonaws.com/x",  # credentials in the authority
        "https://s3.amazonaws.com:8443/x",  # explicit port
        "https://S3.amazonaws.com/x",  # non-canonical case
        "https://s3.amazonaws.com/x#frag",  # fragment
        "https://s3.amazonaws.com/x\n",  # control character
        "file:///etc/passwd",
        "data:text/plain,hello",
        "//s3.amazonaws.com/x",  # scheme-relative
        "",
        b"https://s3.amazonaws.com/x",  # not a string
        None,
    ],
)
def test_disallowed_destinations_are_refused_before_any_connection(monkeypatch, url):
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    seen = _fake_transport(monkeypatch, b"unreached")

    with pytest.raises(vast_result.VastResultError):
        vast_result.fetch_result(url, timeout=1.0)
    assert seen == []


def test_an_allowlisted_destination_is_fetched(monkeypatch):
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    seen = _fake_transport(monkeypatch, b"boot ok\n")

    assert vast_result.fetch_result("https://s3.amazonaws.com/x?sig=a", timeout=7.0) == b"boot ok\n"
    request, timeout = seen[0]
    assert request.full_url == "https://s3.amazonaws.com/x?sig=a"
    assert timeout == 7.0


def test_a_configured_origin_replaces_the_default_rather_than_extending_it(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.vast.ai")
    seen = _fake_transport(monkeypatch, b"logs")

    assert vast_result.fetch_result("https://logs.vast.ai/x", timeout=1.0) == b"logs"
    with pytest.raises(vast_result.VastResultError):
        vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0)
    assert len(seen) == 1


def test_invalid_origin_configuration_refuses_every_fetch(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "http://logs.example.com")
    seen = _fake_transport(monkeypatch, b"unreached")

    with pytest.raises(vast_result.VastResultError):
        vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0)
    assert seen == []


# ---------------------------------------------------------------------------
# fetch_result: transport bounds
# ---------------------------------------------------------------------------
def test_the_fetch_goes_through_the_no_redirect_opener(monkeypatch):
    """the origin check admits exactly the url that was named, so a followed hop would leave the
    vetted origin. the fetch must reach the network through the refusing opener rather than a bare
    urlopen, and a 3xx it raises must surface as a result error.

    asserting the handler set is what makes this a real check: a fetch that stopped building the
    opener would still pass a test that only looked at the raised error.

    `urllib.request.urlopen` is deliberately left alone here. `_urlopen_no_redirect` routes to a
    replaced transport directly, so patching it is what every other test in this file uses to reach
    a fake -- and it is also what would skip the opener this test exists to prove.
    """
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    handlers = []

    class _Refusing:
        def open(self, _request, timeout=None):
            raise _http_error(302)

    def build_opener(*given):
        handlers.extend(given)
        return _Refusing()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)

    with pytest.raises(vast_result.VastResultError) as exc_info:
        vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0)
    assert "302" in str(exc_info.value)
    assert any(isinstance(handler, http_module._NoRedirectHandler) for handler in handlers)


def test_a_not_yet_materialized_result_is_not_an_error(monkeypatch):
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    _fake_transport(monkeypatch, _http_error(404))

    assert vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0) is None


@pytest.mark.parametrize("code", [400, 403, 500, 503])
def test_other_http_statuses_are_terminal(monkeypatch, code):
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    _fake_transport(monkeypatch, _http_error(code))

    with pytest.raises(vast_result.VastResultError) as exc_info:
        vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0)
    assert str(code) in str(exc_info.value)


def test_a_transport_failure_is_a_result_error(monkeypatch):
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    _fake_transport(monkeypatch, urllib.error.URLError("dns"))

    with pytest.raises(vast_result.VastResultError):
        vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0)


def test_a_body_at_the_cap_is_returned(monkeypatch):
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    body = b"a" * vast_result._MAX_RESULT_BODY_BYTES
    _fake_transport(monkeypatch, body)

    assert vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0) == body


def test_an_oversized_body_is_refused_from_its_declaration_without_reading_it(monkeypatch):
    """the cap is enforced against the declared size, so an oversized body is never downloaded."""
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    reads = []

    class _Oversized:
        length = vast_result._MAX_RESULT_BODY_BYTES + 1

        def getcode(self):
            return 200

        def read1(self, size=-1):
            reads.append(size)
            return b"a" * size

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Oversized())

    with pytest.raises(vast_result.VastResultError) as exc_info:
        vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0)
    assert str(vast_result._MAX_RESULT_BODY_BYTES) in str(exc_info.value)
    assert reads == [], "an oversized body must be refused before a single payload byte is read"


def test_reads_never_exceed_one_chunk_at_a_time(monkeypatch):
    """a body at the cap is still read incrementally rather than requested in one allocation."""
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    reads = []
    body = b"a" * vast_result._MAX_RESULT_BODY_BYTES

    class _Recording(_Response):
        def read1(self, size=-1):
            reads.append(size)
            return super().read1(size)

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Recording(body))

    assert vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0) == body
    assert max(reads) <= vast_result._READ_CHUNK_BYTES
    assert sum(reads) == len(body), "must ask for exactly the declared body, no more"


@pytest.mark.parametrize("code", [201, 202, 204, 206])
def test_a_non_200_success_status_is_not_a_materialized_log(monkeypatch, code):
    """urllib raises only for non-2xx, so a fragment or an acknowledgement arrives looking like
    success. accepting one would hand a partial body to the caller as a complete log."""
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_a, **_k: _Response(b"partial", status=code)
    )

    with pytest.raises(vast_result.VastResultError) as exc_info:
        vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0)
    assert str(code) in str(exc_info.value)


def test_a_truncated_body_is_a_result_error(monkeypatch):
    """http.client.HTTPException is not an OSError, so a framing failure would otherwise escape the
    documented VastResultError boundary."""
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)

    class _Truncated:
        # a declared length, so the read is actually reached and the raise below is what is tested.
        length = 8

        def getcode(self):
            return 200

        def read1(self, size=-1):
            raise http.client.IncompleteRead(b"half")

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: _Truncated())

    with pytest.raises(vast_result.VastResultError) as exc_info:
        vast_result.fetch_result("https://s3.amazonaws.com/x", timeout=1.0)
    assert "retrieval failed" in str(exc_info.value), "must land on the transport-failure boundary"


@contextlib.contextmanager
def _one_shot_server(write_response):
    """serve one request from a raw socket, so a test can stage a peer that hangs up mid-body.

    ``http.server`` always frames a complete response, which is the one thing these tests need to
    violate, so the response bytes are written by hand.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve():
        try:
            connection, _ = listener.accept()
        except OSError:
            return
        with connection:
            connection.recv(65536)
            write_response(connection)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1]
    finally:
        listener.close()
        thread.join(timeout=5.0)


def _read_from_server(write_response):
    """run the module's read path against a real peer, returning the body or raising."""
    with _one_shot_server(write_response) as port:
        response = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=30.0)
        declared = vast_result._declared_length(response)
        return vast_result._read_bounded(response, declared, time.monotonic() + 30.0)


def test_a_body_the_peer_never_finished_sending_is_refused():
    """a peer that declares a length and then closes early must not read as a complete log.

    this needs a real socket. ``read1`` reports the short body as a clean EOF instead of raising
    ``IncompleteRead`` the way ``read`` did, so nothing in the read loop distinguishes it from a
    finished body; only the declared count does. a fake that returns bytes from a buffer cannot
    stage a peer that hangs up mid-body.
    """
    declared, sent = 300_000, 100_000

    def hang_up_early(connection):
        connection.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % declared
        )
        connection.sendall(b"a" * sent)

    with pytest.raises(vast_result.VastResultError) as exc_info:
        _read_from_server(hang_up_early)

    assert "truncated" in str(exc_info.value)


def test_a_complete_body_survives_the_completeness_check():
    """the truncation check must not refuse a body the peer did finish sending.

    paired with the test above deliberately: a check that refuses everything would pass that one.
    """
    body = b"a" * 300_000

    def send_all(connection):
        connection.sendall(
            b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body)
        )
        connection.sendall(body)

    assert _read_from_server(send_all) == body


def test_a_close_delimited_body_is_refused_rather_than_read_as_complete():
    """without a declared length there is nothing to detect a peer that hangs up mid-log.

    this is why the length is required rather than merely used when present. a close-delimited
    response carries no framing at all: the truncated body below and a complete one are the same
    bytes on the wire, so no check downstream of this one could tell them apart.
    """

    def truncated_close_delimited(connection):
        connection.sendall(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\npartial log")

    with pytest.raises(vast_result.VastResultError) as exc_info:
        _read_from_server(truncated_close_delimited)

    assert "declared no body length" in str(exc_info.value)


def test_a_chunked_body_is_refused_even_when_it_is_complete():
    """chunked framing is refused for the deadline, not for completeness.

    a chunked ``read1`` parses its chunk-size line through ``fp.readline()``, which loops over
    receives, so a trickling peer keeps one call blocked far past the deadline checked between
    calls. that makes the framing unsafe even when the body is well formed, which is why this
    asserts on a complete one.
    """

    def complete_chunked(connection):
        connection.sendall(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n"
        )

    with pytest.raises(vast_result.VastResultError) as exc_info:
        _read_from_server(complete_chunked)

    assert "declared no body length" in str(exc_info.value)


_TRICKLE_BYTES = 60
_TRICKLE_INTERVAL = 0.05
_TRICKLE_DEADLINE = 0.25


class _TricklingHandler(http.server.BaseHTTPRequestHandler):
    """answer with a complete, well-formed body delivered one byte at a time."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(_TRICKLE_BYTES))
        self.end_headers()
        for _ in range(_TRICKLE_BYTES):
            self.wfile.write(b"a")
            self.wfile.flush()
            time.sleep(_TRICKLE_INTERVAL)

    def log_message(self, *_args):
        return


@pytest.mark.wallclock
def test_a_trickling_peer_cannot_outlast_the_deadline():
    """the deadline has to bound the transfer, not just the gaps between reads.

    a fake cannot prove this. the defect lives inside a real ``HTTPResponse``: ``read(amt)`` loops
    over socket receives until it has ``amt`` bytes or hits EOF, each receive getting a fresh
    inactivity timeout, so a peer trickling under that timeout keeps one call blocked for the whole
    transfer while the deadline check sits outside it, unreached. only a real socket serving real
    bytes slowly reproduces that, so this test asserts on elapsed time against a real server: with
    ``read`` it takes the full 3s transfer, with ``read1`` it stops at the deadline.
    """
    server = http.server.HTTPServer(("127.0.0.1", 0), _TricklingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # the transport timeout is deliberately far above the whole transfer: the point is that no
        # receive ever times out, so the caller's deadline is the only thing that can end this.
        response = urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=30.0)
        declared = vast_result._declared_length(response)
        started = time.monotonic()
        with pytest.raises(vast_result.VastResultError) as exc_info:
            vast_result._read_bounded(response, declared, started + _TRICKLE_DEADLINE)
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()

    assert "deadline" in str(exc_info.value)
    # the untimed read returns only once the peer has sent all 60 bytes, three seconds in.
    assert elapsed < _TRICKLE_BYTES * _TRICKLE_INTERVAL / 2


_CHUNK_LINE_TIMEOUT = 0.4
_CHUNK_LINE_GAP = 0.3
_CHUNK_LINE_DIGITS = 4


@pytest.mark.wallclock
def test_a_chunked_peer_cannot_stretch_one_read_past_the_transport_timeout():
    """the reason chunked framing is refused, measured rather than asserted from the source.

    the deadline is only ever one transport timeout wide, because a call already in progress cannot
    be cut short. on a declared body that holds: one ``read1`` is one receive. on a chunked body it
    does not, and this stages why. the peer trickles the *chunk-size line* one byte at a time under
    the transport timeout, so ``fp.readline()`` inside a single ``read1`` loops over receives with a
    fresh timeout each, and one call blocks for as long as the peer cares to keep typing. this reads
    the raw response directly, since the module now refuses this framing before reaching a read.
    """

    def trickle_the_chunk_size_line(connection):
        connection.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        for byte in b"10\r\n"[:_CHUNK_LINE_DIGITS]:
            connection.sendall(bytes([byte]))
            time.sleep(_CHUNK_LINE_GAP)

    with _one_shot_server(trickle_the_chunk_size_line) as port:
        response = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=_CHUNK_LINE_TIMEOUT)
        assert response.length is None, "chunked framing declares no length"
        started = time.monotonic()
        with contextlib.suppress(Exception):
            response.read1(65536)
        blocked = time.monotonic() - started

    # a single call outlasting its own transport timeout is exactly what makes the deadline
    # unenforceable on this framing, and it scales with the line the peer sends, not with a bound.
    assert blocked > _CHUNK_LINE_TIMEOUT, (
        f"one chunked read1 blocked {blocked:.2f}s against a {_CHUNK_LINE_TIMEOUT}s timeout"
    )
