"""CPU-only containment tests for Vast asynchronous result retrieval."""

from __future__ import annotations

import errno
import socket
import ssl

import pytest

from flash.providers._lifecycle.net import deadline as deadline_module
from flash.providers.vast.client import result as vast_result

_PUBLIC_V4 = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 443))
_PUBLIC_V4_ALT = (
    socket.AF_INET,
    socket.SOCK_STREAM,
    socket.IPPROTO_TCP,
    "",
    ("1.1.1.1", 443),
)
_PUBLIC_V6 = (
    socket.AF_INET6,
    socket.SOCK_STREAM,
    socket.IPPROTO_TCP,
    "",
    ("2606:4700:4700::1111", 443, 0, 0),
)


class _RawSocket:
    def __init__(self):
        self.closed = False
        self.blocking = None

    def setblocking(self, blocking):
        self.blocking = blocking

    def close(self):
        self.closed = True


class _TlsSocket(_RawSocket):
    def __init__(self, response: bytes):
        super().__init__()
        self.response = bytearray(response)
        self.sent = bytearray()
        self.raw_socket = None
        self.handshakes = 0

    def do_handshake(self):
        self.handshakes += 1

    def send(self, data):
        self.sent.extend(data)
        return len(data)

    def recv(self, size):
        data = bytes(self.response[:size])
        del self.response[:size]
        return data

    def close(self):
        super().close()
        if self.raw_socket is not None:
            self.raw_socket.close()


class _Context:
    verify_mode = ssl.CERT_REQUIRED
    check_hostname = True

    def __init__(self, tls_socket: _TlsSocket, *, error: BaseException | None = None):
        self.tls_socket = tls_socket
        self.error = error
        self.calls = []

    def wrap_socket(self, raw_socket, *, server_hostname, do_handshake_on_connect):
        self.calls.append((raw_socket, server_hostname, do_handshake_on_connect))
        if self.error is not None:
            raise self.error
        self.tls_socket.raw_socket = raw_socket
        return self.tls_socket


def _http_response(
    status: int,
    body: bytes = b"",
    headers: tuple[tuple[str, str], ...] = (),
    *,
    include_content_length: bool = True,
) -> bytes:
    reason = {200: "OK", 302: "Found", 404: "Not Found", 500: "Error"}.get(status, "Status")
    fields = list(headers)
    if include_content_length and not any(
        name.lower() == "content-length" for name, _value in fields
    ):
        fields.insert(0, ("Content-Length", str(len(body))))
    raw_headers = b"".join(f"{name}: {value}\r\n".encode() for name, value in fields)
    return f"HTTP/1.1 {status} {reason}\r\n".encode() + raw_headers + b"\r\n" + body


def _install_transport(monkeypatch, response: bytes):
    raw_socket = _RawSocket()
    tls_socket = _TlsSocket(response)
    context = _Context(tls_socket)
    connects = []

    def create(family, sockaddr, deadline_at):
        connects.append((family, sockaddr, deadline_at))
        return raw_socket

    monkeypatch.setattr(vast_result, "_create_pinned_socket", create)
    monkeypatch.setattr(vast_result.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(deadline_module.time, "time", lambda: 100.0)
    return raw_socket, tls_socket, context, connects


def _prepare(monkeypatch, url: str, resolved=None):
    calls = []

    def resolver(host, port, *, type, proto):
        calls.append((host, port, type, proto))
        return list(resolved or [_PUBLIC_V4])

    monkeypatch.setattr(vast_result, "_resolve_result_host", resolver)
    request = vast_result.prepare_result_request(url)
    return request, calls


def test_result_origins_default_when_unset_or_blank(monkeypatch):
    monkeypatch.delenv(vast_result.RESULT_ORIGINS_ENV, raising=False)
    assert vast_result.configured_result_origins() == ("https://s3.amazonaws.com",)
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "")
    assert vast_result.configured_result_origins() == ("https://s3.amazonaws.com",)


def test_configured_result_origins_replace_default_and_match_exactly(monkeypatch):
    monkeypatch.setenv(
        vast_result.RESULT_ORIGINS_ENV,
        "https://logs.example.com,https://other.example.com",
    )
    assert vast_result.configured_result_origins() == (
        "https://logs.example.com",
        "https://other.example.com",
    )
    monkeypatch.setattr(vast_result, "_resolve_result_host", lambda *_args, **_kwargs: [_PUBLIC_V4])
    assert (
        vast_result.prepare_result_request("https://logs.example.com/x").host == "logs.example.com"
    )
    with pytest.raises(vast_result.VastResultError, match="configured HTTPS origin policy"):
        vast_result.prepare_result_request("https://s3.amazonaws.com/x")
    with pytest.raises(vast_result.VastResultError, match="configured HTTPS origin policy"):
        vast_result.prepare_result_request("https://sub.logs.example.com/x")


def test_ipv6_origin_and_request_use_bracketed_authority(monkeypatch):
    origin = "https://[2606:4700:4700::1111]"
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, origin)
    assert vast_result.configured_result_origins() == (origin,)
    request, calls = _prepare(monkeypatch, f"{origin}/signed?token=x", resolved=[_PUBLIC_V6])
    assert request.host == "2606:4700:4700::1111"
    assert request.authority == "[2606:4700:4700::1111]"
    assert calls == [("2606:4700:4700::1111", 443, socket.SOCK_STREAM, socket.IPPROTO_TCP)]
    _raw, tls, context, connects = _install_transport(monkeypatch, _http_response(200, b"ok"))
    assert request.fetch(deadline_at=110.0).body == b"ok"
    assert context.calls[0][1] == "2606:4700:4700::1111"
    assert b"Host: [2606:4700:4700::1111]\r\n" in bytes(tls.sent)
    assert connects[0][1] == ("2606:4700:4700::1111", 443, 0, 0)


@pytest.mark.parametrize(
    "value",
    [
        "http://logs.example.com",
        "https://user@logs.example.com",
        "https://logs.example.com:443",
        "https://[2606:4700:4700::1111]:443",
        "https://2606:4700:4700::1111",
        "https://LOGS.example.com",
        "https://logs.example.com.",
        "https://*.example.com",
        "https://[2606:4700:4700::1111%25eth0]",
        "https://logs.example.com/path",
        "https://logs.example.com?query",
        "https://logs.example.com#fragment",
        "https://logs.example.com#",
        " \t\n",
        "https://logs.example.com, https://other.example.com",
        "https://logs.example.com\n",
        "https://logs.example.com,,https://other.example.com",
        "https://logs.example.com,https://logs.example.com",
    ],
)
def test_result_origin_config_rejects_noncanonical_or_unsafe_values(monkeypatch, value):
    sentinel = "signed-secret-config-sentinel"
    configured = value.replace("query", sentinel)
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, configured)
    with pytest.raises(ValueError, match=vast_result.RESULT_ORIGINS_ENV) as exc_info:
        vast_result.configured_result_origins()
    detail = str(exc_info.value)
    assert vast_result.RESULT_ORIGINS_ENV in detail
    assert "must be a comma-separated list" in detail
    assert configured not in detail
    assert sentinel not in detail


@pytest.mark.parametrize(
    "url",
    [
        "http://logs.example.com/x",
        "https://user:pass@logs.example.com/x",
        "https://logs.example.com:443/x",
        "https://[2606:4700:4700::1111]:443/x",
        "https://2606:4700:4700::1111/x",
        "https://LOGS.example.com/x",
        "https://logs.example.com./x",
        "https://*.example.com/x",
        "https://logs.example.com/x#fragment",
        "https://logs.example.com/x#",
        "https://logs.example.com/x y",
        "https://logs.example.com/x\n",
    ],
)
def test_result_url_rejects_noncanonical_or_unsafe_authorities(monkeypatch, url):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    with pytest.raises(vast_result.VastResultError) as exc_info:
        vast_result.prepare_result_request(url)
    assert url not in str(exc_info.value)


@pytest.mark.parametrize(
    "unsafe",
    [
        "0.0.0.0",
        "127.0.0.1",
        "169.254.1.1",
        "10.0.0.1",
        "192.0.2.1",
        "198.18.0.1",
        "224.0.0.1",
        "240.0.0.1",
        "::",
        "::1",
        "fe80::1",
        "fc00::1",
        "2001:db8::1",
        "2620:4f:8000::1",
        "ff02::1",
        "100::",
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
        "::ffff:224.0.0.1",
        "::ffff:8.8.8.8",
    ],
)
def test_resolution_rejects_every_unsafe_address_class(monkeypatch, unsafe):
    family = socket.AF_INET6 if ":" in unsafe else socket.AF_INET
    sockaddr = (unsafe, 443, 0, 0) if family == socket.AF_INET6 else (unsafe, 443)
    resolved = [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)]
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    monkeypatch.setattr(vast_result, "_resolve_result_host", lambda *_args, **_kwargs: resolved)
    with pytest.raises(vast_result.VastResultError, match="unsafe address"):
        vast_result.prepare_result_request("https://logs.example.com/x")


@pytest.mark.parametrize(
    "unsafe",
    [
        "100.64.0.1",
        "100.127.255.254",
        "fec0::1",
        "feff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
    ],
)
def test_resolution_explicitly_rejects_cgnat_and_ipv6_site_local(monkeypatch, unsafe):
    family = socket.AF_INET6 if ":" in unsafe else socket.AF_INET
    sockaddr = (unsafe, 443, 0, 0) if family == socket.AF_INET6 else (unsafe, 443)
    resolved = [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)]
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    monkeypatch.setattr(vast_result, "_resolve_result_host", lambda *_args, **_kwargs: resolved)
    with pytest.raises(vast_result.VastResultError, match="unsafe address"):
        vast_result.prepare_result_request("https://logs.example.com/x")


def test_ipv6_scope_identifier_is_rejected_before_dns(monkeypatch):
    monkeypatch.setenv(
        vast_result.RESULT_ORIGINS_ENV,
        "https://[2606:4700:4700::1111]",
    )
    resolved = []
    monkeypatch.setattr(
        vast_result,
        "_resolve_result_host",
        lambda *_args, **_kwargs: resolved.append(True),
    )
    with pytest.raises(vast_result.VastResultError, match="configured HTTPS origin policy"):
        vast_result.prepare_result_request("https://[2606:4700:4700::1111%25eth0]/x")
    assert resolved == []


@pytest.mark.parametrize(
    "url",
    [
        "https://logs.example.com/café",
        "https://logs.example.com/x?q=café",
        "https://logs.example.com/\udcff",
        "https://logs.example.com" + chr(0xFF0F) + "x",
        "https://éxample.com/x",
    ],
)
def test_raw_non_ascii_signed_targets_are_rejected_without_dns_or_crash(monkeypatch, url):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    resolved = []
    monkeypatch.setattr(
        vast_result,
        "_resolve_result_host",
        lambda *_args, **_kwargs: resolved.append(True),
    )
    with pytest.raises(
        vast_result.VastResultError, match="configured HTTPS origin policy"
    ) as exc_info:
        vast_result.prepare_result_request(url)
    assert url not in str(exc_info.value)
    assert resolved == []


def test_non_ascii_target_encoding_is_normalized_to_vast_result_error():
    request = vast_result.VastResultRequest(
        host="logs.example.com",
        authority="logs.example.com",
        target="/café",
        addresses=(),
    )
    with pytest.raises(vast_result.VastResultError, match="configured HTTPS origin policy"):
        vast_result._request_bytes(request)


def test_returned_ipv6_scope_identifier_is_rejected(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    scoped = (
        socket.AF_INET6,
        socket.SOCK_STREAM,
        socket.IPPROTO_TCP,
        "",
        ("2606:4700:4700::1111", 443, 0, 3),
    )
    monkeypatch.setattr(vast_result, "_resolve_result_host", lambda *_args, **_kwargs: [scoped])
    with pytest.raises(vast_result.VastResultError, match="invalid address"):
        vast_result.prepare_result_request("https://logs.example.com/x")


def test_empty_dns_result_is_rejected(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    monkeypatch.setattr(vast_result, "_resolve_result_host", lambda *_args, **_kwargs: [])
    with pytest.raises(vast_result.VastResultError, match="no addresses"):
        vast_result.prepare_result_request("https://logs.example.com/x")


@pytest.mark.parametrize(
    "resolved",
    [
        [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 444))],
        [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("2606:4700::1", 443))],
        [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("8.8.8.8", 443, 0, 0),
            )
        ],
    ],
)
def test_resolution_rejects_port_or_family_mismatch(monkeypatch, resolved):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    monkeypatch.setattr(vast_result, "_resolve_result_host", lambda *_args, **_kwargs: resolved)
    with pytest.raises(vast_result.VastResultError, match="invalid address"):
        vast_result.prepare_result_request("https://logs.example.com/x")


def test_mixed_dns_result_fails_before_any_connect(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    monkeypatch.setattr(
        vast_result,
        "_resolve_result_host",
        lambda *_args, **_kwargs: [
            _PUBLIC_V4,
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443)),
        ],
    )
    connects = []
    monkeypatch.setattr(vast_result, "_create_pinned_socket", lambda *args: connects.append(args))
    with pytest.raises(vast_result.VastResultError, match="unsafe address"):
        vast_result.prepare_result_request("https://logs.example.com/x")
    assert connects == []


def test_resolution_happens_once_and_fetch_uses_exact_ipv4_tuple(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, resolver_calls = _prepare(monkeypatch, "https://logs.example.com/x")
    _raw, _tls, _context, connects = _install_transport(monkeypatch, _http_response(200, b"ok"))
    assert request.fetch(deadline_at=110.0).body == b"ok"
    assert resolver_calls == [("logs.example.com", 443, socket.SOCK_STREAM, socket.IPPROTO_TCP)]
    assert connects[0][0:2] == (socket.AF_INET, ("8.8.8.8", 443))


def test_ipv6_family_and_unscoped_sockaddr_are_pinned_verbatim(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(
        monkeypatch,
        "https://logs.example.com/x",
        resolved=[_PUBLIC_V6],
    )
    _raw, _tls, _context, connects = _install_transport(monkeypatch, _http_response(200, b"ok"))
    assert request.fetch(deadline_at=110.0).body == b"ok"
    assert connects[0][0:2] == (socket.AF_INET6, ("2606:4700:4700::1111", 443, 0, 0))


def test_vetted_addresses_share_deadline_and_fall_through_without_reresolving(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, resolver_calls = _prepare(
        monkeypatch,
        "https://logs.example.com/x",
        resolved=[_PUBLIC_V4, _PUBLIC_V4_ALT],
    )
    raw_socket = _RawSocket()
    tls_socket = _TlsSocket(_http_response(200, b"ok"))
    context = _Context(tls_socket)
    attempts = []

    def create(family, sockaddr, timeout):
        attempts.append((family, sockaddr, timeout))
        if len(attempts) == 1:
            raise OSError("first address unavailable")
        return raw_socket

    ticks = iter([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    monkeypatch.setattr(vast_result, "_create_pinned_socket", create)
    monkeypatch.setattr(vast_result.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(deadline_module.time, "time", lambda: next(ticks, 105.0))
    assert request.fetch(deadline_at=110.0).body == b"ok"
    assert resolver_calls == [("logs.example.com", 443, socket.SOCK_STREAM, socket.IPPROTO_TCP)]
    assert [attempt[1] for attempt in attempts] == [("8.8.8.8", 443), ("1.1.1.1", 443)]
    assert [attempt[2] for attempt in attempts] == [110.0, 110.0]


def test_tls_identity_host_header_identity_encoding_and_byte_exact_target(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    url = "https://logs.example.com/a%2Fb/%2f?x=one+two&x=one%20two&blank=&flag&sig=%2F%2f"
    request, _calls = _prepare(monkeypatch, url)
    raw, tls, context, _connects = _install_transport(monkeypatch, _http_response(200, b"ok"))
    assert request.target == "/a%2Fb/%2f?x=one+two&x=one%20two&blank=&flag&sig=%2F%2f"
    assert request.fetch(deadline_at=110.0).body == b"ok"
    assert context.calls == [(raw, "logs.example.com", False)]
    sent = bytes(tls.sent)
    assert sent.startswith(b"GET " + request.target.encode("ascii") + b" HTTP/1.1\r\n")
    assert b"Host: logs.example.com\r\n" in sent
    assert b"Accept-Encoding: identity\r\n" in sent


def test_explicit_empty_query_delimiter_is_preserved(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/signed?")
    assert request.target == "/signed?"
    _raw, tls, _context, _connects = _install_transport(monkeypatch, _http_response(200))
    request.fetch(deadline_at=110.0)
    assert bytes(tls.sent).startswith(b"GET /signed? HTTP/1.1\r\n")


def test_proxy_environment_is_ignored(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    _raw, _tls, _context, connects = _install_transport(monkeypatch, _http_response(200, b"ok"))
    assert request.fetch(deadline_at=110.0).body == b"ok"
    assert connects[0][1] == ("8.8.8.8", 443)


def test_redirect_is_not_followed_and_diagnostic_hides_signed_target(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    secret = "signed-query-secret"
    request, resolver_calls = _prepare(
        monkeypatch,
        f"https://logs.example.com/x?token={secret}",
    )
    _install_transport(
        monkeypatch,
        _http_response(302, headers=(("Location", "https://127.0.0.1/private"),)),
    )
    with pytest.raises(vast_result.VastResultError) as exc_info:
        request.fetch(deadline_at=110.0)
    assert "HTTP 302" in str(exc_info.value)
    assert secret not in str(exc_info.value)
    assert len(resolver_calls) == 1


@pytest.mark.parametrize("status", [404, 500])
def test_response_and_connection_close_on_non_success(monkeypatch, status):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    raw, tls, _context, _connects = _install_transport(monkeypatch, _http_response(status, b"no"))
    if status == 404:
        assert request.fetch(deadline_at=110.0).status == 404
    else:
        with pytest.raises(vast_result.VastResultError, match="HTTP 500"):
            request.fetch(deadline_at=110.0)
    assert raw.closed is True
    assert tls.closed is True


def test_success_closes_response_and_connection(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    raw, tls, _context, _connects = _install_transport(monkeypatch, _http_response(200, b"ok"))
    assert request.fetch(deadline_at=110.0).body == b"ok"
    assert raw.closed is True
    assert tls.closed is True


def test_body_read_failure_closes_response_and_connection(monkeypatch):
    class _ReadFailsTlsSocket(_TlsSocket):
        def recv(self, size):
            if b"\r\n\r\n" not in bytes(self.response):
                raise OSError("body read failed")
            delimiter = self.response.index(b"\r\n\r\n") + 4
            data = bytes(self.response[:delimiter])
            del self.response[:delimiter]
            return data

    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    raw = _RawSocket()
    tls = _ReadFailsTlsSocket(_http_response(200, b"ok"))
    context = _Context(tls)
    monkeypatch.setattr(vast_result, "_create_pinned_socket", lambda *_args: raw)
    monkeypatch.setattr(vast_result.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(deadline_module.time, "time", lambda: 100.0)
    with pytest.raises(vast_result.VastResultError, match="every vetted address"):
        request.fetch(deadline_at=110.0)
    assert raw.closed is True
    assert tls.closed is True


def test_tls_failure_falls_back_to_next_vetted_address(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(
        monkeypatch,
        "https://logs.example.com/x",
        resolved=[_PUBLIC_V4, _PUBLIC_V4_ALT],
    )
    raw_sockets = [_RawSocket(), _RawSocket()]
    first_tls = _TlsSocket(b"")
    second_tls = _TlsSocket(_http_response(200, b"ok"))
    contexts = iter(
        [
            _Context(first_tls, error=ssl.SSLError("handshake failed")),
            _Context(second_tls),
        ]
    )
    addresses = []

    def create(_family, sockaddr, _deadline_at):
        addresses.append(sockaddr)
        return raw_sockets[len(addresses) - 1]

    monkeypatch.setattr(vast_result, "_create_pinned_socket", create)
    monkeypatch.setattr(vast_result.ssl, "create_default_context", lambda: next(contexts))
    monkeypatch.setattr(deadline_module.time, "time", lambda: 100.0)
    assert request.fetch(deadline_at=110.0).body == b"ok"
    assert addresses == [("8.8.8.8", 443), ("1.1.1.1", 443)]
    assert all(sock.closed for sock in raw_sockets)


def test_tls_failure_closes_raw_socket(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    raw = _RawSocket()
    context = _Context(_TlsSocket(b""), error=ssl.SSLError("handshake failed"))
    monkeypatch.setattr(vast_result, "_create_pinned_socket", lambda *_args: raw)
    monkeypatch.setattr(vast_result.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(deadline_module.time, "time", lambda: 100.0)
    with pytest.raises(vast_result.VastResultError, match="every vetted address"):
        request.fetch(deadline_at=110.0)
    assert raw.closed is True


@pytest.mark.parametrize(
    ("size_text", "body"),
    [
        (b"", b"ok"),
        (b"+2", b"ok"),
        (b"0x2", b"ok"),
        (b"1_0", b"x" * 16),
        (b"-2", b"ok"),
        (b" 2", b"ok"),
        (b"2 ", b"ok"),
        (b"\xff", b"ok"),
    ],
)
def test_fetch_rejects_non_http_chunk_size_syntax(monkeypatch, size_text, body):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    secret = "signed-chunk-sentinel"
    request, _calls = _prepare(monkeypatch, f"https://logs.example.com/x?token={secret}")
    response = (
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
        + size_text
        + b"\r\n"
        + body
        + b"\r\n0\r\n\r\n"
    )
    raw, tls, _context, _connects = _install_transport(monkeypatch, response)
    with pytest.raises(vast_result.VastResultError, match="every vetted address") as exc_info:
        request.fetch(deadline_at=110.0)
    assert secret not in str(exc_info.value)
    assert raw.closed is True
    assert tls.closed is True


@pytest.mark.parametrize(
    ("size_text", "body"),
    [
        (b"0002;foo=bar", b"ok"),
        (b"A;foo=bar", b"0123456789"),
        (b"a", b"0123456789"),
    ],
)
def test_fetch_accepts_http_hex_chunk_sizes_and_extensions(monkeypatch, size_text, body):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    response = (
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
        + size_text
        + b"\r\n"
        + body
        + b"\r\n0;done=yes\r\nTrailer: value\r\n\r\n"
    )
    _install_transport(monkeypatch, response)
    assert request.fetch(deadline_at=110.0).body == body


def test_oversized_numeric_content_length_is_sanitized_and_closed(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    secret = "signed-length-sentinel"
    request, _calls = _prepare(monkeypatch, f"https://logs.example.com/x?token={secret}")
    response = _http_response(200, headers=(("Content-Length", "9" * 5_000),))
    raw, tls, _context, _connects = _install_transport(monkeypatch, response)
    with pytest.raises(vast_result.VastResultError, match="1048576-byte limit") as exc_info:
        request.fetch(deadline_at=110.0)
    assert secret not in str(exc_info.value)
    assert raw.closed is True
    assert tls.closed is True


def test_content_length_accepts_leading_zeroes_and_exact_one_mib(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    for declared, body in (
        ("0" * 5_000, b""),
        ("0000002", b"ok"),
        ("1048576", b"x" * 1_048_576),
    ):
        response = _http_response(200, body, headers=(("Content-Length", declared),))
        _install_transport(monkeypatch, response)
        assert request.fetch(deadline_at=110.0).body == body


def test_oversize_content_length_and_stream_are_rejected_and_closed(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    for response in (
        _http_response(200, headers=(("Content-Length", "1048577"),)),
        _http_response(200, b"x" * 1_048_577, include_content_length=False),
    ):
        raw, tls, _context, _connects = _install_transport(monkeypatch, response)
        with pytest.raises(vast_result.VastResultError, match="1048576-byte limit"):
            request.fetch(deadline_at=110.0)
        assert raw.closed is True
        assert tls.closed is True


def test_stream_without_content_length_accepts_exact_one_mib(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    body = b"x" * 1_048_576
    _install_transport(monkeypatch, _http_response(200, body, include_content_length=False))
    assert request.fetch(deadline_at=110.0).body == body


def test_malformed_headers_close_both_sockets(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    raw, tls, _context, _connects = _install_transport(
        monkeypatch,
        b"HTTP/1.1 200 OK\r\nMalformed\r\n\r\n",
    )
    with pytest.raises(vast_result.VastResultError, match="every vetted address"):
        request.fetch(deadline_at=110.0)
    assert raw.closed is True
    assert tls.closed is True


def test_default_socket_creation_closes_on_connect_failure(monkeypatch):
    created = []

    class _ConnectFails(_RawSocket):
        def connect_ex(self, _sockaddr):
            return errno.ECONNREFUSED

    def socket_factory(*_args):
        sock = _ConnectFails()
        created.append(sock)
        return sock

    monkeypatch.setattr(vast_result.socket, "socket", socket_factory)
    monkeypatch.setattr(deadline_module.time, "time", lambda: 1.0)
    with pytest.raises(OSError, match="Connection refused"):
        vast_result._default_create_pinned_socket(socket.AF_INET, ("8.8.8.8", 443), 2.0)
    assert created[0].closed is True


def test_verified_ssl_context_requires_certificates_and_hostname_checks():
    context = vast_result._verified_ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def _install_deadline_phase(monkeypatch, tls_socket, clock):
    raw_socket = _RawSocket()
    context = _Context(tls_socket)
    monkeypatch.setattr(vast_result, "_create_pinned_socket", lambda *_args: raw_socket)
    monkeypatch.setattr(vast_result.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(deadline_module.time, "time", lambda: clock["now"])
    return raw_socket


def test_absolute_deadline_expires_during_tls_handshake(monkeypatch):
    class _SlowHandshake(_TlsSocket):
        def do_handshake(self):
            clock["now"] += 4.0
            if clock["now"] < 110.0:
                raise ssl.SSLWantReadError

    clock = {"now": 100.0}
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    tls = _SlowHandshake(b"")
    raw = _install_deadline_phase(monkeypatch, tls, clock)
    monkeypatch.setattr(vast_result, "_wait_for_socket", lambda *_args, **_kwargs: None)
    with pytest.raises(vast_result.VastResultError, match="deadline expired"):
        request.fetch(deadline_at=110.0)
    assert raw.closed is True
    assert tls.closed is True


def test_absolute_deadline_expires_during_request_write(monkeypatch):
    class _SlowWrite(_TlsSocket):
        def send(self, data):
            clock["now"] += 2.0
            return 1

    clock = {"now": 100.0}
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    tls = _SlowWrite(_http_response(200, b"ok"))
    _install_deadline_phase(monkeypatch, tls, clock)
    with pytest.raises(vast_result.VastResultError, match="deadline expired"):
        request.fetch(deadline_at=110.0)


def test_absolute_deadline_expires_during_response_headers(monkeypatch):
    class _SlowHeaders(_TlsSocket):
        def recv(self, _size):
            clock["now"] += 2.0
            data = bytes(self.response[:1])
            del self.response[:1]
            return data

    clock = {"now": 100.0}
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    tls = _SlowHeaders(_http_response(200, b"ok"))
    _install_deadline_phase(monkeypatch, tls, clock)
    with pytest.raises(vast_result.VastResultError, match="deadline expired"):
        request.fetch(deadline_at=110.0)


def test_absolute_deadline_expires_during_response_body(monkeypatch):
    class _SlowBody(_TlsSocket):
        def recv(self, size):
            if b"\r\n\r\n" in bytes(self.response):
                delimiter = self.response.index(b"\r\n\r\n") + 4
                data = bytes(self.response[:delimiter])
                del self.response[:delimiter]
                return data
            clock["now"] += 6.0
            return super().recv(1)

    clock = {"now": 100.0}
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    tls = _SlowBody(_http_response(200, b"ok"))
    _install_deadline_phase(monkeypatch, tls, clock)
    with pytest.raises(vast_result.VastResultError, match="deadline expired"):
        request.fetch(deadline_at=110.0)


def test_absolute_deadline_expires_during_tcp_connect(monkeypatch):
    class _PendingConnect(_RawSocket):
        def connect_ex(self, _sockaddr):
            return errno.EINPROGRESS

    clock = {"now": 100.0}
    sock = _PendingConnect()
    monkeypatch.setattr(vast_result.socket, "socket", lambda *_args: sock)

    def wait(_sock, *, readable, writable, deadline_at):
        assert readable is False
        assert writable is True
        assert deadline_at == 110.0
        clock["now"] = 111.0
        vast_result._remaining_timeout(deadline_at)

    monkeypatch.setattr(vast_result, "_wait_for_socket", wait)
    monkeypatch.setattr(deadline_module.time, "time", lambda: clock["now"])
    with pytest.raises(TimeoutError, match="deadline expired"):
        vast_result._default_create_pinned_socket(socket.AF_INET, ("8.8.8.8", 443), 110.0)
    assert sock.closed is True


def test_deadline_expiry_before_connect_does_not_open_socket(monkeypatch):
    monkeypatch.setenv(vast_result.RESULT_ORIGINS_ENV, "https://logs.example.com")
    request, _calls = _prepare(monkeypatch, "https://logs.example.com/x")
    connects = []
    monkeypatch.setattr(vast_result, "_create_pinned_socket", lambda *args: connects.append(args))
    monkeypatch.setattr(deadline_module.time, "time", lambda: 110.0)
    with pytest.raises(vast_result.VastResultError, match="deadline expired"):
        request.fetch(deadline_at=110.0)
    assert connects == []
