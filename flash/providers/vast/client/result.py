"""Contained HTTPS retrieval for Vast's asynchronously materialized log results."""

from __future__ import annotations

import errno
import http.client
import ipaddress
import os
import re
import select
import socket
import ssl
import urllib.parse
from dataclasses import dataclass

from flash.providers._lifecycle.net.deadline import remaining_seconds

RESULT_ORIGINS_ENV = "FLASH_VAST_RESULT_ORIGINS"
_DEFAULT_RESULT_ORIGINS = ("https://s3.amazonaws.com",)
_MAX_RESULT_BODY_BYTES = 1_048_576
_MAX_HEADER_BYTES = 65_536
_MAX_CHUNK_LINE_BYTES = 8_192
_READ_SIZE = 65_536

_CONFIG_RULE = (
    "must be a comma-separated list of exact canonical HTTPS origins without credentials, "
    "ports, paths, queries, fragments, wildcards, spaces, or control characters"
)
_RESULT_URL_RULE = "Vast result URL violates the configured HTTPS origin policy"
_HEADER_NAME = re.compile(rb"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_CHUNK_SIZE = re.compile(rb"[0-9A-Fa-f]+")
_STATUS_LINE = re.compile(rb"HTTP/1\.[01] ([0-9]{3})(?:[ \t][\x20-\x7e\t]*)?")
_MAX_RESULT_BODY_DIGITS = str(_MAX_RESULT_BODY_BYTES).encode("ascii")

# python 3.11 and 3.12 differ on parts of ipaddress.is_global. these explicit iana special-use
# ranges make the rejection stable, while the runtime classification below fails closed on any
# additional non-global range known to the interpreter.
_NON_GLOBAL_V4 = tuple(
    ipaddress.ip_network(network)
    for network in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.31.196.0/24",
        "192.52.193.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "192.175.48.0/24",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
    )
)
_NON_GLOBAL_V6 = tuple(
    ipaddress.ip_network(network)
    for network in (
        "::/96",
        "::ffff:0:0/96",
        "64:ff9b::/96",
        "64:ff9b:1::/48",
        "100::/64",
        "2001::/23",
        "2001:db8::/32",
        "2002::/16",
        "2620:4f:8000::/48",
        "3fff::/20",
        "5f00::/16",
        "fc00::/7",
        "fe80::/10",
        "fec0::/10",
        "ff00::/8",
    )
)


class VastResultError(RuntimeError):
    """A contained result URL could not be fetched safely."""


@dataclass(frozen=True)
class _PinnedAddress:
    family: int
    sockaddr: tuple


@dataclass(frozen=True)
class VastResultResponse:
    status: int
    body: bytes


@dataclass(frozen=True)
class VastResultRequest:
    """Validated request target and the one DNS result set it may connect to."""

    host: str
    authority: str
    target: str
    addresses: tuple[_PinnedAddress, ...]

    def fetch(self, *, deadline_at: float) -> VastResultResponse:
        """Fetch once, trying only the pre-resolved addresses within one shared deadline."""
        last_network_error: BaseException | None = None
        for address in self.addresses:
            if remaining_seconds(deadline_at) <= 0:
                break
            try:
                return _fetch_from_address(self, address, deadline_at)
            except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
                last_network_error = exc
        if remaining_seconds(deadline_at) <= 0:
            raise VastResultError("Vast result retrieval deadline expired") from None
        if last_network_error is not None:
            raise VastResultError("Vast result retrieval failed on every vetted address") from None
        raise VastResultError("Vast result retrieval has no vetted address")


def configured_result_origins() -> tuple[str, ...]:
    """Return the operator's exact origin allowlist, or the S3 default when blank or unset."""
    raw = os.environ.get(RESULT_ORIGINS_ENV)
    if raw is None or raw == "":
        return _DEFAULT_RESULT_ORIGINS
    if _has_raw_space_or_control(raw):
        raise ValueError(f"{RESULT_ORIGINS_ENV} {_CONFIG_RULE}")
    parts = raw.split(",")
    if not parts or any(not part for part in parts):
        raise ValueError(f"{RESULT_ORIGINS_ENV} {_CONFIG_RULE}")
    try:
        origins = tuple(_canonical_origin(part) for part in parts)
    except ValueError:
        raise ValueError(f"{RESULT_ORIGINS_ENV} {_CONFIG_RULE}") from None
    if len(set(origins)) != len(origins):
        raise ValueError(f"{RESULT_ORIGINS_ENV} {_CONFIG_RULE}")
    return origins


def prepare_result_request(url: object) -> VastResultRequest:
    """Validate one signed URL and resolve its host exactly once.

    Python's stdlib resolver has no portable deadline control, so ``getaddrinfo`` itself remains
    unbounded. The caller's absolute deadline is checked after this one resolution and shared by
    every subsequent connect, TLS, response, and body-read attempt.
    """
    if not isinstance(url, str) or _has_raw_space_or_control(url):
        raise VastResultError(_RESULT_URL_RULE)
    try:
        split = urllib.parse.urlsplit(url)
        origin, host, authority = _origin_host_and_authority(split, exact_url=url)
        if not split.path.isascii() or not split.query.isascii():
            raise ValueError("signed target must already be ascii")
    except (TypeError, ValueError):
        raise VastResultError(_RESULT_URL_RULE) from None
    try:
        allowed_origins = configured_result_origins()
    except ValueError:
        raise VastResultError("Vast result origin configuration is invalid") from None
    if origin not in allowed_origins:
        raise VastResultError(_RESULT_URL_RULE)

    target = split.path or "/"
    if _has_explicit_query_delimiter(url):
        target += f"?{split.query}"

    try:
        resolved = _resolve_result_host(
            host,
            443,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except OSError:
        raise VastResultError("Vast result host resolution failed") from None
    addresses = _vet_resolved_addresses(resolved)
    return VastResultRequest(host=host, authority=authority, target=target, addresses=addresses)


def _has_raw_space_or_control(value: str) -> bool:
    return any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in value
    )


def _canonical_origin(value: str) -> str:
    split = urllib.parse.urlsplit(value)
    origin, _host, _authority = _origin_host_and_authority(split, exact_url=value)
    if split.path or split.query or split.fragment:
        raise ValueError("origin has non-authority components")
    if origin != value:
        raise ValueError("origin is not canonical")
    return origin


def _origin_host_and_authority(
    split: urllib.parse.SplitResult, *, exact_url: str
) -> tuple[str, str, str]:
    if not exact_url.startswith("https://") or split.scheme != "https":
        raise ValueError("scheme must be canonical https")
    if split.username is not None or split.password is not None or "#" in exact_url:
        raise ValueError("credentials and fragments are forbidden")
    try:
        port = split.port
    except ValueError:
        raise ValueError("invalid port") from None
    if port is not None:
        raise ValueError("explicit ports are forbidden")
    host = split.hostname
    if not host or "%" in host or "*" in host or not host.isascii() or host != host.lower():
        raise ValueError("host is not canonical")
    authority = _canonical_authority(host)
    if split.netloc != authority:
        raise ValueError("authority is not canonical")
    return f"https://{authority}", host, authority


def _canonical_authority(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if all(character in "0123456789." for character in host):
            raise ValueError("numeric host is not a canonical IP address") from None
        labels = host.split(".")
        if any(
            not label
            or len(label) > 63
            or label[0] == "-"
            or label[-1] == "-"
            or not all(character.isalnum() or character == "-" for character in label)
            for label in labels
        ):
            raise ValueError("host is not a canonical DNS name") from None
        return host
    canonical = address.compressed
    if host != canonical:
        raise ValueError("IP address is not canonical")
    return f"[{canonical}]" if address.version == 6 else canonical


def _has_explicit_query_delimiter(url: str) -> bool:
    authority_end = url.find("/", len("https://"))
    query_at = url.find("?")
    return query_at != -1 and (authority_end == -1 or query_at >= authority_end)


def _vet_resolved_addresses(resolved: list[tuple]) -> tuple[_PinnedAddress, ...]:
    if not resolved:
        raise VastResultError("Vast result host resolution returned no addresses")
    addresses: list[_PinnedAddress] = []
    seen: set[tuple[int, tuple]] = set()
    for entry in resolved:
        family, sockaddr = _validated_resolver_entry(entry)
        try:
            address = ipaddress.ip_address(sockaddr[0])
        except (TypeError, ValueError):
            raise VastResultError(
                "Vast result host resolution returned an invalid address"
            ) from None
        if address.version != (4 if family == socket.AF_INET else 6):
            raise VastResultError("Vast result host resolution returned an invalid address family")
        if _is_unsafe_address(address):
            raise VastResultError("Vast result host resolution included an unsafe address")
        key = (family, sockaddr)
        if key not in seen:
            seen.add(key)
            addresses.append(_PinnedAddress(family=family, sockaddr=sockaddr))
    if not addresses:
        raise VastResultError("Vast result host resolution returned no addresses")
    return tuple(addresses)


def _validated_resolver_entry(entry: tuple) -> tuple[int, tuple]:
    if not isinstance(entry, tuple) or len(entry) != 5:
        raise VastResultError("Vast result host resolution returned an invalid address")
    family, socktype, proto, _canonname, sockaddr = entry
    if family not in (socket.AF_INET, socket.AF_INET6):
        raise VastResultError("Vast result host resolution returned an unsupported address")
    if socktype != socket.SOCK_STREAM or proto != socket.IPPROTO_TCP:
        raise VastResultError("Vast result host resolution returned an invalid socket type")
    expected_length = 2 if family == socket.AF_INET else 4
    if (
        not isinstance(sockaddr, tuple)
        or len(sockaddr) != expected_length
        or sockaddr[1] != 443
        or not isinstance(sockaddr[0], str)
        or "%" in sockaddr[0]
        or (family == socket.AF_INET6 and sockaddr[3] != 0)
    ):
        raise VastResultError("Vast result host resolution returned an invalid address")
    return family, sockaddr


def _is_unsafe_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) else None
    if mapped is not None:
        return True
    ranges = _NON_GLOBAL_V4 if isinstance(address, ipaddress.IPv4Address) else _NON_GLOBAL_V6
    return any(address in network for network in ranges) or not address.is_global


def _remaining_timeout(deadline_at: float) -> float:
    remaining = remaining_seconds(deadline_at)
    if remaining <= 0:
        raise TimeoutError("Vast result retrieval deadline expired")
    return remaining


def _verified_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise VastResultError("Vast result TLS verification is unavailable")
    return context


def _request_bytes(request: VastResultRequest) -> bytes:
    try:
        return (
            f"GET {request.target} HTTP/1.1\r\n"
            f"Host: {request.authority}\r\n"
            "Accept-Encoding: identity\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
    except UnicodeEncodeError:
        raise VastResultError(_RESULT_URL_RULE) from None


def _fetch_from_address(
    request: VastResultRequest,
    address: _PinnedAddress,
    deadline_at: float,
) -> VastResultResponse:
    raw_socket = _create_pinned_socket(address.family, address.sockaddr, deadline_at)
    tls_socket: ssl.SSLSocket | None = None
    try:
        _remaining_timeout(deadline_at)
        context = _verified_ssl_context()
        tls_socket = context.wrap_socket(
            raw_socket,
            server_hostname=request.host,
            do_handshake_on_connect=False,
        )
        _remaining_timeout(deadline_at)
        transport = _DeadlineTransport(tls_socket, deadline_at)
        transport.handshake()
        transport.send_all(_request_bytes(request))
        status, headers, buffered_body = transport.read_response_head()
        _remaining_timeout(deadline_at)
        if status == 404:
            return VastResultResponse(status=404, body=b"")
        if status != 200:
            raise VastResultError(f"Vast result retrieval returned HTTP {status}")
        body = _read_response_body(transport, headers, buffered_body)
        _remaining_timeout(deadline_at)
        return VastResultResponse(status=200, body=body)
    finally:
        if tls_socket is not None:
            tls_socket.close()
        else:
            raw_socket.close()


class _DeadlineTransport:
    """Nonblocking TLS I/O bounded by one absolute deadline."""

    def __init__(self, sock: ssl.SSLSocket, deadline_at: float) -> None:
        self._sock = sock
        self._deadline_at = deadline_at
        self._sock.setblocking(False)

    def handshake(self) -> None:
        while True:
            _remaining_timeout(self._deadline_at)
            try:
                self._sock.do_handshake()
            except ssl.SSLWantReadError:
                self._wait(readable=True)
                continue
            except ssl.SSLWantWriteError:
                self._wait(writable=True)
                continue
            _remaining_timeout(self._deadline_at)
            return

    def send_all(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            _remaining_timeout(self._deadline_at)
            try:
                sent = self._sock.send(view)
            except ssl.SSLWantReadError:
                self._wait(readable=True)
                continue
            except (ssl.SSLWantWriteError, BlockingIOError):
                self._wait(writable=True)
                continue
            _remaining_timeout(self._deadline_at)
            if sent <= 0:
                raise OSError("Vast result request write failed")
            view = view[sent:]

    def recv(self, size: int) -> bytes:
        while True:
            _remaining_timeout(self._deadline_at)
            try:
                data = self._sock.recv(size)
            except ssl.SSLWantWriteError:
                self._wait(writable=True)
                continue
            except (ssl.SSLWantReadError, BlockingIOError):
                self._wait(readable=True)
                continue
            _remaining_timeout(self._deadline_at)
            return data

    def read_response_head(self) -> tuple[int, dict[bytes, tuple[bytes, ...]], bytes]:
        data = bytearray()
        delimiter = b"\r\n\r\n"
        while delimiter not in data:
            if len(data) >= _MAX_HEADER_BYTES:
                raise http.client.LineTooLong("Vast result response headers")
            chunk = self.recv(min(_READ_SIZE, _MAX_HEADER_BYTES + 1 - len(data)))
            if not chunk:
                raise http.client.RemoteDisconnected("Vast result response ended before headers")
            data.extend(chunk)
            if len(data) > _MAX_HEADER_BYTES and delimiter not in data:
                raise http.client.LineTooLong("Vast result response headers")
        head, body = bytes(data).split(delimiter, 1)
        return (*_parse_response_head(head), body)

    def _wait(self, *, readable: bool = False, writable: bool = False) -> None:
        _wait_for_socket(
            self._sock,
            readable=readable,
            writable=writable,
            deadline_at=self._deadline_at,
        )


def _parse_response_head(head: bytes) -> tuple[int, dict[bytes, tuple[bytes, ...]]]:
    lines = head.split(b"\r\n")
    match = _STATUS_LINE.fullmatch(lines[0]) if lines else None
    if match is None:
        raise http.client.BadStatusLine(lines[0] if lines else b"")
    status = int(match.group(1))
    if status < 100 or status > 599:
        raise http.client.BadStatusLine(lines[0])
    headers: dict[bytes, list[bytes]] = {}
    for line in lines[1:]:
        if not line or line[:1] in b" \t" or b":" not in line:
            raise http.client.HTTPException("Vast result response contained malformed headers")
        name, value = line.split(b":", 1)
        if _HEADER_NAME.fullmatch(name) is None or _has_invalid_header_value(value):
            raise http.client.HTTPException("Vast result response contained malformed headers")
        headers.setdefault(name.lower(), []).append(value.strip(b" \t"))
    return status, {name: tuple(values) for name, values in headers.items()}


def _has_invalid_header_value(value: bytes) -> bool:
    return any((byte < 0x20 and byte != 0x09) or byte == 0x7F for byte in value)


def _read_response_body(
    transport: _DeadlineTransport,
    headers: dict[bytes, tuple[bytes, ...]],
    buffered: bytes,
) -> bytes:
    transfer_encoding = _single_header_value(headers, b"transfer-encoding")
    content_length = _content_length(headers)
    if transfer_encoding is not None:
        if content_length is not None or transfer_encoding.lower() != b"chunked":
            raise http.client.HTTPException("Vast result response framing is unsupported")
        return _read_chunked_body(transport, buffered)
    if content_length is not None:
        if content_length > _MAX_RESULT_BODY_BYTES:
            raise VastResultError("Vast result body exceeds the 1048576-byte limit")
        return _read_exact_body(transport, buffered, content_length)
    return _read_to_eof(transport, buffered)


def _single_header_value(headers: dict[bytes, tuple[bytes, ...]], name: bytes) -> bytes | None:
    values = headers.get(name)
    if values is None:
        return None
    if len(values) != 1:
        raise http.client.HTTPException("Vast result response framing is ambiguous")
    return values[0]


def _content_length(headers: dict[bytes, tuple[bytes, ...]]) -> int | None:
    values = headers.get(b"content-length")
    if values is None:
        return None
    parts = [part.strip() for value in values for part in value.split(b",")]
    if not parts or any(not part.isdigit() for part in parts) or len(set(parts)) != 1:
        raise http.client.HTTPException("Vast result response Content-Length is invalid")
    normalized = parts[0].lstrip(b"0") or b"0"
    if len(normalized) > len(_MAX_RESULT_BODY_DIGITS) or (
        len(normalized) == len(_MAX_RESULT_BODY_DIGITS) and normalized > _MAX_RESULT_BODY_DIGITS
    ):
        return _MAX_RESULT_BODY_BYTES + 1
    return int(normalized)


def _read_exact_body(transport: _DeadlineTransport, buffered: bytes, size: int) -> bytes:
    body = bytearray(buffered[:size])
    while len(body) < size:
        chunk = transport.recv(min(_READ_SIZE, size - len(body)))
        if not chunk:
            raise http.client.IncompleteRead(bytes(body), size - len(body))
        body.extend(chunk)
    return bytes(body)


def _read_to_eof(transport: _DeadlineTransport, buffered: bytes) -> bytes:
    body = bytearray(buffered)
    if len(body) > _MAX_RESULT_BODY_BYTES:
        raise VastResultError("Vast result body exceeds the 1048576-byte limit")
    while True:
        chunk = transport.recv(min(_READ_SIZE, _MAX_RESULT_BODY_BYTES + 1 - len(body)))
        if not chunk:
            return bytes(body)
        body.extend(chunk)
        if len(body) > _MAX_RESULT_BODY_BYTES:
            raise VastResultError("Vast result body exceeds the 1048576-byte limit")


def _read_chunked_body(transport: _DeadlineTransport, buffered: bytes) -> bytes:
    reader = _BufferedTransportReader(transport, buffered)
    body = bytearray()
    while True:
        line = reader.readline(_MAX_CHUNK_LINE_BYTES)
        size_text = line.split(b";", 1)[0]
        if _CHUNK_SIZE.fullmatch(size_text) is None:
            raise http.client.HTTPException("Vast result response chunk size is invalid")
        chunk_size = int(size_text, 16)
        if chunk_size == 0:
            reader.read_trailers()
            return bytes(body)
        if len(body) + chunk_size > _MAX_RESULT_BODY_BYTES:
            raise VastResultError("Vast result body exceeds the 1048576-byte limit")
        body.extend(reader.readexactly(chunk_size))
        if reader.readexactly(2) != b"\r\n":
            raise http.client.HTTPException("Vast result response chunk is malformed")


class _BufferedTransportReader:
    def __init__(self, transport: _DeadlineTransport, initial: bytes) -> None:
        self._transport = transport
        self._buffer = bytearray(initial)

    def readexactly(self, size: int) -> bytes:
        while len(self._buffer) < size:
            chunk = self._transport.recv(min(_READ_SIZE, size - len(self._buffer)))
            if not chunk:
                raise http.client.IncompleteRead(bytes(self._buffer), size - len(self._buffer))
            self._buffer.extend(chunk)
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result

    def readline(self, limit: int) -> bytes:
        while True:
            delimiter = self._buffer.find(b"\r\n")
            if delimiter >= 0:
                if delimiter > limit:
                    raise http.client.LineTooLong("Vast result response line")
                result = bytes(self._buffer[:delimiter])
                del self._buffer[: delimiter + 2]
                return result
            if len(self._buffer) > limit:
                raise http.client.LineTooLong("Vast result response line")
            chunk = self._transport.recv(min(_READ_SIZE, limit + 1 - len(self._buffer)))
            if not chunk:
                raise http.client.IncompleteRead(bytes(self._buffer))
            self._buffer.extend(chunk)

    def read_trailers(self) -> None:
        total = 0
        while True:
            line = self.readline(_MAX_HEADER_BYTES)
            total += len(line) + 2
            if total > _MAX_HEADER_BYTES:
                raise http.client.LineTooLong("Vast result response trailers")
            if not line:
                return
            if line[:1] in b" \t" or b":" not in line:
                raise http.client.HTTPException("Vast result response trailer is malformed")
            name, value = line.split(b":", 1)
            if _HEADER_NAME.fullmatch(name) is None or _has_invalid_header_value(value):
                raise http.client.HTTPException("Vast result response trailer is malformed")


def _default_wait_for_socket(
    sock: socket.socket,
    *,
    readable: bool,
    writable: bool,
    deadline_at: float,
) -> None:
    timeout = _remaining_timeout(deadline_at)
    readable_sockets = [sock] if readable else []
    writable_sockets = [sock] if writable else []
    ready_read, ready_write, exceptional = select.select(
        readable_sockets,
        writable_sockets,
        [sock],
        timeout,
    )
    _remaining_timeout(deadline_at)
    if exceptional:
        raise OSError("Vast result socket reported an exceptional condition")
    if not ready_read and not ready_write:
        raise TimeoutError("Vast result retrieval deadline expired")


def _default_create_pinned_socket(
    family: int, sockaddr: tuple, deadline_at: float
) -> socket.socket:
    sock = socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
    try:
        sock.setblocking(False)
        result = sock.connect_ex(sockaddr)
        if result not in (0, errno.EISCONN):
            if result not in (errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, errno.EINTR):
                raise OSError(result, os.strerror(result))
            _wait_for_socket(sock, readable=False, writable=True, deadline_at=deadline_at)
            result = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if result:
                raise OSError(result, os.strerror(result))
        _remaining_timeout(deadline_at)
        return sock
    except BaseException:
        sock.close()
        raise


# narrow seams for cpu-only dns, exact-destination, and simulated-time transport tests
_resolve_result_host = socket.getaddrinfo
_create_pinned_socket = _default_create_pinned_socket
_wait_for_socket = _default_wait_for_socket
