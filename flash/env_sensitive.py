"""Bounded detectors for structured credentials that need small parsers."""

from __future__ import annotations

import ipaddress
import re

_MAX_BODY = 256
_HEX_BODY = re.compile(r"[0-9a-fA-F]{32,}")
_MAX_GNUPG_HEADER_BYTES = 64 << 10
_MAX_GNUPG_KEY_BYTES = 1 << 20
_MAX_GNUPG_NODES = 10_000
_MAX_GNUPG_DEPTH = 32
_GNUPG_BLANK = re.compile(rb"\A(?:[ \t]*\r?\n)+")
_GNUPG_LABEL = re.compile(rb"(?:Created|Keygrip|Token|Label|Comment):")
_GNUPG_METADATA = re.compile(rb"(?:Created|Keygrip|Token|Label|Comment):[ \t]*[ -~]{1,1024}\r?\n")
_GNUPG_KEY = re.compile(rb"Key:[ \t]*")
_GNUPG_ROOTS = {b"private-key", b"protected-private-key", b"shadowed-private-key"}
_GNUPG_ALGORITHMS = {b"rsa", b"dsa", b"elg", b"ecc", b"ecdsa", b"eddsa", b"ecdh"}
_GNUPG_PRIVATE_FIELDS = {b"d", b"p", b"q", b"u", b"x"}
_GNUPG_HEX = frozenset(b"0123456789abcdefABCDEF")

_URI_SCHEME = re.compile(rb"(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]{0,31}://")
_MAX_URI_AUTHORITY_BYTES = 768
_URI_HEX = frozenset(b"0123456789abcdefABCDEF")
_URI_USERINFO = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~!$&'()*+,;=:%"
)
_URI_AUTHORITY = _URI_USERINFO | frozenset(b"@[]")
_URI_TERMINATORS = frozenset(b'/?#\t\r\n "<>')


def _is_high_entropy(body: bytes, *, hex_is_issued: bool = False) -> bool:
    """Whether a key body looks issued rather than hand-written."""
    text = body.decode("ascii", "ignore").replace("_", "").replace("-", "")
    if len(set(text)) <= 2:
        return False
    if hex_is_issued and _HEX_BODY.fullmatch(text):
        return True
    return not (text.isalpha() and (text.isupper() or text.islower()))


def _gnupg_expression_start(data: bytes) -> tuple[re.Match[bytes] | None, int | None, bool]:
    """The Key expression start, or whether a recognized header exceeded its bound."""
    at, recognized = 0, False
    limit = min(len(data), _MAX_GNUPG_HEADER_BYTES)
    while at < limit:
        if data.startswith(b"\r\n", at):
            recognized = True
            at += 2
            continue
        if data.startswith(b"\n", at):
            recognized = True
            at += 1
            continue
        if found := _GNUPG_KEY.match(data, at):
            start = found.end()
            while start < len(data) and data[start] in b" \t":
                start += 1
            if start >= _MAX_GNUPG_HEADER_BYTES:
                return found, None, True
            return found, start, False
        if found := _GNUPG_METADATA.match(data, at):
            recognized = True
            at = found.end()
            continue
        line_end = data.find(b"\n", at, limit + 1)
        if line_end >= 0 and not data[at:line_end].strip():
            recognized = True
            at = line_end + 1
            continue
        label = _GNUPG_LABEL.match(data, at)
        if line_end < 0 and len(data) > limit and (recognized or label is not None):
            return label, None, True
        return None, None, False
    if len(data) > limit and recognized:
        return _GNUPG_LABEL.search(data) or _GNUPG_BLANK.match(data), None, True
    return None, None, False


class _SExpression:
    """A bounded parser for the advanced S-expression syntax GnuPG writes."""

    def __init__(self, data: bytes, start: int):
        self.data = data
        self.at = start
        self.end = min(len(data), start + _MAX_GNUPG_KEY_BYTES)
        self.nodes = 0

    def _skip_space(self) -> None:
        while self.at < self.end and self.data[self.at] in b" \t\r\n":
            self.at += 1

    def _atom(self) -> bytes | None:
        if self.at >= self.end:
            return None
        if self.data[self.at] == 0x22:
            return self._quoted()
        if self.data[self.at] == 0x23:
            return self._hex()
        start = self.at
        while self.at < self.end and self.data[self.at] not in b"() \t\r\n":
            self.at += 1
        atom = self.data[start : self.at]
        if not atom or any(byte < 0x21 or byte > 0x7E for byte in atom):
            return None
        if b":" in atom and atom.split(b":", 1)[0].isdigit():
            length, payload = atom.split(b":", 1)
            if int(length) != len(payload):
                return None
            return payload
        return atom

    def _quoted(self) -> bytes | None:
        self.at += 1
        plain = bytearray()
        escaped = False
        while self.at < self.end:
            byte = self.data[self.at]
            self.at += 1
            if escaped:
                plain.append(byte)
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                return bytes(plain)
            elif byte < 0x20:
                return None
            else:
                plain.append(byte)
        return None

    def _hex(self) -> bytes | None:
        self.at += 1
        close = self.data.find(b"#", self.at, self.end)
        if close < 0:
            return None
        encoded = self.data[self.at : close]
        if not encoded or len(encoded) % 2 or any(byte not in _GNUPG_HEX for byte in encoded):
            return None
        self.at = close + 1
        return encoded

    def value(self, depth: int = 0) -> bytes | list[object] | None:
        if depth > _MAX_GNUPG_DEPTH or self.nodes >= _MAX_GNUPG_NODES:
            return None
        self.nodes += 1
        self._skip_space()
        if self.at >= self.end:
            return None
        if self.data[self.at] != 0x28:
            return self._atom()
        self.at += 1
        values: list[object] = []
        while True:
            self._skip_space()
            if self.at >= self.end:
                return None
            if self.data[self.at] == 0x29:
                self.at += 1
                return values
            value = self.value(depth + 1)
            if value is None:
                return None
            values.append(value)


def _named(node: object, names: set[bytes]) -> bool:
    return isinstance(node, list) and bool(node) and isinstance(node[0], bytes) and node[0] in names


def _contains_named(node: object, names: set[bytes]) -> bool:
    if _named(node, names) and isinstance(node, list) and len(node) > 1:
        return True
    return isinstance(node, list) and any(_contains_named(child, names) for child in node[1:])


def _valid_gnupg_key(node: object) -> bool:
    if not isinstance(node, list) or len(node) < 2 or node[0] not in _GNUPG_ROOTS:
        return False
    algorithm = next((child for child in node[1:] if _named(child, _GNUPG_ALGORITHMS)), None)
    if algorithm is None or not isinstance(algorithm, list) or len(algorithm) < 2:
        return False
    if node[0] == b"private-key":
        return _contains_named(algorithm, _GNUPG_PRIVATE_FIELDS)
    marker = {b"protected-private-key": {b"protected"}, b"shadowed-private-key": {b"shadowed"}}[
        node[0]
    ]
    return _contains_named(node, marker)


class GnuPGPrivateKey:
    """A complete native GnuPG private-key envelope."""

    def search(self, data: bytes, /) -> re.Match[bytes] | None:
        found, start, oversized = _gnupg_expression_start(data)
        if oversized:
            return found
        if found is None or start is None or start >= len(data) or data[start] != 0x28:
            return None
        parser = _SExpression(data, start)
        node = parser.value()
        parser._skip_space()
        if parser.at != len(data) or not _valid_gnupg_key(node):
            return None
        return found


def _percent_decoded(value: bytes) -> bytes | None:
    decoded = bytearray()
    at = 0
    while at < len(value):
        if value[at] != 0x25:
            decoded.append(value[at])
            at += 1
            continue
        if at + 2 >= len(value) or any(byte not in _URI_HEX for byte in value[at + 1 : at + 3]):
            return None
        decoded.append(int(value[at + 1 : at + 3], 16))
        at += 3
    return bytes(decoded)


def _valid_uri_host(host: bytes) -> bool:
    if host.startswith(b"["):
        close = host.find(b"]")
        if close < 0:
            return False
        suffix = host[close + 1 :]
        try:
            ipaddress.IPv6Address(host[1:close].decode("ascii"))
        except (UnicodeDecodeError, ipaddress.AddressValueError):
            return False
        if not suffix:
            return True
        port = suffix[1:] if suffix.startswith(b":") else b""
        return bool(port) and port.isdigit() and int(port) <= 65535
    if host.count(b":") > 1:
        return False
    name, separator, port = host.rpartition(b":")
    if not separator:
        name, port = host, b""
    decoded = _percent_decoded(name)
    if decoded is None or not decoded or len(decoded) > 253:
        return False
    if not all(
        byte in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-"
        for byte in decoded
    ):
        return False
    labels = decoded.split(b".")
    if any(
        not label or len(label) > 63 or label.startswith(b"-") or label.endswith(b"-")
        for label in labels
    ):
        return False
    return not separator or (bool(port) and port.isdigit() and int(port) <= 65535)


def _authority(data: bytes, start: int, quote: int | None) -> tuple[bytes | None, bool]:
    at = start
    limit = min(len(data), start + _MAX_URI_AUTHORITY_BYTES)
    while at < limit:
        byte = data[at]
        if byte in _URI_TERMINATORS or byte == quote:
            return data[start:at], False
        if byte not in _URI_AUTHORITY:
            return None, False
        at += 1
    if at == len(data):
        return data[start:at], False
    continuing = data[at] in _URI_AUTHORITY and data[at] != quote
    return (None, True) if continuing else (data[start:at], False)


class UriPassword:
    """A bounded URI authority carrying an issued-looking password in userinfo."""

    def search(self, data: bytes, /) -> re.Match[bytes] | None:
        for found in _URI_SCHEME.finditer(data):
            quote = (
                data[found.start() - 1]
                if found.start() and data[found.start() - 1] in b"\"'"
                else None
            )
            authority, oversized = _authority(data, found.end(), quote)
            if oversized:
                return found
            if not authority:
                continue
            userinfo, marker, host = authority.rpartition(b"@")
            if not marker or not _valid_uri_host(host):
                continue
            username, separator, password = userinfo.partition(b":")
            if not username or not separator or not password:
                continue
            if any(byte not in _URI_USERINFO for byte in userinfo):
                continue
            if _percent_decoded(username) is None:
                continue
            decoded = _percent_decoded(password)
            if decoded is None or not 16 <= len(decoded) <= _MAX_BODY:
                continue
            if all(32 <= byte < 127 for byte in decoded) and _is_high_entropy(decoded):
                return found
        return None
