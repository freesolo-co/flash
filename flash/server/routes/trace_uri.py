"""RFC 3986 URI canonicalization for JSON Schema resource identifiers.

Two spellings of the same `$id` must compare equal, or a local schema referenced in its other form
is classified as external and its secret literals stay in the raw export.
"""

from __future__ import annotations

from socket import AF_INET6, inet_ntop, inet_pton
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit


def _normalize_percent_encoding(value: str, *, fold_decoded: bool = False) -> str:
    """Decode unreserved escapes and uppercase the rest, per RFC 3986 6.2.2.

    `fold_decoded` case-folds the characters an escape decodes to. It is for the host, which is
    case-insensitive: without it `%4A` normalizes to `J` and never matches a plain `j`. Folding the
    whole string instead would lowercase the hex digits of escapes that stay encoded.
    """
    unreserved = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~"
    normalized: list[str] = []
    index = 0
    while index < len(value):
        if index + 2 < len(value) and value[index] == "%":
            escaped = value[index + 1 : index + 3]
            try:
                decoded = chr(int(escaped, 16))
            except ValueError:
                pass
            else:
                if decoded in unreserved:
                    normalized.append(decoded.casefold() if fold_decoded else decoded)
                else:
                    normalized.append(f"%{escaped.upper()}")
                index += 3
                continue
        normalized.append(value[index])
        index += 1
    return "".join(normalized)


def _remove_dot_segments(path: str) -> str:
    input_buffer = path
    output: list[str] = []
    while input_buffer:
        if input_buffer.startswith("../"):
            input_buffer = input_buffer[3:]
        elif input_buffer.startswith("./"):
            input_buffer = input_buffer[2:]
        elif input_buffer.startswith("/./"):
            input_buffer = f"/{input_buffer[3:]}"
        elif input_buffer == "/.":
            input_buffer = "/"
        elif input_buffer.startswith("/../"):
            input_buffer = f"/{input_buffer[4:]}"
            if output:
                output.pop()
        elif input_buffer == "/..":
            input_buffer = "/"
            if output:
                output.pop()
        elif input_buffer in {".", ".."}:
            input_buffer = ""
        else:
            segment_end = input_buffer.find("/", 1 if input_buffer.startswith("/") else 0)
            if segment_end < 0:
                output.append(input_buffer)
                input_buffer = ""
            else:
                output.append(input_buffer[:segment_end])
                input_buffer = input_buffer[segment_end:]
    return "".join(output)


def _safe_urljoin(base: str, ref: str) -> str:
    """Resolve `ref` against `base`, treating an unparseable URI as unresolvable.

    Both arguments come from the recorded payload, so a malformed `$id` or `$ref` is untrusted
    input rather than a bug. Letting `ValueError` escape abandoned the ENTIRE trace after the
    upstream call had already completed -- not even the provider's rejection could be exported.
    """
    try:
        return urljoin(base, ref)
    except ValueError:
        return ref


def _safe_urldefrag(uri: str) -> tuple[str, str]:
    try:
        base, fragment = urldefrag(uri)
    except ValueError:
        return uri, ""
    return base, fragment


def _is_default_port(port: str, default_port: str | None) -> bool:
    """Whether `port` denotes the scheme's default, compared numerically rather than textually.

    `0443` and `443` are the same port: a string comparison kept the padded form, so the same
    resource reached under it was classified external and its secret schema literals survived.
    """
    if default_port is None or not port.isdigit():
        return False
    return int(port) == int(default_port)


def _canonical_ipv6_host(host: str) -> str:
    """Collapse an IPv6 literal to its canonical compressed form, or return it unchanged.

    `[2001:db8::1]` and `[2001:0db8:0:0:0:0:0:1]` name the SAME host, so comparing their textual
    spellings classified a local `$id` referenced in its other form as external and left the local
    target's secret literals in place. A spelling that is not a valid address is left alone: it is
    untrustworthy recorded input, and mapping it onto a valid one would merge distinct resources.
    """
    try:
        packed = inet_pton(AF_INET6, host)
    except (OSError, ValueError, UnicodeEncodeError):
        return host
    try:
        return inet_ntop(AF_INET6, packed)
    except (OSError, ValueError):
        return host


def _canonical_resource_uri(uri: str) -> str:
    try:
        scheme, netloc, path, query, fragment = urlsplit(uri)
    except ValueError:
        # a malformed `$id` (an unterminated IPv6 literal, say) is untrustworthy input from the
        # recorded payload, not a bug here. raising abandoned the WHOLE trace after the upstream
        # call had already been paid for, so the identifier is treated as unresolvable instead and
        # redaction continues conservatively: an unmatched reference redacts rather than exempts.
        return uri
    normalized_scheme = scheme.casefold()
    default_port = {"http": "80", "https": "443"}.get(normalized_scheme)
    userinfo, user_separator, hostport = netloc.rpartition("@")
    if hostport.startswith("[") and (host_end := hostport.find("]")) >= 0:
        suffix = hostport[host_end + 1 :]
        if suffix == ":" or (suffix.startswith(":") and _is_default_port(suffix[1:], default_port)):
            suffix = ""
        hostport = f"[{_canonical_ipv6_host(hostport[1:host_end].casefold())}]{suffix}"
    else:
        host, port_separator, port = hostport.rpartition(":")
        if port_separator and (not port or _is_default_port(port, default_port)):
            hostport = host.casefold()
        else:
            hostport = f"{host.casefold()}:{port}" if port_separator else hostport.casefold()
    # the host is case-insensitive, so a character an escape DECODES to must fold too: `%4A` and a
    # plain `j` are the same host. the casefold above cannot do it (the escape is still encoded) and
    # folding the whole string afterwards would lowercase the hex digits rfc 3986 6.2.2.1 wants
    # uppercase, so the fold is applied to decoded characters only. the userinfo is case-sensitive
    # and keeps the plain normalization. decoding is delimiter-safe here because no unreserved
    # character is a delimiter: `@`, `:` and the brackets stay encoded and the split above holds.
    normalized_netloc = (
        f"{_normalize_percent_encoding(userinfo)}@"
        f"{_normalize_percent_encoding(hostport, fold_decoded=True)}"
        if user_separator
        else _normalize_percent_encoding(hostport, fold_decoded=True)
    )
    normalized_path = _remove_dot_segments(_normalize_percent_encoding(path))
    if normalized_scheme in {"http", "https"} and normalized_netloc and not normalized_path:
        normalized_path = "/"
    return urlunsplit(
        (
            normalized_scheme,
            normalized_netloc,
            normalized_path,
            _normalize_percent_encoding(query),
            _normalize_percent_encoding(fragment),
        )
    )
