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
    """RFC 3986 section 5.2.4, walked by index rather than by re-slicing the remainder.

    The literal transcription of the algorithm replaces the input buffer with a suffix of itself on
    every iteration, which copies nearly the whole remaining path each time. A `$id` is
    caller-controlled and a payload may carry 8 MiB of them, so a path of `a/../` segments made
    canonicalization quadratic: 400 KB took over four seconds, on the persistence path that runs
    AFTER the paid upstream call. `start` advances instead, so each character is read a fixed
    number of times; the emitted output is unchanged.
    """
    output: list[str] = []
    start = 0
    length = len(path)
    while start < length:
        remaining = length - start
        if path.startswith("../", start):
            start += 3
        elif path.startswith("./", start):
            start += 2
        elif path.startswith("/./", start):
            # `/./x` leaves a leading slash for the next round, which is what the buffer rewrite
            # `f"/{buffer[3:]}"` produced; stepping to the final slash of the prefix is the same.
            start += 2
        elif remaining == 2 and path.startswith("/.", start):
            output.append("/")
            start += 2
        elif path.startswith("/../", start):
            start += 3
            if output:
                output.pop()
        elif remaining == 3 and path.startswith("/..", start):
            if output:
                output.pop()
            output.append("/")
            start += 3
        elif remaining == 1 and path[start] == ".":
            start += 1
        elif remaining == 2 and path.startswith("..", start):
            start += 2
        else:
            segment_end = path.find("/", start + 1 if path[start] == "/" else start)
            if segment_end < 0:
                output.append(path[start:])
                start = length
            else:
                output.append(path[start:segment_end])
                start = segment_end
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

    The comparison is made on the zero-stripped TEXT rather than through `int`. A recorded URI is
    untrusted input and may carry thousands of digits, which `int()` refuses to parse at all --
    raising out of sanitization and dropping the whole paid request from storage.
    """
    if default_port is None or not port.isdigit():
        return False
    return _canonical_port(port) == _canonical_port(default_port)


def _canonical_port(port: str) -> str:
    """Strip leading zeros from a numeric port, which are not part of its value.

    Default-port removal alone was not enough: `:0444` and `:444` are the same non-default port, so
    keeping both spellings classified a local `$id` reached under one of them as external and left
    the local definition's secret literals in the raw export. A non-numeric port is untrustworthy
    recorded input and is left alone rather than reinterpreted.

    Stripping is textual for the same reason as above: `int()` rejects an integer literal beyond
    its digit limit, so a URI with a 5000-digit port made sanitization raise. Text handles any
    length and gives the same answer for every port a real URI can carry.
    """
    if not port.isdigit():
        return port
    return port.lstrip("0") or "0"


def _canonical_registered_host(host: str) -> str:
    """Drop a registered name's trailing DNS root label, which names the same host.

    `example.com.` and `example.com` are the same authority, so keeping both spellings classified a
    local `$id` reached under one of them as external and left the local target's secret literals in
    the raw export. Only ONE trailing dot is removed and only from a name that has other content: a
    bare `"."` or `".."` is not a hostname, and rewriting it would merge unrelated authorities.
    """
    if len(host) > 1 and host.endswith(".") and not host.endswith(".."):
        return host[:-1]
    return host


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
        elif suffix.startswith(":"):
            suffix = f":{_canonical_port(suffix[1:])}"
        hostport = f"[{_canonical_ipv6_host(hostport[1:host_end].casefold())}]{suffix}"
    else:
        host, port_separator, port = hostport.rpartition(":")
        if port_separator and (not port or _is_default_port(port, default_port)):
            hostport = _canonical_registered_host(host.casefold())
        elif port_separator:
            hostport = f"{_canonical_registered_host(host.casefold())}:{_canonical_port(port)}"
        else:
            hostport = _canonical_registered_host(hostport.casefold())
    # the host is case-insensitive, so a character an escape DECODES to must fold too: `%4A` and a
    # plain `j` are the same host. the casefold above cannot do it (the escape is still encoded) and
    # folding the whole string afterwards would lowercase the hex digits rfc 3986 6.2.2.1 wants
    # uppercase, so the fold is applied to decoded characters only. the userinfo is case-sensitive
    # and keeps the plain normalization. decoding is delimiter-safe here because no unreserved
    # character is a delimiter: `@`, `:` and the brackets stay encoded and the split above holds.
    # the root-dot cleanup above ran on the ENCODED spelling, but `.` is unreserved, so `%2E`
    # only becomes a dot here. `https://example.com%2E/s` therefore came back out as
    # `example.com.` and was classified as a different authority than `example.com`, which put the
    # local `$id` out of reach and left its secret literals in the raw export. the cleanup is
    # idempotent, so re-applying it to the decoded host is enough; the port was already handled.
    normalized_hostport = _normalize_percent_encoding(hostport, fold_decoded=True)
    if not normalized_hostport.startswith("["):
        host, port_separator, port = normalized_hostport.rpartition(":")
        normalized_hostport = (
            f"{_canonical_registered_host(host)}:{port}"
            if port_separator
            else _canonical_registered_host(normalized_hostport)
        )
    normalized_netloc = (
        f"{_normalize_percent_encoding(userinfo)}@{normalized_hostport}"
        if user_separator
        else normalized_hostport
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
