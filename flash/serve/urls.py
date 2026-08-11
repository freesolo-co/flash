"""Deployment URL helpers."""

from __future__ import annotations

from urllib.parse import urlsplit

FREESOLO_HOSTED_DOMAIN = "freesolo.co"


def is_freesolo_hosted_url(value: str) -> bool:
    """Whether ``value`` addresses Freesolo-operated infrastructure.

    Used to keep a self-hosted plane from sending ``FREESOLO_INTERNAL_KEY`` -- which on such a
    plane is the credential controlling the plane -- to a service the operator does not run.

    Matched on the parsed HOST rather than by substring: `https://freesolo.co.attacker.test`
    contains the domain but is not ours, and `https://SERVE.FREESOLO.CO`, `http://` (cleartext),
    an explicit `:443`, a `user:pw@` prefix and a trailing-dot FQDN all address the hosted service
    while spelling it differently. A schemeless value parses as a path rather than a netloc, so it
    is given a `//` prefix before parsing; otherwise `serve.freesolo.co` would read as safe.
    """
    raw = str(value or "").strip()
    if not raw:
        return False
    probe = raw if "//" in raw.split("?", 1)[0] else "//" + raw
    try:
        host = (urlsplit(probe).hostname or "").strip().rstrip(".").lower()
    except ValueError:  # malformed authority (e.g. a bad ipv6 literal)
        return False
    return host == FREESOLO_HOSTED_DOMAIN or host.endswith("." + FREESOLO_HOSTED_DOMAIN)


def displayable_url(url: str) -> str:
    """``url`` reduced to scheme and host, safe to print in an error or log line.

    A base URL is user-supplied and may carry credentials in its authority
    (``https://user:token@host``) or a secret in its query. Naming the service that answered is
    worth doing, but only scheme and host carry that meaning, so everything else is dropped rather
    than pattern-matched: an allowlist of two fields cannot be outgrown by a new secret-bearing one.
    A value too malformed to parse is reported as a placeholder, never echoed back.

    Sits beside ``is_freesolo_hosted_url`` because both answer questions about a URL's authority
    and must agree on how one is parsed.
    """
    try:
        parsed = urlsplit(url)
    except ValueError:  # malformed authority (e.g. a bad ipv6 literal)
        return "(unparseable url)"
    host = parsed.hostname or ""
    if not host:
        return "(unparseable url)"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return f"{parsed.scheme}://{host}" if parsed.scheme else host


def serving_control_url(value: str) -> str:
    """Return the serving control root without a terminal OpenAI ``/v1`` path."""
    url = str(value or "").rstrip("/")
    if url.endswith("/v1"):
        return url[: -len("/v1")].rstrip("/")
    return url


def openai_base_url(control_url: str) -> str:
    """Return the OpenAI-compatible base URL for a serving control root."""
    control = serving_control_url(control_url)
    return f"{control}/v1" if control else ""


def public_deployment(deployment: dict) -> dict:
    """Return a public copy without private rollback state or the stale legacy URL field."""
    out = dict(deployment)
    out.pop("previous_deployment", None)
    out.pop("verification_generation", None)
    out.pop("url", None)
    return out
