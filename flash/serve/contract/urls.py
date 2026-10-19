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

    Every field is read inside the guard. ``urlsplit`` defers validation to the accessors, so
    ``.port`` raises on ``host:notaport`` long after the split succeeded -- and this helper exists
    to build ERROR messages, so a raise here replaces a friendly failure with a traceback from the
    reporting path itself. There is no "already parsed, so the rest is safe" point to relax at.

    Sits beside ``is_freesolo_hosted_url`` because both answer questions about a URL's authority
    and must agree on how one is parsed.
    """
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        # an ipv6 literal loses its brackets through ``hostname``, and without them a trailing
        # port is unreadable (``2001:db8::1:8443``) and the address itself ambiguous. restore them.
        if ":" in host:
            host = f"[{host}]"
        port = parsed.port
        scheme = parsed.scheme
    except ValueError:  # malformed authority: a bad ipv6 literal, or a port that is not a number
        return "(unparseable url)"
    if not host:
        return "(unparseable url)"
    # `is not None`, not truthiness: port 0 is a real configured value and an invalid endpoint.
    # dropping it renders `http://localhost:0` as `http://localhost`, which names the DEFAULT port
    # and hides the setting the reader has to correct.
    if port is not None:
        host = f"{host}:{port}"
    return f"{scheme}://{host}" if scheme else host


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


PROD_FREESOLO_SERVING_URL = "https://serve.freesolo.co"
DEV_FREESOLO_SERVING_URL = "https://serve-dev.freesolo.co"


def default_serving_url(channel: str | None = None) -> str:
    """Default serving control root for the given release channel.

    Serving and control planes use separate per-channel databases; mixing them causes org FK 23503.
    Dev serving is ``serve-dev.freesolo.co``.
    """
    from flash._internal.channel import CHANNEL

    return DEV_FREESOLO_SERVING_URL if (channel or CHANNEL) == "dev" else PROD_FREESOLO_SERVING_URL


def serving_base_url() -> str:
    """Env-overridable serving control root.

    Standalone planes must target a backend they operate because every request carries the plane's
    ``FREESOLO_INTERNAL_KEY``. Reject hosted URLs whether supplied explicitly or by fallback.

    Lives here rather than in ``flash.serve.deployment.deploy`` so that reading it costs nothing: deploy
    imports httpx at module scope, and ``[project].dependencies`` is empty so the client CLI runs on
    a bare install. A command that only needs to resolve the configured URL must not be the reason
    a user has to install an extra -- and this guard is security-relevant, so the alternative of
    duplicating it into each caller would be worse than the move.
    """
    import os

    from flash.serve.contract.errors import ServingError

    # imported lazily: flash.serve is the CLIENT side, and a module-level import would pull
    # flash.server into every CLI invocation.
    from flash.server.platform.auth import standalone

    configured = (os.environ.get("FREESOLO_SERVING_URL") or "").strip()
    if standalone() and (not configured or is_freesolo_hosted_url(configured)):
        raise ServingError(
            f"FREESOLO_SERVING_URL is {'not set' if not configured else 'a Freesolo-hosted URL'}. "
            "A standalone plane has no serving backend of its own, and using the hosted one would "
            "send FREESOLO_INTERNAL_KEY - the key that controls this plane - to a service you do "
            "not operate. Point FREESOLO_SERVING_URL at your own multi-LoRA deployment, or export "
            "the adapter and serve it yourself (see SELF_HOSTING.md). Training does not require "
            "this."
        )
    return serving_control_url(configured or default_serving_url())


def public_deployment(deployment: dict) -> dict:
    """Return a public copy without private rollback state."""
    out = dict(deployment)
    out.pop("previous_deployment", None)
    out.pop("verification_generation", None)
    return out
