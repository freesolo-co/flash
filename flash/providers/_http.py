"""Shared stdlib REST client for provider API modules.

Provider API clients use the same hardened-retry shape: a Bearer/Content-Type urllib
request, a jittered exponential backoff that retries 5xx/429 and fast-fails other 4xx
with the response body as the actionable detail, and a "failed after N attempts" raise.
This module factors that common core out so the backoff math lives in one place.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

# An unambiguous ``HTTP 404`` token: ``http 404`` bounded so a longer status-LIKE number can't
# match. ``\b`` after ``404`` rejects ``HTTP 4040``/``HTTP 4041`` (digit immediately after), while
# still matching ``HTTP 404:``, ``HTTP 404 Not Found``, and a trailing ``HTTP 404`` at end-of-string.
_HTTP_404_RE = re.compile(r"\bhttp 404\b")


def is_not_found(err: Exception) -> bool:
    """True only when a provider API error represents a genuine HTTP 404 (resource already gone).

    ``request_with_retries`` chains the original urllib ``HTTPError`` as ``__cause__`` for every
    fast-failed 4xx (and on the "failed after N attempts" path), so the status CODE is authoritative
    when a cause is present — anything else is a real failure that must NOT be swallowed. We only
    fall back to a text match when there is no HTTPError cause, and even then only on an unambiguous
    ``HTTP 404`` TOKEN (``_HTTP_404_RE``) — NEVER a bare ``"404"`` substring, and never a longer
    number that just begins with ``404``: the token's trailing ``\\b`` rejects ``HTTP 4040``/``4041``,
    so a transient error whose text embeds such an id is not misread. Mirrors
    ``runpod.api._is_not_found``."""
    cause = getattr(err, "__cause__", None)
    if isinstance(cause, urllib.error.HTTPError):
        return cause.code == 404
    return bool(_HTTP_404_RE.search(str(err).lower()))


class RestClient:
    """Parametrized urllib REST client with jittered-backoff retries.

    ``base_url`` is prefixed onto the ``target`` passed to each call. The key is read
    from ``env_var`` on each request (env-only by design — never persisted) and failures
    raise ``error_cls``.

    ``keys_provider`` (optional) supplies an *ordered* list of API keys to try per call:
    each key runs the full backoff loop, and a key that ends in a failover-class error
    (per ``failover_predicate``) hands off to the next key — used by providers whose key
    is a multi-account pool (see ``runpod.keys``). With no ``keys_provider`` the client
    uses the single ``env_var`` key and behaves exactly as a single-key client.
    """

    def __init__(
        self,
        *,
        env_var: str,
        error_cls: type[Exception],
        base_url: str = "",
        missing_key_message: str | None = None,
        keys_provider: Callable[[], list[str]] | None = None,
        failover_predicate: Callable[[Exception], bool] | None = None,
        extra_headers: dict[str, str] | None = None,
        auth_header_name: str = "Authorization",
        auth_value_format: str = "Bearer {key}",
    ) -> None:
        self.env_var = env_var
        self.error_cls = error_cls
        self.base_url = base_url
        self.missing_key_message = (
            missing_key_message or f"{env_var} not configured on the control-plane host"
        )
        self.keys_provider = keys_provider
        self.failover_predicate = failover_predicate
        # Static headers added to EVERY request (e.g. a custom User-Agent). Lambda Cloud sits
        # behind Cloudflare, which 403s the stdlib default ``Python-urllib/<v>`` UA — so the
        # Lambda client passes a real UA here. The auth + ``Content-Type`` headers are always set
        # by ``request`` and win on a key collision.
        self.extra_headers = dict(extra_headers or {})
        # How the API key is presented. Default is RunPod/Lambda's ``Authorization: Bearer <key>``;
        # a provider can override with a bare ``api_key: <key>`` header instead
        # (``auth_header_name="api_key"``, ``auth_value_format="{key}"``).
        self.auth_header_name = auth_header_name
        self.auth_value_format = auth_value_format

    def api_key(self) -> str:
        key = os.environ.get(self.env_var)
        if not key:
            raise self.error_cls(self.missing_key_message)
        return key

    def _ordered_keys(self) -> list[str]:
        """The keys to try, in order. Single-key clients yield exactly the env key."""
        if self.keys_provider is None:
            return [self.api_key()]
        keys = self.keys_provider()
        if not keys:
            raise self.error_cls(self.missing_key_message)
        return keys

    def request(
        self,
        target: str,
        method: str = "GET",
        body: dict | None = None,
        timeout: float = 30.0,
        key: str | None = None,
    ) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}{target}",
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                **self.extra_headers,
                self.auth_header_name: self.auth_value_format.format(key=key or self.api_key()),
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}

    def _request_one_key(
        self,
        key: str,
        target: str,
        method: str,
        body: dict | None,
        retries: int,
        base_delay: float,
    ) -> Any:
        """One key's full backoff loop (the original single-key behavior)."""
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self.request(target, method=method, body=body, key=key)
            except urllib.error.HTTPError as e:
                if e.code < 500 and e.code != 429:
                    # The response body usually carries the actionable error detail; e.reason
                    # alone (e.g. "Bad Request") is rarely enough to debug a 4xx.
                    detail = ""
                    with contextlib.suppress(Exception):
                        detail = e.read().decode("utf-8", "replace")[:500].strip()
                    suffix = f": {detail}" if detail else ""
                    raise self.error_cls(
                        f"{method} {target} -> HTTP {e.code}: {e.reason}{suffix}"
                    ) from e
                last = e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
            if attempt < retries:
                delay = min(base_delay * (2 ** min(attempt, 6)), 30.0)
                time.sleep(delay * random.uniform(0.7, 1.3))
        # Chain the last exception so callers can inspect it: ``_is_not_found`` keys off the
        # HTTPError code, and the multi-key waterfall's ``failover_predicate`` needs it to see
        # a persistent 429's status (without the cause it can't tell the account is rate/quota
        # limited and would stop instead of trying the next account).
        raise self.error_cls(
            f"{method} {target} failed after {retries + 1} attempts: {last}"
        ) from last

    def request_with_retries_for_key(
        self,
        key: str,
        target: str,
        method: str = "GET",
        body: dict | None = None,
        retries: int = 4,
        base_delay: float = 2.0,
    ) -> Any:
        """Like request_with_retries but always uses the supplied key, bypassing the pool.

        Use this when you need to query each account in the pool independently (e.g.
        list_endpoints aggregation) rather than stopping at the first success.
        """
        return self._request_one_key(key, target, method, body, retries, base_delay)

    def request_with_retries(
        self,
        target: str,
        method: str = "GET",
        body: dict | None = None,
        retries: int = 4,
        base_delay: float = 2.0,
    ) -> Any:
        """REST call hardened against transient network/5xx blips (jittered backoff).

        With a multi-key ``keys_provider``, a key that fails with a failover-class error
        hands off to the next key in the pool; a hard, key-agnostic error (or the last
        key) is raised. Single-key clients try exactly one key — identical to before.
        """
        ordered = self._ordered_keys()
        last_exc: Exception | None = None
        for i, key in enumerate(ordered):
            try:
                return self._request_one_key(key, target, method, body, retries, base_delay)
            except self.error_cls as e:
                last_exc = e
                more_keys = i < len(ordered) - 1
                if more_keys and self.failover_predicate is not None and self.failover_predicate(e):
                    continue
                raise
        # Only reachable if ordered is empty, which _ordered_keys already guards against.
        raise last_exc or self.error_cls(self.missing_key_message)
