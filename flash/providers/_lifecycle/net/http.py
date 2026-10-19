"""Shared stdlib REST client with jittered-backoff retries for provider API modules."""

from __future__ import annotations

import contextlib
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from flash._internal.diagnostics import sanitize_diagnostic
from flash.providers._lifecycle.net.auth import load_provider_key
from flash.providers._lifecycle.net.deadline import remaining_seconds, require_deadline_at

_HTTP_404_RE = re.compile(r"\bhttp 404\b")


def is_not_found(err: Exception) -> bool:
    """True when a provider API error is a genuine HTTP 404."""
    cause = getattr(err, "__cause__", None)
    if isinstance(cause, urllib.error.HTTPError):
        return cause.code == 404
    return bool(_HTTP_404_RE.search(str(err).lower()))


class RestClient:
    """Parametrized urllib REST client with jittered-backoff retries.

    With a ``keys_provider``, keys are tried in order; a failover-class error advances to
    the next key. Without one, the single ``env_var`` key is used.
    """

    def __init__(
        self,
        *,
        env_var: str,
        error_cls: type[Exception],
        base_url: str = "",
        keys_provider: Callable[[], list[str]] | None = None,
        failover_predicate: Callable[[Exception], bool] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.env_var = env_var
        self.error_cls = error_cls
        self.base_url = base_url
        self.missing_key_message = f"{env_var} not configured on the control-plane host"
        self.keys_provider = keys_provider
        self.failover_predicate = failover_predicate
        # Lambda sits behind Cloudflare, which 403s the stdlib default UA — pass a real UA via extra_headers.
        self.extra_headers = dict(extra_headers or {})

    def api_key(self) -> str:
        # read through load_provider_key so preflight and requests use the same stripped value;
        # raw newlines can pass preflight but break provider authorization headers.
        key = load_provider_key(self.env_var)
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
                "Authorization": f"Bearer {key or self.api_key()}",
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
        deadline_at: float | None,
    ) -> Any:
        """One key's full backoff loop (the original single-key behavior)."""
        deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
        last: Exception | None = None
        for attempt in range(retries + 1):
            timeout = 30.0
            if deadline is not None:
                remaining = remaining_seconds(deadline)
                if remaining <= 0:
                    raise self.error_cls(f"{method} {target} deadline exceeded") from None
                timeout = min(timeout, remaining)
            try:
                return self.request(
                    target,
                    method=method,
                    body=body,
                    timeout=timeout,
                    key=key,
                )
            except urllib.error.HTTPError as e:
                if e.code < 500 and e.code != 429:
                    raw = b""
                    with contextlib.suppress(Exception):
                        raw = e.read()
                    reason = sanitize_diagnostic(e.reason, limit=200)
                    err = self.error_cls(f"{method} {target} -> HTTP {e.code}: {reason}")
                    # retain the body as structured metadata for provider-specific classification.
                    err.response_body = raw
                    raise err from e
                last = e
            except json.JSONDecodeError as e:
                # Cloudflare HTML interstitial or truncated body — treat as transient.
                last = e
            except OSError as e:
                last = e
            if attempt < retries:
                delay = min(base_delay * (2 ** min(attempt, 6)), 30.0)
                delay *= random.uniform(0.7, 1.3)
                if deadline is not None:
                    remaining = remaining_seconds(deadline)
                    if remaining <= 0:
                        break
                    delay = min(delay, remaining)
                if delay > 0:
                    time.sleep(delay)
        # Chain last so is_not_found and failover_predicate can inspect the HTTPError code.
        error_kind = type(last).__name__ if last is not None else "unknown error"
        raise self.error_cls(
            f"{method} {target} failed after {retries + 1} attempts ({error_kind})"
        ) from last

    def request_with_retries_for_key(
        self,
        key: str,
        target: str,
        method: str = "GET",
        body: dict | None = None,
        retries: int = 4,
        base_delay: float = 2.0,
        deadline_at: float | None = None,
    ) -> Any:
        """Like request_with_retries but uses the supplied key, bypassing the pool."""
        return self._request_one_key(
            key,
            target,
            method,
            body,
            retries,
            base_delay,
            deadline_at,
        )

    def request_with_retries(
        self,
        target: str,
        method: str = "GET",
        body: dict | None = None,
        retries: int = 4,
        base_delay: float = 2.0,
        deadline_at: float | None = None,
    ) -> Any:
        """REST call with jittered backoff; with a key pool, failover-class errors try the next key."""
        ordered = self._ordered_keys()
        for i, key in enumerate(ordered):
            try:
                return self._request_one_key(
                    key,
                    target,
                    method,
                    body,
                    retries,
                    base_delay,
                    deadline_at,
                )
            except self.error_cls as e:
                if (
                    i < len(ordered) - 1
                    and self.failover_predicate is not None
                    and self.failover_predicate(e)
                ):
                    continue
                raise
        raise AssertionError("unreachable")
