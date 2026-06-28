"""Shared stdlib REST client with jittered-backoff retries for provider API modules."""

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
        # Lambda sits behind Cloudflare, which 403s the stdlib default UA — pass a real UA via extra_headers.
        self.extra_headers = dict(extra_headers or {})
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
                    detail = ""
                    with contextlib.suppress(Exception):
                        detail = e.read().decode("utf-8", "replace")[:500].strip()
                    suffix = f": {detail}" if detail else ""
                    raise self.error_cls(
                        f"{method} {target} -> HTTP {e.code}: {e.reason}{suffix}"
                    ) from e
                last = e
            except json.JSONDecodeError as e:
                # Cloudflare HTML interstitial or truncated body — treat as transient.
                last = e
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
            if attempt < retries:
                delay = min(base_delay * (2 ** min(attempt, 6)), 30.0)
                time.sleep(delay * random.uniform(0.7, 1.3))
        # Chain last so is_not_found and failover_predicate can inspect the HTTPError code.
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
        """Like request_with_retries but uses the supplied key, bypassing the pool."""
        return self._request_one_key(key, target, method, body, retries, base_delay)

    def request_with_retries(
        self,
        target: str,
        method: str = "GET",
        body: dict | None = None,
        retries: int = 4,
        base_delay: float = 2.0,
    ) -> Any:
        """REST call with jittered backoff; with a key pool, failover-class errors try the next key."""
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
        raise last_exc or self.error_cls(self.missing_key_message)
