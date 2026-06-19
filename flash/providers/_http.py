"""Shared stdlib REST client for the provider API modules.

Both ``providers/runpod/api.py`` and ``providers/vast/api.py`` are thin, no-SDK-state
clients with the SAME hardened-retry shape: a Bearer/Content-Type urllib request, a
jittered exponential backoff that retries 5xx/429 and fast-fails other 4xx with the
response body as the actionable detail, and a "failed after N attempts" raise. They
differ only in: the env var that holds the key, the error class, and whether the caller
passes a full URL (RunPod) or a path joined onto a base (Vast). This module factors that
common core out so the backoff math lives in one place.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any


class RestClient:
    """Parametrized urllib REST client with jittered-backoff retries.

    ``base_url`` is prefixed onto the ``target`` passed to each call (empty for the
    RunPod client, which passes full URLs; the Vast base for the Vast client). The key
    is read from ``env_var`` on each request (env-only by design — never persisted) and
    failures raise ``error_cls``.
    """

    def __init__(
        self,
        *,
        env_var: str,
        error_cls: type[Exception],
        base_url: str = "",
        missing_key_message: str | None = None,
    ) -> None:
        self.env_var = env_var
        self.error_cls = error_cls
        self.base_url = base_url
        self.missing_key_message = (
            missing_key_message or f"{env_var} not configured on the control-plane host"
        )

    def api_key(self) -> str:
        key = os.environ.get(self.env_var)
        if not key:
            raise self.error_cls(self.missing_key_message)
        return key

    def request(
        self, target: str, method: str = "GET", body: dict | None = None, timeout: float = 30.0
    ) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}{target}",
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self.api_key()}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}

    def request_with_retries(
        self,
        target: str,
        method: str = "GET",
        body: dict | None = None,
        retries: int = 4,
        base_delay: float = 2.0,
    ) -> Any:
        """REST call hardened against transient network/5xx blips (jittered backoff)."""
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self.request(target, method=method, body=body)
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
        raise self.error_cls(f"{method} {target} failed after {retries + 1} attempts: {last}")
