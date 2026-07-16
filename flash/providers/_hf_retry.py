"""Shared Hugging Face retry helpers for control-plane provider code."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from logging import Logger
from typing import TypeVar

from flash.diagnostics import sanitize_diagnostic
from flash.providers._deadline import remaining_seconds, require_deadline_at

T = TypeVar("T")

HF_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
HF_RETRY_DELAYS_S = (1.0, 3.0, 8.0, 20.0, 60.0)
HF_RETRY_AFTER_MAX_S = 60.0


def hf_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def hf_retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    if not value and hasattr(headers, "items"):
        for key, candidate in headers.items():
            if str(key).lower() == "retry-after":
                value = candidate
                break
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            seconds = (retry_at - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError):
            return None
    return min(HF_RETRY_AFTER_MAX_S, max(0.0, seconds))


def hf_call(
    call: Callable[[], T],
    label: str,
    *,
    logger: Logger,
    sleep: Callable[[float], object] = time.sleep,
    retry_delays: tuple[float, ...] = HF_RETRY_DELAYS_S,
    transient_status_codes: frozenset[int] = HF_TRANSIENT_STATUS_CODES,
    deadline_at: float | None = None,
) -> T:
    deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    for attempt in range(len(retry_delays) + 1):
        try:
            if deadline is not None and remaining_seconds(deadline) <= 0:
                raise TimeoutError(f"{label} exceeded the run wall deadline")
            return call()
        except Exception as exc:
            if hf_status_code(exc) not in transient_status_codes or attempt >= len(retry_delays):
                raise
            retry_after = hf_retry_after(exc)
            delay = retry_after if retry_after is not None else retry_delays[attempt]
            if deadline is not None:
                remaining = remaining_seconds(deadline)
                if remaining <= 0:
                    raise
                delay = min(delay, remaining)
            logger.warning(
                "%s transient Hugging Face error; retrying in %.0fs: %s",
                label,
                delay,
                sanitize_diagnostic(exc, limit=500),
            )
            if delay > 0:
                sleep(delay)
            if deadline is not None and remaining_seconds(deadline) <= 0:
                raise
    raise AssertionError("unreachable")
