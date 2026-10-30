"""Provider-neutral HF-artifact reading + heartbeat provenance.

The control plane observes a remote worker purely through the artifacts it uploads to a private HF
dataset repo: rate-limited text/JSON readers for ``heartbeat.json`` / ``error_*.txt`` / console logs,
plus the provenance predicates that decide whether a heartbeat belongs to THIS attempt or a leftover
prior one. None of this is provider-specific — Modal all poll the same artifact
shape — so it lives here in the shared kernel and no provider package imports another for it.

The worker-side bootstrap (``_instance_bootstrap``) cannot import flash, so it re-implements the
upload half; this module is the read half every poller shares.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from logging import Logger
from typing import Protocol, TypeVar

from flash._internal.diagnostics import sanitize_diagnostic
from flash.adapters.artifacts import attempt_scoped_artifact_name
from flash.providers._lifecycle.instances.poll import _attempt_int
from flash.providers._lifecycle.net.deadline import (
    remaining_seconds,
    require_deadline_at,
)


class TextArtifactReader(Protocol):
    """Read one artifact, allowing a bounded caller to override the captured deadline."""

    def __call__(self, force: bool = False, *, deadline_at: float | None = None) -> str | None: ...


def make_hf_text_reader(
    hf_repo: str,
    path_in_repo: str,
    min_interval_s: float = 45.0,
    *,
    deadline_at: float | None = None,
) -> TextArtifactReader:
    """Rate-limited reader for an HF artifact; returns None until it exists or on any error."""
    deadline = require_deadline_at(deadline_at) if deadline_at is not None else None
    state = {"last": 0.0}

    def read(
        force: bool = False,
        *,
        deadline_at: float | None = deadline,
    ) -> str | None:
        if not hf_repo or (deadline_at is not None and remaining_seconds(deadline_at) <= 0):
            return None
        now = time.time()
        if not force and now - state["last"] < min_interval_s:
            return None
        state["last"] = now
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(
                hf_repo,
                path_in_repo,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
                force_download=True,
            )
            with open(p) as f:
                return f.read()
        except Exception:
            return None

    return read


def make_hf_heartbeat_reader(
    hf_repo: str,
    prefix: str,
    min_interval_s: float = 30.0,
    *,
    deadline_at: float | None = None,
):
    """Rate-limited JSON reader for ``{prefix}/heartbeat.json`` on HF."""
    text_reader = make_hf_text_reader(
        hf_repo,
        f"{prefix}/heartbeat.json",
        min_interval_s,
        deadline_at=deadline_at,
    )

    def read(force: bool = False, *, deadline_at: float | None = deadline_at) -> dict | None:
        raw = text_reader(force=force, deadline_at=deadline_at)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    return read


def heartbeat_reader_for(spec, *, deadline_at: float | None = None):
    """The HF heartbeat reader for a run's spec (None when the run has no hf_repo)."""
    hf_repo = spec.train.hf_repo
    return (
        make_hf_heartbeat_reader(
            hf_repo,
            f"{spec.phase}/{spec.run_id}",
            deadline_at=deadline_at,
        )
        if hf_repo
        else None
    )


def error_artifact_name(phase: str, attempt) -> str:
    """Worker error-artifact filename for one exact bounded attempt identity.

    Shares the writer's definition rather than restating the format: this is the read half of a
    name the worker chose, and the two agreeing is the whole point of the artifact.
    """
    return attempt_scoped_artifact_name("error", phase, attempt)


def make_hf_failure_detail_reader(
    hf_repo: str,
    prefix: str,
    phase: str,
    min_interval_s: float = 45.0,
    attempt: int = 0,
    *,
    deadline_at: float | None = None,
):
    """Reader for worker-uploaded failure artifacts on HF (error/console txt); force-read after terminal failure."""
    # Attempt-scoped to match the worker's error_artifact_name(mode, attempt).
    err_name = error_artifact_name(phase, attempt)
    live_console_name = attempt_scoped_artifact_name("console", phase, attempt)
    error_reader = make_hf_text_reader(
        hf_repo,
        f"{prefix}/{err_name}",
        min_interval_s,
        deadline_at=deadline_at,
    )
    console_reader = make_hf_text_reader(
        hf_repo,
        f"{prefix}/console_{phase}.txt",
        min_interval_s,
        deadline_at=deadline_at,
    )
    live_console_reader = make_hf_text_reader(
        hf_repo,
        f"{prefix}/{live_console_name}",
        min_interval_s,
        deadline_at=deadline_at,
    )

    def read(force: bool = False) -> str | None:
        parts: list[str] = []
        error_text = error_reader(force=force)
        if error_text:
            parts.append(f"--- {err_name} ---\n{error_text}")
        console_text = console_reader(force=force)
        if console_text:
            parts.append(f"--- console_{phase}.txt ---\n{console_text}")
        live_console_text = live_console_reader(force=force)
        if live_console_text:
            parts.append(f"--- {live_console_name} ---\n{live_console_text}")
        return "\n".join(parts) if parts else None

    return read


def _heartbeat_matches_attempt(hb: dict, launch_ts: float | None, current_attempt) -> bool:
    """Require exact attempt and timestamp provenance for one current worker heartbeat."""
    expected_attempt = _attempt_int(current_attempt)
    heartbeat_attempt = _attempt_int(hb.get("attempt"))
    if expected_attempt is None or heartbeat_attempt != expected_attempt:
        return False
    if isinstance(launch_ts, bool) or not isinstance(launch_ts, (int, float)):
        return False
    launch = float(launch_ts)
    ts = hb.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return False
    now = time.time()
    timestamp = float(ts)
    return (
        math.isfinite(launch)
        and launch > 0
        and math.isfinite(timestamp)
        and launch <= timestamp <= now + 120.0
    )


def worker_flagged_retriable(
    heartbeat_reader, *, launch_ts: float | None = None, current_attempt: int | None = None
) -> bool:
    """Honor a retriable heartbeat only when its exact attempt provenance is current."""
    if heartbeat_reader is None:
        return False
    hb = heartbeat_reader(force=True)
    return (
        isinstance(hb, dict)
        and hb.get("retriable") is True
        and _heartbeat_matches_attempt(hb, launch_ts, current_attempt)
    )


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
