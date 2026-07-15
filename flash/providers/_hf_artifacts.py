"""Provider-neutral HF-artifact reading + heartbeat provenance.

The control plane observes a remote worker purely through the artifacts it uploads to a private HF
dataset repo: rate-limited text/JSON readers for ``heartbeat.json`` / ``error_*.txt`` / console logs,
plus the provenance predicates that decide whether a heartbeat belongs to THIS attempt or a leftover
prior one. None of this is provider-specific — RunPod, Lambda, and Vast all poll the same artifact
shape — so it lives here in the shared kernel and no provider package imports another for it.

The worker-side bootstrap (``_instance_bootstrap``) cannot import flash, so it re-implements the
upload half; this module is the read half every poller shares.
"""

from __future__ import annotations

import json
import math
import os
import time

from flash.providers._deadline import deadline_kwargs, remaining_seconds, require_deadline_at
from flash.providers._poll import _attempt_int


def make_hf_text_reader(
    hf_repo: str,
    path_in_repo: str,
    min_interval_s: float = 45.0,
    *,
    deadline_at: float | None = None,
):
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
        **deadline_kwargs(make_hf_text_reader, deadline_at),
    )

    def read(force: bool = False) -> dict | None:
        raw = text_reader(force=force)
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
            **deadline_kwargs(make_hf_heartbeat_reader, deadline_at),
        )
        if hf_repo
        else None
    )


def error_artifact_name(phase: str, attempt) -> str:
    """Worker error-artifact filename for one exact bounded attempt identity."""
    attempt_id = _attempt_int(attempt)
    if attempt_id is None:
        raise ValueError("worker error artifact attempt identity is invalid")
    return f"error_{phase}_attempt{attempt_id}.txt"


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
    error_reader = make_hf_text_reader(
        hf_repo,
        f"{prefix}/{err_name}",
        min_interval_s,
        **deadline_kwargs(make_hf_text_reader, deadline_at),
    )
    console_reader = make_hf_text_reader(
        hf_repo,
        f"{prefix}/console_{phase}.txt",
        min_interval_s,
        **deadline_kwargs(make_hf_text_reader, deadline_at),
    )

    def read(force: bool = False) -> str | None:
        parts: list[str] = []
        error_text = error_reader(force=force)
        if error_text:
            parts.append(f"--- {err_name} ---\n{error_text}")
        console_text = console_reader(force=force)
        if console_text:
            parts.append(f"--- console_{phase}.txt ---\n{console_text}")
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
    return bool(
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
    return bool(
        isinstance(hb, dict)
        and hb.get("retriable") is True
        and _heartbeat_matches_attempt(hb, launch_ts, current_attempt)
    )
