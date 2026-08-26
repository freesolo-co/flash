"""RunPod endpoint names and exact attempt identity matching."""

from __future__ import annotations

import hashlib
import re

from flash.providers.core.base import gpu_short


def endpoint_name(friendly_gpu: str, suffix: str | None = None) -> str:
    """Build an endpoint name with an optional bounded run or attempt suffix."""
    base = f"flash-{gpu_short(friendly_gpu)}"
    if not suffix:
        return base
    safe = "".join(c for c in str(suffix) if c.isalnum() or c == "-").strip("-")[:24]
    return f"{base}-{safe}" if safe else base


def _run_suffix(run_id: str | None) -> str | None:
    """Build the bounded hashed run prefix reserved for attempt endpoint names."""
    if not run_id:
        return None
    digest = hashlib.sha1(run_id.encode()).hexdigest()[:8]
    readable = re.sub(r"[^a-z0-9]", "", run_id.lower())[-3:]
    return f"{readable}{digest}" if readable else digest


def attempt_suffix(run_id: str, attempt: int) -> str:
    """Build a bounded suffix that preserves the complete explicit attempt ordinal."""
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("RunPod attempt identity is invalid")
    run_prefix = _run_suffix(run_id)
    suffix = f"{run_prefix}-a{attempt}"
    if len(suffix) > 24:
        raise ValueError("RunPod attempt identity exceeds the endpoint name budget")
    return suffix


def _endpoint_name_matches_run(name: str, target: str) -> bool:
    canonical = str(name or "").removeprefix("live-")
    return re.fullmatch(re.escape(target) + r"-a[0-9]+", canonical) is not None


def _select_endpoint_resources(resources: dict, target: str) -> list[str]:
    """Return resource ids whose names carry the exact attempt identity for one run."""
    if not target:
        return []
    return [
        uid
        for uid, resource in (resources or {}).items()
        if _endpoint_name_matches_run(getattr(resource, "name", ""), target)
    ]
