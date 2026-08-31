"""RunPod endpoint names and exact attempt identity matching.

One run owns one seed and may execute several attempts, so an endpoint is named for the run plus
the attempt running on it. Every attempt carries an explicit ``-a<n>`` ordinal, including attempt
zero: the old grammar left attempt zero unsuffixed and marked later ones ``r<n>``, which made the
retry ordinal an identity and left the base name ambiguous between "the run" and "its first
attempt".
"""

from __future__ import annotations

import hashlib
import re

from flash.adapters.artifacts import MAX_ATTEMPT_ID
from flash.providers.core.base import gpu_short

_ENDPOINT_NAME_MAX = 64
_RUN_DIGEST_HEX = 16


def endpoint_name(friendly_gpu: str, suffix: str | None = None) -> str:
    """Build an endpoint name with an optional bounded run or attempt suffix."""
    base = f"flash-{gpu_short(friendly_gpu)}"
    if not suffix:
        return base
    safe = "".join(c for c in str(suffix) if c.isalnum() or c == "-").strip("-")
    if not safe:
        return base
    # raise rather than truncate: a clipped suffix can drop the attempt ordinal, and two attempts
    # of one run sharing a name is exactly what this identity exists to prevent.
    if len(base) + len(safe) + 1 > _ENDPOINT_NAME_MAX:
        raise ValueError("RunPod endpoint name exceeds the provider name budget")
    return f"{base}-{safe}"


def run_suffix(run_id: str | None) -> str | None:
    """Build the bounded hashed run prefix reserved for attempt endpoint names."""
    if not run_id:
        return None
    return hashlib.sha256(run_id.encode()).hexdigest()[:_RUN_DIGEST_HEX]


def attempt_suffix(run_id: str, attempt: int) -> str:
    """Build a bounded suffix that preserves the complete explicit attempt ordinal."""
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 0
        or attempt > MAX_ATTEMPT_ID
    ):
        raise ValueError("RunPod attempt identity is invalid")
    suffix = f"{run_suffix(run_id)}-a{attempt}"
    if len(suffix) > _ENDPOINT_NAME_MAX:
        raise ValueError("RunPod attempt identity exceeds the endpoint name budget")
    return suffix


def run_target_of(name: str) -> str | None:
    """The run-scoped endpoint target an attempt name belongs to, or ``None`` if it names no attempt.

    Every attempt endpoint is ``<run target>-a<n>``, so stripping an exact ordinal recovers the
    target that identifies its run across all of that run's attempts.
    """
    canonical = str(name or "").removeprefix("live-")
    if len(canonical) > _ENDPOINT_NAME_MAX:
        return None
    match = re.fullmatch(r"(.+)-a([0-9]+)", canonical)
    if match is None:
        return None
    ordinal = match.group(2)
    attempt = int(ordinal)
    # reject a padded ordinal such as "-a007": it reads as a different attempt than "-a7" while
    # naming the same one.
    if ordinal != str(attempt) or attempt > MAX_ATTEMPT_ID:
        return None
    return match.group(1)


def endpoint_name_matches_run(name: str, target: str) -> bool:
    """True iff ``name`` is an endpoint for ``target``'s run, carrying an exact attempt ordinal."""
    return bool(target) and run_target_of(name) == target


def select_endpoint_resources(resources: dict, target: str) -> list[str]:
    """Return resource ids whose names carry the exact attempt identity for one run."""
    if not target:
        return []
    return [
        uid
        for uid, resource in (resources or {}).items()
        if endpoint_name_matches_run(getattr(resource, "name", ""), target)
    ]
