"""Dynamic per-region health for the instance providers (Lambda / Hyperstack).

A region that just proved SICK — an instance reached OS-level ``active`` but the worker never reached
a *training* heartbeat (cloud-init/boot never came up, or the GPU/driver never initialized) — is
QUARANTINED for a TTL so both the allocator's capacity view AND the launch-time region walk avoid it,
instead of re-rolling the same broken region on every retry. This GENERALIZES the hand-curated static
``HYPERSTACK_BLOCKED_REGIONS`` denylist to ANY region on ANY instance provider, learned at runtime:
the two observed failures — a wedged Lambda ``us-east-1`` (cloud-init never ran) and the broken-driver
Hyperstack ``CANADA-1`` L40 fleet (NVML init fail) — are exactly "active but never trained", so a run
that hits one quarantines that region and the next attempt lands elsewhere.

TTL-based so a TRANSIENT regional blip self-heals without a human editing a denylist; in-process only
(resets on control-plane restart) and strictly best-effort: it can only DEMOTE a region for a bounded
window, never hard-fail a run. The demotion is a strict RANKING preference, not a hard exclusion — a
healthy candidate always outranks a quarantined one, but if a class's only capacity is in quarantined
regions the allocator and the launch-time region walk still REACH it via a last-resort ``ignore_sick``
pass rather than hard-failing, so the run launches there anyway (and simply re-quarantines + escapes
cross-provider if it is still broken). The TTL also lapses and the region is tried fresh.

The quarantine fires only on a HOST-fault signal (``PollResult.host_fault``), which the instance
pollers set just for failures that prove the region/host never got a worker to training — NOT for a
mid-training stall (the region was working), nor ``no_capacity`` (the region is fine, just full), nor
a genuine worker/code error. A healthy box reaches a heartbeat in well under the deadlines, so a
single host-fault signal is trustworthy enough to demote the region for the TTL.
"""

from __future__ import annotations

import os
import threading
import time

# How long a region stays quarantined after a host-fault failure. Long enough to ride out a regional
# outage / a bad fleet image, short enough that a recovered region returns on its own within an hour.
# Override with FLASH_REGION_SICK_TTL_S (seconds); 0 disables the dynamic quarantine entirely (a
# non-positive TTL makes mark_region_sick a no-op). An UNPARSEABLE value is ignored and falls back to
# this default (quarantine stays ENABLED) rather than silently disabling it on a typo.
_DEFAULT_SICK_TTL_S = 1800.0


def sick_ttl_s() -> float:
    raw = os.environ.get("FLASH_REGION_SICK_TTL_S")
    if raw is None:
        return _DEFAULT_SICK_TTL_S
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_SICK_TTL_S


_lock = threading.Lock()
# (provider, REGION) -> epoch when the quarantine expires. REGION is upper-cased so the key is stable
# regardless of how a provider reports case (Hyperstack uppercases; Lambda is lower).
_sick: dict[tuple[str, str], float] = {}


def _key(provider: str, region: str) -> tuple[str, str]:
    return (provider, str(region or "").upper())


def mark_region_sick(provider: str, region: str | None, ttl_s: float | None = None, now: float | None = None) -> None:
    """Quarantine ``(provider, region)`` for ``ttl_s`` (default ``sick_ttl_s()``). No-op on a blank
    region or a non-positive TTL (the quarantine is disabled). Idempotent — re-marking just extends
    the window from ``now``."""
    if not region:
        return
    ttl = sick_ttl_s() if ttl_s is None else ttl_s
    if ttl <= 0:
        return
    with _lock:
        _sick[_key(provider, region)] = (now if now is not None else time.time()) + ttl


def region_is_sick(provider: str, region: str | None, now: float | None = None) -> bool:
    """True while ``(provider, region)`` is within its quarantine window. Expired entries self-heal
    (popped on read), so a recovered region returns automatically once the TTL lapses."""
    if not region:
        return False
    n = now if now is not None else time.time()
    k = _key(provider, region)
    with _lock:
        exp = _sick.get(k)
        if exp is None:
            return False
        if exp <= n:
            _sick.pop(k, None)  # expired -> self-heal
            return False
        return True


def healthy_regions(
    provider: str, regions: list[str], now: float | None = None, ignore_sick: bool = False
) -> list[str]:
    """``regions`` minus those currently quarantined for ``provider`` (order preserved).

    ``ignore_sick=True`` returns ``regions`` unfiltered — the allocator's LAST-RESORT pass when
    every fitting region is quarantined, so the quarantine can only DEMOTE a region (rank it behind
    healthy options), never make a run hard-fail for lack of any candidate (its bounded-demotion
    contract; see the module docstring)."""
    if ignore_sick:
        return list(regions)
    return [r for r in regions if not region_is_sick(provider, r, now=now)]


def sick_regions(provider: str | None = None, now: float | None = None) -> dict[tuple[str, str], float]:
    """Currently-quarantined ``(provider, region) -> expiry`` (diagnostics/tests), optionally for one
    provider. Excludes expired entries."""
    n = now if now is not None else time.time()
    with _lock:
        return {k: v for k, v in _sick.items() if v > n and (provider is None or k[0] == provider)}


def clear() -> None:
    """Drop all quarantine state. For tests / a forced reset; never needed in normal operation
    (entries self-heal at their TTL)."""
    with _lock:
        _sick.clear()
