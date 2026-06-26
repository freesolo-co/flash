"""Dynamic per-region health for the instance providers (Lambda / Hyperstack).

A region that proved SICK (instance reached ``active`` but the worker never reached a *training*
heartbeat — boot/cloud-init never came up or the GPU never initialized) is QUARANTINED for a TTL so
the allocator's capacity view and the launch-time region walk both avoid it instead of re-rolling the
same broken region. Runtime-learned generalization of the static ``HYPERSTACK_BLOCKED_REGIONS``.

Bounded DEMOTION, never a hard fail: a healthy candidate always outranks a quarantined one, but if a
class's only capacity is quarantined the allocator/walk still reach it via a last-resort ``ignore_sick``
pass (it just re-quarantines + escapes cross-provider if still broken). TTL self-heals a transient blip;
in-process only. Fires only on ``PollResult.host_fault`` (region never got a worker to training) — not a
mid-training stall, ``no_capacity``, or a worker/code error.
"""

from __future__ import annotations

import os
import threading
import time

# Quarantine window after a host fault. Override with FLASH_REGION_SICK_TTL_S (seconds); 0/non-positive
# disables (mark_region_sick becomes a no-op); an unparseable value falls back here (stays ENABLED).
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
