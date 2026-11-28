"""Sticky adapter routing over the reserved slots of one model.

Round-robin spreads one checkpoint's requests over every worker, so with many distinct LoRAs
each worker keeps loading and evicting adapters. Instead each adapter ranks the routable slots
by rendezvous hash (stable as slots are quarantined and readmitted; only the adapters of a
changed slot move) and uses its first slot, spilling to its second only while the first is
saturated. A worker also scores at most ``max_adapters`` distinct adapters at once (its vLLM
``max_loras``, the adapters it holds on the GPU together): an adapter that has no room on any
slot waits for one, and after ``wait_seconds`` takes its first slot anyway rather than fail.

The accounting is per router container. The ranking is deterministic, so every container
sends an adapter to the same slots; only the in-flight counts are container-local.
"""

import asyncio
import contextlib
import hashlib
import time

# Bounded so a miscounted or stuck view can only delay, never strand, a request.
ADAPTER_WAIT_SECONDS = 10 * 60


def rendezvous(key, slots):
    """``slots`` ordered by preference for ``key`` (highest random weight first)."""
    return sorted(
        slots,
        key=lambda slot: hashlib.sha256(f"{key}\0{slot}".encode()).digest(),
        reverse=True,
    )


class AdapterAffinity:
    def __init__(
        self, max_adapters, spill_after, *, wait_seconds=ADAPTER_WAIT_SECONDS, clock=time.monotonic
    ):
        self.max_adapters = max(1, int(max_adapters))
        self.spill_after = max(1, int(spill_after))
        self.wait_seconds = wait_seconds
        self._clock = clock
        self._load = {}  # slot -> generations in flight
        self._adapters = {}  # slot -> {adapter key: generations in flight}
        self._changed = None

    def load(self, slot):
        return self._load.get(slot, 0)

    def adapters(self, slot):
        return set(self._adapters.get(slot, {}))

    def choose(self, key, slots):
        """The slot for ``key`` now, or None when no slot has room for another adapter."""
        eligible = [
            slot
            for slot in rendezvous(key, slots)
            if key in self._adapters.get(slot, {})
            or len(self._adapters.get(slot, {})) < self.max_adapters
        ]
        if not eligible:
            return None
        first = eligible[0]
        if (
            self.load(first) >= self.spill_after
            and len(eligible) > 1
            and self.load(eligible[1]) < self.load(first)
        ):
            return eligible[1]
        return first

    @contextlib.asynccontextmanager
    async def route(self, key, slots, weight=1):
        """Hold ``weight`` generations of adapter ``key`` on its chosen slot.

        Choosing and counting never await, so the event loop needs no lock around them.
        """
        slots = list(slots)
        deadline = self._clock() + self.wait_seconds
        while (slot := self.choose(key, slots)) is None:
            remaining = deadline - self._clock()
            if remaining <= 0:
                slot = rendezvous(key, slots)[0]
                break
            if self._changed is None:
                self._changed = asyncio.Event()
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(remaining):
                    await self._changed.wait()
        self._load[slot] = self.load(slot) + weight
        active = self._adapters.setdefault(slot, {})
        active[key] = active.get(key, 0) + weight
        try:
            yield slot
        finally:
            self._load[slot] -= weight
            active[key] -= weight
            if active[key] <= 0:
                del active[key]
            changed, self._changed = self._changed, None
            if changed is not None:
                changed.set()
