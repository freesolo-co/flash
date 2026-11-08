"""Short generation leases preserve cancellation when a controller loses connectivity."""

import asyncio
import contextlib
import time

from flash.serving.src.stream_channel.protocol import CLEANUP_SECONDS, LEASE_SECONDS


class GenerationLeases:
    def __init__(self, max_entries, *, clock=time.monotonic):
        self.entries = {}
        self.max_entries = max_entries
        self.clock = clock

    def renew(self, generation_id):
        if not isinstance(generation_id, str) or not generation_id or len(generation_id) > 512:
            raise ValueError("invalid generation id")
        now = self.clock()
        self.entries = {key: expiry for key, expiry in self.entries.items() if expiry > now}
        if generation_id not in self.entries and len(self.entries) >= self.max_entries:
            raise ValueError("worker generation lease capacity exceeded")
        self.entries[generation_id] = now + LEASE_SECONDS

    async def check(self, generation_id):
        if self.entries.get(generation_id, 0) <= self.clock():
            raise TimeoutError("generation ownership lease expired")

    async def watch(self, generation_id, owner, abort):
        while True:
            delay = self.entries.get(generation_id, 0) - self.clock()
            if delay <= 0:
                try:
                    with contextlib.suppress(Exception):
                        async with asyncio.timeout(CLEANUP_SECONDS):
                            await abort(generation_id)
                finally:
                    owner.cancel()
                return
            await asyncio.sleep(delay)

    def finish(self, generation_id):
        self.entries.pop(generation_id, None)
