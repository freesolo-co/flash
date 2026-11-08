"""Pick a backend before dispatch, never retry a generation on another provider."""

import asyncio
import time


class ProviderSelection:
    def __init__(self, capacity, *, clock=time.monotonic):
        self.capacity = capacity
        self.clock = clock
        self._choices = {}
        self._lock = asyncio.Lock()

    async def choose(self, model):
        async with self._lock:
            choice, expires = self._choices.get(model, (None, 0))
            if self.clock() >= expires:
                choice = "lambda" if await self.capacity(model) else "modal"
                self._choices[model] = choice, self.clock() + 30
            return choice


async def unregister_active(engines, org_id, model_id, expected_generation):
    """Evict an adapter from both live pools without booting an idle GPU."""
    for engine in engines:
        stats = await engine.health.get_current_stats.aio()
        if any((stats.num_total_runners, stats.num_running_inputs, stats.backlog)):
            await engine.unregister.remote.aio(org_id, model_id, expected_generation)
