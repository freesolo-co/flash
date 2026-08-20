"""shared gpu-process mechanics for modal serving containers.

this owns the parts that are identical for every vllm deployment: starting one runtime lazily and
draining the container when its engine core dies. adapter policy, persistence, authorization, and
billing stay with callers.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from flash.serve.runtime import EngineConfig, VllmLoraRuntime


class RuntimeContainer:
    """one lazily started runtime plus the container-draining death handler.

    subclasses supply the engine config. modal entrypoints stay in the caller so each deployment
    keeps its own rpc signatures and policy.
    """

    def __init__(self) -> None:
        self._runtime: VllmLoraRuntime | None = None
        self._runtime_lock = asyncio.Lock()
        self._drained = False

    def engine_config(self) -> EngineConfig:
        raise NotImplementedError("subclasses must supply an engine config")

    @property
    def runtime(self) -> VllmLoraRuntime:
        if self._runtime is None:
            raise RuntimeError("runtime has not been started")
        return self._runtime

    async def start_runtime(self) -> VllmLoraRuntime:
        """construct and start the runtime once, wiring engine death to container draining.

        the runtime is published only after `start()` returns, so a failed start leaves the
        container with no runtime rather than one that looks started and answers requests from a
        half-initialized engine.
        """
        async with self._runtime_lock:
            if self._runtime is None:
                runtime = VllmLoraRuntime(
                    self.engine_config(),
                    on_engine_death=self._drain_on_engine_death,
                )
                await runtime.start()
                self._runtime = runtime
            return self._runtime

    async def close_runtime(self) -> None:
        async with self._runtime_lock:
            if self._runtime is not None:
                await self._runtime.close()
                self._runtime = None

    async def _drain_on_engine_death(self, _health: Any) -> None:
        """stop accepting work so later demand starts a healthy container.

        a dead engine core cannot be repaired in place, so the container stops fetching inputs and
        lets modal route new work to a fresh replica. draining once is enough.
        """
        if self._drained:
            return
        self._drained = True
        print(
            "serving: vllm engine core is dead; draining this container so later demand starts a "
            "healthy engine",
            flush=True,
        )
        try:
            from modal.experimental import stop_fetching_inputs

            stop_fetching_inputs()
        except Exception:
            # without a drain hook the only safe move is to end the container outright.
            os._exit(1)
