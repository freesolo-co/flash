"""shared gpu-process mechanics for modal serving containers.

this owns the parts that are identical for every vllm deployment: starting one runtime lazily,
draining the container when its engine core dies, and giving a dynamically built class a stable
module-level identity. adapter policy, persistence, authorization, and billing stay with callers.
"""

from __future__ import annotations

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
        if self._runtime is None:
            runtime = VllmLoraRuntime(
                self.engine_config(),
                on_engine_death=self._drain_on_engine_death,
            )
            await runtime.start()
            self._runtime = runtime
        return self._runtime

    async def close_runtime(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            await runtime.close()

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


def bind_module_class(namespace: dict[str, Any], cls: type, class_name: str) -> type:
    """give a dynamically created class a real module-level identity.

    modal validates that a registered class lives at module scope and re-imports it by name in the
    container, so the rename and the module binding must happen before any class decorator runs.
    a decorator returns a wrapper holding the user class separately, so renaming afterwards would
    rename the wrapper instead and every instance would register under the original name.
    """
    if not class_name or not class_name.isidentifier():
        raise ValueError("class_name must be a valid python identifier")
    cls.__name__ = class_name
    cls.__qualname__ = class_name
    namespace[class_name] = cls
    return cls
