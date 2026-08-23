"""concise stdout progress for packaged serving startup."""

from __future__ import annotations

import json
import time

_STARTED_AT = time.perf_counter()


def emit_boot_progress(phase: str, /, **context: object) -> None:
    """emit one flushed, single-line startup marker with only caller-selected context."""

    elapsed = time.perf_counter() - _STARTED_AT
    fields = "".join(
        f" {name}={json.dumps(str(value), ensure_ascii=True)}" for name, value in context.items()
    )
    print(f"flash-serving boot elapsed={elapsed:.3f}s phase={phase}{fields}", flush=True)
