"""concise stdout progress for packaged serving startup."""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Literal

_STARTED_AT = time.perf_counter()
_MIB = 1024 * 1024
_FILESYSTEM_USAGE_STAGES = frozenset(
    {
        "cache-prepared",
        "engine-constructed",
        "serving-ready",
    }
)


def emit_boot_progress(phase: str, /, **context: object) -> None:
    """emit one flushed, single-line startup marker with only caller-selected context."""

    elapsed = time.perf_counter() - _STARTED_AT
    fields = "".join(
        f" {name}={json.dumps(str(value), ensure_ascii=True)}" for name, value in context.items()
    )
    print(f"flash-serving boot elapsed={elapsed:.3f}s phase={phase}{fields}", flush=True)


def emit_filesystem_usage(
    stage: Literal["cache-prepared", "engine-constructed", "serving-ready"],
    cache_root: str | os.PathLike[str],
) -> None:
    """emit privacy-safe root and cache filesystem usage without affecting startup."""

    if stage not in _FILESYSTEM_USAGE_STAGES:
        raise ValueError("filesystem usage stage is invalid")
    context: dict[str, object] = {"stage": stage}
    for name, target in (("root", "/"), ("cache", cache_root)):
        try:
            usage = shutil.disk_usage(target)
        except OSError:
            context[f"{name}_status"] = "unavailable"
            continue
        context.update(
            {
                f"{name}_total_mib": usage.total // _MIB,
                f"{name}_used_mib": usage.used // _MIB,
                f"{name}_free_mib": usage.free // _MIB,
            }
        )
    emit_boot_progress("filesystem-usage", **context)
