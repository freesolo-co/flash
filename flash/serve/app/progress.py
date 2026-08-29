"""concise stdout progress for packaged serving startup."""

from __future__ import annotations

import json
import os
import shutil
import stat
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


def boot_elapsed_seconds() -> float:
    """return elapsed process boot time for persisted replica telemetry."""

    return max(0.0, time.perf_counter() - _STARTED_AT)


def emit_boot_progress(phase: str, /, **context: object) -> None:
    """emit one flushed, single-line startup marker with only caller-selected context."""

    elapsed = boot_elapsed_seconds()
    fields = "".join(
        f" {name}={json.dumps(str(value), ensure_ascii=True)}" for name, value in context.items()
    )
    print(f"flash-serving boot elapsed={elapsed:.3f}s phase={phase}{fields}", flush=True)


def _cache_tree_logical_bytes(cache_root: str | os.PathLike[str]) -> tuple[int, bool]:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(cache_root, directory_flags)
    seen_files: set[tuple[int, int]] = set()

    def scan(directory_fd: int) -> tuple[int, bool]:
        total = 0
        complete = True
        try:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    try:
                        details = entry.stat(follow_symlinks=False)
                        if stat.S_ISDIR(details.st_mode):
                            child_fd = os.open(
                                entry.name,
                                directory_flags,
                                dir_fd=directory_fd,
                            )
                            child_complete = True
                            try:
                                child_total, child_complete = scan(child_fd)
                            finally:
                                try:
                                    os.close(child_fd)
                                except OSError:
                                    child_complete = False
                            total += child_total
                            complete = complete and child_complete
                        elif stat.S_ISREG(details.st_mode):
                            identity = (details.st_dev, details.st_ino)
                            if identity not in seen_files:
                                seen_files.add(identity)
                                total += details.st_size
                    except OSError:
                        complete = False
        except OSError:
            complete = False
        return total, complete

    complete = True
    try:
        total, complete = scan(root_fd)
    finally:
        try:
            os.close(root_fd)
        except OSError:
            complete = False
    return total, complete


def emit_filesystem_usage(
    stage: Literal["cache-prepared", "engine-constructed", "serving-ready"],
    cache_root: str | os.PathLike[str],
) -> None:
    """emit privacy-safe root and cache filesystem usage without affecting startup."""

    if stage not in _FILESYSTEM_USAGE_STAGES:
        raise ValueError("filesystem usage stage is invalid")
    context: dict[str, object] = {"stage": stage}
    try:
        usage = shutil.disk_usage("/")
    except OSError:
        context["root_status"] = "unavailable"
    else:
        context.update(
            {
                "root_total_mib": usage.total // _MIB,
                "root_used_mib": usage.used // _MIB,
                "root_free_mib": usage.free // _MIB,
            }
        )
    try:
        cache_logical_bytes, cache_complete = _cache_tree_logical_bytes(cache_root)
    except OSError:
        context["cache_status"] = "unavailable"
    else:
        context["cache_logical_bytes"] = cache_logical_bytes
        if not cache_complete:
            context["cache_status"] = "partial"
    emit_boot_progress("filesystem-usage", **context)
