"""Back-compat shim: worker packaging moved to flash.providers._worker (shared kernel)."""

from __future__ import annotations

from flash.providers._worker import (  # noqa: F401
    BAKED_PER_SM_ARCHES,
    DEFAULT_EXECUTION_TIMEOUT_MS,
    LATEST_CHALK_MAIN_SHA,
    WORKER_IMAGE,
    WORKER_SYSTEM_DEPS,
    build_worker_env,
    drop_unmounted_cache_env,
    logger,
    os,
    resolve_worker_deps,
    strip_runpod_volume_env,
    weight_cache_env,
    worker_image_for_gpu,
)
