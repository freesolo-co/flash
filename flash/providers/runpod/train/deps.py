"""Back-compat shim: worker packaging moved to flash.providers._worker (shared kernel)."""

from __future__ import annotations

from flash.providers._worker import (  # noqa: F401
    _REMOVED_OPTIMIZATION_ENV,
    _RUNTIME_SECRET_KEYS,
    _WEIGHT_CACHE_MOUNT,
    BAKED_PER_SM_ARCHES,
    DEFAULT_CHALK_SPEC,
    DEFAULT_CHALK_VERSION,
    DEFAULT_EXECUTION_TIMEOUT_MS,
    LATEST_CHALK_MAIN_SHA,
    WORKER_DEPS,
    WORKER_IMAGE,
    WORKER_IMAGE_PER_SM_ENV,
    WORKER_IMAGE_TEMPLATE_ENV,
    WORKER_SYSTEM_DEPS,
    _append_tag_suffix,
    _effective_worker_env,
    _truthy,
    build_worker_env,
    chalk_extra_pip,
    drop_unmounted_cache_env,
    logger,
    os,
    resolve_worker_deps,
    strip_runpod_volume_env,
    weight_cache_env,
    worker_image_for_gpu,
)
