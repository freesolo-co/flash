"""RunPod Flash fine-tuning endpoints (queue-based, one dedicated GPU per run)."""

from __future__ import annotations

import os

from flash.providers.runpod.train.deps import (  # noqa: F401
    DEFAULT_CHALK_SPEC,
    DEFAULT_EXECUTION_TIMEOUT_MS,
    WORKER_DEPS,
    WORKER_IMAGE,
    WORKER_SYSTEM_DEPS,
    _effective_worker_env,
    build_worker_env,
    chalk_extra_pip,
    logger,
    resolve_worker_deps,
    strip_runpod_volume_env,
    worker_image_for_gpu,
)
from flash.providers.runpod.train.endpoints import (  # noqa: F401
    _ENDPOINT_CACHE,
    FLASH_SDK_LOCK,
    _patch_runpod_backoff,
    _run_suffix,
    _select_endpoint_resources,
    _train_body,
    endpoint_name,
    get_train_endpoint,
    isolate_flash_state,
    min_cuda_for,
    stop_endpoint,
    terminate_endpoint,
)


def upload_code(repo: str | None = None) -> str:
    """Upload the ``flash`` package to the run's HF artifact repo."""
    from huggingface_hub import HfApi

    import flash

    if not repo:
        raise RuntimeError(
            "hf_repo must be set (the run's [train] hf_repo: HF dataset repo for code + artifacts)"
        )
    token = os.environ.get("HF_TOKEN")
    pkg_dir = os.path.realpath(os.path.dirname(os.path.abspath(flash.__file__)))
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
    # create_repo(exist_ok=True) won't flip an existing public repo private; force it explicitly.
    api.update_repo_settings(repo_id=repo, repo_type="dataset", private=True)
    api.upload_folder(
        folder_path=pkg_dir,
        path_in_repo="code/flash",
        repo_id=repo,
        repo_type="dataset",
        ignore_patterns=["__pycache__/*", "*.pyc"],
        # delete_patterns purges renamed/orphaned modules so the worker never re-imports stale code.
        delete_patterns=["**"],
    )
    return repo
