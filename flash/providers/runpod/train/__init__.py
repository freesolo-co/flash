"""RunPod Flash fine-tuning endpoints (queue-based, one dedicated GPU per run)."""

from __future__ import annotations

import os
import time
from io import BytesIO

from flash.providers._hf_retry import hf_call, hf_status_code
from flash.providers.runpod.train.deps import (  # noqa: F401
    DEFAULT_CHALK_SPEC,
    DEFAULT_EXECUTION_TIMEOUT_MS,
    LATEST_CHALK_MAIN_SHA,
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

_CODE_SNAPSHOT_COMPLETE = ".flash-code-snapshot-complete"


_hf_status_code = hf_status_code


def _hf_call(call, label: str):
    return hf_call(call, label, logger=logger, sleep=time.sleep)


def _is_hf_not_found(exc: BaseException) -> bool:
    return _hf_status_code(exc) == 404 or exc.__class__.__name__ == "RepositoryNotFoundError"


def _ensure_private_artifact_repo(api, repo: str) -> None:
    try:
        _hf_call(
            lambda: api.repo_info(repo_id=repo, repo_type="dataset"),
            f"lookup artifact repo {repo}",
        )
    except Exception as exc:
        if not _is_hf_not_found(exc):
            raise
        _hf_call(
            lambda: api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True),
            f"create artifact repo {repo}",
        )
    # create_repo(exist_ok=True) won't flip an existing public repo private; force it explicitly.
    _hf_call(
        lambda: api.update_repo_settings(repo_id=repo, repo_type="dataset", private=True),
        f"force artifact repo private {repo}",
    )


def upload_code(repo: str | None = None, *, code_prefix: str | None = None) -> str:
    """Upload the ``flash`` package to its content-addressed HF artifact prefix."""
    from huggingface_hub import HfApi

    import flash
    from flash.runner import flash_code_prefix

    if not repo:
        raise RuntimeError(
            "hf_repo must be set (the run's [train] hf_repo: HF dataset repo for code + artifacts)"
        )
    token = os.environ.get("HF_TOKEN")
    pkg_dir = os.path.realpath(os.path.dirname(os.path.abspath(flash.__file__)))
    api = HfApi(token=token)
    _ensure_private_artifact_repo(api, repo)
    code_prefix = code_prefix or flash_code_prefix()
    code_marker = f"{code_prefix}/{_CODE_SNAPSHOT_COMPLETE}"
    if _hf_call(
        lambda: api.file_exists(repo_id=repo, filename=code_marker, repo_type="dataset"),
        f"check flash code snapshot {repo}:{code_marker}",
    ):
        return repo
    _hf_call(
        lambda: api.upload_folder(
            folder_path=pkg_dir,
            path_in_repo=code_prefix,
            repo_id=repo,
            repo_type="dataset",
            ignore_patterns=["__pycache__/*", "*.pyc", "*.pyo"],
        ),
        f"upload flash code to {repo}:{code_prefix}",
    )
    _hf_call(
        lambda: api.upload_file(
            path_or_fileobj=BytesIO(b"complete\n"),
            path_in_repo=code_marker,
            repo_id=repo,
            repo_type="dataset",
        ),
        f"mark flash code snapshot complete {repo}:{code_marker}",
    )
    return repo
