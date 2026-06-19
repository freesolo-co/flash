"""RunPod Flash fine-tuning endpoints (queue-based, one dedicated GPU per run).

Flash provisions a dedicated RunPod GPU (RTX 4090 / 5090, no Docker), installs
``WORKER_DEPS``, runs the handler, returns the metrics dict, and scales to zero.

Flash's live ("ad-hoc") provisioning does not bundle local project code, so the
handler fetches the ``flash`` package from the HF dataset repo (uploaded by
``upload_code`` before submit), adds it to ``PYTHONPATH``, and runs
``flash.engine.worker`` to train. The worker streams the adapter + checkpoints to
the same HF repo for serving and preemption-resilient resume.

This is a package: the worker dependency stack + per-run env / chalk selection live in
``.deps`` (the leaf), the endpoint lifecycle + worker handler in ``.endpoints``; this
``__init__`` owns code upload + submit and re-exports the package's public surface so the
import path ``flash.providers.runpod.train`` is unchanged.
"""

from __future__ import annotations

import asyncio
import inspect
import os

# Re-export the package's public surface so ``from flash.providers.runpod.train import <name>``
# (callers in providers/runpod, providers/vast, runner, and the tests) keeps working unchanged.
from flash.providers.runpod.train.deps import (  # noqa: F401
    DEFAULT_CHALK_SPEC,
    DEFAULT_EXECUTION_TIMEOUT_MS,
    WORKER_DEPS,
    WORKER_IMAGE,
    WORKER_SYSTEM_DEPS,
    _chalk_selected,
    _effective_worker_env,
    build_worker_env,
    chalk_extra_pip,
    logger,
    resolve_worker_deps,
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
from flash.spec import JobSpec


def upload_code(repo: str | None = None) -> str:
    """Upload the ``flash`` package to the run's HF artifact repo.

    ``repo`` is the per-run artifact repo (``spec.train.hf_repo``); the worker fetches
    ``code/**`` from the same repo it is given in the submit payload, so the code must land in
    that per-run repo.

    The worker downloads ``code/**`` to ``/runcode``. Verifiers-only: there are no built-in
    example environments to ship — Hub/installed envs are pip-installed on the worker (see
    ``registry.worker_pip_for_env``).

    Only the ``flash`` package is uploaded, NOT the client's project tree. Managed runs must
    reference a published Hub env by ``id`` (``slm env push`` to publish a local env first); the
    worker pip-installs the env wheel.
    """
    from huggingface_hub import HfApi

    import flash

    if not repo:
        raise RuntimeError(
            "hf_repo must be set (the run's [train] hf_repo: HF dataset repo for code + artifacts)"
        )
    token = os.environ.get("HF_TOKEN")
    pkg_dir = os.path.dirname(os.path.abspath(flash.__file__))
    api = HfApi(token=token)
    # Run artifact repos are always private (they carry run code, adapters, and metrics).
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
    # create_repo(exist_ok=True) is a no-op on an EXISTING repo, so `private=True` above does NOT
    # change the visibility of a repo that was created earlier as public. Force private explicitly
    # so a reused/public artifact repo can't leak run code/adapters/metrics under the always-private
    # invariant. (Idempotent: a no-op on a repo that is already private.)
    api.update_repo_settings(repo_id=repo, repo_type="dataset", private=True)
    api.upload_folder(
        folder_path=pkg_dir,
        path_in_repo="code/flash",
        repo_id=repo,
        repo_type="dataset",
        ignore_patterns=["__pycache__/*", "*.pyc"],
    )
    return repo


def submit_train(spec: JobSpec, seed: int, log=None) -> dict:
    """Provision a dedicated GPU via Flash, run training, return the metrics dict."""
    timeout_s = max(60, int(spec.gpu.max_wall_seconds))
    from flash.envs.registry import worker_hub_env_ids, worker_pip_for_env

    handler = get_train_endpoint(
        spec.gpu.type,
        execution_timeout_ms=timeout_s * 1000,
        name_suffix=_run_suffix(spec.run_id),
        disk_gb=spec.gpu.disk_gb,
        spec=spec,
    )
    payload = {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "env": build_worker_env(spec, seed),
        # extra_pip is installed by the worker for EVERY job (baked-image RunPod _train_body and
        # Vast bootstrap both pip-install it), so it's where the chalk spec must go to reach a
        # default run — see chalk_extra_pip().
        "extra_pip": (list(spec.environment.pip) or worker_pip_for_env(spec.environment.id))
        + chalk_extra_pip(spec),
        "hub_env_ids": worker_hub_env_ids(spec.environment.id, spec.environment.params),
    }
    if log is not None:
        print(
            f"submitting Flash job: gpu={spec.gpu.type} phase={spec.phase} "
            f"seed={seed} model={spec.model}",
            file=log,
            flush=True,
        )

    async def _call():
        res = handler(payload)
        if inspect.isawaitable(res):
            res = await res
        return res

    out = asyncio.run(_call())
    if not isinstance(out, dict):
        raise RuntimeError(f"flash job returned no metrics: {out!r}")
    return out
