"""Provider-neutral worker packaging: the deps/image/env every rent-a-box or serverless worker ships, plus upload_code for the HF code snapshot. Shared kernel — no provider package imports another for this."""

from __future__ import annotations

import os
import time
from io import BytesIO

from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
from flash._internal.logging import get_logger
from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS
from flash.core.spec import (
    CONTROL_PLANE_OWNED_ENV_KEYS,
    MANAGED_TEACHER_CREDENTIAL_ENV_KEYS,
    PUBLIC_URL_ENV,
    TEACHER_CAPABILITY_ENV,
    JobSpec,
    require_matching_seed,
)
from flash.providers.artifacts.hf import hf_call, hf_status_code
from flash.providers.base import get_gpu_info
from flash.teacher.retry_contract import OPD_RESUME_REVISION_ENV

# pinned literal (not __name__): keeps the logger stream named "flash.providers.runpod.train"
# after this module moved out of flash/providers/runpod/train.py, so operator log filters stay stable.
# the literal is an OPERATOR-FACING contract, not a module path: it must not track package renames,
# or the log filters and dashboards it exists to protect break.
logger = get_logger("flash.providers.runpod.train")


WORKER_IMAGE = "ghcr.io/freesolo-co/flash-worker:cu128"

# MUST mirror the bake matrix in .github/workflows/bake-kernel-cache.yml. Unlisted arches fall
# back to WORKER_IMAGE (no -smXX tag built) rather than failing at docker pull.
BAKED_PER_SM_ARCHES = frozenset({"sm80", "sm86", "sm89", "sm90", "sm120", "sm100"})


def worker_image_for_gpu(friendly_gpu: str | None) -> str:
    """Return the worker Docker image for a GPU class (per-SM kernel-cache tag or base)."""
    if friendly_gpu:
        info = get_gpu_info(friendly_gpu)
        # Per-SM baked kernel-cache image is always used for baked arches (skips ~10-15 min
        # cold-start JIT). Unbaked arches fall through to the base image to avoid a 404 docker pull.
        if info.sm in BAKED_PER_SM_ARCHES:
            return f"{WORKER_IMAGE}-{info.sm}"
    return WORKER_IMAGE


DEFAULT_EXECUTION_TIMEOUT_MS = 6 * 3600 * 1000  # 6h cap


# optimization toggles dropped in pr #175. filter declared runtime secrets so dead keys cannot
# reach the worker, including allocator settings that crash grpo vllm sleep mode.
_REMOVED_OPTIMIZATION_ENV = frozenset(
    {
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "RL_VLLM_SLEEP",
        "SFT_PER_DEVICE_BS",
        "FLASH_ALLOC_AUTO",
        "TORCHDYNAMO_DISABLE",
        "VLLM_USE_V1",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_FLASH_ATTN_VERSION",
        "FLASH_DISABLE_FA2",
        "FLASH_DISABLE_FA3",
        "FLASH_ROPE_KERNEL",
        "FLASH_QKV_KERNEL",
        "FLASH_MLP_KERNEL",
        "FLASH_EMBED_KERNEL",
        "FLASH_FP8_BASE",
        "FLASH_TRITON_LORA",
        "FLASH_WORKER_DEPS",
        "FLASH_WORKER_EXTRA_DEPS",
        # chalk selected an install source for kernels that installed against an in-process
        # trainer.model. verl runs the model in a child interpreter, so nothing could engage and the
        # whole surface was deleted. the key is filtered rather than merely forgotten: an unfiltered
        # key still reaches the worker, where it now configures nothing, and a run that sets it would
        # otherwise get silence instead of the warning that says it stopped mattering.
        "FLASH_CHALK_SPEC",
    }
)


_WEIGHT_CACHE_MOUNT = "/runpod-volume"


def weight_cache_env(mount: str = _WEIGHT_CACHE_MOUNT) -> dict[str, str]:
    """Env pointing the base-model prefetch at the persistent volume mount.

    Sets FLASH_WEIGHT_CACHE_DIR (not HF_HOME) so only the trusted public base model lands on the
    shared multi-tenant mount; reward/env HF downloads stay in the ephemeral per-worker cache. JIT
    caches are never redirected -- sharing compiled artifacts across tenants is unsafe.
    """
    return {"FLASH_WEIGHT_CACHE_DIR": f"{mount}/hf-cache/hub"}


def strip_runpod_volume_env(env: dict, mount: str = _WEIGHT_CACHE_MOUNT) -> dict:
    """Remove the RunPod weight-cache redirect from an env bound for a non-RunPod worker (mutates)."""
    for k in [k for k, v in env.items() if str(v).startswith(mount)]:
        env.pop(k, None)
    return env


def build_worker_env(
    spec: JobSpec,
    seed: int,
    runtime_secrets: dict[str, str] | None = None,
) -> dict:
    """Per-run env passed to the worker from managed control-plane inputs."""
    canonical_seed = require_matching_seed(spec, seed)
    declared_managed_credentials = sorted(
        set(spec.environment.secrets) & MANAGED_TEACHER_CREDENTIAL_ENV_KEYS
    )
    if declared_managed_credentials:
        raise ValueError(
            "environment secrets must not include managed teacher credential names: "
            + ", ".join(declared_managed_credentials)
        )
    # GRPO and OPD run a verl vLLM rollout (`actor_rollout_ref.rollout.name=vllm`), and verl
    # leaves rollout.enable_sleep_mode defaulted True, so the engine always builds a
    # CuMemAllocator -- which asserts outright on "expandable_segments:True"
    # (vllm/device_allocator/cumem.py:132, pytorch#147851). Both therefore take the
    # non-expandable conf.
    _alloc_conf = (
        "expandable_segments:True"
        if str(getattr(spec, "algorithm", "")).lower() == "sft"
        else "garbage_collection_threshold:0.8,max_split_size_mb:256"
    )
    env: dict[str, str] = {
        "RUN_ID": spec.run_id,
        "FLASH_ARM": "runpod",
        "BENCH_HF_MODEL": spec.model,
        "SEED": str(canonical_seed),
        "PYTORCH_CUDA_ALLOC_CONF": _alloc_conf,
        "PYTORCH_ALLOC_CONF": _alloc_conf,
    }
    for key in (
        "HF_TOKEN",
        "GITHUB_TOKEN",
    ):
        # Stripped, and a blank value forwards NOTHING. The worker's git askpass and HF client both
        # branch on presence, so a whitespace-only credential is worse than an absent one: it turns
        # an anonymous public fetch into an authenticated request with a malformed token, which
        # GitHub and HF reject outright.
        value = (os.environ.get(key) or "").strip()
        if value:
            env[key] = value
    env["HF_REPO"] = spec.train.hf_repo
    if getattr(spec.gpu, "network_volume", None):
        env.update(weight_cache_env())
    # runtime secrets and declared env secrets may never clobber a control-plane-owned key:
    # the canonical seed, run id, hf repo, and arm are set above, and a runtime seed override
    # would break the authoritative-seed invariant regardless of how environment.secrets was
    # populated. removed keys are filtered here too. the two sets are disjoint and answer
    # different questions: control-plane ownership prevents overrides such as SEED, while removed
    # optimization keys configure nothing and must not silently reach the worker.
    allowed_runtime_secrets = {
        k
        for k in (set(DEFAULT_RUNTIME_SECRET_KEYS) | set(spec.environment.secrets))
        if k.upper() not in CONTROL_PLANE_OWNED_ENV_KEYS
        and k.upper() not in _REMOVED_OPTIMIZATION_ENV
    }
    for k, v in (runtime_secrets or {}).items():
        if k in allowed_runtime_secrets and v:
            env[k] = str(v)
    # the pinned resume revision is control-plane-owned transport, not a user-declared secret.
    # lifecycle removes caller input and supplies only the gate-approved sha on a required replacement.
    env.pop(OPD_RESUME_REVISION_ENV, None)
    resume_revision = (runtime_secrets or {}).get(OPD_RESUME_REVISION_ENV)
    if resume_revision:
        env[OPD_RESUME_REVISION_ENV] = str(resume_revision)

    # managed teacher provider credentials stay control-plane-only even if a caller declares or
    # supplies the same names as runtime secrets. opd receives only the control-panel origin and its
    # attempt-scoped teacher capability.
    for key in MANAGED_TEACHER_CREDENTIAL_ENV_KEYS:
        env.pop(key, None)
    env.pop(PUBLIC_URL_ENV, None)
    env.pop(TEACHER_CAPABILITY_ENV, None)
    if str(getattr(spec, "algorithm", "")).lower() == "opd":
        public_url = str((runtime_secrets or {}).get(PUBLIC_URL_ENV) or "").strip()
        capability = str((runtime_secrets or {}).get(TEACHER_CAPABILITY_ENV) or "").strip()
        if not public_url or not capability:
            raise RuntimeError("managed opd control-panel teacher transport is missing")
        env[PUBLIC_URL_ENV] = public_url
        env[TEACHER_CAPABILITY_ENV] = capability
    # declared runtime secrets can carry any name, so their names are listed explicitly for the
    # redactors (flash._internal.diagnostics and the provider bootstraps): the name-shape
    # heuristic alone would let AWS_SECRET_ACCESS_KEY-style values through. set last so no
    # runtime secret can clobber it -- SECRET_ENV_KEYS_ENV is control-plane-owned, so a job cannot
    # declare it as a secret and have its value overwritten by this list.
    secret_keys = (set(allowed_runtime_secrets) | {TEACHER_CAPABILITY_ENV}) & set(env)
    # the list is comma-joined, so a name containing a comma would arrive at every redactor as two
    # unrelated names and its value would never be recognized. [environment] secrets rejects those
    # names at declaration; this is the fail-closed guard, because emitting a silently ambiguous
    # list is how a credential reaches diagnostics verbatim.
    ambiguous = sorted(key for key in secret_keys if "," in key)
    if ambiguous:
        raise RuntimeError(
            f"secret env name(s) contain the {SECRET_ENV_KEYS_ENV} delimiter: {ambiguous}"
        )
    if secret_keys:
        env[SECRET_ENV_KEYS_ENV] = ",".join(sorted(secret_keys))
    return env


_CODE_SNAPSHOT_COMPLETE = ".flash-code-snapshot-complete"


def _hf_call(call, label: str, *, deadline_at: float | None = None):
    return hf_call(
        call,
        label,
        logger=logger,
        sleep=time.sleep,
        deadline_at=deadline_at,
    )


def _ensure_private_artifact_repo(
    api,
    repo: str,
    *,
    deadline_at: float | None = None,
) -> None:
    try:
        _hf_call(
            lambda: api.repo_info(repo_id=repo, repo_type="dataset"),
            f"lookup artifact repo {repo}",
            deadline_at=deadline_at,
        )
    except Exception as exc:
        if hf_status_code(exc) != 404 and exc.__class__.__name__ != "RepositoryNotFoundError":
            raise
        _hf_call(
            lambda: api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True),
            f"create artifact repo {repo}",
            deadline_at=deadline_at,
        )
    # create_repo(exist_ok=True) won't flip an existing public repo private; force it explicitly.
    _hf_call(
        lambda: api.update_repo_settings(repo_id=repo, repo_type="dataset", private=True),
        f"force artifact repo private {repo}",
        deadline_at=deadline_at,
    )


def upload_code(
    repo: str | None = None,
    *,
    code_prefix: str | None = None,
    deadline_at: float | None = None,
) -> str:
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
    _ensure_private_artifact_repo(api, repo, deadline_at=deadline_at)
    code_prefix = code_prefix or flash_code_prefix()
    code_marker = f"{code_prefix}/{_CODE_SNAPSHOT_COMPLETE}"
    if _hf_call(
        lambda: api.file_exists(repo_id=repo, filename=code_marker, repo_type="dataset"),
        f"check flash code snapshot {repo}:{code_marker}",
        deadline_at=deadline_at,
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
        deadline_at=deadline_at,
    )
    _hf_call(
        lambda: api.upload_file(
            path_or_fileobj=BytesIO(b"complete\n"),
            path_in_repo=code_marker,
            repo_id=repo,
            repo_type="dataset",
        ),
        f"mark flash code snapshot complete {repo}:{code_marker}",
        deadline_at=deadline_at,
    )
    return repo
