"""Provider-neutral worker packaging: the deps/image/env every rent-a-box or serverless worker ships, plus upload_code for the HF code snapshot. Shared kernel — no provider package imports another for this."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from io import BytesIO

from flash._logging import get_logger
from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS
from flash.envs.registry import FREESOLO_WORKER_SPEC
from flash.opd_retry_contract import OPD_RESUME_REVISION_ENV
from flash.providers._hf_retry import hf_call, hf_status_code
from flash.providers.base import get_gpu_info
from flash.spec import (
    CONTROL_PANEL_URL_ENV,
    MANAGED_TEACHER_CREDENTIAL_ENV_KEYS,
    RESERVED_WORKER_ENV_KEYS,
    TEACHER_CAPABILITY_ENV,
    JobSpec,
    require_matching_seed,
    validate_worker_env_reserved,
)

# pinned literal (not __name__): keeps the logger stream named "flash.providers.runpod.train"
# after this module moved out of flash/providers/runpod/train.py, so operator log filters stay stable.
logger = get_logger("flash.providers.runpod.train")


# vllm 0.19.1: first vllm compatible with transformers 5.x; vllm>=0.20 pins torch 2.11
# (CUDA-13 wheels) which reports no GPU on 12.8/12.9 drivers common on 4090/5090 hosts.
WORKER_DEPS = [
    "torch==2.10.0",
    "transformers>=5.6,<5.13",
    "tokenizers>=0.22",
    "tiktoken>=0.12",
    "peft>=0.19",
    "vllm==0.19.1",
    # FlashInfer: vLLM's Blackwell-native attention backend. vllm 0.19.1 pins flashinfer-python==0.6.6
    # but treats it as an OPTIONAL extra (the plain `vllm` install does not pull it), so a consumer-
    # Blackwell (sm120) / B200 rollout would silently fall back to a PTX-fragile default attention
    # without it. Pin the matching 0.6.6 so the worker image carries the FLASHINFER attention backend
    # (resolve_blackwell_attention_backends picks FLASHINFER when it imports). No-op elsewhere.
    "flashinfer-python==0.6.6",
    "bitsandbytes>=0.49",
    # resolve_verl_python shells out to `uv` to provision the isolated verl interpreter whenever
    # FLASH_VERL_PYTHON is unset. sft and opd are verl-only, so on the no-image path that call is
    # unconditional and a missing `uv` fails the run with FileNotFoundError before training.
    "uv>=0.5",
    "datasets>=4.7,<6",
    # >=0.2.54: includes robust JSONL loading and corrected package metadata.
    FREESOLO_WORKER_SPEC,
    # >=1.2.0: built-in RateLimit-header-aware 429 retry for HF downloads (base-model pulls) and
    # paginated Hub API calls. Must match Dockerfile.worker's floor (the baked image is the default
    # run path; this list only installs on the no-image/live-function path) -- see
    # tests/test_kernel_fingerprint.py::test_huggingface_hub_floor_is_in_lockstep.
    "huggingface_hub>=1.2.0",
    "accelerate>=1.4",
    # HF `kernels` Hub NOT pinned: torch2.10-compatible versions crash `import transformers` (LayerRepository API mismatch).
    "wandb>=0.17",
    # fla from git: PyPI wheel is a broken stub missing fla.modules. SHA-pinned for reproducibility;
    # keep in lockstep with Dockerfile.worker. fla kept on ALL arches — worker ensures tilelang
    # backend on sm90 before model import (fla #640: chunk_bwd miscompute with Triton>=3.4 on Hopper)
    # and OPTS OUT of tilelang on sm100 (B200) where tilelang's chunk_bwd_dqkwg miscomputes grads
    # (worker _force_fla_triton_gdn_on_sm100; upstream default-gates tilelang to Hopper since fla #975).
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@f0e213dbd8b5fb90c3c7eca869ac1706d5377139",
    # tilelang version-pinned with the worker image and flash/engine/worker/perf/__init__.py runtime reinstall.
    "tilelang==0.1.11",
    "apache-tvm-ffi==0.1.11",  # pin: 0.1.12 double-registers TVM-FFI -> `import tilelang` aborts
    # causal_conv1d NOT pip-listed: CUDA extension compiled in Dockerfile.worker with TORCH_CUDA_ARCH_LIST.
]
WORKER_SYSTEM_DEPS = ["build-essential"]  # Triton/Inductor need a C compiler

WORKER_IMAGE = "ghcr.io/freesolo-co/flash-worker:cu128"

# MUST mirror the bake matrix in .github/workflows/bake-kernel-cache.yml. Unlisted arches fall
# back to WORKER_IMAGE (no -smXX tag built) rather than failing at docker pull.
BAKED_PER_SM_ARCHES = frozenset({"sm80", "sm86", "sm89", "sm90", "sm120", "sm100"})


@dataclass(frozen=True)
class WorkerImageOverride:
    """A worker-image override and the registry credential needed to pull it.

    The CUDA and disk floors this used to carry were removed with their env vars in #906: the GPU
    class floor already encodes the driver requirement, and container disk belongs in the run's own
    ``[gpu] disk_gb``. Only the auth id is left, because it cannot be derived from the image ref.
    """

    image: str
    registry_auth_id: str = ""


def worker_image_override() -> WorkerImageOverride | None:
    """Parse the FLASH_WORKER_IMAGE override and its registry credential.

    FLASH_WORKER_IMAGE holds the image ref; FLASH_WORKER_IMAGE_REGISTRY_AUTH holds the
    provider-side registry-credential id a private image needs to pull. The credential cannot be
    derived from the image ref, so it stays configurable.

    The image's CUDA and disk floors are NOT configurable: the GPU class floor (min_cuda_modern)
    already encodes the real constraint, and a run that needs more container disk sets
    ``[gpu] disk_gb`` in its own spec, where the requirement is recorded with the run.
    """
    image = os.environ.get("FLASH_WORKER_IMAGE", "").strip()
    if not image:
        return None
    return WorkerImageOverride(
        image=image,
        registry_auth_id=os.environ.get("FLASH_WORKER_IMAGE_REGISTRY_AUTH", "").strip(),
    )


def worker_image_for_gpu(friendly_gpu: str | None, *, allow_default: bool = True) -> str | None:
    """Return the worker Docker image for a GPU class (per-SM kernel-cache tag or base), respecting the FLASH_WORKER_IMAGE override."""
    override = worker_image_override()
    if override:
        return override.image
    if friendly_gpu and allow_default:
        info = get_gpu_info(friendly_gpu)
        # Per-SM baked kernel-cache image is always used for baked arches (skips ~10-15 min
        # cold-start JIT). Unbaked arches fall through to the base image to avoid a 404 docker pull.
        if info.sm in BAKED_PER_SM_ARCHES:
            return f"{WORKER_IMAGE}-{info.sm}"
    return WORKER_IMAGE if allow_default else None


def resolve_worker_deps() -> list[str]:
    """Return the pinned worker dependency list."""
    return list(WORKER_DEPS)


DEFAULT_EXECUTION_TIMEOUT_MS = 6 * 3600 * 1000  # 6h cap


# Optimization toggles dropped in PR #175 (deterministic behavior). Filtered from [worker_env]
# to prevent recipes re-injecting e.g. expandable_segments (crashes GRPO vLLM sleep mode).
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
    shared multi-tenant mount; reward/env HF downloads stay in the ephemeral per-worker cache.
    JIT caches are never redirected — sharing compiled artifacts across tenants is unsafe.
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
    """Per-run env passed to the worker (platform creds + recipe overrides)."""
    canonical_seed = require_matching_seed(spec, seed)
    validate_worker_env_reserved(spec.worker_env)
    declared_managed_credentials = sorted(
        set(spec.environment.secrets) & MANAGED_TEACHER_CREDENTIAL_ENV_KEYS
    )
    if declared_managed_credentials:
        raise ValueError(
            "environment secrets must not include managed teacher credential names: "
            + ", ".join(declared_managed_credentials)
        )
    # GRPO and OPD run a verl vLLM rollout (`actor_rollout_ref.rollout.name=vllm`), and verl leaves
    # rollout.enable_sleep_mode defaulted True, so the engine always builds a CuMemAllocator -- which
    # asserts outright on "expandable_segments:True" (vllm/device_allocator/cumem.py:132,
    # pytorch#147851). Both therefore take the non-expandable conf.
    #
    # SFT does NOT qualify: verl.trainer.sft_trainer is a pure FSDP trainer with no rollout, no vLLM
    # engine, and no CuMemAllocator, so it keeps the expandable allocator its large generate/logit
    # tensors need against fragmentation. The predicate is "generates through verl", not "runs on
    # verl" (codex[bot]).
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
    for k, v in (getattr(spec, "worker_env", None) or {}).items():
        ku = str(k).upper()
        if ku in _REMOVED_OPTIMIZATION_ENV:
            logger.warning(
                "ignoring removed optimization toggle %s in [worker_env] (flash is fully "
                "managed; behavior is deterministic)",
                k,
            )
            continue
        env[str(k)] = str(v)
    # runtime secrets and declared env secrets may never clobber a control-plane-owned key: the
    # canonical seed, run id, hf repo, and arm are set above, and a runtime seed override would break
    # the authoritative-seed invariant regardless of how environment.secrets was populated.
    # removed keys are filtered here too. the two sets are disjoint and answer different questions --
    # RESERVED_WORKER_ENV_KEYS is ownership (a caller may not override SEED), _REMOVED_OPTIMIZATION_ENV
    # is deadness (the key configures nothing now) -- so checking only ownership left
    # [environment].secrets as a second door that delivered a dead key with none of the warning that
    # the [worker_env] path emits.
    allowed_runtime_secrets = {
        k
        for k in (set(DEFAULT_RUNTIME_SECRET_KEYS) | set(spec.environment.secrets))
        if k.upper() not in RESERVED_WORKER_ENV_KEYS and k.upper() not in _REMOVED_OPTIMIZATION_ENV
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
    env.pop(CONTROL_PANEL_URL_ENV, None)
    env.pop(TEACHER_CAPABILITY_ENV, None)
    if str(getattr(spec, "algorithm", "")).lower() == "opd":
        control_panel_url = str((runtime_secrets or {}).get(CONTROL_PANEL_URL_ENV) or "").strip()
        capability = str((runtime_secrets or {}).get(TEACHER_CAPABILITY_ENV) or "").strip()
        if not control_panel_url or not capability:
            raise RuntimeError("managed opd control-panel teacher transport is missing")
        env[CONTROL_PANEL_URL_ENV] = control_panel_url
        env[TEACHER_CAPABILITY_ENV] = capability
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
