"""Provider-neutral worker packaging: the deps/image/env every rent-a-box or serverless worker ships, plus upload_code for the HF code snapshot. Shared kernel — no provider package imports another for this."""

from __future__ import annotations

import os
import time
from io import BytesIO

from flash._logging import get_logger
from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS
from flash.envs.registry import FREESOLO_WORKER_SPEC
from flash.providers._hf_retry import hf_call, hf_status_code
from flash.providers.base import get_gpu_info
from flash.spec import JobSpec

# Literal name so the logger stays "flash.providers.runpod.train" after the package split — tests assert this.
logger = get_logger("flash.providers.runpod.train")


# vllm 0.19.1: first vllm compatible with transformers 5.x; vllm>=0.20 pins torch 2.11
# (CUDA-13 wheels) which reports no GPU on 12.8/12.9 drivers common on 4090/5090 hosts.
WORKER_DEPS = [
    "torch==2.10.0",
    "transformers>=5.6,<5.13",
    "trl>=1.6,<1.7",
    "peft>=0.19",
    "vllm==0.19.1",
    # FlashInfer: vLLM's Blackwell-native attention backend. vllm 0.19.1 pins flashinfer-python==0.6.6
    # but treats it as an OPTIONAL extra (the plain `vllm` install does not pull it), so a consumer-
    # Blackwell (sm120) / B200 rollout would silently fall back to a PTX-fragile default attention
    # without it. Pin the matching 0.6.6 so the worker image carries the FLASHINFER attention backend
    # (force_vllm_backend_for_sm120). No-op on non-Blackwell archs.
    "flashinfer-python==0.6.6",
    "bitsandbytes>=0.49",
    "datasets>=4.7,<6",
    # >=0.2.54: includes robust JSONL loading and corrected package metadata.
    FREESOLO_WORKER_SPEC,
    "huggingface_hub>=0.25",
    "accelerate>=1.4",
    # HF `kernels` Hub NOT pinned: torch2.10-compatible versions crash `import transformers` (LayerRepository API mismatch).
    "wandb>=0.17",
    # fla from git: PyPI wheel is a broken stub missing fla.modules. SHA-pinned for reproducibility;
    # keep in lockstep with Dockerfile.worker. fla kept on ALL arches — worker ensures tilelang
    # backend on sm90 before model import (fla #640: chunk_bwd miscompute with Triton>=3.4 on Hopper).
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@f0e213dbd8b5fb90c3c7eca869ac1706d5377139",
    # tilelang version-pinned in lockstep with Dockerfile.worker + perf.py runtime reinstall.
    "tilelang==0.1.11",
    "apache-tvm-ffi==0.1.11",  # pin: 0.1.12 double-registers TVM-FFI -> `import tilelang` aborts
    # causal_conv1d NOT pip-listed: CUDA extension compiled in Dockerfile.worker with TORCH_CUDA_ARCH_LIST.
    # freesolo-chalk NOT in base deps: chalk_extra_pip() appends DEFAULT_CHALK_SPEC to extra_pip per job.
]
WORKER_SYSTEM_DEPS = ["build-essential"]  # Triton/Inductor need a C compiler

WORKER_IMAGE = "ghcr.io/freesolo-co/flash-worker:cu128"
WORKER_IMAGE_TEMPLATE_ENV = "FLASH_WORKER_IMAGE_TEMPLATE"
WORKER_IMAGE_PER_SM_ENV = "FLASH_WORKER_IMAGE_PER_SM"

# MUST mirror the bake matrix in .github/workflows/bake-kernel-cache.yml. Unlisted arches fall
# back to WORKER_IMAGE (no -smXX tag built) rather than failing at docker pull.
BAKED_PER_SM_ARCHES = frozenset({"sm80", "sm86", "sm89", "sm90", "sm120"})


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _append_tag_suffix(image: str, suffix: str) -> str:
    slash = image.rfind("/")
    colon = image.rfind(":")
    if colon > slash:
        return f"{image[:colon]}:{image[colon + 1 :]}-{suffix}"
    return f"{image}-{suffix}"


def worker_image_for_gpu(friendly_gpu: str | None, *, allow_default: bool = True) -> str | None:
    """Return the worker Docker image for a GPU class (per-SM kernel-cache tag or base), respecting the FLASH_WORKER_IMAGE override."""
    override = os.environ.get("FLASH_WORKER_IMAGE", "").strip()
    if override:
        return override
    if friendly_gpu and allow_default:
        info = get_gpu_info(friendly_gpu)
        template = os.environ.get(WORKER_IMAGE_TEMPLATE_ENV, "").strip()
        if template:
            return template.format(
                base_image=WORKER_IMAGE,
                gpu=info.name,
                gpu_short=info.short,
                sm=info.sm,
                sm_num=info.sm.removeprefix("sm"),
            )
        if _truthy(os.environ.get(WORKER_IMAGE_PER_SM_ENV)) and info.sm in BAKED_PER_SM_ARCHES:
            return _append_tag_suffix(WORKER_IMAGE, info.sm)
        # arch not in BAKED_PER_SM_ARCHES: fall through to base image to avoid a 404 docker pull
    return WORKER_IMAGE if allow_default else None


def resolve_worker_deps() -> list[str]:
    """Return the pinned worker dependency list."""
    return list(WORKER_DEPS)


def _effective_worker_env(spec=None) -> dict[str, str]:
    """os.environ overlaid with spec.worker_env — mirrors what build_worker_env sends the worker."""
    eff: dict[str, str] = dict(os.environ)
    for k, v in (getattr(spec, "worker_env", None) or {}).items():
        eff[str(k)] = str(v)
    return eff


# Chalk is published to PUBLIC PyPI on every version bump (chalk .github/workflows/publish.yml).
# Pin the exact PUBLIC version so worker + kernel-cache-bake pods install with NO GitHub auth — the
# chalk repo is INTERNAL, so a git+https default made installs fail wherever GITHUB_TOKEN is absent
# (bake pod, tokenless workers). freesolo-chalk 0.5.0 = chalk PR #21 fafe167 (standalone, Liger fully
# replaced, + GRPO op + LoRA grad-gate). Bump DEFAULT_CHALK_VERSION when the merged kernel surface moves.
DEFAULT_CHALK_VERSION = "0.5.0"
DEFAULT_CHALK_SPEC = f"freesolo-chalk=={DEFAULT_CHALK_VERSION}"
# Provenance only (kernel-cache fingerprint / traceability): the chalk commit this version was cut from.
LATEST_CHALK_MAIN_SHA = "fafe167158eb3b7c947e7182ea4eddc351e4ca7d"


def chalk_extra_pip(spec=None) -> list[str]:
    """Return chalk pip spec(s) for the worker's extra_pip; resolved against the effective worker env."""
    spec_str = _effective_worker_env(spec).get("FLASH_CHALK_SPEC", "").strip() or DEFAULT_CHALK_SPEC
    import shlex

    return [d for d in shlex.split(spec_str) if d.strip()]


DEFAULT_EXECUTION_TIMEOUT_MS = 6 * 3600 * 1000  # 6h cap


_RUNTIME_SECRET_KEYS = DEFAULT_RUNTIME_SECRET_KEYS


# Optimization toggles dropped in PR #175 (deterministic behavior). Filtered from [worker_env]
# to prevent recipes re-injecting e.g. expandable_segments (crashes GRPO vLLM sleep mode).
# FLASH_CHALK_SPEC is deliberately absent — it's still a supported install-source override.
_REMOVED_OPTIMIZATION_ENV = frozenset(
    {
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "RL_VLLM_SLEEP",
        "FLASH_ALLOC_AUTO",
        "TORCHDYNAMO_DISABLE",
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


def drop_unmounted_cache_env(env: dict, mount: str = _WEIGHT_CACHE_MOUNT) -> dict:
    """Strip mount-rooted cache vars if the volume isn't actually mounted (mutates+returns)."""
    if os.path.isdir(mount):
        return env
    for k in [k for k, v in env.items() if str(v).startswith(mount)]:
        env.pop(k, None)
    return env


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
    # GRPO uses a non-expandable alloc conf: expandable_segments crashes the vLLM sleep-mode engine.
    # The worker upgrades RL to expandable when it resolves sleep=OFF (finalize_alloc_conf_for_sleep).
    # SFT and OPD have NO vLLM sleep engine and allocate large HF generate/logit tensors, so both take
    # the anti-fragmentation expandable allocator up front: OPD was previously lumped in with RL here
    # (`not in ("sft",)`) and ran non-expandable despite having no sleep engine, so a long-completion /
    # large-vocab OPD job fragmented and OOM'd on a GPU the VRAM estimate said would fit (codex[bot]).
    _needs_nonexpandable = str(getattr(spec, "algorithm", "")).lower() not in ("sft", "opd")
    _alloc_conf = (
        "garbage_collection_threshold:0.8,max_split_size_mb:256"
        if _needs_nonexpandable
        else "expandable_segments:True"
    )
    env: dict[str, str] = {
        "RUN_ID": spec.run_id,
        "FLASH_ARM": "runpod",
        "BENCH_HF_MODEL": spec.model,
        "PYTORCH_CUDA_ALLOC_CONF": _alloc_conf,
        "PYTORCH_ALLOC_CONF": _alloc_conf,
    }
    for key in (
        "HF_TOKEN",
        "GITHUB_TOKEN",
    ):
        if os.environ.get(key):
            env[key] = os.environ[key]
    env["HF_REPO"] = spec.train.hf_repo
    if getattr(spec.gpu, "network_volume", None):
        env.update(weight_cache_env())
    if spec.train.steps is not None:
        env["RL_STEPS"] = str(spec.train.steps)
    if spec.train.epochs is not None:
        env["SFT_EPOCHS"] = str(spec.train.epochs)
    for k in (
        "SFT_PER_DEVICE_BS",
        "VLLM_USE_V1",
        "FLASH_CHALK_SPEC",  # install-source override; kernel selection is fixed in chalk_kernels
    ):
        # Forward when SET, even if empty: an explicit "" is a meaningful override.
        if os.environ.get(k) is not None:
            env[k] = os.environ[k]
    # RUN_ID/HF_REPO/FLASH_ARM are control-plane-owned: overriding them would orphan artifacts.
    _RESERVED_WORKER_ENV = {"RUN_ID", "HF_REPO", "FLASH_ARM"}
    for k, v in (getattr(spec, "worker_env", None) or {}).items():
        ku = str(k).upper()
        if ku in _RESERVED_WORKER_ENV:
            continue
        if ku in _REMOVED_OPTIMIZATION_ENV:
            logger.warning(
                "ignoring removed optimization toggle %s in [worker_env] (flash is fully "
                "managed; behavior is deterministic)",
                k,
            )
            continue
        env[str(k)] = str(v)
    allowed_runtime_secrets = set(_RUNTIME_SECRET_KEYS) | set(spec.environment.secrets)
    for k, v in (runtime_secrets or {}).items():
        if k in allowed_runtime_secrets and v:
            env[k] = str(v)
    return env


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
