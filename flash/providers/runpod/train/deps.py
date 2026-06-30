"""Worker dependency stack + per-run env / chalk-kernel selection (leaf module)."""

from __future__ import annotations

import os

from flash._logging import get_logger
from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS
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
    # >=0.2.51: includes robust JSONL loading and restored full package metadata.
    "freesolo>=0.2.51",
    "huggingface_hub>=0.25",
    "accelerate>=1.4",
    # HF `kernels` Hub NOT pinned: torch2.10-compatible versions crash `import transformers` (LayerRepository API mismatch).
    "wandb>=0.17",
    "liger-kernel>=0.5",
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
        return f"{image[:colon]}:{image[colon + 1:]}-{suffix}"
    return f"{image}-{suffix}"


def worker_image_for_gpu(friendly_gpu: str | None, *, allow_default: bool = True) -> str | None:
    """Return the RunPod worker image for a GPU class, respecting FLASH_WORKER_IMAGE override."""
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


DEFAULT_CHALK_SPEC = "freesolo-chalk>=0.1.0,<0.2.0"


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
    # RL uses a non-expandable alloc conf: expandable_segments crashes GRPO vLLM sleep mode.
    # Worker upgrades to expandable when it resolves sleep=OFF (finalize_alloc_conf_for_sleep).
    _is_rl = str(getattr(spec, "algorithm", "")).lower() not in ("sft",)
    _alloc_conf = (
        "garbage_collection_threshold:0.8,max_split_size_mb:256"
        if _is_rl
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
