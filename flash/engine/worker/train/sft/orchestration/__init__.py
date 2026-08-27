"""Shared SFT orchestration values with no entrypoint or runner dependency."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from functools import partial

import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.model.adapter as _worker_adapter
import flash.engine.worker.perf as _worker_perf
import flash.engine.worker.runtime.state as _worker_state
from flash._internal.diagnostics import SECRET_ENV_KEYS_ENV
from flash.adapters.lora_rank import alpha_from_adapter_config, rank_from_adapter_config
from flash.engine.plan.recipe import RECIPE as RECIPE
from flash.engine.plan.steps import validate_save_steps as validate_save_steps
from flash.engine.profiling.sft_workload import prepare_sft_workload as prepare_sft_workload
from flash.engine.profiling.sft_workload import sft_tokens_for_updates as sft_tokens_for_updates
from flash.engine.worker.entry.sft import _model_arch_dims as _model_arch_dims
from flash.engine.worker.entry.sft import sft_under_ran as sft_under_ran
from flash.engine.worker.io.heartbeat import liveness_heartbeat as liveness_heartbeat
from flash.engine.worker.runtime.kernel_warmup import KERNEL_CACHE_ENV_SUBDIRS
from flash.engine.worker.runtime.rng import seed_training_rngs as seed_training_rngs
from flash.engine.worker.train.sft.setup.checkpoints import (
    _VerlCheckpointWatcher as _VerlCheckpointWatcher,
)
from flash.engine.worker.train.sft.setup.config import (
    _LORAPLUS_READY_MARKER as _LORAPLUS_READY_MARKER,
)
from flash.engine.worker.train.sft.setup.config import (
    _VERL_OPTIMIZER_IMPL as _VERL_OPTIMIZER_IMPL,
)
from flash.engine.worker.train.sft.setup.config import (
    _VERL_OPTIMIZER_NAME as _VERL_OPTIMIZER_NAME,
)
from flash.engine.worker.train.sft.setup.config import (
    _render_sft_dataset_module as _render_sft_dataset_module,
)
from flash.engine.worker.train.sft.setup.config import _write_sft_parquet as _write_sft_parquet
from flash.engine.worker.train.sft.setup.config import build_sft_overrides as build_sft_overrides
from flash.engine.worker.verl.capabilities import fused_ce_backend as fused_ce_backend
from flash.engine.worker.verl.capabilities import resolve_verl_loggers as resolve_verl_loggers
from flash.engine.worker.verl.checkpoints import latest_global_step_dir as latest_global_step_dir
from flash.engine.worker.verl.checkpoints import restore_verl_resume
from flash.engine.worker.verl.child_io import (
    SHIM_FRAGMENT_FAILED_EXIT_CODE as SHIM_FRAGMENT_FAILED_EXIT_CODE,
)
from flash.engine.worker.verl.child_io import parse_wandb_link as parse_wandb_link
from flash.engine.worker.verl.child_io import (
    render_sitecustomize_bootstrap as render_sitecustomize_bootstrap,
)
from flash.engine.worker.verl.child_io import shim_marker_file as shim_marker_file
from flash.engine.worker.verl.child_io import (
    verify_applied_shim_markers as verify_applied_shim_markers,
)
from flash.engine.worker.verl.child_io import verl_step_number as verl_step_number
from flash.engine.worker.verl.process import run_verl_training as run_verl_training

_SFT_LORAPLUS_RATIO = 16.0
_MAX_ZERO_GRAD_STEPS = 2


@dataclass(frozen=True)
class _SftPaths:
    workdir: str
    data_dir: str
    image_dir: str
    local_dir: str
    export_root: str


@dataclass(frozen=True)
class _SftOptions:
    spec: object
    env: object
    started_at: float
    gpu_probe: dict
    model_id: str
    model_revision: str
    epochs: int
    learning_rate: float
    effective_batch: int
    max_steps: int
    save_at_steps: tuple[int, ...]
    save_every: int
    gpu_count: int
    paths: _SftPaths


@dataclass(frozen=True)
class _SftData:
    rows: list[dict]
    multimodal: bool
    processor: object | None
    profile: object
    max_length: int
    realized_max_length: int
    train_file: str


@dataclass(frozen=True)
class _SftModelSetup:
    download_seconds: float
    setup_seconds: float
    lora_rank: int
    lora_alpha: int
    target_modules: object
    exclude_modules: str | None
    warmstart_adapter: str | None
    fused_ce: bool
    train_batch_size: int
    micro_batch: int
    update_horizon: int
    loop_epochs: int
    save_freq: int
    gradient_checkpointing: bool
    reentrant_gradient_checkpointing: bool


@dataclass(frozen=True)
class _SftCapabilities:
    python_bin: str
    caps: dict
    gdn_hybrid: bool
    gdn_module: str


@dataclass(frozen=True)
class _SftChild:
    python_bin: str
    loggers: list[str]
    project_name: str
    experiment_name: str
    gdn_reset_arch: str | None
    gdn_hybrid: bool
    resume_step: int
    watcher: object
    child_env: dict[str, str]
    command: list[str]
    world_size: int
    micro_batch: int
    shim_markers: str
    expected_shims: tuple[str, ...]


@dataclass
class _SftProgress:
    values: dict[str, float | int | None]
    zero_grad_steps: list[int]
    observed_grad_norms: list[float]
    loss_curve: list[float]
    train_tokens: int
    loraplus_applied: bool
    wandb_link: dict[str, str | None]
    shim_markers: str = ""
    expected_shims: tuple[str, ...] = ()
    shims_verified: bool = False


@dataclass(frozen=True)
class _SftVerified:
    actor_dir: str
    final_step: int
    train_tokens: int


@dataclass(frozen=True)
class _SftOutputs:
    adapter_dir: str
    train_wall: float
    device_peak_gpu_gb: float


def _cached_model_path(model_id: str, model_revision: str) -> str:
    from huggingface_hub import snapshot_download

    from flash.engine.worker.io.hf import _shared_weight_cache_dir
    from flash.engine.worker.perf import RetriableInfraError

    for cache_dir in (_shared_weight_cache_dir(), None):
        try:
            return snapshot_download(
                repo_id=model_id,
                revision=model_revision or None,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        except Exception:
            continue
    raise RetriableInfraError(
        f"model {model_id} not resolvable from the local HF cache after prefetch"
    )


def _warmstart_adapter_path(
    model_id: str,
    model_revision: str,
    expected_rank: int,
    expected_alpha: int,
    targeting=None,
) -> str | None:
    """Stage and verify this run's warm-start source adapter."""
    spec = _worker_state.JOB_SPEC
    source = spec.train.init_from_adapter if spec else ""
    if not source:
        return None
    adapter_dir = _worker_adapter._download_adapter(source)
    if not adapter_dir:
        raise RuntimeError("the prepared warm-start adapter could not be downloaded")
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    rank = rank_from_adapter_config(config, source=config_path)
    if rank != expected_rank:
        raise ValueError(
            f"warm-start adapter rank {rank} does not match the prepared train.lora_rank "
            f"{expected_rank}; rank changes are not supported"
        )
    alpha = alpha_from_adapter_config(config, source=config_path)
    if alpha != expected_alpha:
        raise ValueError(
            f"warm-start adapter alpha {alpha} does not match the prepared train.lora_alpha "
            f"{expected_alpha}; alpha changes are not supported"
        )
    if targeting is None:
        _worker_adapter.validate_warmstart_adapter(config, model_id, adapter_dir)
    else:
        _worker_adapter.validate_warmstart_adapter(config, model_id, adapter_dir, targeting)
    base = str(config.get("base_model_name_or_path") or "").strip()
    if base and base != model_id:
        raise ValueError("warm-start adapter base model does not match the target model")
    revision = str(config.get("revision") or "").strip()
    if revision and model_revision and revision != model_revision:
        raise ValueError("warm-start adapter revision does not match the target model revision")
    return adapter_dir


def _verl_image_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        raise ValueError("multimodal message content must be text or content blocks")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("multimodal message content blocks must be objects")
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif block.get("type") == "image":
            parts.append("<image>")
        else:
            raise ValueError(f"unsupported multimodal SFT content block {block.get('type')!r}")
    return "".join(parts)


_restore_verl_resume = partial(restore_verl_resume, job_label="SFT")


def _durable_required_save_steps(required_steps: tuple[int, ...], resume_step: int) -> set[int]:
    candidates = [step for step in required_steps if step <= resume_step]
    if not candidates:
        return set()
    if not _worker_state.HF_REPO:
        raise RuntimeError("required SFT saves have no artifact repository")
    durable: set[int] = set()
    for step in candidates:
        marker = f"{_worker_hf.hf_prefix()}/checkpoints/step-{step}/adapter/adapter_config.json"
        try:
            exists = _worker_hf.hf_api().file_exists(
                repo_id=_worker_state.HF_REPO,
                filename=marker,
                repo_type="dataset",
            )
        except Exception as error:
            raise _worker_perf.RetriableInfraError(
                f"could not verify required SFT save step {step} on hf"
            ) from error
        if exists:
            durable.add(step)
    return durable


def _seed_resume_lifecycle(watcher, required_steps: tuple[int, ...], resume_step: int) -> None:
    watcher.lifecycle.seed_resumed_step(resume_step, frozenset(required_steps))
    for step in _durable_required_save_steps(required_steps, resume_step):
        watcher.lifecycle.mark_deployable_published(step)
        watcher.lifecycle.mark_discovered(step)


_CHILD_ENV_EXACT = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_HUB_DISABLE_XET",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "FLASH_VERL_PYTHON",
    }
    | KERNEL_CACHE_ENV_SUBDIRS.keys()
)
_CHILD_ENV_PREFIXES = (
    "CUDA_",
    "NCCL_",
    "TORCH_",
    "PYTORCH_",
    "VERL_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
    "LC_",
    "FLA_",
)


def _build_verl_child_env(*, shim_dir: str, wandb_enabled: bool) -> dict[str, str]:
    applied_secret_names = frozenset(
        name.strip() for name in os.environ.get(SECRET_ENV_KEYS_ENV, "").split(",") if name.strip()
    )
    child = {
        key: value
        for key, value in os.environ.items()
        if key in _CHILD_ENV_EXACT or key.startswith(_CHILD_ENV_PREFIXES)
    }
    if wandb_enabled:
        child.update({key: value for key, value in os.environ.items() if key.startswith("WANDB_")})
    for name in applied_secret_names:
        child.pop(name, None)
    if wandb_enabled and "WANDB_API_KEY" in os.environ:
        child["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    parent_pythonpath = (
        os.environ.get("PYTHONPATH", "") if "PYTHONPATH" not in applied_secret_names else ""
    )
    child["PYTHONPATH"] = os.pathsep.join(item for item in (shim_dir, parent_pythonpath) if item)
    child["PYTHONUNBUFFERED"] = "1"
    child["HYDRA_FULL_ERROR"] = "1"
    child["HF_HUB_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    return child


def _probe_gpu_in_subprocess(requested_gpu: str | None, exact_type: str = "") -> dict:
    script = r"""
import json
import sys

from flash.engine.worker.perf.lifecycle import wait_for_gpu

requested, exact = json.loads(sys.argv[1])
wait_for_gpu(requested, gpu_type=exact)
import torch
print("FLASH_GPU_PROBE=" + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    "capability": list(torch.cuda.get_device_capability(0)),
}), flush=True)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, json.dumps([requested_gpu, exact_type])],
            capture_output=True,
            text=True,
            timeout=150,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise _worker_perf.RetriableInfraError(
            "gpu readiness probe failed in its subprocess"
        ) from error
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.returncode != 0:
        raise _worker_perf.RetriableInfraError(
            f"gpu readiness probe exited with status {result.returncode}"
        )
    for line in result.stdout.splitlines():
        if line.startswith("FLASH_GPU_PROBE="):
            return json.loads(line.split("=", 1)[1])
    raise _worker_perf.RetriableInfraError("gpu readiness probe returned no device metadata")


class _NvidiaSmiPeakSampler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                if result.returncode == 0:
                    values = [
                        float(value.strip())
                        for value in result.stdout.splitlines()
                        if value.strip()
                    ]
                    if values:
                        self.peak_mib = max(self.peak_mib, max(values))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass

    def start(self):
        self._thread.start()
        return self

    def stop_gb(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2)
        return round(self.peak_mib / 1024, 3)


def _resolve_sft_vocab_size(model_id: str, model_revision: str) -> int:
    from flash.core.catalog import resolve_vocab_size

    return resolve_vocab_size(model_id, model_revision)


def _resolve_sft_grad_accum(effective_batch: int, **kwargs):
    from flash.engine.plan.vram import sft_grad_accum

    return sft_grad_accum(effective_batch, **kwargs)


def _resolve_sft_gradient_checkpointing(model_id: str, max_length: int, **kwargs) -> bool:
    return _worker_perf.grad_checkpointing_on(model_id, max_length, **kwargs)


def _resolve_sft_reentrant_gradient_checkpointing(model_id: str) -> bool:
    return _worker_perf.grpo_use_reentrant(model_id)


def _resolve_sft_fused_ce_backend(caps):
    return fused_ce_backend(caps)


def _sft_profile_max_length(profile) -> int:
    return profile.max_length


def _sft_liger_config() -> dict[str, bool]:
    return {"use_liger": False}
