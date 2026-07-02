"""Structured job specification shared by CLI/API/runner and GPU workers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .catalog import DEFAULT_GPU, DEFAULT_MODEL, normalize_algorithm

_FALSE_STRINGS = {"", "0", "false", "no", "off", "none"}


def _str_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a string-or-list knob to a tuple of strings; a bare string is one element, not iterated."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(s for s in (str(x) for x in value) if s)


def coerce_bool(value: Any) -> bool:
    """Parse a bool from loosely-typed sources; treats "false"/"0"/"no"/"off" as False."""
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    return bool(value)


def _coerce_str_map(value: Any) -> dict[str, str]:
    """Coerce to dict[str, str]; non-dict input returns empty dict."""
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


def _coerce_wandb(value: Any) -> WandbSpec:
    """Coerce to WandbSpec; non-dict input returns default."""
    if not isinstance(value, dict):
        return WandbSpec()

    def _label(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    return WandbSpec(project=_label(value.get("project")), run_name=_label(value.get("run_name")))


def _volume_gb(value: Any, default: int = 100) -> int:
    """Parse volume size in GB; non-positive / non-numeric / missing values return default."""
    if isinstance(value, bool):
        # bool is an int subclass; reject to avoid int(True)==1 becoming a 1 GB volume.
        return default
    try:
        gb = int(value)
    except (TypeError, ValueError):
        return default
    return gb if gb > 0 else default


def _opt_int(value: Any) -> int | None:
    """Parse optional int; rejects bools (bool is int subclass — int(True)==1 is a footgun)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return int(value)


def _opt_float(value: Any) -> float | None:
    """Parse optional float; rejects bools (mirrors _opt_int)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return float(value)


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    # Pip requirements the GPU worker needs for this environment; empty means "use defaults"
    # (resolved via worker_pip_for_env in spec_payload / provider submit). An explicit
    # [environment] pip is the escape hatch.
    pip: tuple[str, ...] = ()
    # Names only — values sent out-of-band via runtime_secrets, never stored in spec.
    secrets: tuple[str, ...] = ()
    # Resolved once in control plane to avoid GitHub rate-limits on cold spawn waves.
    resolved_sha: str = ""


# The single RNG seed every training job uses. Flash trains exactly one adapter per run (no
# multi-seed); the value is fixed so runs are reproducible and the artifact path carries no seed
# segment. It still reaches the worker via the ``SEED`` env (data shuffling / init in sft.py/rl.py).
FIXED_SEED = 42


@dataclass(frozen=True)
class TrainSpec:
    steps: int | None = None
    epochs: int | None = None
    lora_rank: int = 32
    lora_alpha: int = 64
    # Artifact-store adapter ref output by `flash status`:
    # ``<hf_repo>:<phase>/<run_id>``.
    init_from_adapter: str = ""
    # PLATFORM-MANAGED: control-plane-assigned HF artifact repo; user-supplied values are ignored.
    hf_repo: str = ""
    # None -> worker's tuned recipe default.
    learning_rate: float | None = None
    batch_size: int | None = None
    max_length: int | None = None
    save_every: int | None = None
    max_steps: int | None = None
    max_examples: int | None = None
    group_size: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    kl_penalty_coef: float | None = None
    advantage_clip: float | None = None
    thinking_length_penalty_coef: float | None = None
    stop_sequences: tuple[str, ...] = ()
    # On-policy distillation (algorithm="opd"). None/"" => worker resolves the recipe default.
    teacher_model: str = ""
    # align | seqkd | uld — cross-tokenizer strategy (see docs/on-policy-distillation.md).
    tokenizer_alignment: str = ""
    teacher_top_logprobs: int | None = None


@dataclass(frozen=True)
class GpuSpec:
    # gpu.type does NOT pin — allocator re-picks cheapest fitting validated class at submit time.
    type: str = DEFAULT_GPU
    disk_gb: int = 60
    max_wall_seconds: int = 24 * 3600
    max_retries: int = 5
    # PLATFORM-MANAGED: runner assigns weight-cache volume; None = cold download.
    network_volume: str | None = None
    network_volume_gb: int = 100


@dataclass(frozen=True)
class WandbSpec:
    project: str | None = None
    run_name: str | None = None


@dataclass(frozen=True)
class JobSpec:
    model: str = DEFAULT_MODEL
    algorithm: str = "grpo"
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    gpu: GpuSpec = field(default_factory=GpuSpec)
    run_id: str = "local"
    # Per-run env overrides forwarded to the GPU worker; never put secrets here.
    worker_env: dict[str, str] = field(default_factory=dict)
    # "catalog" (curated models only) or "allow" (any HF model that fits the GPU).
    model_policy: str = "catalog"
    thinking: bool = False
    wandb: WandbSpec = field(default_factory=WandbSpec)

    @property
    def phase(self) -> str:
        return "rl" if self.algorithm == "grpo" else self.algorithm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobSpec:
        env = data.get("environment") or {}
        # Reject stale payloads carrying a local `path`; worker only runs published env ids.
        if isinstance(env, dict) and env.get("path"):
            raise ValueError(
                "local environment paths are no longer supported; the worker only runs "
                "published Freesolo environment ids"
            )
        train = data.get("train") or {}
        gpu = data.get("gpu") or {}
        return cls(
            model=data.get("model", cls.model),
            algorithm=normalize_algorithm(data.get("algorithm", cls.algorithm)),
            environment=EnvironmentSpec(
                id=env.get("id", ""),
                params=dict(env.get("params") or {}),
                pip=tuple(str(p) for p in env.get("pip") or ()),
                secrets=_str_tuple(env.get("secrets")),
                resolved_sha=str(env.get("resolved_sha") or ""),
            ),
            train=TrainSpec(
                steps=_opt_int(train.get("steps")),
                epochs=_opt_int(train.get("epochs")),
                lora_rank=int(train.get("lora_rank", 32)),
                lora_alpha=int(train.get("lora_alpha", 64)),
                init_from_adapter=str(train.get("init_from_adapter") or ""),
                hf_repo=str(train.get("hf_repo") or ""),
                learning_rate=_opt_float(train.get("learning_rate")),
                batch_size=_opt_int(train.get("batch_size")),
                max_length=_opt_int(train.get("max_length")),
                save_every=_opt_int(train.get("save_every")),
                max_steps=_opt_int(train.get("max_steps")),
                max_examples=_opt_int(train.get("max_examples")),
                group_size=_opt_int(train.get("group_size")),
                temperature=_opt_float(train.get("temperature")),
                max_tokens=_opt_int(train.get("max_tokens")),
                kl_penalty_coef=_opt_float(train.get("kl_penalty_coef")),
                advantage_clip=_opt_float(train.get("advantage_clip")),
                thinking_length_penalty_coef=_opt_float(train.get("thinking_length_penalty_coef")),
                stop_sequences=_str_tuple(train.get("stop_sequences")),
                teacher_model=str(train.get("teacher_model") or ""),
                tokenizer_alignment=str(train.get("tokenizer_alignment") or ""),
                teacher_top_logprobs=_opt_int(train.get("teacher_top_logprobs")),
            ),
            gpu=GpuSpec(
                type=gpu.get("type", DEFAULT_GPU),
                disk_gb=int(gpu.get("disk_gb", 60)),
                max_wall_seconds=int(gpu.get("max_wall_seconds", 24 * 3600)),
                max_retries=int(gpu.get("max_retries", 5)),
                # network_volume/network_volume_gb round-trip so the runner-assigned weight cache
                # survives the to_dict()->from_dict() hops in _with_model_disk / _spec_with_gpu /
                # _assign_managed_hf_repo before deploy.
                network_volume=gpu.get("network_volume"),
                network_volume_gb=_volume_gb(gpu.get("network_volume_gb")),
            ),
            run_id=data.get("run_id", "local"),
            worker_env=_coerce_str_map(data.get("worker_env")),
            model_policy=data.get("model_policy", "catalog"),
            thinking=coerce_bool(data.get("thinking", False)),
            wandb=_coerce_wandb(data.get("wandb")),
        )

    @classmethod
    def from_json(cls, raw: str) -> JobSpec:
        return cls.from_dict(json.loads(raw))


def load_job_spec_from_env() -> JobSpec | None:
    """Load FLASH_JOB_SPEC_JSON or FLASH_JOB_SPEC_PATH if present on a worker node."""
    raw = os.environ.get("FLASH_JOB_SPEC_JSON")
    if raw:
        return JobSpec.from_json(raw)
    path = os.environ.get("FLASH_JOB_SPEC_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return JobSpec.from_json(f.read())
    return None
