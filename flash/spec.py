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
    epochs: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    lora_rank: int = field(default=32, metadata={"introduced_in": "0.2.0"})
    lora_alpha: int = field(default=64, metadata={"introduced_in": "0.2.0"})
    # Artifact-store adapter ref output by `flash status`:
    # ``<hf_repo>:<phase>/<run_id>``.
    init_from_adapter: str = field(default="", metadata={"introduced_in": "0.2.0"})
    # internal: immutable source dataset commit used for a prepared warm-start. parsed only from
    # control-plane jobspec payloads; the public config schema does not accept this key.
    init_from_adapter_revision: str = ""
    # PLATFORM-MANAGED: control-plane-assigned HF artifact repo; user-supplied values are ignored.
    hf_repo: str = field(default="", metadata={"introduced_in": "0.2.0"})
    # None -> worker's tuned recipe default.
    learning_rate: float | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    batch_size: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    max_context_tokens: int | None = field(default=None, metadata={"introduced_in": "0.2.49"})
    save_every: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    max_steps: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    max_examples: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    group_size: int | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    temperature: float | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    max_completion_tokens: int | None = field(default=None, metadata={"introduced_in": "0.2.49"})
    kl_penalty_coef: float | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    advantage_clip: float | None = field(default=None, metadata={"introduced_in": "0.2.0"})
    thinking_length_penalty_coef: float | None = field(
        default=None, metadata={"introduced_in": "0.2.0"}
    )
    # OPD only: weight of the terminal-EOS behaviour-cloning term (recipe default when None). 0
    # disables it. Raise it for a student that keeps running past the length cap without emitting EOS.
    opd_eos_loss_coef: float | None = field(default=None, metadata={"introduced_in": "0.2.55"})
    # OPD only: the managed teacher to distil from, stored as its resolved Fireworks model id (parse
    # canonicalizes any alias / spaced / raw-id form via recipe.resolve_teacher, e.g. "glm-5.2" =>
    # "accounts/fireworks/models/glm-5p2"). "" => the recipe default (GLM 5.2).
    teacher_model: str = field(default="", metadata={"introduced_in": "0.2.56"})
    stop_sequences: tuple[str, ...] = field(default=(), metadata={"introduced_in": "0.2.0"})
    # Canonical JSON of vLLM StructuredOutputsParams kwargs ("" = unconstrained). Normalized once
    # at parse time (schema/fields.py) so worker/hub/API hops carry one stable string form.
    structured_outputs: str = field(default="", metadata={"introduced_in": "0.2.56"})


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
    algorithm: str = "sft"
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
        """Return the public/API representation of this job specification."""
        data = asdict(self)
        train = data["train"]
        train.pop("init_from_adapter_revision", None)
        if train.get("init_from_adapter"):
            train.pop("lora_rank", None)
        return data

    def to_internal_dict(self) -> dict[str, Any]:
        """Return the complete control-plane and worker representation."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_internal_dict(), sort_keys=True)

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
                epochs=_opt_int(train.get("epochs")),
                lora_rank=int(train.get("lora_rank", 32)),
                lora_alpha=int(train.get("lora_alpha", 64)),
                init_from_adapter=str(train.get("init_from_adapter") or ""),
                init_from_adapter_revision=str(train.get("init_from_adapter_revision") or ""),
                hf_repo=str(train.get("hf_repo") or ""),
                learning_rate=_opt_float(train.get("learning_rate")),
                batch_size=_opt_int(train.get("batch_size")),
                max_context_tokens=_opt_int(train.get("max_context_tokens")),
                save_every=_opt_int(train.get("save_every")),
                max_steps=_opt_int(train.get("max_steps")),
                max_examples=_opt_int(train.get("max_examples")),
                group_size=_opt_int(train.get("group_size")),
                temperature=_opt_float(train.get("temperature")),
                max_completion_tokens=_opt_int(train.get("max_completion_tokens")),
                kl_penalty_coef=_opt_float(train.get("kl_penalty_coef")),
                advantage_clip=_opt_float(train.get("advantage_clip")),
                thinking_length_penalty_coef=_opt_float(train.get("thinking_length_penalty_coef")),
                opd_eos_loss_coef=_opt_float(train.get("opd_eos_loss_coef")),
                teacher_model=str(train.get("teacher_model") or ""),
                stop_sequences=_str_tuple(train.get("stop_sequences")),
                structured_outputs=str(train.get("structured_outputs") or ""),
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
