"""Structured job specification shared by CLI/API/orchestrator and GPU workers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .catalog import ALGORITHMS as ALGORITHMS  # re-exported for spec consumers
from .catalog import DEFAULT_MODEL, normalize_algorithm

_FALSE_STRINGS = {"", "0", "false", "no", "off", "none"}


def _coerce_bool(value: Any) -> bool:
    """Parse a bool from loosely-typed spec sources (JSON/env/persisted dicts).

    bool(...) on a string is truthy for ANY non-empty string, so "false"/"0" would
    wrongly become True; treat the usual falsey strings as False.
    """
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    return bool(value)


@dataclass(frozen=True)
class EnvironmentSpec:
    id: str = "gsm8k"
    params: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    # Pip requirements the GPU worker needs for this environment (verifiers/Hub envs).
    # Filled in client-side from the local install manifest so the managed control
    # plane never depends on client-local state; empty means "derive on the server".
    pip: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainSpec:
    steps: int | None = None
    epochs: int | None = None
    lora_rank: int = 32
    lora_alpha: int = 64
    seeds: tuple[int, ...] = (0,)
    # Artifact-store adapter prefix (``<phase>/<run_id>/seed<N>``) to initialize the
    # LoRA from instead of training fresh — e.g. a GRPO run continuing an SFT adapter.
    init_from_adapter: str = ""


@dataclass(frozen=True)
class GpuSpec:
    type: str = "RTX 5090"
    # GPU substrate: "auto" (cheapest across providers at submit time), "runpod", or
    # "vast" (verified datacenters only).
    provider: str = "auto"
    # The raw user gpu.type input ("cheapest"/"auto" or a concrete class), always set
    # by config parsing. The orchestrator re-allocates the class at submit time iff
    # this is a policy word — ``type`` is then just the parse-time provisional; a
    # concrete ``requested`` pins the class and the allocator only picks the provider.
    requested: str = ""
    # Carried into the submit-time allocator (None -> AUTOSLM_GPU_ALLOW_UNVALIDATED).
    allow_unvalidated: bool | None = None
    disk_gb: int = 60
    max_wall_seconds: int = 24 * 3600
    # Auto-resubmit budget for infra-shaped failures (worker loss / stall / timeout);
    # each retry resumes from the latest streamed checkpoint.
    max_retries: int = 2
    # OPT-IN persistent RunPod network volume mounted at /runpod-volume, used as a
    # cross-run HF model cache (repeat runs skip the model download). Trade-offs: it
    # pins the run to the volume's datacenter (smaller GPU pool — usually the bigger
    # cost) and the volume bills monthly while it exists. Off (None) by default.
    network_volume: str | None = None
    network_volume_gb: int = 100
    datacenter: str | None = None  # e.g. "EU-RO-1"; required pool pin for the volume


@dataclass(frozen=True)
class JobSpec:
    model: str = DEFAULT_MODEL
    algorithm: str = "grpo"
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    gpu: GpuSpec = field(default_factory=GpuSpec)
    run_id: str = "local"
    # "catalog" (curated models only) or "allow" (any HF model that fits the GPU).
    model_policy: str = "catalog"
    # Thinking/reasoning mode (thinking-capable models only). One flag per run, consumed
    # identically by SFT rendering, RL rollouts, and serving (decoding parity).
    thinking: bool = False

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
        train = data.get("train") or {}
        gpu = data.get("gpu") or {}
        return cls(
            model=data.get("model", cls.model),
            algorithm=normalize_algorithm(data.get("algorithm", cls.algorithm)),
            environment=EnvironmentSpec(
                id=env.get("id", "gsm8k"),
                params=dict(env.get("params") or {}),
                path=env.get("path"),
                pip=tuple(str(p) for p in env.get("pip") or ()),
            ),
            train=TrainSpec(
                steps=train.get("steps"),
                epochs=train.get("epochs"),
                lora_rank=int(train.get("lora_rank", 32)),
                lora_alpha=int(train.get("lora_alpha", 64)),
                seeds=tuple(int(s) for s in train.get("seeds", (0,))),
                init_from_adapter=str(train.get("init_from_adapter") or ""),
            ),
            gpu=GpuSpec(
                type=gpu.get("type", "RTX 5090"),
                provider=gpu.get("provider", "auto"),
                requested=gpu.get("requested", ""),
                allow_unvalidated=gpu.get("allow_unvalidated"),
                disk_gb=int(gpu.get("disk_gb", 60)),
                max_wall_seconds=int(gpu.get("max_wall_seconds", 24 * 3600)),
                max_retries=int(gpu.get("max_retries", 2)),
                network_volume=gpu.get("network_volume"),
                network_volume_gb=int(gpu.get("network_volume_gb", 100)),
                datacenter=gpu.get("datacenter"),
            ),
            run_id=data.get("run_id", "local"),
            model_policy=data.get("model_policy", "catalog"),
            thinking=_coerce_bool(data.get("thinking", False)),
        )

    @classmethod
    def from_json(cls, raw: str) -> JobSpec:
        return cls.from_dict(json.loads(raw))


def load_job_spec_from_env() -> JobSpec | None:
    """Load AUTOSLM_JOB_SPEC_JSON or AUTOSLM_JOB_SPEC_PATH if present on a worker node."""
    raw = os.environ.get("AUTOSLM_JOB_SPEC_JSON")
    if raw:
        return JobSpec.from_json(raw)
    path = os.environ.get("AUTOSLM_JOB_SPEC_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return JobSpec.from_json(f.read())
    return None
