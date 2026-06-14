"""Structured job specification shared by CLI/API/orchestrator and GPU workers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .catalog import ALGORITHMS as ALGORITHMS  # re-exported for spec consumers
from .catalog import DEFAULT_MODEL, normalize_algorithm

_FALSE_STRINGS = {"", "0", "false", "no", "off", "none"}


def _str_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a string-or-list knob (e.g. stop_sequences) to a tuple of strings.

    A bare string is ONE element — never iterated into characters ("</s>" must not become
    ('<','/','s','>')). None and empty strings -> () (no stop configured); empty entries
    in a list are dropped."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(s for s in (str(x) for x in value) if s)


def _coerce_bool(value: Any) -> bool:
    """Parse a bool from loosely-typed spec sources (JSON/env/persisted dicts).

    bool(...) on a string is truthy for ANY non-empty string, so "false"/"0" would
    wrongly become True; treat the usual falsey strings as False.
    """
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    return bool(value)


def _opt_int(value: Any) -> int | None:
    """Parse an optional int from a loosely-typed spec source; None stays None.

    Rejects JSON booleans: ``bool`` is an ``int`` subclass in Python, so ``int(True)`` would
    silently coerce a stray boolean train knob to 1 (and ``False`` to 0). Mirrors
    config_schema._opt_num — a bool is a type error, not a number.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return int(value)


def _opt_float(value: Any) -> float | None:
    """Parse an optional float from a loosely-typed spec source; None stays None.

    Rejects JSON booleans (``bool`` is an ``int`` subclass) so a stray boolean train knob is
    not silently coerced to 0.0/1.0; mirrors config_schema._opt_num.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return float(value)


@dataclass(frozen=True)
class EnvironmentSpec:
    # Verifiers/Prime Hub env slug ("owner/name") or installed/local env id. No default:
    # a run must name an environment explicitly (validated in config_schema / the worker).
    id: str = ""
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
    # Optimizer/batching knobs (SFT + GRPO). None -> the worker's tuned recipe default.
    # batch_size is the GLOBAL/effective batch (SFT: grad-accum is sized to hit it; GRPO:
    # prompts per optimizer step). max_length is the SFT max sequence length. save_every
    # is the checkpoint interval in optimizer steps.
    learning_rate: float | None = None
    batch_size: int | None = None
    max_length: int | None = None
    save_every: int | None = None
    # GRPO recipe knobs (datums parity), shipped by the SDK in [train] (NOT in
    # [environment.params], which is forwarded verbatim to the verifiers env loader).
    # None/() -> recipe default. group_size = completions per prompt; temperature = rollout
    # sampling temp; max_tokens = completion budget; kl_penalty_coef = KL beta;
    # advantage_clip = centered-advantage clamp; thinking_length_penalty_coef =
    # per-<think>-token reward deduction; stop_sequences = rollout stop strings.
    group_size: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    kl_penalty_coef: float | None = None
    advantage_clip: float | None = None
    thinking_length_penalty_coef: float | None = None
    stop_sequences: tuple[str, ...] = ()


@dataclass(frozen=True)
class GpuSpec:
    type: str = "RTX 5090"
    # GPU substrate: "auto" (cheapest RunPod class at submit time) or "runpod".
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
                id=env.get("id", ""),
                params=dict(env.get("params") or {}),
                path=env.get("path"),
                pip=tuple(str(p) for p in env.get("pip") or ()),
            ),
            train=TrainSpec(
                steps=_opt_int(train.get("steps")),
                epochs=_opt_int(train.get("epochs")),
                lora_rank=int(train.get("lora_rank", 32)),
                lora_alpha=int(train.get("lora_alpha", 64)),
                seeds=tuple(int(s) for s in train.get("seeds", (0,))),
                init_from_adapter=str(train.get("init_from_adapter") or ""),
                learning_rate=_opt_float(train.get("learning_rate")),
                batch_size=_opt_int(train.get("batch_size")),
                max_length=_opt_int(train.get("max_length")),
                save_every=_opt_int(train.get("save_every")),
                group_size=_opt_int(train.get("group_size")),
                temperature=_opt_float(train.get("temperature")),
                max_tokens=_opt_int(train.get("max_tokens")),
                kl_penalty_coef=_opt_float(train.get("kl_penalty_coef")),
                advantage_clip=_opt_float(train.get("advantage_clip")),
                thinking_length_penalty_coef=_opt_float(train.get("thinking_length_penalty_coef")),
                stop_sequences=_str_tuple(train.get("stop_sequences")),
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
