"""Structured job specification shared by CLI/API/runner and GPU workers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .catalog import DEFAULT_GPU, DEFAULT_MODEL, normalize_algorithm

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


def coerce_bool(value: Any) -> bool:
    """Parse a bool from loosely-typed sources (JSON request bodies / env / persisted dicts).

    bool(...) on a string is truthy for ANY non-empty string, so "false"/"0"/"no" would
    wrongly become True; treat the usual falsey strings (see ``_FALSE_STRINGS``) as False, so
    e.g. JSON ``"is_new": "false"`` is parsed as False. An already-bool value passes through.
    """
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    return bool(value)


def _coerce_str_map(value: Any) -> dict[str, str]:
    """Coerce a loosely-typed spec field into a ``dict[str, str]``.

    A malformed persisted spec (or programmatic caller) can set a mapping field to a non-dict;
    `.items()` on that would crash `from_dict` with AttributeError. Treat a non-dict as empty,
    mirroring how the other nested fields tolerate missing/garbage input.
    """
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


def _coerce_wandb(value: Any) -> WandbSpec:
    """Coerce a loosely-typed ``wandb`` spec field into a ``WandbSpec``.

    A malformed/older persisted spec can set ``wandb`` to a non-dict (e.g. a bare string), and
    ``(value or {}).get(...)`` would crash ``from_dict`` with AttributeError on the worker. Treat
    a non-dict as empty (default naming), mirroring ``_coerce_str_map``. String-coerce + trim the
    leaves so a non-string label can't reach the W&B SDK / run-name path; blank -> None (default).
    """
    if not isinstance(value, dict):
        return WandbSpec()

    def _label(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    return WandbSpec(project=_label(value.get("project")), run_name=_label(value.get("run_name")))


def _opt_int(value: Any) -> int | None:
    """Parse an optional int from a loosely-typed spec source; None stays None.

    Rejects JSON booleans: ``bool`` is an ``int`` subclass in Python, so ``int(True)`` would
    silently coerce a stray boolean train knob to 1 (and ``False`` to 0). Mirrors the
    bool rejection in schema._train_int — a bool is a type error, not a number.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return int(value)


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    """Coerce a loosely-typed scalar to an int WITHOUT silently truncating.

    Plain ``int(...)`` accepts ``2.9`` (-> 2) and ``True`` (-> 1), so a malformed spec from a
    JSON/env/TOML source would quietly provision a different topology than intended. We require an
    exact integer: a bool is rejected (a JSON/TOML boolean is not a count), and a float is accepted
    only if it is whole (``2.0`` -> 2) — a fractional value (``2.9``) fails loudly. ``minimum``, if
    given, enforces a lower bound (raises if below it). Shared with the rollout-pool config
    (flash.pool.config._strict_int) so the strict-int contract lives in exactly one place.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got bool {value!r}")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be a whole number, got non-integer float {value!r}")
        result = int(value)
    else:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__} {value!r}")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _opt_float(value: Any) -> float | None:
    """Parse an optional float from a loosely-typed spec source; None stays None.

    Rejects JSON booleans (``bool`` is an ``int`` subclass) so a stray boolean train knob is
    not silently coerced to 0.0/1.0; mirrors the bool rejection in schema._train_float.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return float(value)


@dataclass(frozen=True)
class EnvironmentSpec:
    # Verifiers/Prime Hub env slug ("owner/name") or installed/local env id. No default:
    # a run must name an environment explicitly (validated in schema / the worker).
    id: str = ""
    params: dict[str, Any] = field(default_factory=dict)
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
    # Per-run HuggingFace artifact repo ("owner/name") for this run's adapter/checkpoint/
    # code storage AND serving. PLATFORM-MANAGED, not a user field: the control plane assigns
    # it server-side in runner.submit_job (a per-run private dataset under the operator's
    # namespace, written by the operator HF_TOKEN). A user-supplied value is ignored by
    # schema.spec_from_dict; this field carries the control-plane-assigned repo to the worker.
    hf_repo: str = ""
    # Optimizer/batching knobs (SFT + GRPO). None -> the worker's tuned recipe default.
    # batch_size is the GLOBAL/effective batch (SFT: grad-accum is sized to hit it; GRPO:
    # prompts per optimizer step). max_length is the SFT max sequence length. save_every
    # is the checkpoint interval in optimizer steps.
    learning_rate: float | None = None
    batch_size: int | None = None
    max_length: int | None = None
    save_every: int | None = None
    # SFT caps (None/0 -> no cap). max_steps caps optimizer steps (cheap pre-flight smoke);
    # max_examples truncates the SFT dataset.
    max_steps: int | None = None
    max_examples: int | None = None
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
    # Disaggregated (async) GRPO rollout: number of GPUs in the node dedicated to the vLLM rollout
    # server, the rest train (see engine.rollout_bench.select_rollout_split). 0 = colocate (the
    # current single-GPU TRL path). >0 gets a multi-GPU node (size derived = one trainer + inference).
    inference_gpus: int = 0


@dataclass(frozen=True)
class GpuSpec:
    # The parse-time provisional GPU class (cheapest VALIDATED class that fits the model). GPU
    # pinning is gone: the submit-time allocator always re-picks the cheapest fitting validated
    # class across ALL providers, so a config's gpu.type does NOT pin — ``type`` is just the
    # offline sizing/display default and the carrier the runner overwrites with the
    # actually-allocated class.
    type: str = DEFAULT_GPU
    disk_gb: int = 60
    max_wall_seconds: int = 24 * 3600
    # Auto-resubmit budget for infra-shaped failures (worker loss / stall / timeout);
    # each retry resumes from the latest streamed checkpoint. Raised from 2 -> 6 so the runtime
    # MIG-walk has headroom: RunPod can hand back a Blackwell MIG slice for a validated GPU *type*,
    # and each MIG hit (exclude_class) costs a retry to walk OFF that class onto a full GPU. With the
    # allocator now skipping the known MIG-substituted types up front (runpod_mig_risk), this is the
    # safety net for any MIG slice that still slips through on an unflagged type.
    max_retries: int = 6
    # OPT-IN persistent RunPod network volume mounted at /runpod-volume, used as a
    # cross-run HF model cache (repeat runs skip the model download). Trade-offs: it
    # pins the run to the volume's datacenter (smaller GPU pool — usually the bigger
    # cost) and the volume bills monthly while it exists. Off (None) by default.
    # RunPod-specific: network_volume/datacenter are read only by the RunPod provider
    # and ignored by Vast (which rents single-GPU instances with no network volume).
    network_volume: str | None = None
    network_volume_gb: int = 100
    datacenter: str | None = None  # e.g. "EU-RO-1"; required pool pin for the volume
    # OPT-IN per-run provider pin. Unlike gpu.type (no pin — the submit-time allocator always
    # re-picks the cheapest fitting validated CLASS across ALL providers), provider pins which
    # SUBSTRATE the allocator may use: "vast" or "runpod" restricts allocation to that provider;
    # None (default) keeps the cross-provider cheapest-wins behavior. Used for A/B-ing one provider
    # against the full pool. The allocator raises a clear error if the pinned provider isn't
    # available/configured.
    provider: str | None = None


@dataclass(frozen=True)
class WandbSpec:
    # Optional W&B naming, defined in the [wandb] config table (first-class spec config, NOT
    # env vars). project/run_name are non-secret labels; the actual WANDB_API_KEY stays an
    # env-var secret. None -> the worker's defaults ("flash" project, "flash-<phase>-<run_id>-
    # seedN" run name).
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
    # Per-run worker-environment overrides merged into the GPU worker's env (highest precedence
    # over the control-plane os.environ allowlist). The escape hatch for A/B kernel experiments
    # that must differ PER RUN, not globally: e.g. an optimizer or LoRA-init override on just the
    # experiment run while others keep the global default. Forwarded verbatim (string values);
    # never set secrets here.
    worker_env: dict[str, str] = field(default_factory=dict)
    # "catalog" (curated models only) or "allow" (any HF model that fits the GPU).
    model_policy: str = "catalog"
    # Thinking/reasoning mode (thinking-capable models only). One flag per run, consumed
    # identically by SFT rendering, RL rollouts, and serving (decoding parity). OFF by default
    # (operator preference: training defaults to no-reasoning; set thinking = true to enable).
    thinking: bool = False
    # Optional W&B run naming from the [wandb] config table. Carried as typed spec config
    # (round-tripped in the job-spec JSON the worker reads), not as environment variables.
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
        # Defense-in-depth: a stale/older payload may still carry a local `path`. The worker only
        # runs published Hub env ids, so reject it here rather than silently dropping it.
        if isinstance(env, dict) and env.get("path"):
            raise ValueError(
                "local environment paths are no longer supported; the worker only runs "
                "published Hub env ids"
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
            ),
            train=TrainSpec(
                steps=_opt_int(train.get("steps")),
                epochs=_opt_int(train.get("epochs")),
                lora_rank=int(train.get("lora_rank", 32)),
                lora_alpha=int(train.get("lora_alpha", 64)),
                seeds=tuple(int(s) for s in train.get("seeds", (0,))),
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
                inference_gpus=(
                    _strict_int(train.get("inference_gpus"), name="train.inference_gpus", minimum=0)
                    if train.get("inference_gpus") is not None
                    else 0
                ),
            ),
            gpu=GpuSpec(
                type=gpu.get("type", DEFAULT_GPU),
                disk_gb=int(gpu.get("disk_gb", 60)),
                max_wall_seconds=int(gpu.get("max_wall_seconds", 24 * 3600)),
                max_retries=int(gpu.get("max_retries", 6)),
                network_volume=gpu.get("network_volume"),
                network_volume_gb=int(gpu.get("network_volume_gb", 100)),
                datacenter=gpu.get("datacenter"),
                provider=gpu.get("provider"),
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


def gpus_per_node(spec: Any) -> int:
    """GPUs the run's worker node must provision = ONE trainer card + ``train.inference_gpus``.

    There is no ``[gpu].count`` field: the node size is DERIVED from the rollout topology. A
    colocated run (``inference_gpus == 0``, the single-process TRL path) needs 1 GPU; a
    disaggregated async-GRPO run (``inference_gpus == N > 0``) needs ``N + 1`` — N cards for the
    vLLM rollout server and one trainer card (see engine.rollout_bench.select_rollout_split, whose
    ``train_gpus = total_gpus - inference_gpus``, and the "keep it to one trainer card"
    guidance in validate_disaggregated_requirement). Used to set the RunPod endpoint's ``gpu_count``
    and the worker's ``FLASH_GPU_COUNT`` so multi-GPU jobs actually request multiple GPUs instead of
    silently falling back to 1. Tolerant of a missing/partial spec (defaults to 1).
    """
    train = getattr(spec, "train", None)
    try:
        inference_gpus = int(getattr(train, "inference_gpus", 0) or 0)
    except (TypeError, ValueError):
        inference_gpus = 0
    return max(1, inference_gpus + 1) if inference_gpus > 0 else 1


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
