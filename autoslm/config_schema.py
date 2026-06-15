"""Parse AutoSLM TOML configs into worker JobSpecs."""

from __future__ import annotations

import math
import os
import tomllib
from typing import Any

from .catalog import normalize_algorithm, resolve_model
from .providers import PROVIDER_NAMES
from .providers.base import (
    POLICY_NAMES,
    SUPPORTED,
    UnsupportedGpuError,
    canonical_gpu,
    is_validated,
    providers_for,
    resolve_gpu_policy,
    unvalidated_allowed,
)
from .worker_spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec


def _train_int(train_raw: dict, key: str, *, minimum: int) -> int | None:
    """Validate an optional integer [train] knob (>= minimum) -> ConfigError (HTTP 400).

    None stays None (recipe default). Rejects bools, non-numbers, non-integers, and
    out-of-range values at parse time instead of letting them reach a provisioned worker.
    """
    v = train_raw.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"train.{key} must be an integer")
    # Check finiteness BEFORE int(v): int(inf) raises OverflowError and int(nan) ValueError
    # (the former would be a 500); reject both as a clean 400.
    if not math.isfinite(v) or float(v) != int(v):
        raise ConfigError(f"train.{key} must be a finite integer")
    v = int(v)
    if v < minimum:
        raise ConfigError(f"train.{key} must be >= {minimum}")
    return v


def _train_float(
    train_raw: dict, key: str, *, minimum: float, exclusive: bool = False
) -> float | None:
    """Validate an optional float [train] knob -> ConfigError (HTTP 400). None stays None."""
    v = train_raw.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"train.{key} must be a number")
    v = float(v)
    # nan/inf slip past the range checks below (nan compares false, inf passes any minimum)
    # and would reach TRL optimizer/sampling settings; reject them as a 400 here.
    if not math.isfinite(v):
        raise ConfigError(f"train.{key} must be a finite number")
    if exclusive and v <= minimum:
        raise ConfigError(f"train.{key} must be > {minimum}")
    if not exclusive and v < minimum:
        raise ConfigError(f"train.{key} must be >= {minimum}")
    return v


def _train_stops(train_raw: dict) -> tuple[str, ...]:
    """Validate stop_sequences -> ConfigError. A string is ONE stop (never char-split);
    a list must hold strings; empties are dropped; anything else is rejected."""
    v = train_raw.get("stop_sequences")
    if v is None:
        return ()
    if isinstance(v, str):
        return (v,) if v else ()
    if not isinstance(v, (list, tuple)):
        raise ConfigError("train.stop_sequences must be a string or a list of strings")
    for s in v:
        if not isinstance(s, str):
            raise ConfigError("train.stop_sequences entries must be strings")
    return tuple(s for s in v if s)


class ConfigError(ValueError):
    pass


def load_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def spec_from_file(
    path: str,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    extra_configs: list[str] | None = None,
) -> JobSpec:
    base_dir = os.path.dirname(os.path.abspath(path))
    raw = load_toml(path)
    # Composed configs: later files override earlier keys (deep merge).
    for extra in extra_configs or []:
        _deep_merge(raw, load_toml(extra))
    # `--set key=value` dotted overrides (highest precedence).
    for item in overrides or []:
        _apply_override(raw, item)
    return spec_from_dict(raw, base_dir=base_dir, run_id=run_id)


def _deep_merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _coerce_scalar(value: str):
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _apply_override(raw: dict, item: str) -> None:
    if "=" not in item:
        raise ConfigError(f"--set must be key=value, got {item!r}")
    key, value = item.split("=", 1)
    parts = key.strip().split(".")
    node = raw
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ConfigError(f"--set path {key!r} traverses a non-table value")
    leaf = parts[-1]
    # support list values like seeds=[0,1]
    val = value.strip()
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        node[leaf] = [_coerce_scalar(x.strip()) for x in inner.split(",") if x.strip()]
    else:
        node[leaf] = _coerce_scalar(val)


def spec_from_dict(raw: dict[str, Any], base_dir: str = ".", run_id: str | None = None) -> JobSpec:
    try:
        model = raw["model"]
    except KeyError as exc:
        raise ConfigError("config must set `model`") from exc

    try:
        algorithm = normalize_algorithm(raw.get("algorithm"))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    model_policy = (raw.get("model_policy") or "catalog").lower()
    if model_policy not in ("catalog", "allow"):
        raise ConfigError('model_policy must be "catalog" or "allow"')
    thinking = raw.get("thinking", False)
    if not isinstance(thinking, bool):
        raise ConfigError("thinking must be a boolean")

    env_raw = raw.get("environment") or {}
    train_raw = raw.get("train") or {}
    gpu_raw = raw.get("gpu") or {}

    # Smart allocation is the default: an omitted gpu.type means "the cheapest GPU
    # (across providers) that fits the model", re-resolved live at submit time. The
    # original request survives in gpu.requested so the orchestrator knows whether
    # it may re-allocate (policy words) or must honor a concrete pin.
    requested_gpu = str(gpu_raw.get("requested") or gpu_raw.get("type") or "auto")
    provider = str(gpu_raw.get("provider") or "auto").strip().lower()
    if provider not in ("auto", *PROVIDER_NAMES):
        allowed = '", "'.join(("auto", *PROVIDER_NAMES))
        raise ConfigError(f'gpu.provider must be "{allowed}"')
    allow_unval = gpu_raw.get("allow_unvalidated")
    if allow_unval is not None and not isinstance(allow_unval, bool):
        raise ConfigError("gpu.allow_unvalidated must be a boolean")
    try:
        # Parse-time provisional: "cheapest"/"auto" resolve to the cheapest validated
        # RunPod-static class that fits (deterministic offline; open models sized from
        # HF metadata); concrete names are canonicalized. The submit-time allocator
        # re-resolves policy words live across providers.
        gpu_type = resolve_gpu_policy(
            requested_gpu, model, allow_unvalidated=allow_unval, algorithm=algorithm
        )
    except UnsupportedGpuError as exc:
        raise ConfigError(str(exc)) from exc
    pinned = requested_gpu.strip().lower() not in POLICY_NAMES
    if pinned and provider != "auto" and provider not in providers_for(gpu_type):
        raise ConfigError(
            f"gpu type {gpu_type!r} is not available on provider {provider!r} "
            f"(providers: {', '.join(providers_for(gpu_type))})"
        )
    if (
        pinned
        and not is_validated(gpu_type, provider if provider != "auto" else None)
        and not unvalidated_allowed(allow_unval)
    ):
        raise ConfigError(
            f"gpu type {gpu_type!r} has not passed AutoSLM's live validation smoke"
            f"{' on ' + provider if provider != 'auto' else ''} "
            f"(validated: {', '.join(SUPPORTED)}). Set gpu.allow_unvalidated = true "
            f"(or AUTOSLM_GPU_ALLOW_UNVALIDATED=1) to use it anyway."
        )
    try:
        info = resolve_model(model, algorithm, policy=model_policy, gpu=gpu_type)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if thinking and info.thinking == "none":
        raise ConfigError(
            f"{model} does not support thinking mode (its chat template has no "
            f"<think> support); pick a thinking-capable model — `slm models` lists "
            f"each model's thinking capability"
        )
    if not thinking and info.thinking == "always":
        raise ConfigError(
            f"{model} always emits <think> reasoning and cannot run with thinking "
            f"disabled; set thinking = true"
        )
    if thinking and info.thinking == "unknown":
        print(
            f"warning: open-model policy: cannot verify that {model}'s chat template "
            f"supports thinking mode; the run proceeds with enable_thinking=true"
        )

    spec = JobSpec(
        model=model,
        algorithm=algorithm,
        environment=EnvironmentSpec(
            id=str(env_raw.get("id") or ""),
            params=dict(env_raw.get("params") or {}),
            pip=tuple(str(p) for p in env_raw.get("pip") or ()),
        ),
        train=TrainSpec(
            steps=train_raw.get("steps"),
            epochs=train_raw.get("epochs"),
            lora_rank=int(train_raw.get("lora_rank", 32)),
            lora_alpha=int(train_raw.get("lora_alpha", 64)),
            seeds=tuple(int(s) for s in train_raw.get("seeds", (0,))),
            init_from_adapter=str(train_raw.get("init_from_adapter") or ""),
            learning_rate=_train_float(train_raw, "learning_rate", minimum=0.0, exclusive=True),
            batch_size=_train_int(train_raw, "batch_size", minimum=1),
            max_length=_train_int(train_raw, "max_length", minimum=1),
            save_every=_train_int(train_raw, "save_every", minimum=1),
            group_size=_train_int(train_raw, "group_size", minimum=1),
            temperature=_train_float(train_raw, "temperature", minimum=0.0),
            max_tokens=_train_int(train_raw, "max_tokens", minimum=1),
            kl_penalty_coef=_train_float(train_raw, "kl_penalty_coef", minimum=0.0),
            advantage_clip=_train_float(train_raw, "advantage_clip", minimum=0.0),
            thinking_length_penalty_coef=_train_float(
                train_raw, "thinking_length_penalty_coef", minimum=0.0
            ),
            stop_sequences=_train_stops(train_raw),
        ),
        gpu=GpuSpec(
            type=gpu_type,
            provider=provider,
            requested=requested_gpu,
            allow_unvalidated=allow_unval,
            disk_gb=int(gpu_raw.get("disk_gb", 60)),
            max_wall_seconds=int(gpu_raw.get("max_wall_seconds", 24 * 3600)),
            max_retries=int(gpu_raw.get("max_retries", 2)),
            network_volume=gpu_raw.get("network_volume"),
            network_volume_gb=int(gpu_raw.get("network_volume_gb", 100)),
            datacenter=gpu_raw.get("datacenter"),
        ),
        run_id=run_id or raw.get("run_id", "local"),
        model_policy=model_policy,
        thinking=thinking,
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: JobSpec) -> None:
    if not spec.train.seeds:
        raise ConfigError("train.seeds must contain at least one seed")
    try:
        canonical_gpu(spec.gpu.type)
    except UnsupportedGpuError as exc:
        raise ConfigError(str(exc)) from exc
    # GRPO is step-driven; SFT is epoch-driven. Reject a non-positive explicit count
    # for whichever the algorithm consumes, so an invalid config fails here instead of
    # provisioning a worker that silently falls back to a default count.
    if spec.algorithm == "grpo" and spec.train.steps is not None and spec.train.steps <= 0:
        raise ConfigError("train.steps must be positive for GRPO")
    if spec.algorithm == "sft" and spec.train.epochs is not None and spec.train.epochs <= 0:
        raise ConfigError("train.epochs must be positive for SFT")
    # Verifiers-only: every run must name an environment by its verifiers/Prime Hub slug
    # via [environment] id. There is no default environment and no local path mode.
    if not spec.environment.id:
        raise ConfigError(
            "config must set [environment] id (a verifiers/Prime Hub env slug, e.g. "
            '"owner/name"); there is no local path mode'
        )
    if spec.train.lora_rank <= 0:
        raise ConfigError("train.lora_rank must be positive")
    # GRPO recipe knobs (optional): validate the ones that are set. A None means
    # "the worker applies its recipe default", so only constrain explicit values.
    if spec.train.group_size is not None and spec.train.group_size <= 0:
        raise ConfigError("train.group_size must be positive")
    if spec.train.temperature is not None and spec.train.temperature < 0:
        raise ConfigError("train.temperature must be non-negative")
    if spec.train.max_tokens is not None and spec.train.max_tokens <= 0:
        raise ConfigError("train.max_tokens must be positive")
    if spec.train.kl_penalty_coef is not None and spec.train.kl_penalty_coef < 0:
        raise ConfigError("train.kl_penalty_coef must be non-negative")
    if spec.train.advantage_clip is not None and spec.train.advantage_clip < 0:
        raise ConfigError("train.advantage_clip must be non-negative")
    if spec.train.thinking_length_penalty_coef is not None and not (
        0.0 <= spec.train.thinking_length_penalty_coef <= 1.0
    ):
        raise ConfigError("train.thinking_length_penalty_coef must be between 0 and 1")
    # lora_alpha scales the adapter contribution; 0 (or negative) trains a paid run
    # that produces a no-op adapter (zero scaling at serve). Reject up front.
    if spec.train.lora_alpha <= 0:
        raise ConfigError("train.lora_alpha must be positive")
