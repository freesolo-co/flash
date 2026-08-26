"""Strict decoding and validation helpers for persisted job-spec records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import MISSING
from typing import Any

PREPARATION_ENVELOPE_VERSION = 1


def validate_persisted_spec_envelope(snapshot: object) -> int:
    """Validate and return the current persisted preparation version."""
    if not isinstance(snapshot, Mapping):
        raise ValueError("persisted effective preparation is malformed")
    if "version" not in snapshot:
        raise ValueError("persisted preparation envelope version is required")
    version = snapshot["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("persisted preparation envelope version must be a positive integer")
    if version != PREPARATION_ENVELOPE_VERSION:
        raise ValueError(f"unsupported persisted preparation envelope version {version}")
    return version


def persisted_default(field: Any) -> Any:
    """Return a dataclass field's documented persisted default."""
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    raise TypeError(f"persisted field {field.name} has no default")


def validated_section(
    data: dict[str, Any],
    name: str,
    allowed: set[str],
) -> dict[str, Any]:
    """Read one nested persisted block and reject null, wrong types, and unknown keys."""
    section = data.get(name, {})
    if not isinstance(section, dict):
        raise TypeError(f"{name} must be an object")
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"{name} has unknown key(s): {', '.join(unknown)}")
    return section


def persisted_bool(value: Any, *, name: str) -> bool:
    """Decode an exact persisted boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def persisted_int(value: Any, *, name: str, optional: bool = False) -> int | None:
    """Decode an exact persisted integer, excluding booleans and floats."""
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        suffix = " or null" if optional else ""
        raise TypeError(f"{name} must be an integer{suffix}")
    return value


def persisted_float(value: Any, *, name: str, optional: bool = False) -> float | None:
    """Decode one finite persisted numeric value without string or boolean coercion."""
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        suffix = " or null" if optional else ""
        raise TypeError(f"{name} must be a finite number{suffix}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def persisted_str(value: Any, *, name: str, optional: bool = False) -> str | None:
    """Decode an exact persisted string."""
    if value is None and optional:
        return None
    if not isinstance(value, str):
        suffix = " or null" if optional else ""
        raise TypeError(f"{name} must be a string{suffix}")
    return value


def persisted_dict(value: Any, *, name: str) -> dict[str, Any]:
    """Decode an exact persisted object without truthiness fallback."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return dict(value)


def persisted_sequence(
    value: Any,
    *,
    name: str,
    entry_type: type,
) -> tuple[Any, ...]:
    """Decode a list or tuple whose entries all have one exact type."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list")
    if any(
        not isinstance(entry, entry_type) or (entry_type is int and isinstance(entry, bool))
        for entry in value
    ):
        raise TypeError(f"{name} entries must be {entry_type.__name__}s")
    return tuple(value)


def validated_persisted_providers(
    gpu: dict[str, Any], gpu_type: str, gpu_type_fallbacks: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """Return persisted provider preferences cross-checked against gpu classes."""
    from flash.providers.core.registry import PROVIDER_NAMES, validated_provider_preferences

    provider = persisted_str(gpu.get("provider", ""), name="gpu.provider")
    assert provider is not None
    provider = provider.strip().lower()
    providers = validated_provider_preferences(
        gpu.get("providers", ()), allow_empty="providers" not in gpu
    )
    if provider and providers:
        raise ValueError("gpu.provider and gpu.providers cannot both be set")
    if provider or providers or gpu_type:
        from flash.providers.core.base import providers_for

        if provider and provider not in PROVIDER_NAMES:
            raise ValueError(f"unknown gpu.provider {provider!r}")
        for candidate in (gpu_type, *gpu_type_fallbacks):
            if candidate and provider and provider not in providers_for(candidate):
                raise ValueError(
                    f"gpu.provider {provider!r} cannot provision gpu.type {candidate!r}"
                )
    return provider, providers


def validate_resolved_spec_semantics(spec: Any) -> None:
    """Validate resolved persisted values independently of current catalog eligibility."""
    train = spec.train
    gpu = spec.gpu

    positive_optional = (
        ("train.epochs", train.epochs),
        ("train.learning_rate", train.learning_rate),
        ("train.batch_size", train.batch_size),
        ("train.prompts_per_step", train.prompts_per_step),
        ("train.max_context_tokens", train.max_context_tokens),
        ("train.save_every", train.save_every),
        ("train.group_size", train.group_size),
        ("train.max_completion_tokens", train.max_completion_tokens),
    )
    for name, value in positive_optional:
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("train.lora_rank", train.lora_rank),
        ("train.lora_alpha", train.lora_alpha),
        ("gpu.disk_gb", gpu.disk_gb),
        ("gpu.network_volume_gb", gpu.network_volume_gb),
        ("gpu.max_wall_seconds", gpu.max_wall_seconds),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("train.max_examples", train.max_examples),
        ("train.max_steps", train.max_steps),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name, value in (
        ("train.temperature", train.temperature),
        ("train.kl_penalty_coef", train.kl_penalty_coef),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name, value in (
        ("train.entropy_quantile", train.entropy_quantile),
        ("train.thinking_length_penalty_coef", train.thinking_length_penalty_coef),
    ):
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if gpu.max_retries < 0:
        raise ValueError("gpu.max_retries must be nonnegative")
    if gpu.count < 1 or gpu.count > 8:
        raise ValueError("gpu.count must be between 1 and 8")

    defaults = type(train)()
    inapplicable = {
        "sft": (
            "structured_outputs",
            "group_size",
            "temperature",
            "max_completion_tokens",
            "kl_penalty_coef",
            "entropy_quantile",
            "thinking_length_penalty_coef",
            "teacher_model",
            "credit_assignment",
            "stop_sequences",
            "prompts_per_step",
        ),
        "opd": (
            "entropy_quantile",
            "thinking_length_penalty_coef",
            "credit_assignment",
            "batch_size",
        ),
        "grpo": ("teacher_model", "batch_size"),
    }
    for name in inapplicable.get(spec.algorithm, ()):
        if getattr(train, name) != getattr(defaults, name):
            raise ValueError(f"train.{name} does not apply to {spec.algorithm}")

    if spec.algorithm == "grpo":
        from flash.core.grpo import resolve_grpo_rollout_shape

        resolve_grpo_rollout_shape(train.prompts_per_step, train.group_size)
    if spec.algorithm == "opd" and train.kl_penalty_coef == 0.0:
        raise ValueError("train.kl_penalty_coef must be positive for opd")


def _decode_environment(data: dict[str, Any]) -> Any:
    from dataclasses import fields

    from flash.core.spec import EnvironmentPackageSpec, EnvironmentSpec

    env = validated_section(data, "environment", {item.name for item in fields(EnvironmentSpec)})
    raw_package = env.get("package")
    if raw_package is not None and not isinstance(raw_package, dict):
        raise TypeError("environment.package must be an object or null")
    package = None
    if isinstance(raw_package, dict):
        allowed = {item.name for item in fields(EnvironmentPackageSpec)}
        unknown = sorted(set(raw_package) - allowed)
        if unknown:
            raise ValueError("environment.package has unknown key(s): " + ", ".join(unknown))
        package = EnvironmentPackageSpec(
            artifact_revision=persisted_str(
                raw_package.get("artifact_revision", ""),
                name="environment.package.artifact_revision",
            ),
            archive_sha256=persisted_str(
                raw_package.get("archive_sha256", ""),
                name="environment.package.archive_sha256",
            ),
            manifest_sha256=persisted_str(
                raw_package.get("manifest_sha256", ""),
                name="environment.package.manifest_sha256",
            ),
        )
    return EnvironmentSpec(
        id=persisted_str(env.get("id", ""), name="environment.id"),
        params=persisted_dict(env.get("params", {}), name="environment.params"),
        pip=persisted_sequence(env.get("pip", ()), name="environment.pip", entry_type=str),
        secrets=persisted_sequence(
            env.get("secrets", ()), name="environment.secrets", entry_type=str
        ),
        resolved_sha=persisted_str(env.get("resolved_sha", ""), name="environment.resolved_sha"),
        package=package,
    )


def _decode_train(data: dict[str, Any], algorithm: str) -> Any:
    from dataclasses import fields

    from flash.core.spec import (
        DEFAULT_CREDIT_ASSIGNMENT,
        TrainSpec,
        _coerce_credit_assignment,
        parse_max_steps,
    )

    train_fields = {item.name: item for item in fields(TrainSpec)}
    train = validated_section(data, "train", set(train_fields))
    rank = persisted_int(
        train.get("lora_rank", persisted_default(train_fields["lora_rank"])),
        name="train.lora_rank",
    )
    assert rank is not None
    alpha = persisted_int(train.get("lora_alpha", 2 * rank), name="train.lora_alpha")
    assert alpha is not None
    if rank <= 0:
        raise ValueError("train.lora_rank must be positive")
    if alpha <= 0:
        raise ValueError("train.lora_alpha must be positive")
    credit_raw = persisted_str(
        train.get("credit_assignment", DEFAULT_CREDIT_ASSIGNMENT),
        name="train.credit_assignment",
    )
    assert credit_raw is not None
    if algorithm in {"grpo", "opd"} and train.get("batch_size") is not None:
        raise ValueError(
            f"train.batch_size does not apply to {algorithm}; use train.prompts_per_step"
        )
    return TrainSpec(
        epochs=persisted_int(train.get("epochs"), name="train.epochs", optional=True),
        lora_rank=rank,
        lora_alpha=alpha,
        init_from_adapter=persisted_str(
            train.get("init_from_adapter", ""), name="train.init_from_adapter"
        ),
        init_from_adapter_revision=persisted_str(
            train.get("init_from_adapter_revision", ""),
            name="train.init_from_adapter_revision",
        ),
        hf_repo=persisted_str(train.get("hf_repo", ""), name="train.hf_repo"),
        learning_rate=persisted_float(
            train.get("learning_rate"), name="train.learning_rate", optional=True
        ),
        batch_size=persisted_int(train.get("batch_size"), name="train.batch_size", optional=True),
        prompts_per_step=persisted_int(
            train.get("prompts_per_step"), name="train.prompts_per_step", optional=True
        ),
        max_context_tokens=persisted_int(
            train.get("max_context_tokens"), name="train.max_context_tokens", optional=True
        ),
        save_every=persisted_int(train.get("save_every"), name="train.save_every", optional=True),
        max_steps=parse_max_steps(
            persisted_int(train.get("max_steps"), name="train.max_steps", optional=True)
        ),
        save_at_steps=persisted_sequence(
            train.get("save_at_steps", ()), name="train.save_at_steps", entry_type=int
        ),
        max_examples=persisted_int(
            train.get("max_examples"), name="train.max_examples", optional=True
        ),
        group_size=persisted_int(train.get("group_size"), name="train.group_size", optional=True),
        temperature=persisted_float(
            train.get("temperature"), name="train.temperature", optional=True
        ),
        max_completion_tokens=persisted_int(
            train.get("max_completion_tokens"),
            name="train.max_completion_tokens",
            optional=True,
        ),
        kl_penalty_coef=persisted_float(
            train.get("kl_penalty_coef"), name="train.kl_penalty_coef", optional=True
        ),
        entropy_quantile=persisted_float(
            train.get("entropy_quantile"), name="train.entropy_quantile", optional=True
        ),
        thinking_length_penalty_coef=persisted_float(
            train.get("thinking_length_penalty_coef"),
            name="train.thinking_length_penalty_coef",
            optional=True,
        ),
        teacher_model=persisted_str(train.get("teacher_model", ""), name="train.teacher_model"),
        stop_sequences=persisted_sequence(
            train.get("stop_sequences", ()), name="train.stop_sequences", entry_type=str
        ),
        structured_outputs=persisted_str(
            train.get("structured_outputs", ""), name="train.structured_outputs"
        ),
        credit_assignment=_coerce_credit_assignment(credit_raw),
    )


def _decode_gpu(data: dict[str, Any]) -> Any:
    from dataclasses import fields

    from flash.core.spec import GpuSpec, _parse_persisted_gpu_types

    gpu = validated_section(data, "gpu", {item.name for item in fields(GpuSpec)})
    gpu_type, fallbacks = _parse_persisted_gpu_types(gpu)
    provider, providers = validated_persisted_providers(gpu, gpu_type, fallbacks)
    return GpuSpec(
        type=gpu_type,
        provider=provider,
        providers=providers,
        disk_gb=persisted_int(gpu.get("disk_gb", 60), name="gpu.disk_gb"),
        max_wall_seconds=persisted_int(
            gpu.get("max_wall_seconds", 24 * 3600), name="gpu.max_wall_seconds"
        ),
        max_retries=persisted_int(gpu.get("max_retries", 5), name="gpu.max_retries"),
        network_volume=persisted_str(
            gpu.get("network_volume"), name="gpu.network_volume", optional=True
        ),
        network_volume_gb=persisted_int(
            gpu.get("network_volume_gb", 100), name="gpu.network_volume_gb"
        ),
        count=persisted_int(gpu.get("count", 1), name="gpu.count"),
        type_fallbacks=fallbacks,
    )


def _decode_wandb(data: dict[str, Any]) -> Any:
    from dataclasses import fields

    from flash.core.spec import WandbSpec

    raw = data.get("wandb", {})
    if not isinstance(raw, dict):
        raise TypeError("wandb must be an object")
    unknown = sorted(set(raw) - {item.name for item in fields(WandbSpec)})
    if unknown:
        raise ValueError(f"wandb has unknown key(s): {', '.join(unknown)}")
    return WandbSpec(
        project=persisted_str(raw.get("project"), name="wandb.project", optional=True),
        run_name=persisted_str(raw.get("run_name"), name="wandb.run_name", optional=True),
    )


def decode_persisted_job_spec(data: dict[str, Any]) -> Any:
    """Decode one complete persisted JobSpec without current catalog activation checks."""
    from dataclasses import fields

    from flash.core.catalog import normalize_algorithm
    from flash.core.spec import (
        FIXED_SEED,
        JobSpec,
        _model_revision,
        parse_seed,
        require_project_id,
    )

    if not isinstance(data, dict):
        raise TypeError("job spec must be an object")
    unknown = sorted(set(data) - {item.name for item in fields(JobSpec)})
    if unknown:
        raise ValueError(f"job spec has unknown key(s): {', '.join(unknown)}")
    model = persisted_str(data.get("model", JobSpec.model), name="model")
    algorithm_raw = persisted_str(data.get("algorithm", JobSpec.algorithm), name="algorithm")
    project_raw = persisted_str(data.get("project", ""), name="project")
    assert model is not None
    assert algorithm_raw is not None
    assert project_raw is not None
    algorithm = normalize_algorithm(algorithm_raw)
    if algorithm != algorithm_raw:
        raise ValueError("algorithm must use its canonical lowercase spelling")
    model_revision = _model_revision(
        persisted_str(data.get("model_revision", JobSpec.model_revision), name="model_revision")
    )
    model_revision_auto = persisted_bool(
        data.get("model_revision_auto", False), name="model_revision_auto"
    )
    if model_revision and not model_revision_auto:
        raise ValueError("model_revision requires model_revision_auto=True")
    spec = JobSpec(
        model=model,
        model_revision=model_revision,
        algorithm=algorithm,
        environment=_decode_environment(data),
        train=_decode_train(data, algorithm),
        gpu=_decode_gpu(data),
        run_id=persisted_str(data.get("run_id", "local"), name="run_id"),
        thinking=persisted_bool(data.get("thinking", False), name="thinking"),
        wandb=_decode_wandb(data),
        seed=parse_seed(persisted_int(data.get("seed", FIXED_SEED), name="seed")),
        model_revision_auto=model_revision_auto,
        model_revision_force_pin=persisted_bool(
            data.get("model_revision_force_pin", False), name="model_revision_force_pin"
        ),
        gpu_count_auto=persisted_bool(data.get("gpu_count_auto", False), name="gpu_count_auto"),
        workload_profile_input_digest=persisted_str(
            data.get("workload_profile_input_digest", ""),
            name="workload_profile_input_digest",
        ),
        workload_profile_producer_version=persisted_str(
            data.get("workload_profile_producer_version", ""),
            name="workload_profile_producer_version",
        ),
        workload_profile=persisted_dict(data.get("workload_profile", {}), name="workload_profile"),
        project=require_project_id(project_raw) if project_raw.strip() else "",
    )
    validate_resolved_spec_semantics(spec)
    return spec
