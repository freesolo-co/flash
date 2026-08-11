"""Parse Flash TOML configs into worker JobSpecs."""

from __future__ import annotations

import re
import sys
import tomllib
from collections.abc import Collection
from dataclasses import fields as dataclass_fields
from typing import Any

from flash.core.catalog import normalize_algorithm, resolve_model, serving_lora_rank_cap
from flash.core.spec import (
    FIXED_SEED,
    MANAGED_GPU_KEYS,
    EnvironmentSpec,
    GpuSpec,
    JobSpec,
    TrainSpec,
    WandbSpec,
    parse_seed,
    require_project_id,
)
from flash.providers import PROVIDER_NAMES
from flash.providers.base import (
    GPU_INFO,
    UnsupportedGpuError,
    canonical_gpu,
    get_gpu_info,
    providers_for,
    provisional_gpu,
)
from flash.schema.fields import (
    ConfigError,
    _coerce_scalar,
    _environment_pip,
    _environment_secrets,
    _require_environment_ref,
    _section_int,
    _train_credit_assignment,
    _train_float,
    _train_int,
    _train_stops,
    _train_structured_outputs,
    _train_teacher,
    _wandb_spec,
)

_OWNER_REPO_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_RUN_ID_RE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
# canonical short checkpoint references name a run alias or a saved checkpoint. immutable adapter
# revisions additionally lock that checkpoint identity to the exact hugging face commit.
_CHECKPOINT_REF_RE = re.compile(rf"^(?P<run_id>{_RUN_ID_RE})(?:/step-(?P<step>\d{{1,18}}))?$")
_ADAPTER_REVISION_RE = re.compile(
    rf"^(?P<run_id>{_RUN_ID_RE})@(?:final|step-(?P<step>0|[1-9]\d{{0,17}}))\."
    r"(?P<hf_revision>[0-9a-f]{40})$"
)
# INTERNAL artifact-store locator (`<owner>/<repo>:<phase>/<run_id>[/checkpoints/step-N]`); built by
# the control plane from run metadata and consumed by the worker — not accepted from users anywhere.
_ADAPTER_STORAGE_REF_RE = re.compile(
    rf"^(?P<repo>{_OWNER_REPO_RE}/{_OWNER_REPO_RE}):(?P<phase>sft|rl|opd)/"
    rf"(?P<run_id>{_RUN_ID_RE})(?P<checkpoint>/checkpoints/step-\d+)?$"
)


def parse_checkpoint_ref(text: str) -> tuple[str, int | None] | None:
    """Parse the canonical short reference: `<run_id>` or `<run_id>/step-N` -> (run_id, step|None)."""
    match = _CHECKPOINT_REF_RE.fullmatch(str(text or "").strip())
    if match is None:
        return None
    step = match.group("step")
    return match.group("run_id"), int(step) if step is not None else None


def parse_adapter_revision(text: str) -> tuple[str, int | None, str] | None:
    """Parse a locked immutable adapter revision into ``(run_id, step|None, hf_revision)``."""
    match = _ADAPTER_REVISION_RE.fullmatch(str(text or "").strip())
    if match is None:
        return None
    step = match.group("step")
    return (
        match.group("run_id"),
        int(step) if step is not None else None,
        match.group("hf_revision"),
    )


def format_checkpoint_ref(run_id: str, step: int | None = None) -> str:
    """Format the canonical short reference: `<run_id>` or `<run_id>/step-N`."""
    return f"{run_id}/step-{int(step)}" if step is not None else str(run_id)


def format_adapter_revision(run_id: str, step: int | None, hf_revision: str) -> str:
    """Format and validate a canonical immutable adapter revision."""
    suffix = f"step-{int(step)}" if step is not None else "final"
    revision = f"{run_id}@{suffix}.{hf_revision}"
    if parse_adapter_revision(revision) is None:
        raise ValueError("invalid immutable adapter revision components")
    return revision


def checkpoint_storage_ref(hf_repo: str, phase: str, run_id: str, step: int | None = None) -> str:
    """Internal storage reference for a run's adapter (or one saved step) on the artifact store."""
    suffix = f"/checkpoints/step-{int(step)}" if step is not None else ""
    return f"{hf_repo}:{phase}/{run_id}{suffix}"


def parse_adapter_storage_ref(text: str) -> tuple[str, str] | None:
    """Parse an internal storage reference -> (hf_repo, artifact prefix), or None."""
    match = _ADAPTER_STORAGE_REF_RE.fullmatch(str(text or "").strip())
    if match is None:
        return None
    repo, phase, run_id, checkpoint = match.groups()
    return repo, f"{phase}/{run_id}{checkpoint or ''}"


def normalize_env_name_segment(value: str) -> str | None:
    """normalize one env-name segment to ``[a-z0-9][a-z0-9._-]*``.

    lowercase, collapse invalid runs to ``-``, strip edge dashes, and return none when no usable
    alphanumeric content remains. cli and server share this grammar.
    """
    segment = re.sub(r"[^a-z0-9._-]+", "-", str(value or "").lower()).strip("-")
    if segment in {"", ".", ".."} or not re.search(r"[a-z0-9]", segment):
        return None
    return segment


def load_toml(path: str) -> dict[str, Any]:
    # a missing path, a directory, or an unreadable file is a user mistake, not a flash bug. only
    # FileNotFoundError is in _USER_ERRORS, so IsADirectoryError / PermissionError would otherwise
    # dump a raw traceback (and FileNotFoundError still surfaced a bare [Errno 2] string). route the
    # whole OSError family through ConfigError so the cli prints one clean `error:` line instead.
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError as exc:
        raise ConfigError(
            f"config file not found: {path} (run `flash env setup` to scaffold configs/sft.toml)"
        ) from exc
    except IsADirectoryError as exc:
        raise ConfigError(
            f"{path} is a directory, not a TOML config (point train at a .toml file)"
        ) from exc
    except OSError as exc:
        # permission denied, ENOTDIR on a path component, symlink loops, etc.
        raise ConfigError(f"cannot read config {path}: {exc.strerror or exc}") from exc


def spec_and_train_keys_from_file(
    path: str,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    extra_configs: list[str] | None = None,
    *,
    project_required: bool = False,
) -> tuple[JobSpec, frozenset[str]]:
    """parse a config and retain the raw authored [train] keys for schema preflights."""
    raw = load_toml(path)
    for extra in extra_configs or []:
        _deep_merge(raw, load_toml(extra))
    for item in overrides or []:
        _apply_override(raw, item)
    train_raw = raw.get("train")
    authored_train_keys = frozenset(train_raw) if isinstance(train_raw, dict) else frozenset()
    return (
        spec_from_dict(raw, run_id=run_id, project_required=project_required),
        authored_train_keys,
    )


def _deep_merge(base: dict, extra: dict) -> None:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


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
    # support list values like pip=["a","b"]
    val = value.strip()
    # wandb leaves are string labels — don't coerce to int/bool
    if parts[0] == "wandb":
        node[leaf] = val
    elif val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        node[leaf] = [_coerce_scalar(x.strip()) for x in inner.split(",") if x.strip()]
    else:
        node[leaf] = _coerce_scalar(val)


def _job_seed(raw: dict[str, Any]) -> int:
    try:
        return parse_seed(raw.get("seed", FIXED_SEED))
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from None


def _init_from_adapter_ref(train_raw: dict[str, Any]) -> str:
    ref_raw = train_raw.get("init_from_adapter")
    if ref_raw is None:
        return ""
    if not isinstance(ref_raw, str):
        raise ConfigError("train.init_from_adapter must be a string")
    ref = ref_raw.strip()
    if not ref:
        return ""
    if parse_checkpoint_ref(ref) is not None:
        return ref
    raise ConfigError(
        "train.init_from_adapter must be `<run_id>` (continue that run's trained adapter) or "
        "`<run_id>/step-N` (warm-start from a checkpoint listed by `flash runs checkpoint`)"
    )


# unknown tables are rejected LOUDLY: a stray [grpo] table silently dropped grpo knobs and trained
# at 16x-cost defaults. platform-managed fields (run_id; per-section hf_repo, gpu disk/volume,
# environment resolved_sha) are assigned by the control plane, so to_dict() omits them and this
# parser rejects a user who sets them.
_TOP_LEVEL_KEYS = frozenset(
    {
        "model",
        "model_revision",
        "algorithm",
        "thinking",
        "seed",
        "environment",
        "train",
        "gpu",
        "wandb",
        "project",
    }
)
# runner-assigned [gpu] fields (MANAGED_GPU_KEYS, single-sourced in flash.core.spec) are excluded from the
# user-facing surface. GpuSpec still carries them so the internal JobSpec.from_dict round trip
# preserves the runner's disk sizing, weight-cache volume, and platform retry/wall-clock policy.
_GPU_KEYS = frozenset(item.name for item in dataclass_fields(GpuSpec)) - MANAGED_GPU_KEYS
# [environment] user-authorable keys, derived from EnvironmentSpec (mirrors _GPU_KEYS) so a new field
# is accepted automatically; resolved_sha is control-plane-pinned (see _assign_resolved_env_sha).
# pip is authorable: worker_pip_for_env returns only Flash's own worker requirement, so a scorer that
# imports a third-party dependency has no other way to get it onto the worker, and the missing import
# surfaces as a zero reward at training time. The submit paths append these to worker_pip_for_env
# rather than replacing it, so the worker requirement cannot be displaced by an override.
_ENV_MANAGED_KEYS = frozenset({"resolved_sha"})
_ENVIRONMENT_KEYS = (
    frozenset(item.name for item in dataclass_fields(EnvironmentSpec)) - _ENV_MANAGED_KEYS
)
TRAIN_KEY_MIN_VERSIONS = {
    item.name: str(item.metadata["introduced_in"])
    for item in dataclass_fields(TrainSpec)
    if "introduced_in" in item.metadata
}
TRAIN_SCHEMA_KEYS = frozenset(TRAIN_KEY_MIN_VERSIONS)


def train_schema_metadata() -> dict[str, str]:
    """return the deterministic [train] field-to-release mapping."""
    return {key: TRAIN_KEY_MIN_VERSIONS[key] for key in sorted(TRAIN_KEY_MIN_VERSIONS)}


def validate_train_keys(keys: Collection[str]) -> None:
    """reject names outside the canonical TrainSpec field surface."""
    unknown = sorted(set(keys) - TRAIN_SCHEMA_KEYS)
    if unknown:
        raise ConfigError(
            f"[train] unknown key(s): {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(TRAIN_SCHEMA_KEYS))})"
        )


def _validate_top_level(
    raw: dict[str, Any], project_required: bool
) -> tuple[str, str, str, str, bool]:
    """Validate the top-level config section."""
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        hint = ""
        if {"grpo", "sft", "opd"} & set(unknown):
            hint = (
                " - algorithm knobs (group_size, batch_size, max_completion_tokens, ...) "
                "belong under [train], not a per-algorithm table"
            )
        noun = "section(s)" if any(isinstance(raw[key], dict) for key in unknown) else "key(s)"
        raise ConfigError(
            f"unknown config {noun}: {', '.join(unknown)} "
            f"(allowed tables: environment, train, gpu, wandb){hint}"
        )
    try:
        model = raw["model"]
    except KeyError as exc:
        raise ConfigError("config must set `model`") from exc
    # an unhashable model (toml array / `[model]` table) would typeerror on models.get() downstream,
    # escaping the callers' configerror/valueerror guards -> 500; type-check like the other scalars.
    if not isinstance(model, str) or not model.strip():
        raise ConfigError('config `model` must be a model id string (e.g. "Qwen/Qwen3.5-4B")')
    model_revision_raw = raw.get("model_revision", "")
    if not isinstance(model_revision_raw, str):
        raise ConfigError("model_revision must be a string")
    model_revision = model_revision_raw.strip()
    project_raw = raw.get("project", "")
    try:
        if project_required:
            project = require_project_id(project_raw)
        elif "project" not in raw:
            project = ""
        elif not isinstance(project_raw, str):
            raise TypeError("project must be a string")
        elif not project_raw.strip():
            project = ""
        else:
            project = require_project_id(project_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc

    try:
        algorithm = normalize_algorithm(raw.get("algorithm"))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    thinking = raw.get("thinking", False)
    if not isinstance(thinking, bool):
        raise ConfigError("thinking must be a boolean")
    return model, model_revision, project, algorithm, thinking


def _validate_environment_section(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Validate the environment section, returning it with the parsed pip and secrets tuples."""
    # use `is none` not `or {}`: a present-but-non-dict value (e.g. `environment = false`) must hit the type check.
    env_raw = raw.get("environment")
    if env_raw is None:
        env_raw = {}
    if not isinstance(env_raw, dict):
        raise ConfigError("[environment] must be a table")
    unknown_env = sorted(set(env_raw) - _ENVIRONMENT_KEYS)
    if unknown_env:
        raise ConfigError(
            f"[environment] unknown key(s): {', '.join(unknown_env)} "
            f"(allowed: {', '.join(sorted(_ENVIRONMENT_KEYS))})"
        )
    # validate environment sub-fields before environmentspec coercion. missing or none keeps the
    # default; every present non-none value, including false, must have the correct type so malformed
    # input fails clearly instead of becoming {} or an opaque dict conversion error.
    if env_raw.get("params") is not None and not isinstance(env_raw["params"], dict):
        raise ConfigError("[environment] params must be a table")
    environment_pip = _environment_pip(env_raw.get("pip"))
    environment_secrets = _environment_secrets(env_raw.get("secrets"))
    return env_raw, environment_pip, environment_secrets


def _validate_train_section(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate the train section."""
    train_raw = raw.get("train")
    if train_raw is None:
        train_raw = {}
    if not isinstance(train_raw, dict):
        raise ConfigError("[train] must be a table")
    validate_train_keys(train_raw)
    return train_raw


def _validate_gpu_section(
    raw: dict[str, Any],
    *,
    model: str,
    model_revision: str,
    algorithm: str,
    train_raw: dict[str, Any],
    thinking: bool,
) -> tuple[str, str, dict[str, int]]:
    """Validate the gpu section."""
    gpu_raw = raw.get("gpu")
    if gpu_raw is None:
        gpu_raw = {}
    if not isinstance(gpu_raw, dict):
        raise ConfigError("[gpu] must be a table")
    unknown_gpu = sorted(set(gpu_raw) - _GPU_KEYS)
    if unknown_gpu:
        raise ConfigError(
            f"[gpu] unknown key(s): {', '.join(unknown_gpu)} "
            f"(allowed: {', '.join(sorted(_GPU_KEYS))})"
        )
    # cards a single training worker occupies (1..8); count > 1 provisions a multi-gpu pod.
    gpu_count = _section_int(gpu_raw, "gpu", "count", minimum=1, maximum=8)
    gpu_options = {}
    if gpu_count is not None:
        gpu_options["count"] = gpu_count

    provider_raw = gpu_raw.get("provider", "")
    if not isinstance(provider_raw, str):
        raise ConfigError("gpu.provider must be a string")
    gpu_provider = provider_raw.strip().lower()
    if gpu_provider and gpu_provider not in PROVIDER_NAMES:
        raise ConfigError(
            f"gpu.provider must be one of {', '.join(PROVIDER_NAMES)}; got {provider_raw!r}"
        )

    gpu_type_raw = gpu_raw.get("type", "")
    if not isinstance(gpu_type_raw, str):
        raise ConfigError("gpu.type must be a string")
    gpu_type = ""
    if gpu_type_raw.strip():
        try:
            gpu_type = canonical_gpu(gpu_type_raw)
        except UnsupportedGpuError as exc:
            raise ConfigError(f"gpu.type: {exc}") from exc
        gpu_info = GPU_INFO.get(gpu_type)
        if gpu_info is None or not gpu_info.validated:
            raise ConfigError(f"gpu.type {gpu_type!r} must name an active validated GPU class")
        if gpu_provider and gpu_provider not in providers_for(gpu_type):
            raise ConfigError(
                f"gpu.provider {gpu_provider!r} cannot provision gpu.type {gpu_type!r}"
            )

    from flash.providers.allocator import geometry_safe_gpu_cap

    preflight_gpu_count = geometry_safe_gpu_cap(
        model, gpu_count or 1, model_revision=model_revision
    )
    try:
        # called for its rejection, not its return: it raises when no validated class can hold the
        # run, which is the parse-time "this is unplaceable" gate. the class it picks is offline
        # sizing/display only -- the allocator re-resolves auto runs at submit time.
        provisional_gpu(
            model,
            algorithm=algorithm,
            train=train_raw,
            thinking=thinking,
            # sized against the shape the allocator may actually rent. sizing a --gpus n run
            # against one card rejected it here before sharding was ever considered, which made
            # the flag inert for exactly the large runs it exists to serve.
            gpu_count=preflight_gpu_count,
        )
        if gpu_type and not model_revision:
            from flash.providers.allocator import required_vram_gb

            required_vram = required_vram_gb(
                model,
                algorithm,
                train=train_raw,
                thinking=thinking,
            )
            # required_vram is the whole-run floor, so it may only be compared against a single
            # card's vram when the run is confined to a single card. above that the allocator
            # shards the run across a combination and applies its own multi-card fit test
            # (allocator.py:181), which this gate must not pre-empt: a pinned 141 gb class with
            # gpu.count=2 holds 282 gb and is rejected here on a 180 gb floor it clears.
            single_card = gpu_count is None or gpu_count <= 1
            if single_card and get_gpu_info(gpu_type).vram_gb < required_vram:
                raise ConfigError(
                    f"gpu.type {gpu_type!r} has {get_gpu_info(gpu_type).vram_gb} GB VRAM, "
                    f"but this run requires at least {required_vram} GB"
                )
    except (UnsupportedGpuError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    return gpu_type, gpu_provider, gpu_options


def _validate_algorithm_model_consistency(
    model: str, algorithm: str, thinking: bool, train_raw: dict[str, Any]
) -> tuple[str, int, int]:
    """Validate algorithm and model-info consistency."""
    try:
        info = resolve_model(model, algorithm)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if thinking and info.thinking == "none":
        raise ConfigError(
            f"{model} does not support thinking mode (its chat template has no "
            f"<think> support); pick a thinking-capable model — `flash models list` lists "
            f"each model's thinking capability"
        )
    if not thinking and info.thinking == "always":
        raise ConfigError(
            f"{model} always emits <think> reasoning and cannot run with thinking "
            f"disabled; set thinking = true"
        )
    init_from_adapter = _init_from_adapter_ref(train_raw)
    if algorithm == "sft" and init_from_adapter:
        raise ConfigError(
            "train.init_from_adapter is supported only for GRPO and OPD continue-in-place runs; "
            "SFT adapter continuation is not supported"
        )
    if init_from_adapter and "lora_rank" in train_raw:
        raise ConfigError(
            "train.lora_rank cannot be set with train.init_from_adapter because source adapter "
            "rank metadata is authoritative"
        )
    if init_from_adapter and "lora_alpha" in train_raw:
        raise ConfigError(
            "train.lora_alpha cannot be set with train.init_from_adapter because source adapter "
            "alpha metadata is authoritative"
        )
    lora_rank = _train_int(train_raw, "lora_rank", minimum=1) or 32
    # unset -> the tuned 2 x rank default. authored -> the user's value, which need not be 2 x rank.
    lora_alpha = _train_int(train_raw, "lora_alpha", minimum=1) or 2 * lora_rank
    max_lora_rank = serving_lora_rank_cap(info)
    if not init_from_adapter and max_lora_rank is not None and lora_rank > max_lora_rank:
        raise ConfigError(
            f"train.lora_rank={lora_rank} exceeds {model}'s serving max_lora_rank="
            f"{max_lora_rank}; lower train.lora_rank or raise the serving cap "
            "after real-GPU validation"
        )
    return init_from_adapter, lora_rank, lora_alpha


def _validate_wandb_section(raw: dict[str, Any]) -> WandbSpec:
    """Validate the wandb section."""
    return _wandb_spec(raw.get("wandb"))


def spec_from_dict(
    raw: dict[str, Any], run_id: str | None = None, *, project_required: bool = False
) -> JobSpec:
    model, model_revision, project, algorithm, thinking = _validate_top_level(raw, project_required)
    env_raw, environment_pip, environment_secrets = _validate_environment_section(raw)
    train_raw = _validate_train_section(raw)
    gpu_type, gpu_provider, gpu_options = _validate_gpu_section(
        raw,
        model=model,
        model_revision=model_revision,
        algorithm=algorithm,
        train_raw=train_raw,
        thinking=thinking,
    )
    init_from_adapter, lora_rank, lora_alpha = _validate_algorithm_model_consistency(
        model, algorithm, thinking, train_raw
    )
    wandb_spec = _validate_wandb_section(raw)

    try:
        train_spec = TrainSpec(
            epochs=_train_int(train_raw, "epochs", minimum=1),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            init_from_adapter=init_from_adapter,
            hf_repo="",  # assigned server-side; see submit_job._assign_managed_hf_repo
            learning_rate=_train_float(train_raw, "learning_rate", minimum=0.0, exclusive=True),
            batch_size=_train_int(train_raw, "batch_size", minimum=1),
            max_context_tokens=_train_int(train_raw, "max_context_tokens", minimum=1),
            save_every=_train_int(train_raw, "save_every", minimum=1),
            group_size=_train_int(train_raw, "group_size", minimum=1),
            temperature=_train_float(train_raw, "temperature", minimum=0.0),
            max_completion_tokens=_train_int(train_raw, "max_completion_tokens", minimum=1),
            kl_penalty_coef=_train_float(train_raw, "kl_penalty_coef", minimum=0.0),
            entropy_quantile=_train_float(train_raw, "entropy_quantile", minimum=0.0, maximum=1.0),
            thinking_length_penalty_coef=_train_float(
                train_raw, "thinking_length_penalty_coef", minimum=0.0, maximum=1.0
            ),
            # opd-only managed teacher alias, validated against the allow-list; "" -> default glm 5.2.
            teacher_model=_train_teacher(train_raw),
            stop_sequences=_train_stops(train_raw),
            structured_outputs=_train_structured_outputs(train_raw),
            credit_assignment=_train_credit_assignment(train_raw),
            # minimum=0: explicit 0 means "no cap" per trainspec contract
            max_steps=train_raw.get("max_steps"),
            max_examples=_train_int(train_raw, "max_examples", minimum=0),
            save_at_steps=train_raw.get("save_at_steps"),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc

    spec = JobSpec(
        model=model,
        model_revision=model_revision,
        algorithm=algorithm,
        environment=EnvironmentSpec(
            id=str(env_raw.get("id") or ""),
            params=dict(env_raw.get("params") or {}),
            pip=environment_pip,
            secrets=environment_secrets,
        ),
        train=train_spec,
        gpu=GpuSpec(
            type=gpu_type,
            provider=gpu_provider,
            **gpu_options,
        ),
        run_id=run_id or "local",  # server-assigned at create_run; never user-set
        seed=_job_seed(raw),
        thinking=thinking,
        wandb=wandb_spec,
        project=project,
    )
    _validate_spec(spec)
    if spec.train.structured_outputs and thinking:
        # thinking + a constraint is supported: the rollout worker sets a reasoning parser on the vLLM
        # engine so the guided grammar is held until </think>, constraining only the answer. Surface
        # the runtime behavior (informational, not a warning) so the interaction is unambiguous — the
        # <think> reasoning phase is NOT constrained.
        print(
            "note: train.structured_outputs with thinking = true constrains only the answer after "
            "the </think> reasoning phase (the model reasons freely first, via a reasoning-aware "
            "guided-decoding gate); set thinking = false to constrain from the first token instead",
            file=sys.stderr,
        )
    return spec


# map each algorithm to meaningful [train] knobs its worker cannot consume. rejecting them prevents
# shared-table rollout options from being silently ignored by sft and vice versa.
_INAPPLICABLE_TRAIN_KNOBS: dict[str, dict[str, str]] = {
    "sft": {
        "structured_outputs": (
            "only applies to rollout algorithms (grpo, opd); SFT trains on dataset completions "
            "and never generates"
        ),
        "group_size": (
            "only applies to rollout algorithms (grpo, opd); it sizes the generations per prompt, "
            "and SFT never generates"
        ),
        "temperature": (
            "only applies to rollout algorithms (grpo, opd); SFT trains on dataset completions "
            "and never samples"
        ),
        "max_completion_tokens": (
            "only applies to rollout algorithms (grpo, opd); SFT never generates, so cap the "
            "training rows with max_context_tokens instead"
        ),
        "kl_penalty_coef": (
            "only applies to rollout algorithms (grpo, opd); SFT's objective has no KL term"
        ),
        "entropy_quantile": "only applies to grpo; SFT has no rollout tokens to rank by entropy",
        "thinking_length_penalty_coef": (
            "only applies to grpo; it penalizes generated reasoning length, and SFT never generates"
        ),
        "teacher_model": "only applies to opd; SFT trains on dataset completions, not a teacher",
        "credit_assignment": (
            "only applies to grpo; it distributes rollout advantage across turns, and SFT has no "
            "rollouts"
        ),
        "stop_sequences": (
            "only applies to rollout algorithms (grpo, opd); they bound sampling, and SFT never "
            "generates"
        ),
    },
    "opd": {
        "entropy_quantile": (
            "only applies to grpo; opd distils every sampled token against the teacher rather than "
            "ranking tokens by entropy"
        ),
        "thinking_length_penalty_coef": (
            "only applies to grpo; it shapes the grpo reward, and opd optimizes a distillation "
            "objective with no reward term"
        ),
        "credit_assignment": (
            "only applies to grpo; it distributes group-relative advantage, and opd has no "
            "advantages to assign"
        ),
    },
    "grpo": {
        "teacher_model": (
            "only applies to opd; grpo optimizes its environment reward and has no teacher"
        ),
    },
}


# unset value per [train] field, read off TrainSpec so a changed default cannot silently turn an
# omitted knob into a rejection.
_TRAIN_DEFAULTS = {item.name: item.default for item in dataclass_fields(TrainSpec)}


def _reject_inapplicable_train_knobs(spec: JobSpec) -> None:
    """reject [train] knobs the selected algorithm cannot consume.

    inspect meaningful values, not presence, because serialized JobSpec dictionaries contain every
    unset field and must survive client/server and public-status re-parsing.
    """
    inapplicable = _INAPPLICABLE_TRAIN_KNOBS.get(spec.algorithm)
    if not inapplicable:
        return
    for key, reason in inapplicable.items():
        # unset sentinels differ by field (None, "", (), and credit_assignment's non-falsy
        # "per_episode"), so compare against the dataclass default rather than assuming one
        # falsy spelling. a value equal to the default is indistinguishable from unset and is
        # what a full-dict round trip carries, so it must not be rejected.
        if getattr(spec.train, key) == _TRAIN_DEFAULTS[key]:
            continue
        raise ConfigError(f"train.{key} {reason}")


def _validate_grpo(spec: JobSpec) -> None:
    """validate the grpo group-size and prompt-budget constraints."""
    if spec.train.group_size is not None and spec.train.group_size < 2:
        raise ConfigError(
            "train.group_size must be >= 2 for GRPO (advantages are group-relative, so a "
            "prompt needs at least two generations to compare against)"
        )
    _validate_on_policy_prompt_budget(spec, "grpo")


def _validate_on_policy_prompt_budget(spec: JobSpec, algorithm: str) -> None:
    """reject a context that leaves no prompt room once the completion budget is reserved.

    each algorithm resolves its completion length off its own recipe, so read it through the same
    helper its worker does, since a shared number would reject runs the worker accepts. the worker
    clamps the requested context down to the model architecture before subtracting
    (`train/rl/inputs.py`, `opd_train_runner`), and clamping can only shrink the budget, so
    checking the unclamped value here is never stricter than the worker's own enforcement.
    """
    if not spec.train.max_context_tokens:
        return
    from flash.engine.plan.vram import grpo_completion_len, opd_completion_len

    resolve = grpo_completion_len if algorithm == "grpo" else opd_completion_len
    max_completion = resolve(spec.train.max_completion_tokens, spec.thinking)
    if spec.train.max_context_tokens - max_completion < 1:
        raise ConfigError(
            f"[train] max_context_tokens ({spec.train.max_context_tokens}) leaves no prompt budget "
            f"after max_completion_tokens ({max_completion}) for {algorithm}; set "
            "max_context_tokens > max_completion_tokens. Rejected at parse time so an invalid "
            "budget fails before a GPU worker is provisioned."
        )


def _validate_opd(spec: JobSpec) -> None:
    """validate opd-specific training constraints."""
    if spec.train.kl_penalty_coef == 0.0:
        raise ConfigError(
            "[train] kl_penalty_coef must be > 0 for opd because it scales the gkd "
            "distillation objective; omit it to use the default 1.0 or set a positive value"
        )
    _validate_on_policy_prompt_budget(spec, "opd")


# each algorithm's spec-level contract lives in one validator, dispatched by name. sft has no entry:
# its one rule (structured_outputs) moved into _INAPPLICABLE_TRAIN_KNOBS, and it needs no row-count
# requirement because an sft quote is backed by a workload profile that materializes and tokenizes
# the real dataset -- an omitted or zero max_examples means "every row" and is measured, not guessed.
_ALGO_VALIDATORS = {
    "grpo": _validate_grpo,
    "opd": _validate_opd,
}


def _validate_spec(spec: JobSpec) -> None:
    if spec.gpu.type:
        try:
            canonical_gpu(spec.gpu.type)
        except UnsupportedGpuError as exc:
            raise ConfigError(str(exc)) from exc
        gpu_info = GPU_INFO.get(spec.gpu.type)
        if gpu_info is None or not gpu_info.validated:
            raise ConfigError("gpu.type must name an active validated GPU class")
        if spec.gpu.provider and spec.gpu.provider not in providers_for(spec.gpu.type):
            raise ConfigError(
                f"gpu.provider {spec.gpu.provider!r} cannot provision gpu.type {spec.gpu.type!r}"
            )
    if spec.gpu.provider and spec.gpu.provider not in PROVIDER_NAMES:
        raise ConfigError(f"unknown gpu.provider {spec.gpu.provider!r}")
    _reject_inapplicable_train_knobs(spec)
    validator = _ALGO_VALIDATORS.get(spec.algorithm)
    if validator is not None:
        validator(spec)
    if not spec.environment.id:
        raise ConfigError(
            "config must set [environment] id (upload an environment with "
            '`flash env push --project <project-uuid> --name <name>` and paste the returned id, e.g. "your-name/your-env"); '
            "there is no local path mode"
        )
    _require_environment_ref(
        spec.environment.id,
        '[environment] id must be a Freesolo environment id (for example "your-name/your-env")',
    )
