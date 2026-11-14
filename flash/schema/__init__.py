"""Parse Flash TOML configs into worker JobSpecs."""

from __future__ import annotations

import re
import sys
import tomllib
from collections.abc import Collection, Mapping
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any

from flash._internal.channel import CLI_NAME
from flash.core.catalog import (
    normalize_algorithm,
    resolve_model,
    serving_lora_rank_cap,
)
from flash.core.grpo import resolve_grpo_rollout_shape
from flash.core.spec import (
    FIXED_SEED,
    MANAGED_ENVIRONMENT_KEYS,
    MANAGED_GPU_KEYS,
    MANAGED_TOP_LEVEL_KEYS,
    EnvironmentSpec,
    GpuSpec,
    JobSpec,
    TrainSpec,
    WandbSpec,
    parse_seed,
    require_project_id,
)
from flash.providers.core.base import (
    UnsupportedGpuError,
    providers_for,
    provisional_gpu,
    provisional_gpu_count,
)
from flash.providers.core.registry import PROVIDER_NAMES, validated_provider_preferences
from flash.schema.fields import (
    ConfigError,
    _coerce_scalar,
    _environment_pip,
    _environment_secrets,
    _require_environment_ref,
    _train_credit_assignment,
    _train_float,
    _train_int,
    _train_stops,
    _train_structured_outputs,
    _wandb_spec,
)

# the smallest rank the parser accepts, and so the smallest a source adapter can turn out to have.
# unresolved warm starts use it instead of the serialization default for permissive client-side
# sizing. it is a vram lower bound, not a dollar lower bound because cost-ranked hardware selection
# is non-monotonic across shapes. keep it aligned with the `lora_rank` parser floor.
MIN_LORA_RANK = 1

_OWNER_REPO_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_RUN_ID_RE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
_CHECKPOINT_REF_RE = re.compile(
    rf"^(?P<run_id>{_RUN_ID_RE})/(?:(?P<final>final)|step-(?P<step>0|[1-9][0-9]{{0,17}}))$"
)
# INTERNAL artifact-store locator (`<owner>/<repo>:<phase>/<run_id>[/checkpoints/step-N]`); built by
# the control plane from run metadata and consumed by the worker — not accepted from users anywhere.
_ADAPTER_STORAGE_REF_RE = re.compile(
    rf"^(?P<repo>{_OWNER_REPO_RE}/{_OWNER_REPO_RE}):(?P<phase>sft|rl|custom)/"
    rf"(?P<run_id>{_RUN_ID_RE})(?P<checkpoint>/checkpoints/step-\d+)?$"
)


def parse_checkpoint_ref(text: str) -> tuple[str, int | None] | None:
    """parse `<run_id>/final` or `<run_id>/step-N` into its canonical components."""
    if not isinstance(text, str):
        return None
    match = _CHECKPOINT_REF_RE.fullmatch(text)
    if match is None:
        return None
    step = match.group("step")
    return match.group("run_id"), int(step) if step is not None else None


def format_checkpoint_ref(run_id: str, step: int | None = None) -> str:
    """format and validate one canonical permanent checkpoint identity."""
    if not isinstance(run_id, str) or re.fullmatch(_RUN_ID_RE, run_id) is None:
        raise ValueError("invalid run_id for checkpoint identity")
    if step is None:
        checkpoint_id = f"{run_id}/final"
    else:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        checkpoint_id = f"{run_id}/step-{step}"
    if parse_checkpoint_ref(checkpoint_id) is None:
        raise ValueError("invalid checkpoint identity components")
    return checkpoint_id


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
    segment = re.sub(r"[^a-z0-9._-]+", "-", str(value or "").lower()).lstrip("._-").rstrip("-")
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
            f"config file not found: {path} (run `{CLI_NAME} env setup` to scaffold configs/sft.toml)"
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
    """Parse a new training request and retain authored [train] keys for schema preflights."""
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
        try:
            node[leaf] = tomllib.loads(f"value = {val}")["value"]
        except tomllib.TOMLDecodeError:
            # Preserve the existing shorthand [STOP1,STOP2]. Quoted TOML values
            # above must be parsed before splitting, since they may contain commas.
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
        "train.init_from_adapter must be `<run_id>/final` or `<run_id>/step-N` "
        f"from `{CLI_NAME} runs checkpoint`"
    )


# unknown tables are rejected LOUDLY: a stray [grpo] table silently dropped grpo knobs and trained
# at 16x-cost defaults. platform-managed fields (run_id; per-section hf_repo, gpu disk/volume,
# environment resolved_sha) are assigned by the control plane, so to_dict() omits them and this
# parser rejects a user who sets them.
#
# derived rather than listed, for the same reason the gpu key set already is: what a user may author
# is exactly what the public payload carries, and `to_dict()` builds that by removing the managed
# top-level registry. spelling the survivors out by hand made this a second copy of that
# subtraction, and the two could disagree only by drifting -- a new managed field left off this list
# is a field the parser invites a user to set and the serializer then silently drops.
_TOP_LEVEL_KEYS = (
    frozenset(item.name for item in dataclass_fields(JobSpec)) - MANAGED_TOP_LEVEL_KEYS
)
# runner-assigned [gpu] fields (MANAGED_GPU_KEYS, single-sourced in flash.core.spec) are excluded from the
# user-facing surface. GpuSpec still carries them so the internal JobSpec.from_dict round trip
# preserves the runner's disk sizing, weight-cache volume, and platform retry/wall-clock policy.
# GPU class, card count, and provider are allocator-owned: the allocator picks the cheapest fitting
# shape and the runner records it in the worker spec. `spot` is the only authorable [gpu] key.
_RETIRED_GPU_KEYS = frozenset({"type", "type_fallbacks", "count", "provider", "providers"})
_GPU_KEYS = (
    frozenset(item.name for item in dataclass_fields(GpuSpec))
    - MANAGED_GPU_KEYS
    - _RETIRED_GPU_KEYS
)


def _is_derived_gpu_value(key: str, value: Any, *, spot: bool) -> bool:
    """True for the empty/derived values ``JobSpec.to_dict()`` itself emits for allocator-owned keys.

    Every stored public spec carries ``count = 1`` (the digest-stable placeholder), so resubmitting a
    spec copied from ``flash runs status`` must still parse; the value never reaches allocation.
    """
    if key == "count":
        return type(value) is int and value == 1
    if key == "type":
        return isinstance(value, str) and not value.strip()
    if key == "provider":
        return isinstance(value, str) and value.strip().lower() in (("", "spot") if spot else ("",))
    return False


# [environment] user-authorable keys, derived from environmentspec (mirrors _gpu_keys) so a new field
# is accepted automatically; resolved_sha and package are controller-managed by staging and named once
# in the managed environment registry rather than restated here.
# pip is authorable: worker_pip_for_env returns only Flash's own worker requirement, so a scorer that
# imports a third-party dependency has no other way to get it onto the worker, and the missing import
# surfaces as a zero reward at training time. The submit paths append these to worker_pip_for_env
# rather than replacing it, so the worker requirement cannot be displaced by an override.
_ENVIRONMENT_KEYS = (
    frozenset(item.name for item in dataclass_fields(EnvironmentSpec)) - MANAGED_ENVIRONMENT_KEYS
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


# the optimizer batch has a different name per algorithm because it is a different quantity, and one
# name for both was a silent trap: under sft the packaged-dataset estimate turns `batch_size` into
# examples_per_update, while under grpo the key IS prompts-per-step. so the standard sft
# out-of-memory workaround, `batch_size = 1`, copied into an rl config meant one prompt per optimizer
# update -- the run trained, logged and billed, and nothing errored. rejecting the wrong name makes
# that copy a parse error naming the right key instead of a quiet, expensive misconfiguration.
_ALGORITHM_ONLY_TRAIN_KEYS = {
    "batch_size": ("sft",),
    "prompts_per_step": ("grpo",),
}


def validate_train_keys_for_algorithm(train_raw: Mapping[str, Any], algorithm: str) -> None:
    """reject [train] keys that carry an AUTHORED value not applicable to this algorithm.

    Keyed on the value, not the key's presence. ``to_dict()`` emits the full field surface, so a
    grpo spec round-trips carrying ``batch_size: null``; rejecting on presence alone would make
    every spec fail to reparse its own output and break resubmit, warm-start and server reparse.
    A null is the absence of an authored value, which is exactly what the user is being asked for.
    """
    for key, allowed in _ALGORITHM_ONLY_TRAIN_KEYS.items():
        if train_raw.get(key) is None or algorithm in allowed:
            continue
        counterpart = next(
            other
            for other, other_allowed in _ALGORITHM_ONLY_TRAIN_KEYS.items()
            if algorithm in other_allowed
        )
        raise ConfigError(
            f"[train] {key} does not apply to {algorithm}: use {counterpart} instead. "
            f"{key} is {'/'.join(allowed)}-only, and the two are different quantities -- "
            f"{counterpart} is the optimizer batch itself, so copying an sft batch_size here "
            "would change how many prompts each update trains on."
        )


def _validate_top_level(
    raw: dict[str, Any], project_required: bool
) -> tuple[str, str, str, str, bool]:
    """Validate the top-level config section."""
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        hint = ""
        if {"grpo", "sft"} & set(unknown):
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
        raise ConfigError('config `model` must be a model id string (e.g. "Qwen/Qwen3.5-9B")')
    model_revision = ""
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
    raw: dict[str, Any], algorithm: str
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
    environment_secrets = _environment_secrets(env_raw.get("secrets"), algorithm)
    return env_raw, environment_pip, environment_secrets


def _validate_train_section(raw: dict[str, Any], algorithm: str) -> dict[str, Any]:
    """Validate the train section."""
    train_raw = raw.get("train")
    if train_raw is None:
        train_raw = {}
    if not isinstance(train_raw, dict):
        raise ConfigError("[train] must be a table")
    validate_train_keys(train_raw)
    if algorithm == "custom":
        from flash.schema.custom import TRAIN_DEFAULTS, TRAIN_KEYS

        if set(train_raw) - TRAIN_KEYS:
            raise ConfigError(f"[train] custom mode accepts only: {', '.join(sorted(TRAIN_KEYS))}")
        defaults = dict(TRAIN_DEFAULTS)
        if train_raw.get("init_from_adapter"):
            # a warm start inherits the source adapter's rank and alpha.
            defaults.pop("lora_rank")
        return {**defaults, **train_raw}
    validate_train_keys_for_algorithm(train_raw, algorithm)
    return train_raw


def _validate_gpu_section(
    raw: dict[str, Any],
    *,
    model: str,
    model_revision: str,
    algorithm: str,
    train_raw: dict[str, Any],
    init_from_adapter: str,
    thinking: bool,
) -> tuple[GpuSpec, bool]:
    """Validate the GPU section and return its canonical spec plus auto-sizing provenance."""
    gpu_raw = raw.get("gpu")
    if gpu_raw is None:
        gpu_raw = {}
    if not isinstance(gpu_raw, dict):
        raise ConfigError("[gpu] must be a table")
    unknown_gpu = sorted(set(gpu_raw) - _GPU_KEYS - _RETIRED_GPU_KEYS)
    if unknown_gpu:
        raise ConfigError(
            f"[gpu] unknown key(s): {', '.join(unknown_gpu)} "
            f"(allowed: {', '.join(sorted(_GPU_KEYS))})"
        )
    spot = gpu_raw.get("spot", False)
    if type(spot) is not bool:
        raise ConfigError("gpu.spot must be a boolean")
    # GPU class, card count, and provider are allocator-owned. a stored public spec still carries
    # `type = ""`, `count = 1`, and `provider = ""` (or the "spot" provider a Spot spec derives) for
    # digest stability, so only values a user could have meant as a choice are rejected.
    retired = sorted(
        key
        for key in _RETIRED_GPU_KEYS & set(gpu_raw)
        if not _is_derived_gpu_value(key, gpu_raw[key], spot=spot)
    )
    if retired:
        raise ConfigError(
            f"[gpu] {', '.join(retired)} is no longer supported: Flash picks the GPU class, card "
            "count, and provider automatically. Delete the line; `spot = true` is the only [gpu] "
            "setting."
        )
    max_count = gpu_raw.get("max_count", 0)
    if type(max_count) is not int or not 1 <= max_count <= 8:
        if "max_count" in gpu_raw:
            raise ConfigError("gpu.max_count must be an integer from 1 to 8")
        max_count = 0
    gpu_spec = GpuSpec(provider="spot" if spot else "", spot=spot, max_count=max_count)

    # the card count is never authored: every run auto-sizes to the smallest fitting geometry-safe
    # shape, at most `max_count` cards when set. custom runs are bounded by their placement shapes
    # (one Spot card, or up to two Modal cards).
    requested_gpu_count = max_count or None
    if algorithm == "custom":
        from flash.schema.custom import validate_custom_gpu_count

        custom_count = validate_custom_gpu_count(model, gpu_spec)
        requested_gpu_count = (
            custom_count
            if requested_gpu_count is None or custom_count is None
            else min(custom_count, requested_gpu_count)
        )
        if max_count and requested_gpu_count is not None:
            # store the ceiling clipped to the custom placement shape (one Spot card, two Modal
            # cards) so allocation's custom shape check never sees an unsupported exact width.
            gpu_spec = replace(gpu_spec, max_count=requested_gpu_count)
    unresolved_warmstart_rank = bool(init_from_adapter) and "lora_rank" not in train_raw
    # the client cannot resolve a run id to the source adapter metadata, so rank 32 is only a
    # serialization placeholder here. keep geometry checks and reject shapes that cannot fit even at
    # the minimum valid rank; the server replaces this value from adapter_config.json before its
    # authoritative allocation-time vram check.
    preflight_train = (
        {**train_raw, "lora_rank": MIN_LORA_RANK} if unresolved_warmstart_rank else train_raw
    )
    try:
        preflight_gpu_count = provisional_gpu_count(
            model,
            algorithm,
            train=preflight_train,
            thinking=thinking,
            geometry_model_revision=model_revision,
            gpu_count=requested_gpu_count,
        )
        # called for its rejection, not its return: it raises when no validated class can hold the
        # run, which is the parse-time "this is unplaceable" gate. every count reaches this boundary
        # after the geometry cap, so an unsafe eight-card width cannot leak into sizing.
        provisional_gpu(
            model,
            algorithm=algorithm,
            train=preflight_train,
            thinking=thinking,
            geometry_model_revision=model_revision,
            gpu_count=preflight_gpu_count,
            authored_gpu_ceiling=requested_gpu_count,
        )
    except OverflowError as exc:
        raise ConfigError("training dimensions exceed the supported numeric range") from exc
    except (UnsupportedGpuError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    return gpu_spec, True


def _validate_algorithm_model_consistency(
    model: str,
    algorithm: str,
    thinking: bool,
    train_raw: dict[str, Any],
    *,
    init_from_adapter: str,
) -> tuple[str, int, int]:
    """Validate algorithm and model-info consistency."""
    try:
        info = resolve_model(model, algorithm)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if thinking and info.thinking == "none":
        raise ConfigError(
            f"{model} does not support thinking mode (its chat template has no "
            f"<think> support); pick a thinking-capable model — `{CLI_NAME} models list` lists "
            f"each model's thinking capability"
        )
    if not thinking and info.thinking == "always":
        raise ConfigError(
            f"{model} always emits <think> reasoning and cannot run with thinking "
            f"disabled; set thinking = true"
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
    lora_rank = _train_int(train_raw, "lora_rank", minimum=MIN_LORA_RANK) or 32
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
    from flash.schema.custom import parse_custom_section

    custom = parse_custom_section(raw.get("custom"), algorithm)
    env_raw, environment_pip, environment_secrets = _validate_environment_section(raw, algorithm)
    train_raw = _validate_train_section(raw, algorithm)
    if algorithm == "grpo":
        try:
            resolve_grpo_rollout_shape(
                train_raw.get("prompts_per_step"),
                train_raw.get("group_size"),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
    init_from_adapter = _init_from_adapter_ref(train_raw)
    gpu_spec, gpu_count_auto = _validate_gpu_section(
        raw,
        model=model,
        model_revision=model_revision,
        algorithm=algorithm,
        train_raw=train_raw,
        init_from_adapter=init_from_adapter,
        thinking=thinking,
    )
    if custom:
        gpu_spec = replace(gpu_spec, max_retries=0, max_wall_seconds=custom["max_wall_seconds"])
    init_from_adapter, lora_rank, lora_alpha = _validate_algorithm_model_consistency(
        model,
        algorithm,
        thinking,
        train_raw,
        init_from_adapter=init_from_adapter,
    )
    wandb_spec = _validate_wandb_section(raw)

    try:
        train_spec = TrainSpec(
            epochs=_train_int(train_raw, "epochs", minimum=1),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            init_from_adapter=init_from_adapter,
            hf_repo="",  # assigned server-side; see submit_job._assign_managed_artifact_repo
            learning_rate=_train_float(train_raw, "learning_rate", minimum=0.0, exclusive=True),
            batch_size=_train_int(train_raw, "batch_size", minimum=1),
            prompts_per_step=_train_int(train_raw, "prompts_per_step", minimum=1),
            max_context_tokens=_train_int(train_raw, "max_context_tokens", minimum=1),
            save_every=_train_int(train_raw, "save_every", minimum=1),
            group_size=_train_int(train_raw, "group_size", minimum=1),
            temperature=_train_float(train_raw, "temperature", minimum=0.0),
            max_completion_tokens=_train_int(train_raw, "max_completion_tokens", minimum=1),
            mask_truncated_completions=train_raw.get("mask_truncated_completions"),
            kl_penalty_coef=_train_float(train_raw, "kl_penalty_coef", minimum=0.0),
            thinking_length_penalty_coef=_train_float(
                train_raw, "thinking_length_penalty_coef", minimum=0.0, maximum=1.0
            ),
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
        gpu_count_auto=gpu_count_auto,
        algorithm=algorithm,
        environment=EnvironmentSpec(
            id=str(env_raw.get("id") or ""),
            params=dict(env_raw.get("params") or {}),
            pip=environment_pip,
            secrets=environment_secrets,
        ),
        train=train_spec,
        gpu=gpu_spec,
        run_id=run_id or "local",  # server-assigned at create_run; never user-set
        seed=_job_seed(raw),
        thinking=thinking,
        wandb=wandb_spec,
        project=project,
        custom=custom,
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
        "mask_truncated_completions": "only applies to single-turn grpo rollouts",
        "structured_outputs": (
            "only applies to grpo; SFT trains on dataset completions and never generates"
        ),
        "group_size": (
            "only applies to grpo; it sizes the generations per prompt, and SFT never generates"
        ),
        "temperature": (
            "only applies to grpo; SFT trains on dataset completions and never samples"
        ),
        "max_completion_tokens": (
            "only applies to grpo; SFT never generates, so cap the "
            "training rows with max_context_tokens instead"
        ),
        "kl_penalty_coef": ("only applies to grpo; SFT's objective has no KL term"),
        "thinking_length_penalty_coef": (
            "only applies to grpo; it penalizes generated reasoning length, and SFT never generates"
        ),
        "credit_assignment": (
            "only applies to grpo; it distributes rollout advantage across turns, and SFT has no "
            "rollouts"
        ),
        "stop_sequences": ("only applies to grpo; they bound sampling, and SFT never generates"),
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
    """validate the grpo rollout shape and prompt-budget constraints."""
    try:
        resolve_grpo_rollout_shape(spec.train.prompts_per_step, spec.train.group_size)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    _validate_on_policy_prompt_budget(spec, "grpo")


def _validate_on_policy_prompt_budget(spec: JobSpec, algorithm: str) -> None:
    """reject a context that leaves no prompt room once the completion budget is reserved.

    each algorithm resolves its completion length off its own recipe, so read it through the same
    helper its worker does, since a shared number would reject runs the worker accepts. the worker
    clamps the requested context down to the model architecture before subtracting
    (`train/rl/inputs.py`), and clamping can only shrink the budget, so
    checking the unclamped value here is never stricter than the worker's own enforcement.
    """
    if not spec.train.max_context_tokens:
        return
    from flash.engine.plan.vram import grpo_completion_len

    max_completion = grpo_completion_len(spec.train.max_completion_tokens, spec.thinking)
    if spec.train.max_context_tokens - max_completion < 1:
        raise ConfigError(
            f"[train] max_context_tokens ({spec.train.max_context_tokens}) leaves no prompt budget "
            f"after max_completion_tokens ({max_completion}) for {algorithm}; set "
            "max_context_tokens > max_completion_tokens. Rejected at parse time so an invalid "
            "budget fails before a GPU worker is provisioned."
        )


# each algorithm's spec-level contract lives in one validator, dispatched by name. sft has no entry:
# its one rule (structured_outputs) moved into _INAPPLICABLE_TRAIN_KNOBS, and it needs no row-count
# requirement because an sft quote is backed by a workload profile that materializes and tokenizes
# the real dataset -- an omitted or zero max_examples means "every row" and is measured, not guessed.
_ALGO_VALIDATORS = {
    "grpo": _validate_grpo,
}


def _validate_spec(spec: JobSpec) -> None:
    if spec.gpu.provider and spec.gpu.providers:
        raise ConfigError("gpu.provider and gpu.providers cannot both be set")
    try:
        validated_provider_preferences(spec.gpu.providers, allow_empty=True)
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    # the gpu spec owns class canonicalization and active-catalog validation. this layer only checks
    # the provider relationship, which depends on the complete job spec rather than the field itself.
    for gpu_type in spec.gpu.acceptable_types:
        if spec.gpu.provider and spec.gpu.provider not in providers_for(gpu_type):
            raise ConfigError(
                f"gpu.provider {spec.gpu.provider!r} cannot provision gpu.type {gpu_type!r}"
            )
    if spec.gpu.provider and spec.gpu.provider not in PROVIDER_NAMES:
        raise ConfigError(f"unknown gpu.provider {spec.gpu.provider!r}")
    _reject_inapplicable_train_knobs(spec)
    validator = _ALGO_VALIDATORS.get(spec.algorithm)
    if validator is not None:
        validator(spec)
    if spec.algorithm == "custom":
        from flash.schema.custom import validate_custom_spec

        validate_custom_spec(spec)
        return
    if not spec.environment.id:
        raise ConfigError(
            "config must set [environment] id (upload an environment with "
            f'`{CLI_NAME} env push --project <project-uuid> --name <name>` and paste the returned id, e.g. "your-org/your-project/your-env"); '
            "there is no local path mode"
        )
    _require_environment_ref(
        spec.environment.id,
        '[environment] id must be a Freesolo environment id (for example "your-org/your-project/your-env")',
    )
