"""Parse Flash TOML configs into worker JobSpecs."""

from __future__ import annotations

import re
import sys
import tomllib
from collections.abc import Collection
from dataclasses import fields as dataclass_fields
from typing import Any

from flash.catalog import normalize_algorithm, resolve_model, serving_lora_rank_cap
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
    _environment_secrets,
    _require_environment_ref,
    _section_int,
    _train_float,
    _train_int,
    _train_stops,
    _train_structured_outputs,
    _train_teacher,
    _wandb_spec,
    _worker_env,
)
from flash.spec import (
    FIXED_SEED,
    EnvironmentSpec,
    GpuSpec,
    JobSpec,
    TrainSpec,
    parse_seed,
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
    """Normalize one env-name segment to the shared grammar ``[a-z0-9][a-z0-9._-]*``.

    Lowercases, collapses runs of other characters to ``-``, strips edge dashes. Returns None
    when nothing usable remains (empty, ``.``/``..``, or no alphanumeric). Shared by the CLI's
    pre-publish name normalization and the server's authoritative publish-slug validation.
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


def spec_from_file(
    path: str,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    extra_configs: list[str] | None = None,
) -> JobSpec:
    spec, _ = spec_and_train_keys_from_file(
        path,
        run_id=run_id,
        overrides=overrides,
        extra_configs=extra_configs,
    )
    return spec


def spec_and_train_keys_from_file(
    path: str,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    extra_configs: list[str] | None = None,
) -> tuple[JobSpec, frozenset[str]]:
    """parse a config and retain the raw authored [train] keys for schema preflights."""
    raw = load_toml(path)
    for extra in extra_configs or []:
        _deep_merge(raw, load_toml(extra))
    for item in overrides or []:
        _apply_override(raw, item)
    train_raw = raw.get("train")
    authored_train_keys = frozenset(train_raw) if isinstance(train_raw, dict) else frozenset()
    return spec_from_dict(raw, run_id=run_id), authored_train_keys


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
        "`<run_id>/step-N` (warm-start from a checkpoint listed by `flash checkpoints`)"
    )


# unknown tables are rejected loudly: a stray [grpo] table silently dropped grpo knobs and trained
# at 16x-cost defaults. managed fields remain recognized in their canonical sections so a
# round trip through public serialization does not fail re-validation on submit.
_TOP_LEVEL_KEYS = frozenset(
    {
        "model",
        "model_revision",
        "algorithm",
        "model_policy",
        "thinking",
        "seed",
        "environment",
        "train",
        "gpu",
        "worker_env",
        "wandb",
        "run_id",
    }
)
_GPU_KEYS = frozenset(item.name for item in dataclass_fields(GpuSpec))
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


def spec_from_dict(raw: dict[str, Any], run_id: str | None = None) -> JobSpec:
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        hint = ""
        if {"grpo", "sft", "opd"} & set(unknown):
            hint = (
                " - GRPO/SFT/opd knobs (group_size, batch_size, max_completion_tokens, ...) "
                "belong under [train], not a [grpo]/[sft]/[opd] table"
            )
        noun = "section(s)" if any(isinstance(raw[key], dict) for key in unknown) else "key(s)"
        raise ConfigError(
            f"unknown config {noun}: {', '.join(unknown)} "
            f"(allowed tables: environment, train, gpu, wandb, worker_env){hint}"
        )
    try:
        model = raw["model"]
    except KeyError as exc:
        raise ConfigError("config must set `model`") from exc
    # An unhashable model (TOML array / `[model]` table) would TypeError on MODELS.get() downstream,
    # escaping the callers' ConfigError/ValueError guards -> 500; type-check like the other scalars.
    if not isinstance(model, str) or not model.strip():
        raise ConfigError('config `model` must be a model id string (e.g. "Qwen/Qwen3.5-4B")')
    model_revision_raw = raw.get("model_revision", "")
    if not isinstance(model_revision_raw, str):
        raise ConfigError("model_revision must be a string")
    model_revision = model_revision_raw.strip()

    try:
        algorithm = normalize_algorithm(raw.get("algorithm"))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    model_policy = "catalog"  # not a user knob; "allow" path exists for internal use only
    thinking = raw.get("thinking", False)
    if not isinstance(thinking, bool):
        raise ConfigError("thinking must be a boolean")

    # Use `is None` not `or {}`: a present-but-non-dict value (e.g. `environment = false`) must hit the type check.
    env_raw = raw.get("environment")
    if env_raw is None:
        env_raw = {}
    if not isinstance(env_raw, dict):
        raise ConfigError("[environment] must be a table")
    # Validate the [environment] sub-fields before they reach EnvironmentSpec(...). The
    # constructor's ``dict(... or {})`` / ``tuple(str(p) for p in ... or ())`` papers over a falsy
    # value (false -> {}/()) but a present-but-wrong-typed value otherwise crashes opaquely or
    # silently misbehaves: ``params = "x"`` -> ``dict("x")`` ValueError, ``params = 1`` ->
    # ``dict(1)`` TypeError (a 500), and ``pip = "x"`` is char-split into ('x',) (the worker then
    # tries to install bogus one-char packages). A MISSING sub-field — absent OR ``None`` (e.g.
    # JSON ``null``) — keeps its default; any present, NON-None value must be the right type. A
    # falsy ``params = false`` is still rejected, mirroring the section-level rule that
    # ``environment = false`` must fail rather than silently coerce. Mirrors the ``must be a
    # table`` style; a string is never char-split.
    if env_raw.get("params") is not None and not isinstance(env_raw["params"], dict):
        raise ConfigError("[environment] params must be a table")
    if env_raw.get("pip") is not None and not isinstance(env_raw["pip"], (list, tuple)):
        raise ConfigError("[environment] pip must be a list of strings")
    if env_raw.get("pip") is not None and not all(isinstance(p, str) for p in env_raw["pip"]):
        raise ConfigError("[environment] pip entries must be strings")
    environment_secrets = _environment_secrets(env_raw.get("secrets"))
    train_raw = raw.get("train")
    if train_raw is None:
        train_raw = {}
    if not isinstance(train_raw, dict):
        raise ConfigError("[train] must be a table")
    validate_train_keys(train_raw)
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
    gpu_max_retries = _section_int(gpu_raw, "gpu", "max_retries", minimum=0)
    gpu_max_wall_seconds = _section_int(gpu_raw, "gpu", "max_wall_seconds", minimum=60)
    gpu_options = {}
    if gpu_max_retries is not None:
        gpu_options["max_retries"] = gpu_max_retries
    if gpu_max_wall_seconds is not None:
        gpu_options["max_wall_seconds"] = gpu_max_wall_seconds

    provider_raw = gpu_raw.get("provider", "")
    if not isinstance(provider_raw, str):
        raise ConfigError("gpu.provider must be a string")
    gpu_provider = provider_raw.strip().lower()
    if gpu_provider and gpu_provider not in PROVIDER_NAMES:
        raise ConfigError(
            f"gpu.provider must be one of {', '.join(PROVIDER_NAMES)}; got {provider_raw!r}"
        )

    exact_type_raw = gpu_raw.get("exact_type", "")
    if not isinstance(exact_type_raw, str):
        raise ConfigError("gpu.exact_type must be a string")
    exact_type = ""
    if exact_type_raw.strip():
        try:
            exact_type = canonical_gpu(exact_type_raw)
        except UnsupportedGpuError as exc:
            raise ConfigError(f"gpu.exact_type: {exc}") from exc
        exact_info = GPU_INFO.get(exact_type)
        if exact_info is None or not exact_info.validated:
            raise ConfigError(
                f"gpu.exact_type {exact_type!r} must name an active validated GPU class"
            )
        if gpu_provider and gpu_provider not in providers_for(exact_type):
            raise ConfigError(
                f"gpu.provider {gpu_provider!r} cannot provision gpu.exact_type {exact_type!r}"
            )

    try:
        # offline sizing/display only; allocator re-resolves at submit time.
        gpu_type = provisional_gpu(model, algorithm=algorithm, train=train_raw, thinking=thinking)
        if exact_type and not model_revision:
            from flash.providers.allocator import required_vram_gb

            required_vram = required_vram_gb(
                model,
                algorithm,
                train=train_raw,
                thinking=thinking,
            )
            if get_gpu_info(exact_type).vram_gb < required_vram:
                raise ConfigError(
                    f"gpu.exact_type {exact_type!r} has {get_gpu_info(exact_type).vram_gb} GB VRAM, "
                    f"but this run requires at least {required_vram} GB"
                )
    except (UnsupportedGpuError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    try:
        info = resolve_model(model, algorithm, policy=model_policy, gpu=exact_type or gpu_type)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if thinking and info.thinking == "none":
        raise ConfigError(
            f"{model} does not support thinking mode (its chat template has no "
            f"<think> support); pick a thinking-capable model — `flash models` lists "
            f"each model's thinking capability"
        )
    if not thinking and info.thinking == "always":
        raise ConfigError(
            f"{model} always emits <think> reasoning and cannot run with thinking "
            f"disabled; set thinking = true"
        )
    if thinking and info.thinking == "unknown":
        # stderr keeps stdout clean for machine-readable callers
        print(
            f"warning: open-model policy: cannot verify that {model}'s chat template "
            f"supports thinking mode; the run proceeds with enable_thinking=true",
            file=sys.stderr,
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
    lora_rank = _train_int(train_raw, "lora_rank", minimum=1) or 32
    max_lora_rank = serving_lora_rank_cap(info)
    if not init_from_adapter and max_lora_rank is not None and lora_rank > max_lora_rank:
        raise ConfigError(
            f"train.lora_rank={lora_rank} exceeds {model}'s serving max_lora_rank="
            f"{max_lora_rank}; lower train.lora_rank or raise the serving cap "
            "after real-GPU validation"
        )

    worker_env = _worker_env(raw.get("worker_env"))
    wandb_spec = _wandb_spec(raw.get("wandb"))

    try:
        train_spec = TrainSpec(
            epochs=_train_int(train_raw, "epochs", minimum=1),
            lora_rank=lora_rank,
            lora_alpha=_train_int(train_raw, "lora_alpha", minimum=1) or 64,
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
            advantage_clip=_train_float(train_raw, "advantage_clip", minimum=0.0),
            thinking_length_penalty_coef=_train_float(
                train_raw, "thinking_length_penalty_coef", minimum=0.0, maximum=1.0
            ),
            # opd-only managed teacher alias, validated against the allow-list; "" -> default glm 5.2.
            teacher_model=_train_teacher(train_raw),
            stop_sequences=_train_stops(train_raw),
            structured_outputs=_train_structured_outputs(train_raw),
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
            pip=tuple(str(p) for p in env_raw.get("pip") or ()),
            secrets=environment_secrets,
        ),
        train=train_spec,
        gpu=GpuSpec(
            type=gpu_type,
            provider=gpu_provider,
            exact_type=exact_type,
            **gpu_options,
        ),
        run_id=run_id or "local",  # server-assigned at create_run; never user-set
        seed=_job_seed(raw),
        worker_env=worker_env,
        model_policy=model_policy,
        thinking=thinking,
        wandb=wandb_spec,
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


def _validate_sft(spec: JobSpec) -> None:
    """validate sft row-count and structured-output constraints."""
    if int(spec.train.max_examples or 0) <= 0:
        raise ConfigError(
            "train.max_examples must be set to a positive row count for SFT "
            "(use the full dataset row count for an uncapped run)"
        )
    if spec.train.structured_outputs:
        # SFT never generates — a constraint here would silently do nothing; reject at parse time
        # like other no-op configs (see the opd kl_penalty_coef=0 guard).
        raise ConfigError(
            "train.structured_outputs only applies to rollout algorithms (grpo, opd); "
            "SFT trains on dataset completions and never generates"
        )


def _validate_grpo(spec: JobSpec) -> None:
    """validate the grpo group-size constraint."""
    if spec.train.group_size is not None and spec.train.group_size < 2:
        raise ConfigError(
            "train.group_size must be >= 2 for GRPO (TRL needs at least two generations "
            "per prompt to calculate advantages)"
        )


def _validate_opd(spec: JobSpec) -> None:
    """validate opd-specific training constraints."""
    if spec.train.kl_penalty_coef == 0.0:
        raise ConfigError(
            "[train] kl_penalty_coef must be > 0 for opd because it scales the gkd "
            "distillation objective; omit it to use the default 1.0 or set a positive value"
        )
    if spec.train.max_context_tokens:
        # Mirror run_opd's prompt-budget guard at PARSE time: a context budget that leaves no room
        # for any prompt after the completion budget is rejected here, BEFORE a paid worker is
        # provisioned (wait_for_gpu + model prefetch + tokenizer/adapter load), instead of failing
        # deterministically only after GPU setup. max_completion resolves exactly as the worker does:
        # explicit [train] max_completion_tokens, else the recipe thinking/non-thinking default.
        from flash.engine.vram import opd_completion_len

        max_completion = opd_completion_len(spec.train.max_completion_tokens, spec.thinking)
        if spec.train.max_context_tokens - max_completion < 1:
            raise ConfigError(
                f"[train] max_context_tokens ({spec.train.max_context_tokens}) leaves no prompt budget "
                f"after max_completion_tokens ({max_completion}) for opd; set "
                f"max_context_tokens > max_completion_tokens. Rejected at parse time so an "
                f"invalid budget fails before a GPU worker is provisioned."
            )

        # short hybrid-mamba contexts are safe because the worker compares vllm's derived scheduler
        # budget with the catalogued block size and supplies an explicit floor only when required.


# Each algorithm's spec-level contract lives in ONE validator, dispatched by name so a new algorithm
# adds a function + a map entry rather than another scattered ``if spec.algorithm == ...`` block.
_ALGO_VALIDATORS = {"sft": _validate_sft, "grpo": _validate_grpo, "opd": _validate_opd}


def _validate_spec(spec: JobSpec) -> None:
    try:
        canonical_gpu(spec.gpu.type)
    except UnsupportedGpuError as exc:
        raise ConfigError(str(exc)) from exc
    if spec.gpu.provider and spec.gpu.provider not in PROVIDER_NAMES:
        raise ConfigError(f"unknown gpu.provider {spec.gpu.provider!r}")
    if spec.gpu.exact_type:
        exact = GPU_INFO.get(spec.gpu.exact_type)
        if exact is None or not exact.validated:
            raise ConfigError("gpu.exact_type must name an active validated GPU class")
        if spec.gpu.provider and spec.gpu.provider not in providers_for(spec.gpu.exact_type):
            raise ConfigError(
                f"gpu.provider {spec.gpu.provider!r} cannot provision "
                f"gpu.exact_type {spec.gpu.exact_type!r}"
            )
    validator = _ALGO_VALIDATORS.get(spec.algorithm)
    if validator is not None:
        validator(spec)
    if not spec.environment.id:
        raise ConfigError(
            "config must set [environment] id (upload an environment with "
            '`flash env push --name <name>` and paste the returned id, e.g. "your-name/your-env"); '
            "there is no local path mode"
        )
    _require_environment_ref(
        spec.environment.id,
        '[environment] id must be a Freesolo environment id (for example "your-name/your-env")',
    )
