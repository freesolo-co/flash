"""Parse Flash TOML configs into worker JobSpecs."""

from __future__ import annotations

import re
import sys
import tomllib
from typing import Any

from flash.catalog import normalize_algorithm, resolve_model
from flash.providers.base import (
    UnsupportedGpuError,
    canonical_gpu,
    provisional_gpu,
)
from flash.schema.fields import (
    ConfigError,
    _coerce_scalar,
    _environment_secrets,
    _require_environment_ref,
    _train_float,
    _train_int,
    _train_stops,
    _wandb_spec,
    _worker_env,
)
from flash.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec

_OWNER_REPO_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_RUN_ID_RE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
_ADAPTER_REF_RE = re.compile(
    rf"^(?P<repo>{_OWNER_REPO_RE}/{_OWNER_REPO_RE}):(?P<phase>sft|rl)/"
    rf"(?P<run_id>{_RUN_ID_RE})$"
)


def load_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def spec_from_file(
    path: str,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    extra_configs: list[str] | None = None,
) -> JobSpec:
    raw = load_toml(path)
    for extra in extra_configs or []:
        _deep_merge(raw, load_toml(extra))
    for item in overrides or []:
        _apply_override(raw, item)
    return spec_from_dict(raw, run_id=run_id)


def _deep_merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


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


def _init_from_adapter_ref(train_raw: dict[str, Any]) -> str:
    ref_raw = train_raw.get("init_from_adapter")
    if ref_raw is None:
        return ""
    if not isinstance(ref_raw, str):
        raise ConfigError("train.init_from_adapter must be a string")
    ref = ref_raw.strip()
    if not ref:
        return ""
    if _ADAPTER_REF_RE.match(ref):
        return ref
    raise ConfigError(
        "train.init_from_adapter must be the full adapter_ref emitted by `flash status` "
        "(<owner>/<repo>:<phase>/<run_id>)"
    )


# Unknown tables are rejected loudly: a stray [grpo] table silently dropped GRPO knobs and trained
# at 16x-cost defaults. Platform-managed keys (gpu, run_id, hf_repo) remain recognized (not
# rejected) so a round-tripped JobSpec.to_dict() doesn't fail re-validation on submit.
_TOP_LEVEL_KEYS = frozenset(
    {
        "model",
        "algorithm",
        "model_policy",
        "thinking",
        "environment",
        "train",
        "gpu",
        "worker_env",
        "wandb",
        "run_id",
    }
)
_TRAIN_KEYS = frozenset(
    {
        "steps",
        "epochs",
        "lora_rank",
        "lora_alpha",
        "init_from_adapter",
        "hf_repo",
        "learning_rate",
        "batch_size",
        "max_length",
        "save_every",
        "group_size",
        "temperature",
        "max_tokens",
        "kl_penalty_coef",
        "advantage_clip",
        "thinking_length_penalty_coef",
        "stop_sequences",
        "max_steps",
        "max_examples",
    }
)
def spec_from_dict(raw: dict[str, Any], run_id: str | None = None) -> JobSpec:
    # Only reject table-valued unknowns — callers pass harmless scalar flags like dry_run alongside spec.
    unknown = sorted(k for k in set(raw) - _TOP_LEVEL_KEYS if isinstance(raw[k], dict))
    if unknown:
        hint = ""
        if {"grpo", "sft"} & set(unknown):
            hint = (
                " — GRPO/SFT knobs (group_size, batch_size, max_tokens, …) belong under [train], "
                "not a [grpo]/[sft] table"
            )
        raise ConfigError(
            f"unknown config section(s): {', '.join(unknown)} "
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
    if env_raw.get("path"):
        raise ConfigError(
            "local environment paths are no longer supported — remove `path` and reference a "
            "Freesolo environment `id` returned by `flash env push --name <name>`"
        )
    # Validate sub-fields explicitly: pip="x" would char-split into bogus packages; params=1 crashes opaquely.
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
    unknown_train = sorted(set(train_raw) - _TRAIN_KEYS)
    if unknown_train:
        raise ConfigError(
            f"[train] unknown key(s): {', '.join(unknown_train)} "
            f"(allowed: {', '.join(sorted(_TRAIN_KEYS))})"
        )
    gpu_raw = raw.get("gpu")
    if gpu_raw is None:
        gpu_raw = {}
    if not isinstance(gpu_raw, dict):
        raise ConfigError("[gpu] must be a table")

    try:
        # Offline sizing/display only; allocator re-resolves at submit time.
        gpu_type = provisional_gpu(model, algorithm=algorithm, train=train_raw, thinking=thinking)
    except UnsupportedGpuError as exc:
        raise ConfigError(str(exc)) from exc
    try:
        info = resolve_model(model, algorithm, policy=model_policy, gpu=gpu_type)
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

    worker_env = _worker_env(raw.get("worker_env"))
    wandb_spec = _wandb_spec(raw.get("wandb"))

    spec = JobSpec(
        model=model,
        algorithm=algorithm,
        environment=EnvironmentSpec(
            id=str(env_raw.get("id") or ""),
            params=dict(env_raw.get("params") or {}),
            pip=tuple(str(p) for p in env_raw.get("pip") or ()),
            secrets=environment_secrets,
        ),
        train=TrainSpec(
            steps=_train_int(train_raw, "steps", minimum=1),
            epochs=_train_int(train_raw, "epochs", minimum=1),
            lora_rank=_train_int(train_raw, "lora_rank", minimum=1) or 32,
            lora_alpha=_train_int(train_raw, "lora_alpha", minimum=1) or 64,
            init_from_adapter=_init_from_adapter_ref(train_raw),
            hf_repo="",  # assigned server-side; see submit_job._assign_managed_hf_repo
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
                train_raw, "thinking_length_penalty_coef", minimum=0.0, maximum=1.0
            ),
            stop_sequences=_train_stops(train_raw),
            # minimum=0: explicit 0 means "no cap" per TrainSpec contract
            max_steps=_train_int(train_raw, "max_steps", minimum=0),
            max_examples=_train_int(train_raw, "max_examples", minimum=0),
        ),
        gpu=GpuSpec(type=gpu_type),
        run_id=run_id or "local",  # server-assigned at create_run; never user-set
        worker_env=worker_env,
        model_policy=model_policy,
        thinking=thinking,
        wandb=wandb_spec,
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: JobSpec) -> None:
    try:
        canonical_gpu(spec.gpu.type)
    except UnsupportedGpuError as exc:
        raise ConfigError(str(exc)) from exc
    if spec.algorithm == "grpo" and spec.train.steps is not None and spec.train.steps <= 0:
        raise ConfigError("train.steps must be positive for GRPO")
    if spec.algorithm == "sft" and spec.train.epochs is not None and spec.train.epochs <= 0:
        raise ConfigError("train.epochs must be positive for SFT")
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
    if spec.train.lora_rank <= 0:
        raise ConfigError("train.lora_rank must be positive")
    # lora_alpha=0 produces a no-op adapter (zero scaling at serve) — reject up front.
    if spec.train.lora_alpha <= 0:
        raise ConfigError("train.lora_alpha must be positive")
