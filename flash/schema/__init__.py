"""Parse Flash TOML configs into worker JobSpecs."""

from __future__ import annotations

import sys
import tomllib
from typing import Any, Literal, overload

from flash.catalog import ModelInfo, normalize_algorithm, resolve_model
from flash.providers.base import (
    UnsupportedGpuError,
    canonical_gpu,
    provisional_gpu,
)
from flash.schema.fields import (
    ConfigError,
    _coerce_scalar,
    _require_slug,
    _train_float,
    _train_int,
    _train_stops,
    _wandb_spec,
    _worker_env,
)
from flash.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec


def load_toml(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


@overload
def spec_from_file(
    path: str,
    run_id: str | None = ...,
    overrides: list[str] | None = ...,
    extra_configs: list[str] | None = ...,
    *,
    return_info: Literal[False] = ...,
) -> JobSpec: ...
@overload
def spec_from_file(
    path: str,
    run_id: str | None = ...,
    overrides: list[str] | None = ...,
    extra_configs: list[str] | None = ...,
    *,
    return_info: Literal[True],
) -> tuple[JobSpec, ModelInfo]: ...
def spec_from_file(
    path: str,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    extra_configs: list[str] | None = None,
    *,
    return_info: bool = False,
) -> JobSpec | tuple[JobSpec, ModelInfo]:
    """Parse (and validate) a TOML config into a JobSpec.

    With ``return_info=True`` also returns the ``ModelInfo`` that parsing already resolved, so
    a caller (the CLI) can reuse it instead of calling ``resolve_model`` a second time — which
    for ``model_policy="allow"`` would re-probe HF metadata and re-emit the policy warning.
    """
    raw = load_toml(path)
    # Composed configs: later files override earlier keys (deep merge).
    for extra in extra_configs or []:
        _deep_merge(raw, load_toml(extra))
    # `--set key=value` dotted overrides (highest precedence).
    for item in overrides or []:
        _apply_override(raw, item)
    # Branch so each call passes a literal return_info (matches the typed overloads).
    if return_info:
        return spec_from_dict(raw, run_id=run_id, return_info=True)
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
    # support list values like seeds=[0,1]
    val = value.strip()
    # [wandb] leaves are string-valued labels (project / run name); a numeric- or
    # bool-looking value like `--set wandb.run_name=123` is still the string label the
    # user intends. Preserve it as a string instead of coercing it to int/float/bool
    # (which _wandb_spec's string validation would otherwise reject).
    if parts[0] == "wandb":
        node[leaf] = val
    elif val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        node[leaf] = [_coerce_scalar(x.strip()) for x in inner.split(",") if x.strip()]
    else:
        node[leaf] = _coerce_scalar(val)


# Recognized config keys. Anything else is a typo or a knob in the wrong place — reject it loudly
# rather than silently ignoring it and training (expensively) against defaults. The classic trap:
# putting GRPO knobs under a `[grpo]` table (they belong under `[train]`), which used to be dropped
# without a peep — a run would then use the default rollout (16x more completions) at 16x the cost.
_TOP_LEVEL_KEYS = frozenset(
    {"model", "algorithm", "model_policy", "thinking",
     "environment", "train", "gpu", "worker_env", "wandb", "run_id"}
)
_TRAIN_KEYS = frozenset(
    {"steps", "epochs", "lora_rank", "lora_alpha", "seeds", "init_from_adapter", "hf_repo",
     "learning_rate", "batch_size", "max_length", "save_every", "group_size", "temperature",
     "max_tokens", "kl_penalty_coef", "advantage_clip", "thinking_length_penalty_coef",
     "stop_sequences", "max_steps", "max_examples"}
)


@overload
def spec_from_dict(
    raw: dict[str, Any], run_id: str | None = ..., *, return_info: Literal[False] = ...
) -> JobSpec: ...
@overload
def spec_from_dict(
    raw: dict[str, Any], run_id: str | None = ..., *, return_info: Literal[True]
) -> tuple[JobSpec, ModelInfo]: ...
def spec_from_dict(
    raw: dict[str, Any], run_id: str | None = None, *, return_info: bool = False
) -> JobSpec | tuple[JobSpec, ModelInfo]:
    # Reject unknown config SECTIONS (table-valued top-level keys) — the footgun is a `[grpo]`
    # table holding rollout knobs that actually belong under `[train]`, silently dropped + run at
    # 16x-cost defaults. We only flag tables, not scalars: callers (e.g. the MCP handler) pass
    # through harmless scalar control flags like `dry_run`/`background` alongside the spec.
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

    try:
        algorithm = normalize_algorithm(raw.get("algorithm"))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    model_policy = (raw.get("model_policy") or "catalog").lower()
    if model_policy not in ("catalog", "allow"):
        raise ConfigError('model_policy must be "catalog" or "allow"')
    thinking = raw.get("thinking", False)  # reasoning mode OFF by default (operator preference)
    if not isinstance(thinking, bool):
        raise ConfigError("thinking must be a boolean")

    # ``is None`` (not ``or {}``): a missing section defaults to an empty table, but a present-
    # but-non-dict value (e.g. ``environment = false``) must reach the "must be a table" check
    # rather than being silently coerced to ``{}`` and bypassing validation.
    env_raw = raw.get("environment")
    if env_raw is None:
        env_raw = {}
    if not isinstance(env_raw, dict):
        raise ConfigError("[environment] must be a table")
    # Local environment paths are gone: a run names a published Hub env by [environment] id.
    # A stray `path` (alone or alongside `id`) is a stale config — reject it loudly instead of
    # silently ignoring the key and training against the wrong/missing env.
    if env_raw.get("path"):
        raise ConfigError(
            "local environment paths are no longer supported — remove `path` and reference a "
            'published Hub `id` ("owner/name")'
        )
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

    # GPU allocation is fully automatic: the submit-time allocator always picks the cheapest
    # fitting class across ALL providers — there is no GPU pin. Any gpu.type in the config is
    # ignored. ``provisional_gpu`` computes the offline RunPod-static cheapest-that-fits for
    # sizing/display only; the live allocator re-resolves it at submit time.
    try:
        # No GPU pin / no validation gate: the cheapest fitting class (every class eligible).
        # The submit-time allocator re-resolves the cheapest fitting class live across providers.
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
        # stderr, not stdout: spec_from_dict runs inside flash/mcp/server.py, which speaks a
        # one-JSON-object-per-line protocol on stdout — a warning line there corrupts the stream.
        print(
            f"warning: open-model policy: cannot verify that {model}'s chat template "
            f"supports thinking mode; the run proceeds with enable_thinking=true",
            file=sys.stderr,
        )

    # worker_env is the lower-level per-run escape hatch ([worker_env] table, string-valued,
    # secret-guarded). The optional [wandb] naming table is a separate, typed spec field
    # (JobSpec.wandb) — NOT folded into worker_env env vars.
    worker_env = _worker_env(raw.get("worker_env"))
    wandb_spec = _wandb_spec(raw.get("wandb"))

    spec = JobSpec(
        model=model,
        algorithm=algorithm,
        environment=EnvironmentSpec(
            id=str(env_raw.get("id") or ""),
            params=dict(env_raw.get("params") or {}),
            pip=tuple(str(p) for p in env_raw.get("pip") or ()),
        ),
        train=TrainSpec(
            steps=_train_int(train_raw, "steps", minimum=1),
            epochs=_train_int(train_raw, "epochs", minimum=1),
            lora_rank=_train_int(train_raw, "lora_rank", minimum=1) or 32,
            lora_alpha=_train_int(train_raw, "lora_alpha", minimum=1) or 64,
            seeds=tuple(int(s) for s in train_raw.get("seeds", (0,))),
            init_from_adapter=str(train_raw.get("init_from_adapter") or ""),
            hf_repo=str(train_raw.get("hf_repo") or ""),
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
            # SFT caps: max_steps caps optimizer steps (cheap pre-flight smoke); max_examples
            # truncates the SFT dataset. minimum=0 so an explicit 0 means "no cap" (matches the
            # TrainSpec "None/0 -> no cap" contract); the worker reads these from [train].
            max_steps=_train_int(train_raw, "max_steps", minimum=0),
            max_examples=_train_int(train_raw, "max_examples", minimum=0),
        ),
        gpu=GpuSpec(
            type=gpu_type,
            disk_gb=int(gpu_raw.get("disk_gb", 60)),
            max_wall_seconds=int(gpu_raw.get("max_wall_seconds", 24 * 3600)),
            max_retries=int(gpu_raw.get("max_retries", 2)),
            network_volume=gpu_raw.get("network_volume"),
            network_volume_gb=int(gpu_raw.get("network_volume_gb", 100)),
            datacenter=gpu_raw.get("datacenter"),
        ),
        run_id=run_id or raw.get("run_id", "local"),
        worker_env=worker_env,
        model_policy=model_policy,
        thinking=thinking,
        wandb=wandb_spec,
    )
    _validate_spec(spec)
    # ``info`` was resolved above (the same ModelInfo this spec validated against). Hand it back
    # when asked so the CLI reuses it rather than re-resolving (a second HF probe under "allow").
    return (spec, info) if return_info else spec


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
    # The id must be a full Prime Hub slug "owner/name": exactly one slash, both parts
    # non-empty. A bare id like "gsm8k" passes the presence check but then the worker runs
    # `prime env install gsm8k` (invalid — Prime needs owner/name) and fails after provisioning.
    _require_slug(
        spec.environment.id,
        '[environment] id must be a published Prime Hub slug "owner/name"',
    )
    if spec.train.lora_rank <= 0:
        raise ConfigError("train.lora_rank must be positive")
    # The per-run HF artifact repo (adapters/checkpoints/code + serving) is required: there
    # is no operator-wide default anymore. It must look like "owner/name" (exactly one slash,
    # both parts non-empty) — a malformed value would reach the worker/serve as an unusable id.
    if not spec.train.hf_repo:
        raise ConfigError(
            "train.hf_repo is required: the HF dataset repo for this run's adapters/checkpoints, "
            'e.g. "owner/name"'
        )
    _require_slug(
        spec.train.hf_repo,
        'train.hf_repo must be a HuggingFace repo of the form "owner/name"',
    )
    # GRPO recipe knobs (group_size/temperature/max_tokens/kl_penalty_coef/advantage_clip/
    # thinking_length_penalty_coef) are range-validated at parse time by the _train_int/
    # _train_float coercers above (including the thinking_length_penalty_coef <= 1.0 upper
    # bound), so no re-check is needed here.
    # lora_alpha scales the adapter contribution; 0 (or negative) trains a paid run
    # that produces a no-op adapter (zero scaling at serve). Reject up front.
    if spec.train.lora_alpha <= 0:
        raise ConfigError("train.lora_alpha must be positive")
