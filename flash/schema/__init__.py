"""Parse Flash TOML configs into worker JobSpecs."""

from __future__ import annotations

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
    # Composed configs: later files override earlier keys (deep merge).
    for extra in extra_configs or []:
        _deep_merge(raw, load_toml(extra))
    # `--set key=value` dotted overrides (highest precedence).
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


def _init_from_adapter_ref(train_raw: dict[str, Any]) -> str:
    ref = str(train_raw.get("init_from_adapter") or "")
    if not ref:
        return ""
    repo, sep, prefix = ref.partition(":")
    if sep and repo.count("/") == 1 and prefix.startswith(("sft/", "rl/")):
        return ref
    raise ConfigError(
        "train.init_from_adapter must be the full adapter_ref emitted by `flash status` "
        "(<owner>/<repo>:<phase>/<run_id>/seed<N>)"
    )


# Recognized config keys. Anything else is a typo or a knob in the wrong place — reject it loudly
# rather than silently ignoring it and training (expensively) against defaults. The classic trap:
# putting GRPO knobs under a `[grpo]` table (they belong under `[train]`), which used to be dropped
# without a peep — a run would then use the default rollout (16x more completions) at 16x the cost.
#
# Some of these are platform-MANAGED, not user knobs: `gpu`, `model_policy`, `run_id`, and
# `train.hf_repo` are ignored if a user sets them (the control plane derives/assigns them). They
# remain RECOGNIZED — not rejected — because a round-tripped JobSpec (spec.to_dict(), which the
# control plane re-parses on submit) still carries them; rejecting would break that re-validation.
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
        "seeds",
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
# Allowed values for the OPT-IN [gpu] provider pin (mirrors providers.PROVIDER_NAMES); unset keeps
# cross-provider cheapest-wins allocation.
_GPU_PROVIDERS = frozenset({"runpod", "vast"})


def spec_from_dict(raw: dict[str, Any], run_id: str | None = None) -> JobSpec:
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
    # model_policy (curated "catalog" vs any-fitting-HF-model "allow") is NOT a user knob: managed
    # runs always use the curated catalog, so a user-supplied model_policy is ignored. (The "allow"
    # path still exists in resolve_model for internal use, but a submitted config can't select it.)
    model_policy = "catalog"
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
    # Local environment paths are gone: a run names a published Freesolo env by [environment] id.
    # A stray `path` (alone or alongside `id`) is a stale config — reject it loudly instead of
    # silently ignoring the key and training against the wrong/missing env.
    if env_raw.get("path"):
        raise ConfigError(
            "local environment paths are no longer supported — remove `path` and reference a "
            "Freesolo environment `id` returned by `flash env push --name <name>`"
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

    # [gpu] provider is the one real user knob in the otherwise platform-managed [gpu] table: an
    # OPT-IN per-run provider pin ("vast" / "runpod") that restricts the submit-time allocator to a
    # single substrate (for A/B-ing one provider against the full pool). Unset -> cross-provider
    # cheapest-wins (the default, no behavior change). Validate it here so a typo fails at parse
    # time rather than as an opaque "provider not available" at submit.
    gpu_provider = gpu_raw.get("provider")
    if gpu_provider is not None:
        if not isinstance(gpu_provider, str):
            raise ConfigError("[gpu] provider must be a string")
        gpu_provider = gpu_provider.strip().lower() or None
    if gpu_provider is not None and gpu_provider not in _GPU_PROVIDERS:
        raise ConfigError(
            f"[gpu] provider must be one of {sorted(_GPU_PROVIDERS)} (or unset for "
            f"cross-provider allocation), got {gpu_raw.get('provider')!r}"
        )

    # GPU allocation is fully automatic: the submit-time allocator always picks the cheapest
    # fitting validated class across ALL providers — there is no GPU pin. A config's gpu.type
    # is not a user knob. ``provisional_gpu`` computes the offline RunPod-static
    # cheapest-validated-that-fits for sizing/display only; the allocator re-resolves it at
    # submit time.
    try:
        # No GPU pin: the cheapest fitting VALIDATED class (the pool the deployed control plane
        # accepts). The submit-time allocator re-resolves it across providers.
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
    # secret-guarded; the worker reads it for the per-run chalk/kernel opt-in). The optional
    # [wandb] naming table is a separate, typed spec field (JobSpec.wandb) — NOT folded into
    # worker_env env vars.
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
            seeds=tuple(int(s) for s in train_raw.get("seeds", (0,))),
            init_from_adapter=_init_from_adapter_ref(train_raw),
            # hf_repo is assigned by the control plane (a per-run private dataset under the
            # operator's namespace, written by the operator HF_TOKEN); a user-supplied
            # [train] hf_repo is ignored. See flash.runner.submit_job._assign_managed_hf_repo.
            hf_repo="",
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
        # GPU allocation, disk sizing, retry budget, and network volumes are all platform-managed:
        # the submit-time allocator picks the cheapest fitting validated GPU across providers, disk
        # is raised to the model's minimum server-side, and the infra knobs are operator defaults.
        # A user [gpu] table is ignored EXCEPT the opt-in provider pin (validated above); gpu_type
        # here is the offline sizing/display provisional, re-resolved at submit.
        gpu=GpuSpec(type=gpu_type, provider=gpu_provider),
        run_id=run_id or "local",  # server-assigned (new_run_id at create_run); never user-set
        worker_env=worker_env,
        model_policy=model_policy,
        thinking=thinking,
        wandb=wandb_spec,
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
    # Every run must name a Freesolo environment by [environment] id.
    # There is no default environment and no local path mode.
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
    # NOTE: the per-run HF artifact repo (train.hf_repo) is NOT validated here — it is no longer a
    # user field. The control plane assigns it server-side (a per-run private dataset under the
    # operator's namespace) in flash.runner.submit_job; see _assign_managed_hf_repo.
    # GRPO recipe knobs (group_size/temperature/max_tokens/kl_penalty_coef/advantage_clip/
    # thinking_length_penalty_coef) are range-validated at parse time by the _train_int/
    # _train_float coercers above (including the thinking_length_penalty_coef <= 1.0 upper
    # bound), so no re-check is needed here.
    # lora_alpha scales the adapter contribution; 0 (or negative) trains a paid run
    # that produces a no-op adapter (zero scaling at serve). Reject up front.
    if spec.train.lora_alpha <= 0:
        raise ConfigError("train.lora_alpha must be positive")
