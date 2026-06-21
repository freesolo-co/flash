"""Parse Flash TOML configs into worker JobSpecs."""

from __future__ import annotations

import tomllib
from typing import Any

from flash.catalog import normalize_algorithm, resolve_model
from flash.providers import PROVIDER_NAMES
from flash.providers.base import (
    POLICY_NAMES,
    SUPPORTED,
    UnsupportedGpuError,
    canonical_gpu,
    is_validated,
    providers_for,
    resolve_gpu_policy,
    unvalidated_allowed,
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


def spec_from_file(
    path: str,
    run_id: str | None = None,
    overrides: list[str] | None = None,
    extra_configs: list[str] | None = None,
    *,
    skip_net: bool = False,
) -> JobSpec:
    raw = load_toml(path)
    # Composed configs: later files override earlier keys (deep merge).
    for extra in extra_configs or []:
        _deep_merge(raw, load_toml(extra))
    # `--set key=value` dotted overrides (highest precedence).
    for item in overrides or []:
        _apply_override(raw, item)
    return spec_from_dict(raw, run_id=run_id, skip_net=skip_net)


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


def spec_from_dict(
    raw: dict[str, Any], run_id: str | None = None, *, skip_net: bool = False
) -> JobSpec:
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

    env_raw = raw.get("environment") or {}
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
    train_raw = raw.get("train") or {}
    gpu_raw = raw.get("gpu") or {}

    # Smart allocation is the default: an omitted gpu.type means "the cheapest GPU
    # (across providers) that fits the model", re-resolved live at submit time. The
    # original request survives in gpu.requested so the runner knows whether
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
        # GPU class that fits (across providers, deterministic offline; open models
        # sized from HF metadata); concrete names are canonicalized. The submit-time
        # allocator re-resolves policy words live across providers.
        gpu_type = resolve_gpu_policy(
            requested_gpu,
            model,
            allow_unvalidated=allow_unval,
            algorithm=algorithm,
            train=train_raw,
            thinking=thinking,
            skip_net=skip_net,
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
            f"gpu type {gpu_type!r} has not passed Flash's live validation smoke"
            f"{' on ' + provider if provider != 'auto' else ''} "
            f"(validated: {', '.join(SUPPORTED)}). Set gpu.allow_unvalidated = true to use it anyway."
        )
    try:
        info = resolve_model(model, algorithm, policy=model_policy, gpu=gpu_type, skip_net=skip_net)
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
        print(
            f"warning: open-model policy: cannot verify that {model}'s chat template "
            f"supports thinking mode; the run proceeds with enable_thinking=true"
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
            # minimum=0 so `eval_every_steps = 0` explicitly disables (matches "0/None disables");
            # negatives are rejected.
            eval_every_steps=_train_int(train_raw, "eval_every_steps", minimum=0),
            # How many held-out rows each mid-run eval pass scores (a fixed seeded random sample);
            # minimum=0 so an explicit `eval_examples = 0` is accepted as the documented "use the
            # built-in default (64)" no-op (matches TrainSpec/eval_config, which map 0/None -> 64);
            # negatives are rejected. None -> built-in default (64).
            eval_examples=_train_int(train_raw, "eval_examples", minimum=0),
            # SFT caps: max_steps caps optimizer steps (cheap pre-flight smoke); max_examples
            # truncates the SFT dataset. minimum=0 so an explicit 0 means "no cap" (matches the
            # TrainSpec "None/0 -> no cap" contract); the worker reads these from [train].
            max_steps=_train_int(train_raw, "max_steps", minimum=0),
            max_examples=_train_int(train_raw, "max_examples", minimum=0),
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
    # A separate eval env ([environment.params] eval_env_id) is also prime-installed on the worker
    # (worker_hub_env_ids), so it must be a full "owner/name" slug too — else a bare eval id passes
    # --dry-run but fails `prime env install` after a GPU is provisioned.
    if "eval_env" in spec.environment.params:
        # Legacy alias: `eval_env` is no longer mapped (the worker installs only eval_env_id, and
        # a stray `eval_env` would be forwarded into load_environment). Reject at parse rather than
        # silently evaluating against the training env.
        raise ConfigError(
            "[environment.params] eval_env is no longer supported; use eval_env_id "
            '(a published Prime Hub slug "owner/name")'
        )
    eval_ref = spec.environment.params.get("eval_env_id")
    if eval_ref:
        _require_slug(
            str(eval_ref),
            '[environment.params] eval_env_id must be a published Prime Hub slug "owner/name"',
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
