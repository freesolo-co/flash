"""Parse Flash TOML configs into worker JobSpecs."""

from __future__ import annotations

import re
import sys
import tomllib
from typing import Any

from flash.catalog import normalize_algorithm, resolve_model, serving_lora_rank_cap
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
    _train_str,
    _wandb_spec,
    _worker_env,
)
from flash.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec

_OWNER_REPO_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_RUN_ID_RE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
# The ONE public checkpoint/adapter reference grammar: `<run_id>` (a run's trained adapter) or
# `<run_id>/step-N` (a specific saved checkpoint listed by `flash checkpoints`). The control plane
# resolves it to the internal storage reference below; users never see or write storage refs.
_CHECKPOINT_REF_RE = re.compile(rf"^(?P<run_id>{_RUN_ID_RE})(?:/step-(?P<step>\d{{1,18}}))?$")
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
    if step is None:
        return match.group("run_id"), None
    try:
        return match.group("run_id"), int(step)
    except ValueError:
        return None


def format_checkpoint_ref(run_id: str, step: int | None = None) -> str:
    """Format the canonical short reference: `<run_id>` or `<run_id>/step-N`."""
    return f"{run_id}/step-{int(step)}" if step is not None else str(run_id)


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
    if parse_checkpoint_ref(ref) is not None:
        return ref
    raise ConfigError(
        "train.init_from_adapter must be `<run_id>` (continue that run's trained adapter) or "
        "`<run_id>/step-N` (warm-start from a checkpoint listed by `flash checkpoints`)"
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
        # on-policy distillation (algorithm="opd")
        "teacher_model",
    }
)


def spec_from_dict(raw: dict[str, Any], run_id: str | None = None) -> JobSpec:
    # Only reject table-valued unknowns — callers pass harmless scalar flags like dry_run alongside spec.
    unknown = sorted(k for k in set(raw) - _TOP_LEVEL_KEYS if isinstance(raw[k], dict))
    if unknown:
        hint = ""
        if {"grpo", "sft", "opd"} & set(unknown):
            hint = (
                " — GRPO/SFT/opd knobs (group_size, batch_size, max_tokens, teacher_model, …) "
                "belong under [train], not a [grpo]/[sft]/[opd] table"
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
    if algorithm == "opd" and "FIREWORKS_API_KEY" not in environment_secrets:
        # OPD cannot run without the teacher key, so make it a REQUIRED declared secret: the
        # client (runtime_secrets_from_local_env) and server (_runtime_secrets) both raise on a
        # missing required secret, so a keyless opd run fails fast before any GPU is provisioned.
        # It is a name-only declaration (value stays out-of-band). This declaration is how the key
        # is collected/allowed for opd — each gate unions the spec's declared secrets on top of the
        # global default, which no longer carries FIREWORKS_API_KEY (so SFT/GRPO don't receive it).
        environment_secrets = (*environment_secrets, "FIREWORKS_API_KEY")
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
    _teacher = _train_str(train_raw, "teacher_model")
    if algorithm == "opd" and _teacher:
        # Reject an unpriced teacher override at parse time: falling back to the default rate would
        # make both the submit-time quote and the final charge wrong for a differently-priced model.
        from flash.cost.facts import TEACHER_USD_PER_1M

        if _teacher not in TEACHER_USD_PER_1M:
            raise ConfigError(
                f"[train] teacher_model {_teacher!r} has no pricing entry, so its cost quote and "
                f"charge would silently use the default rate. Use a priced teacher "
                f"({', '.join(sorted(TEACHER_USD_PER_1M))}) or add an entry in flash/cost/facts.py."
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
    lora_rank = _train_int(train_raw, "lora_rank", minimum=1) or 32
    max_lora_rank = serving_lora_rank_cap(info)
    if max_lora_rank is not None and lora_rank > max_lora_rank:
        raise ConfigError(
            f"train.lora_rank={lora_rank} exceeds {model}'s serving max_lora_rank="
            f"{max_lora_rank}; lower train.lora_rank or raise the serving cap "
            "after real-GPU validation"
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
            lora_rank=lora_rank,
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
            teacher_model=_train_str(train_raw, "teacher_model"),
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
    if spec.algorithm == "sft" and int(spec.train.max_examples or 0) <= 0:
        raise ConfigError(
            "train.max_examples must be set to a positive row count for SFT "
            "(use the full dataset row count for an uncapped run)"
        )
    if spec.algorithm == "opd" and spec.train.steps is not None and spec.train.steps <= 0:
        # OPD is step-driven (on-policy sampling), like GRPO — not epoch-driven. The teacher
        # key (FIREWORKS_API_KEY) is auto-declared as a required secret in spec_from_dict, so a
        # keyless opd run already fails fast in the client/server runtime-secret gate.
        raise ConfigError("train.steps must be positive for opd")
    if spec.algorithm == "opd" and spec.train.max_length:
        # Mirror run_opd's prompt-budget guard at PARSE time: a max_length that leaves no room for
        # any prompt after the completion budget is rejected here, BEFORE a paid worker is
        # provisioned (wait_for_gpu + model prefetch + tokenizer/adapter load), instead of failing
        # deterministically only after GPU setup. max_completion resolves exactly as the worker does:
        # explicit [train] max_tokens, else the recipe thinking/non-thinking default.
        from flash.engine.recipe import RECIPE

        max_completion = int(
            spec.train.max_tokens
            or (
                RECIPE.opd.max_completion_len_thinking
                if spec.thinking
                else RECIPE.opd.max_completion_len
            )
        )
        if spec.train.max_length - max_completion < 1:
            raise ConfigError(
                f"[train] max_length ({spec.train.max_length}) leaves no prompt budget after "
                f"max_tokens ({max_completion}) for opd; set max_length > max_tokens. Rejected at "
                f"parse time so an invalid budget fails before a GPU worker is provisioned."
            )
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
