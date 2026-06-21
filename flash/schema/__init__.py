"""Parse Flash TOML configs into worker JobSpecs."""

from __future__ import annotations

import os
import sys
import tomllib
from typing import Any

from flash.catalog import MODELS, normalize_algorithm, resolve_model
from flash.providers.base import (
    UnsupportedGpuError,
    canonical_gpu,
    provisional_gpu,
)
from flash.schema.fields import (
    ConfigError,
    _coerce_scalar,
    _require_int,
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
     "stop_sequences", "max_steps", "max_examples", "inference_gpus"}
)


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
        # Pass [train] so the open-model ("allow") fit check is disaggregated-aware: an
        # inference_gpus>0 run sizes per-GPU (a big HF model fits as a split), matching the
        # disaggregated-aware provisional_gpu above instead of rejecting it on the colocate total.
        info = resolve_model(model, algorithm, policy=model_policy, gpu=gpu_type, train=train_raw)
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
            # GPUs in the node dedicated to the disaggregated vLLM rollout server (0 = colocate,
            # the default). >0 needs a multi-GPU node ([gpu] count = trainer + inference); the
            # count>inference_gpus cross-check is in _validate_spec. minimum=0 so an explicit
            # `inference_gpus = 0` (colocate) is accepted, not rejected as below-minimum.
            inference_gpus=_train_int(train_raw, "inference_gpus", minimum=0) or 0,
        ),
        gpu=GpuSpec(
            type=gpu_type,
            # count is the paid multi-GPU topology (trainer + inference_gpus); reject non-integer /
            # bool values up front rather than silently truncating (2.9 -> 2) or coercing (true -> 1)
            # and provisioning a different split than requested. (>=1 is re-asserted in _validate_spec.)
            count=_require_int(gpu_raw.get("count"), "gpu.count", minimum=1, default=1),
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


def validate_topology(spec: JobSpec) -> None:
    """Generic multi-GPU / disaggregated-rollout topology checks ([gpu] count, [train] inference_gpus).

    These cross-field guards depend only on the spec (plus the catalog head count, when the model is
    a catalog entry) — NOT on resolving/renting a GPU — so they catch an unfittable topology at submit
    before a paid node is provisioned. Factored out of ``_validate_spec`` so ``runner.submit_job`` can
    re-run them on a ``JobSpec`` that reached it WITHOUT going through ``spec_from_dict`` (a directly
    constructed or ``JobSpec.from_dict``-rehydrated spec, e.g. a programmatic submission), which would
    otherwise bypass the schema guard and only fail on the paid worker.
    """
    if spec.gpu.count < 1:
        raise ConfigError("gpu.count must be >= 1")
    if spec.train.inference_gpus < 0:
        raise ConfigError("train.inference_gpus must be >= 0")
    # A multi-GPU node is ONLY used by the disaggregated GRPO path (gpu.count = trainer GPUs +
    # inference_gpus). With inference_gpus == 0 the worker takes the colocated/single-process path
    # and never touches the extra cards, so gpu.count > 1 would silently bill for GPUs that cannot
    # affect the run. Reject it up front (it also catches SFT, where inference_gpus is rejected
    # above, so SFT is always single-GPU here).
    if spec.gpu.count > 1 and spec.train.inference_gpus == 0:
        raise ConfigError(
            f"gpu.count ({spec.gpu.count}) > 1 requires train.inference_gpus > 0 (the "
            "disaggregated GRPO rollout is the only multi-GPU path; gpu.count = trainer GPUs + "
            "inference_gpus). Set inference_gpus, or use gpu.count = 1."
        )
    if spec.train.inference_gpus > 0:
        # The disaggregated async rollout (vLLM server on dedicated GPUs) is a GRPO-only path —
        # SFT has no rollout engine, so inference_gpus would just strand paid GPUs.
        if spec.algorithm != "grpo":
            raise ConfigError(
                "train.inference_gpus is only valid for grpo (the disaggregated rollout server); "
                "SFT has no rollout engine"
            )
        # Need at least one trainer GPU left after carving off the inference GPUs.
        if spec.gpu.count <= spec.train.inference_gpus:
            raise ConfigError(
                f"gpu.count ({spec.gpu.count}) must be greater than train.inference_gpus "
                f"({spec.train.inference_gpus}) — at least one GPU must train "
                "(gpu.count = trainer GPUs + inference_gpus)"
            )
        # train_gpus>1 (2:1, 3:1, 2:2) runs the trainer as a DDP group via `accelerate launch`
        # (run_rl's disaggregated launcher re-execs the worker across the train devices). Supported.
        # Reject a tensor-parallel split the model's head count can't satisfy BEFORE renting: vLLM
        # requires num_attention_heads % inference_gpus == 0 for TP (inference_gpus>1). For a catalog
        # model with a declared head count this is known at submit time — e.g. MiniCPM5-1B (16 heads)
        # with inference_gpus=3 is impossible (16 % 3 != 0). Catching it here avoids charging the user
        # for a multi-GPU node that only fails at the worker's pre-server-boot guard. Open-model runs
        # declare no head count and rely on the worker guard.
        #
        # The divisibility law only constrains TENSOR parallelism (the default). The dp opt-in
        # (FLASH_DISAGG_PARALLEL=dp) replicates the whole server per card (tp=1) so head count is
        # irrelevant — but ONLY for a model the worker actually serves data-parallel, i.e. a MoE.
        # vLLM rejects offline data parallelism for DENSE models, so the worker (engine.worker.run_rl)
        # downgrades a dense `dp` request back to TP and then enforces head divisibility; a dense
        # catalog model with an indivisible split under `dp` would otherwise pass schema yet crash the
        # worker's TP guard after a paid multi-GPU rent. So skip the check only when dp is requested
        # AND the catalog model is MoE (the worker will honor dp). For the MoE 35B-A3B (16 heads) under
        # the default tp, a 1:3 split is still correctly rejected here. Must mirror the worker exactly:
        # tp unless FLASH_DISAGG_PARALLEL=="dp" AND the model is MoE.
        _disagg_is_dp = (os.environ.get("FLASH_DISAGG_PARALLEL") or "").strip().lower() == "dp"
        _skip_for_dp = _disagg_is_dp and bool(getattr(MODELS.get(spec.model), "is_moe", False))
        if spec.train.inference_gpus > 1 and spec.model in MODELS and not _skip_for_dp:
            _heads = int(getattr(MODELS[spec.model], "num_attention_heads", 0) or 0)
            if _heads and _heads % spec.train.inference_gpus != 0:
                _valid = [d for d in range(1, _heads + 1) if _heads % d == 0 and d < spec.gpu.count]
                raise ConfigError(
                    f"train.inference_gpus ({spec.train.inference_gpus}) is an invalid tensor-parallel "
                    f"split for {spec.model}: it has {_heads} attention heads and vLLM requires "
                    f"heads % inference_gpus == 0. Valid inference_gpus for this model (and gpu.count "
                    f"= {spec.gpu.count}): {_valid}"
                )
        # Reject an OPTIONAL disaggregated split of a vision-language catalog model. VL checkpoints
        # (Qwen3.5/3.6) train/serve TEXT-ONLY: the colocate rollout engine skips the vision tower via
        # patch_vllm_language_model_only, but the disaggregated `trl vllm-serve` server has NO
        # language-model-only flag (see disaggregated.TRL_VLLM_SERVE_FLAGS), so it would load the full
        # model incl. the vision tower — extra VRAM and, on RTX 5090-class cards, the vision-attention
        # PTX failure the colocate patch dodges. The dense Qwen3.5 line colocates fine on one card, so
        # a disaggregated split buys nothing and only strands the tower; reject it at submit before a
        # paid multi-GPU rent. The ONE exception is a requires_disaggregated VL model (the 35B-A3B):
        # it MUST run disaggregated and does so on H200-class GPUs where the tower fits and the PTX
        # issue doesn't apply — that path stays allowed. (Open-model VL checkpoints aren't catalog
        # entries, so the worker remains their catch-all.)
        _info = MODELS.get(spec.model)
        if (
            _info is not None
            and getattr(_info, "is_vl", False)
            and not getattr(_info, "requires_disaggregated", False)
        ):
            raise ConfigError(
                f"{spec.model} is a vision-language checkpoint trained text-only; the disaggregated "
                "rollout server (trl vllm-serve) cannot skip its vision tower, so a disaggregated "
                "split (train.inference_gpus>0) would load the full model — extra VRAM and an RTX "
                "5090 vision-attention PTX failure. This model colocates on a single GPU: use gpu.count "
                "= 1 with train.inference_gpus = 0 (colocated GRPO)."
            )


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
    # Multi-GPU / disaggregated-rollout topology ([gpu] count, [train] inference_gpus).
    validate_topology(spec)
    # Catalog-level disaggregation requirements (requires_disaggregated / single_trainer_only).
    # These live on the catalog entry, so they apply only to catalog models and are known here
    # WITHOUT resolving/renting. submit_job re-runs this on the resolved ModelInfo, but doing it at
    # parse time too means local-only validators (the MCP create_training_run dry-run, the server's
    # /spec parse, CLI --dry-run via load) reject a 35B-A3B colocate or multi-trainer split here
    # instead of letting it pass dry-run and fail only at submit.
    if spec.model in MODELS:
        from flash.engine.rollout_bench import validate_disaggregated_requirement

        _info = MODELS[spec.model]
        try:
            validate_disaggregated_requirement(
                requires_disaggregated=bool(getattr(_info, "requires_disaggregated", False)),
                algorithm=spec.algorithm,
                inference_gpus=spec.train.inference_gpus,
                single_trainer_only=bool(getattr(_info, "single_trainer_only", False)),
                gpu_count=spec.gpu.count,
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
