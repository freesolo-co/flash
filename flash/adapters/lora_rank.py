"""LoRA adapter metadata parsing and control-plane preflight helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from flash.core.catalog import lora_expert_count, serving_context_cap, serving_lora_rank_cap


class ServingPreflightError(ValueError):
    """Serving-path preflight rejection. Homed here (dependency-light) so unstructured
    preparation can raise/catch it without importing the heavy flash.serve.deployment.preflight module."""


if TYPE_CHECKING:
    from flash.core.spec import JobSpec

AdapterConfigLoader = Callable[[str, str | None, str | None], Mapping[str, Any]]


@dataclass(frozen=True)
class AdapterMetadata:
    rank: int
    alpha: int


@dataclass(frozen=True)
class AdapterArtifactIdentity:
    digest: str
    config_sha256: str
    weight_filename: str
    weight_identity: str

    def to_dict(self) -> dict[str, str]:
        return {
            "digest": self.digest,
            "config_sha256": self.config_sha256,
            "weight_filename": self.weight_filename,
            "weight_identity": self.weight_identity,
        }


def resolve_adapter_ref(adapter_ref: str) -> tuple[str, str] | None:
    """Resolve the INTERNAL adapter storage ref into ``(repo, artifact_prefix)``.

    Users write the short ``<run_id>[/step-N]`` form (see ``flash.schema.parse_checkpoint_ref``);
    the control plane resolves it against the source run's metadata into the storage reference
    the worker receives (``flash.runner.lifecycle.preparation._prepare_init_from_adapter``). per-step deployable
    adapters live at the identical ``<prefix>/adapter`` layout in the artifact repo (see
    ``publish_deployable_checkpoint``), so the same download path serves both.
    """
    from flash.schema import parse_adapter_storage_ref

    return parse_adapter_storage_ref(adapter_ref)


def adapter_config_path_from_ref(adapter_ref: str) -> tuple[str, str]:
    resolved = resolve_adapter_ref(adapter_ref)
    if resolved is None:
        raise ValueError(
            "train.init_from_adapter could not be resolved to an internal adapter storage ref"
        )
    repo, prefix = resolved
    return repo, f"{prefix}/adapter/adapter_config.json"


def _positive_int(value: Any, *, source: str, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"could not verify adapter metadata: {source} has invalid {field}")
    parsed: int | None = None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            parsed = int(value)
    elif isinstance(value, Decimal):
        # load_hf_adapter_config parses json floats as decimal for exact textual fidelity.
        if value.is_finite() and value == value.to_integral_value():
            parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+]?[0-9]+", text):
            parsed = int(text)
        else:
            try:
                numeric = Decimal(text)
            except InvalidOperation:
                numeric = Decimal("NaN")
            if numeric.is_finite() and numeric == numeric.to_integral_value():
                parsed = int(numeric)
    if parsed is None:
        raise ValueError(f"could not verify adapter metadata: {source} has invalid {field}")
    if parsed <= 0:
        raise ValueError(
            f"could not verify adapter metadata: {source} has non-positive {field} {parsed}"
        )
    return parsed


def _file_identity(file_info: Any) -> str:
    # the producer is `HfApi.list_repo_tree`, which yields `RepoFile` -- whose `__init__` always
    # sets `path`/`size`/`blob_id` and builds `lfs` as a `BlobLfsInfo` (field `sha256`, no `oid`).
    # verified at the declared floor `huggingface-hub>=1.2.0` and at the installed 1.27.0, so the
    # mapping-shaped `lfs` and the `lfs.oid` spelling have no producer in the supported range.
    # `getattr` stays only on `lfs` itself, which is legitimately optional (non-LFS files), and on
    # the entry object, because a listing also yields `RepoFolder` -- no size, no blob_id -- which
    # must fail closed rather than hash to a folder's tree id.
    lfs = getattr(file_info, "lfs", None)
    oid = getattr(lfs, "sha256", None)
    size = getattr(lfs, "size", None)
    blob_id = getattr(file_info, "blob_id", None)
    size = size if size is not None else getattr(file_info, "size", None)
    immutable = oid or blob_id
    if not immutable:
        raise ValueError("source adapter weight metadata has no immutable content identity")
    return f"{immutable}:{size if size is not None else 'unknown'}"


def resolve_hf_dataset_revision(repo: str, token: str | None = None) -> str:
    """Resolve a dataset's current revision to an immutable commit SHA."""
    try:
        from huggingface_hub import HfApi

        revision = str(HfApi(token=token).repo_info(repo, repo_type="dataset").sha or "").strip()
    except Exception as exc:
        raise ValueError("could not pin source adapter revision") from exc
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise ValueError("source adapter revision is not an immutable commit SHA")
    return revision.lower()


def _contains_decimal(value: Any) -> bool:
    if isinstance(value, Decimal):
        return True
    if isinstance(value, Mapping):
        return any(_contains_decimal(key) or _contains_decimal(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_decimal(item) for item in value)
    return False


def _typed_canonical_tree(value: Any) -> Any:
    """Encode JSON values and decimals into an injective, type-aware canonical tree."""
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, Mapping):
        return [
            "object",
            [
                [_typed_canonical_tree(key), _typed_canonical_tree(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ],
        ]
    if isinstance(value, (list, tuple)):
        return ["array", [_typed_canonical_tree(item) for item in value]]
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_config_bytes(config: Mapping[str, Any]) -> bytes:
    if not _contains_decimal(config):
        return json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    tree = _typed_canonical_tree(config)
    encoded = json.dumps(tree, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return b"typed-json-decimal-v1\0" + encoded


def adapter_artifact_identity(
    adapter_ref: str,
    config: Mapping[str, Any],
    token: str | None = None,
    revision: str | None = None,
) -> AdapterArtifactIdentity:
    """Bind adapter config semantics and the required weight objects' immutable metadata.

    Binds whatever peft would load, which for a save past its shard size is the complete
    index-referenced shard set rather than a single file. Asking for the two single-file names by
    path could not see a sharded adapter at all, so warm-starting from one failed here -- at
    submission, before the worker that would have loaded it ever ran.

    A single-file adapter's digest is unchanged by that: it still binds exactly its one name and
    identity, so an in-flight run's stored identity still compares equal.
    """
    from flash.adapters.artifacts import loadable_adapter_weight_files

    resolved = resolve_adapter_ref(adapter_ref)
    if resolved is None:
        raise ValueError("source adapter reference is invalid")
    repo, prefix = resolved
    config_bytes = _canonical_config_bytes(config)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    folder = f"{prefix}/adapter"
    try:
        from huggingface_hub import HfApi

        infos = list(
            HfApi(token=token).list_repo_tree(
                repo_id=repo,
                path_in_repo=folder,
                repo_type="dataset",
                recursive=False,
                revision=revision,
            )
        )
    except Exception as exc:
        raise ValueError("could not verify source adapter weight identity") from exc
    by_name = {str(getattr(info, "path", "")).rsplit("/", 1)[-1]: info for info in infos}
    selected = loadable_adapter_weight_files(by_name)
    if not selected:
        raise ValueError("source adapter has no required weight file")
    identities = [_file_identity(by_name[name]) for name in selected]
    # every selected object, because a sharded adapter's content is the union of its shards: binding
    # only the first would let a later attempt rewrite the rest and still compare equal.
    bound = "\0".join(
        f"{name}\0{identity}" for name, identity in zip(selected, identities, strict=True)
    )
    payload = f"v1\0{config_sha256}\0{bound}".encode()
    return AdapterArtifactIdentity(
        digest=hashlib.sha256(payload).hexdigest(),
        config_sha256=config_sha256,
        weight_filename=",".join(selected),
        weight_identity=",".join(identities),
    )


def _pattern_values(config: Mapping[str, Any], *, field: str, source: str) -> list[int]:
    value = config.get(field)
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise ValueError(f"could not verify adapter metadata: {source} has invalid {field}")
    return [_positive_int(item, source=source, field=field) for item in value.values()]


def _max_adapter_value(
    config: Mapping[str, Any], *, scalar: str, pattern: str, label: str, source: str
) -> int:
    """The largest value PEFT records for one adapter dimension, across both places it writes it.

    Rank and alpha are the same shape of question -- a default scalar (``r``, ``lora_alpha``) plus an
    optional per-module override map -- and the answer is the maximum, because that is what serving
    has to be able to hold. Sharing the walk keeps the two from drifting: an adapter whose rank came
    from ``rank_pattern`` but whose alpha came from the scalar must still be read the same way.
    """
    if not isinstance(config, Mapping):
        raise ValueError(f"could not verify adapter metadata: {source} is not a JSON object")
    values = _pattern_values(config, field=pattern, source=source)
    if config.get(scalar) is not None:
        values.append(_positive_int(config[scalar], source=source, field=scalar))
    if not values:
        raise ValueError(
            f"could not verify adapter metadata: {source} has no LoRA {label} metadata"
        )
    return max(values)


def rank_from_adapter_config(config: Mapping[str, Any], *, source: str) -> int:
    """Return the maximum LoRA rank across default and per-module metadata."""
    return _max_adapter_value(
        config, scalar="r", pattern="rank_pattern", label="rank", source=source
    )


_LORA_A_INFIX = ".lora_A."
_LORA_B_INFIX = ".lora_B."
# PEFT appends this rung for every parameter wrapper nested inside another one.
_NESTED_WRAPPER_RUNG = ".base_layer"


@dataclass(frozen=True)
class DeclaredLoraRanks:
    """The default rank, per-module overrides, and parameter-target module paths in a config."""

    default: int | None = None
    by_module: Mapping[str, int] = field(default_factory=dict)
    stacked_rank_modules: tuple[str, ...] = ()
    stacked_rank_multiplier: int | None = None

    def __bool__(self) -> bool:
        return self.default is not None or bool(self.by_module)


def _declared_lora_rank_context(
    config: Mapping[str, Any], *, default: int | None, by_module: Mapping[str, int]
) -> DeclaredLoraRanks:
    stacked_rank_modules: list[str] = []
    targets = config.get("target_parameters")
    if isinstance(targets, list):
        for target in targets:
            if not isinstance(target, str):
                continue
            module, separator, _ = target.rpartition(".")
            if separator and module:
                stacked_rank_modules.append(module)

    base_model = str(config.get("base_model_name_or_path") or "").strip()
    return DeclaredLoraRanks(
        default=default,
        by_module=by_module,
        stacked_rank_modules=tuple(stacked_rank_modules),
        stacked_rank_multiplier=lora_expert_count(base_model),
    )


def strict_declared_lora_ranks(
    config: Mapping[str, Any], *, source: str = "adapter_config.json"
) -> DeclaredLoraRanks:
    """validate and preserve PEFT rank declarations for an authoritative load boundary."""
    if not isinstance(config, Mapping):
        raise ValueError(f"{source} must be an object")

    default = None
    if "r" in config:
        value = config["r"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{source} r must be a positive integer")
        default = value

    by_module: dict[str, int] = {}
    if "rank_pattern" in config:
        pattern = config["rank_pattern"]
        if not isinstance(pattern, Mapping):
            raise ValueError(f"{source} rank_pattern must be an object")
        for module, value in pattern.items():
            if not isinstance(module, str) or not module.strip():
                raise ValueError(f"{source} rank_pattern keys must be non-empty strings")
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{source} rank_pattern values must be positive integers")
            try:
                re.compile(rf"(.*\.)?({module})$")
            except re.error as exc:
                raise ValueError(
                    f"{source} rank_pattern contains invalid regex {module!r}"
                ) from exc
            by_module[module] = value

    return _declared_lora_rank_context(config, default=default, by_module=by_module)


def _rank_for_module(module_path: str, declared: DeclaredLoraRanks) -> int | None:
    """Resolve a module's rank with PEFT's ordered, anchored ``rank_pattern`` matching."""
    for module, rank in declared.by_module.items():
        if not module:
            continue
        try:
            # entries are regexes in PEFT, deliberately not escaped here either
            if re.match(rf"(.*\.)?({module})$", module_path):
                return rank
        except re.error:
            continue
    return declared.default


def _module_uses_target_parameters(module_path: str, declared: DeclaredLoraRanks) -> bool:
    """Whether this serialized module came from one of PEFT's targeted parameters.

    PEFT wraps the module that *owns* a targeted parameter, keeping the parameter name off the
    serialized path. When several targeted parameters share one owner, the wrappers nest and every
    wrapper after the first appends a ``base_layer`` rung, so a single owner produces keys like::

        mlp.experts                          (the outermost wrapper)
        mlp.experts.base_layer               (the one nested inside it)

    Those inner rungs carry the same stacked axis as the outer one, so peeling the ``base_layer``
    rungs off before matching is what keeps a valid adapter from being measured against the scalar
    rank.
    """
    candidates = [module_path]
    trimmed = module_path
    while trimmed.endswith(_NESTED_WRAPPER_RUNG):
        trimmed = trimmed[: -len(_NESTED_WRAPPER_RUNG)]
        candidates.append(trimmed)
    return any(
        candidate == module or candidate.endswith(f".{module}")
        for candidate in candidates
        for module in declared.stacked_rank_modules
    )


def lora_tensor_rank_disagrees(key: str, shape: Any, declared: DeclaredLoraRanks) -> bool:
    """Return whether a 2-D LoRA weight provably contradicts its configured rank.

    Ordinary module weights carry exactly ``r`` on the LoRA axis. A 3-D parameter targeted through
    PEFT's ``target_parameters`` serializes under its parent module and stacks every model expert on
    that axis, so only that module may carry exactly ``r * num_experts``. The distinction must be per
    module because one config may contain ordinary ``target_modules`` and fused parameters together.
    """
    if not declared:
        return False
    is_a = _LORA_A_INFIX in key
    is_b = _LORA_B_INFIX in key
    if is_a == is_b:  # neither, or a key somehow claiming both
        return False
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return False
    dims = [d for d in shape if isinstance(d, int) and not isinstance(d, bool) and d > 0]
    if len(dims) != 2:
        return False

    infix = _LORA_A_INFIX if is_a else _LORA_B_INFIX
    module_path = key.partition(infix)[0]
    rank = _rank_for_module(module_path, declared)
    if rank is None:
        return False
    axis = dims[0] if is_a else dims[1]
    if _module_uses_target_parameters(module_path, declared):
        multiplier = declared.stacked_rank_multiplier
        expected = rank * multiplier if multiplier is not None else rank
        return axis != expected
    return axis != rank


def alpha_from_adapter_config(config: Mapping[str, Any], *, source: str) -> int:
    """Return the maximum LoRA alpha across default and per-module metadata."""
    return _max_adapter_value(
        config, scalar="lora_alpha", pattern="alpha_pattern", label="alpha", source=source
    )


def _normalized_model(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def inspect_adapter_config(
    config: Mapping[str, Any], *, source: str, target_model: str
) -> AdapterMetadata:
    """Validate PEFT compatibility and return source-authoritative adapter metadata."""
    if not isinstance(config, Mapping):
        raise ValueError(f"could not verify adapter metadata: {source} is not a JSON object")
    if str(config.get("peft_type") or "").strip().upper() != "LORA":
        raise ValueError(f"could not verify adapter metadata: {source} peft_type must be LORA")
    # both fields are REQUIRED, not "checked when present": every warm-start source is a
    # flash-owned adapter resolved from a run or checkpoint reference, and the one exporter every
    # publish path funnels through stamps `base_model_name_or_path = model_id` unconditionally
    # (engine/worker/verl/checkpoints.py) while the builder always sets `task_type = CAUSAL_LM`
    # (engine/worker/model/adapter.py). treating a blank value as "no opinion" made the base-model
    # match SKIP itself, so an adapter trained on a different base passed preflight and was
    # inherited into the run -- the check failing open is worse than an old artifact failing loudly.
    task_type = str(config.get("task_type") or "").strip().upper()
    if task_type != "CAUSAL_LM":
        raise ValueError(f"could not verify adapter metadata: {source} task_type must be CAUSAL_LM")
    base_model = _normalized_model(config.get("base_model_name_or_path"))
    if not base_model:
        raise ValueError(
            f"could not verify adapter metadata: {source} does not name its base model"
        )
    if base_model != _normalized_model(target_model):
        raise ValueError(
            f"train.init_from_adapter base model {base_model!r} does not match target model "
            f"{target_model!r}"
        )
    rank = rank_from_adapter_config(config, source=source)
    alpha = alpha_from_adapter_config(config, source=source)
    max_lora_rank = serving_lora_rank_cap(target_model)
    if max_lora_rank is not None and rank > max_lora_rank:
        raise ValueError(
            f"train.init_from_adapter has rank {rank}, exceeding {target_model}'s serving "
            f"max_lora_rank={max_lora_rank}; use a lower-rank adapter or raise the serving cap "
            "after real-GPU validation"
        )
    return AdapterMetadata(rank=rank, alpha=alpha)


def load_hf_adapter_config(
    adapter_ref: str, token: str | None = None, revision: str | None = None
) -> Mapping[str, Any]:
    """Read ``adapter_config.json`` for a Flash adapter ref from Hugging Face datasets."""
    repo, filename = adapter_config_path_from_ref(adapter_ref)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - package extra is present in supported installs
        raise ValueError(
            "could not verify train.init_from_adapter metadata: huggingface_hub is not installed"
        ) from exc
    try:
        local = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            token=token,
            revision=revision,
        )
    except Exception as exc:
        raise ValueError(
            f"could not verify train.init_from_adapter metadata: failed to read {repo}:{filename}"
        ) from exc
    try:
        with open(local, encoding="utf-8") as f:
            config = json.load(f, parse_float=Decimal)
    except Exception as exc:
        raise ValueError(
            f"could not verify train.init_from_adapter metadata: invalid JSON in {repo}:{filename}"
        ) from exc
    if not isinstance(config, Mapping):
        raise ValueError(
            f"could not verify train.init_from_adapter metadata: {repo}:{filename} is not a JSON object"
        )
    return config


def preflight_init_adapter_lora_rank(
    spec: JobSpec,
    *,
    token: str | None = None,
    config_loader: AdapterConfigLoader | None = None,
) -> AdapterMetadata | None:
    """Validate and return metadata for a continued adapter without comparing child knobs."""
    adapter_storage_ref = (spec.train.init_from_adapter or "").strip()
    if not adapter_storage_ref:
        return None
    repo, filename = adapter_config_path_from_ref(adapter_storage_ref)
    loader = config_loader or load_hf_adapter_config
    return inspect_adapter_config(
        loader(adapter_storage_ref, token, spec.train.init_from_adapter_revision or None),
        source=f"{repo}:{filename}",
        target_model=spec.model,
    )


def serving_completion_token_capacity(spec: JobSpec, *, prompt_allowance: int) -> int | None:
    """Return completion tokens available after reserving serving prompt context."""
    cap = serving_context_cap(spec.model)
    if cap is None:
        return None
    return max(0, cap - max(0, int(prompt_allowance)))


def _effective_train_context(
    spec: JobSpec, *, completion_tokens: int | None = None
) -> tuple[int, str] | None:
    from flash.engine.plan.vram import grpo_rollout_seq_len, opd_rollout_seq_len

    max_completion_tokens = (
        completion_tokens if completion_tokens is not None else spec.train.max_completion_tokens
    )
    if spec.algorithm == "grpo":
        return (
            grpo_rollout_seq_len(
                spec.train.max_context_tokens or 0,
                max_completion_tokens,
                spec.thinking,
            ),
            (
                "train.max_context_tokens / train.max_completion_tokens "
                "(GRPO rollout prompt+completion)"
            ),
        )
    if spec.algorithm == "opd":
        return (
            opd_rollout_seq_len(
                spec.train.max_context_tokens or 0,
                max_completion_tokens,
                spec.thinking,
            ),
            (
                "train.max_context_tokens / train.max_completion_tokens "
                "(OPD rollout prompt+completion)"
            ),
        )
    effective = int(spec.train.max_context_tokens or 0)
    return (effective, "train.max_context_tokens") if effective > 0 else None


def preflight_train_context_within_serving(
    spec: JobSpec,
    *,
    completion_tokens: int | None = None,
    prompt_allowance: int = 0,
) -> None:
    """Reject a run whose training or completion context exceeds serving ``max_model_len``."""
    cap = serving_context_cap(spec.model)
    if cap is None:
        return
    if completion_tokens is not None:
        capacity = serving_completion_token_capacity(spec, prompt_allowance=prompt_allowance)
        assert capacity is not None
        if completion_tokens > capacity:
            raise ValueError(
                f"train.max_completion_tokens effective budget ({completion_tokens}) cannot fit "
                f"{spec.model}'s serving max_model_len={cap} after reserving {prompt_allowance} "
                f"tokens for the serving prompt; lower train.max_completion_tokens to <= "
                f"{capacity}."
            )
    resolved = _effective_train_context(spec, completion_tokens=completion_tokens)
    if resolved is None:
        return
    effective, knob = resolved
    if effective > cap:
        raise ValueError(
            f"{knob}={effective} exceeds {spec.model}'s serving max_model_len={cap}: a LoRA trained "
            f"at a longer context than it is served wastes compute and learns positions never used "
            f"at inference. Lower it to <= {cap}, or raise the serving context after real-GPU "
            f"validation."
        )
