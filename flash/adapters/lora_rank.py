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

from flash.core.catalog import serving_context_cap, serving_lora_rank_cap


class ServingPreflightError(ValueError):
    """Serving-path preflight rejection. Homed here (dependency-light) so unstructured
    preparation can raise/catch it without importing the heavy flash.serve.preflight module."""


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
    the worker receives (``flash.runner._prepare_init_from_adapter``). Per-step deployable
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
    lfs = getattr(file_info, "lfs", None)
    if isinstance(lfs, Mapping):
        oid = lfs.get("sha256") or lfs.get("oid")
        size = lfs.get("size")
    else:
        oid = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
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


@dataclass(frozen=True)
class DeclaredLoraRanks:
    """The ranks a config declares, and whether a fused stacked rank axis is possible at all.

    ``by_module`` holds the ``rank_pattern`` overrides keyed by the pattern PEFT matches against the
    module path; ``default`` is the ``r`` every non-overridden module gets. Kept apart rather than
    collapsed to one summary number because PEFT resolves the rank PER MODULE
    (``lora.LoraModel._create_and_replace``: ``rank_pattern.get(matched_key, config.r)``), so any
    single number is wrong for every module it did not come from.

    ``allows_stacked_rank_axis`` is true only when ``target_parameters`` is non-empty. That is what
    routes a module through ``lora.ParamWrapper``, the one code path that builds
    ``nn.Linear(in_features, r * num_experts)`` and so legitimately carries a rank axis larger than
    ``r``. Ordinary ``target_modules`` never do, and their axis must equal the module's rank exactly.
    """

    default: int | None = None
    by_module: Mapping[str, int] = field(default_factory=dict)
    allows_stacked_rank_axis: bool = False

    def __bool__(self) -> bool:
        return self.default is not None or bool(self.by_module)


def declared_lora_ranks(config: Mapping[str, Any]) -> DeclaredLoraRanks:
    """Read the per-module LoRA rank structure out of ``adapter_config.json``.

    Falsy when nothing readable is declared, which leaves the caller with no basis to refuse.
    """
    if not isinstance(config, Mapping):
        return DeclaredLoraRanks()
    try:
        default = _positive_int(config["r"], source="adapter_config.json", field="r")
    except (KeyError, ValueError):
        default = None
    by_module: dict[str, int] = {}
    pattern = config.get("rank_pattern")
    if isinstance(pattern, Mapping):
        for module, value in pattern.items():
            try:
                by_module[str(module)] = _positive_int(
                    value, source="adapter_config.json", field="rank_pattern"
                )
            except ValueError:
                continue
    targets = config.get("target_parameters")
    return DeclaredLoraRanks(
        default=default,
        by_module=by_module,
        allows_stacked_rank_axis=bool(targets) and not isinstance(targets, (str, bytes)),
    )


def _rank_for_module(key: str, declared: DeclaredLoraRanks) -> int | None:
    """The rank this tensor's own module is configured for, mirroring PEFT's pattern matching.

    PEFT matches a ``rank_pattern`` entry against the module path either exactly or as a suffix
    (``peft.tuners.tuners_utils.get_pattern_key``), so the longest matching entry wins and anything
    unmatched falls back to ``r``.
    """
    matched = [
        (module, rank)
        for module, rank in declared.by_module.items()
        if module and (f".{module}." in key or key.startswith(f"{module}.") or module in key)
    ]
    if matched:
        return max(matched, key=lambda pair: len(pair[0]))[1]
    return declared.default


def lora_tensor_rank_disagrees(key: str, shape: Any, declared: DeclaredLoraRanks) -> bool:
    """True only when a weight's own shape provably contradicts its module's configured rank.

    PEFT writes ``lora_A`` as ``[r, in_features]`` and ``lora_B`` as ``[out_features, r]``, so the
    rank is legible from the tensor itself rather than taken on the config's word. That second
    reading is the point: ``adapter_config.json`` and the tensors beside it are produced by
    different code paths and can disagree, and the config is the half a serving engine sizes its
    LoRA slots from.

    The axis must EQUAL the rank resolved for that module. Equality is the rule because loading a
    rank-64 tensor into a rank-32 LoRA layer is a size mismatch, and accepting any multiple would
    wave through exactly that. The single exception is a config declaring ``target_parameters``,
    where ``lora.ParamWrapper`` stacks every expert on the rank axis
    (``nn.Linear(in_features, r * num_experts)``) and a positive multiple is the correct shape.
    Applying that exception unconditionally is what would let the original 35B report through: its
    config carried ``target_parameters: None`` with ``experts`` as an ordinary ``target_modules``
    entry, so its rank-8192 tensors had no fused layout to justify them.

    False for everything whose shape cannot answer the question at all -- ``modules_to_save``
    copies, biases, ``lora_embedding_A``/``_B`` (which the infixes deliberately do not match), and
    3-D stacked layouts. Guessing at those would refuse adapters that are fine.
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
    rank = _rank_for_module(key, declared)
    if rank is None:
        return False
    axis = dims[0] if is_a else dims[1]
    return axis % rank != 0 if declared.allows_stacked_rank_axis else axis != rank


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
    task_type = str(config.get("task_type") or "").strip().upper()
    if task_type and task_type != "CAUSAL_LM":
        raise ValueError(f"could not verify adapter metadata: {source} task_type must be CAUSAL_LM")
    base_model = _normalized_model(config.get("base_model_name_or_path"))
    if base_model and base_model != _normalized_model(target_model):
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
