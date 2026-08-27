"""immutable import-light records for serving deployment control."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, fields
from pathlib import PurePosixPath
from typing import Literal, TypeAlias

from flash.schema import parse_checkpoint_ref

from ._canonical import canonical_json
from ._urls import validate_modal_public_url

Provider: TypeAlias = Literal["modal"]
RepoType: TypeAlias = Literal["model", "dataset"]
Modality: TypeAlias = Literal["text", "multimodal"]
DeploymentStatus: TypeAlias = Literal[
    "ready",
    "provisioning",
    "failed",
    "outcome_unknown",
    "absent",
]
DeploymentErrorCode: TypeAlias = Literal[
    "authentication_failed",
    "conflict",
    "invalid_request",
    "provider_rejected",
    "readiness_failed",
    "resource_ambiguous",
    "transport_failed",
]
DeploymentErrorReason: TypeAlias = Literal[
    "artifact_cleanup_conflict",
    "artifact_cleanup_delete_rejected",
    "artifact_cleanup_delete_unknown",
    "artifact_cleanup_observation_failed",
    "readiness_deadline_unproven",
]

_DEPLOYMENT_STATUSES = frozenset({"ready", "provisioning", "failed", "outcome_unknown", "absent"})
_DEPLOYMENT_ERROR_CODES = frozenset(
    {
        "authentication_failed",
        "conflict",
        "invalid_request",
        "provider_rejected",
        "readiness_failed",
        "resource_ambiguous",
        "transport_failed",
    }
)
_DEPLOYMENT_ERROR_REASONS = frozenset(
    {
        "artifact_cleanup_conflict",
        "artifact_cleanup_delete_rejected",
        "artifact_cleanup_delete_unknown",
        "artifact_cleanup_observation_failed",
        "readiness_deadline_unproven",
    }
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_HEX_40_RE = re.compile(r"[0-9a-f]{40}")
_HEX_64_RE = re.compile(r"[0-9a-f]{64}")
_IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_MODAL_PROVIDER_ID_PATTERNS = {
    "app": re.compile(r"ap-[A-Za-z0-9]{22}"),
    "function": re.compile(r"fu-[A-Za-z0-9]{22}"),
    "secret": re.compile(r"st-[A-Za-z0-9]{22}"),
    "volume": re.compile(r"vo-[A-Za-z0-9]{22}"),
}
_SAFE_SUBFOLDER_PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REPO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


def _require_nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty unpadded string")
    return value


def _require_optional_nonempty(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty(value, name)


def _require_exact_digest(value: object, name: str, pattern: re.Pattern[str]) -> str:
    text = _require_nonempty(value, name)
    if pattern.fullmatch(text) is None:
        raise ValueError(f"{name} must be an exact lowercase digest")
    return text


def _require_identifier(value: object, name: str) -> str:
    text = _require_nonempty(value, name)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        raise ValueError(f"{name} is invalid")
    return text


def _require_int_type(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    integer = _require_int_type(value, name)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _require_optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


def _canonical_float(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return 0.0 if normalized == 0 else normalized


def _require_nonnegative_number(value: object, name: str) -> float:
    normalized = _canonical_float(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return normalized


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _validate_repo_id(value: object) -> str:
    repo_id = _require_nonempty(value, "artifact_repo_id")
    if _REPO_ID_RE.fullmatch(repo_id) is None:
        raise ValueError("artifact_repo_id must be an exact owner/name repository id")
    return repo_id


def _validate_repo_type(value: object) -> RepoType:
    if type(value) is not str or value not in {"model", "dataset"}:
        raise ValueError("artifact_repo_type must be model or dataset")
    return value


def _validate_subfolder(value: object) -> str:
    subfolder = _require_nonempty(value, "artifact_subfolder")
    if "\\" in subfolder or subfolder.startswith("/") or subfolder.endswith("/"):
        raise ValueError("artifact_subfolder must be a safe relative posix path")
    path = PurePosixPath(subfolder)
    if str(path) != subfolder or not path.parts:
        raise ValueError("artifact_subfolder must be a canonical relative posix path")
    if any(
        part in {".", ".."} or _SAFE_SUBFOLDER_PART_RE.fullmatch(part) is None
        for part in path.parts
    ):
        raise ValueError("artifact_subfolder contains an unsafe path component")
    return subfolder


def _validate_structured_default(value: object) -> None:
    if value is None:
        return
    text = _require_nonempty(value, "structured_outputs_default_json")
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("structured_outputs_default_json must be valid json") from exc
    if type(parsed) is not dict:
        raise ValueError("structured_outputs_default_json must encode an object")
    if canonical_json(parsed) != text:
        raise ValueError("structured_outputs_default_json must use canonical sorted json")


@dataclass(frozen=True, slots=True)
class EngineIdentity:
    """every engine-wide field that determines serving compatibility."""

    served_model: str
    model_revision: str
    tokenizer_model: str
    tokenizer_revision: str
    image_digest: str
    modality: Modality
    runtime_family: str
    dtype: str
    quantization: str | None
    kv_cache_dtype: str | None
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int | None
    max_loras: int
    max_cpu_loras: int
    max_lora_rank: int
    gpu_memory_utilization: float
    cpu_offload_gb: float
    image_limit: int | None
    mm_processor_cache_gb: float
    enable_tower_connector_lora: bool
    reasoning_parser: str | None
    trust_remote_code: bool
    engine_args_fingerprint: str
    tokenizer_kwargs_fingerprint: str
    processor_kwargs_fingerprint: str

    def __post_init__(self) -> None:
        validate_engine_identity(self)

    @property
    def canonical_json(self) -> str:
        payload = {entry.name: getattr(self, entry.name) for entry in fields(self)}
        return canonical_json(payload)

    @property
    def engine_id(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def adapter_capacity(self) -> int:
        """return the explicit number of adapters validated for this engine."""

        return self.max_cpu_loras


def validate_engine_identity(identity: EngineIdentity) -> None:
    """validate one complete exact engine identity, including runtime invariants."""

    if type(identity) is not EngineIdentity:
        raise ValueError("engine must be an exact EngineIdentity")
    _require_nonempty(identity.served_model, "served_model")
    _require_exact_digest(identity.model_revision, "model_revision", _HEX_40_RE)
    _require_nonempty(identity.tokenizer_model, "tokenizer_model")
    _require_exact_digest(identity.tokenizer_revision, "tokenizer_revision", _HEX_40_RE)
    _require_exact_digest(identity.image_digest, "image_digest", _IMAGE_DIGEST_RE)
    if identity.modality not in {"text", "multimodal"}:
        raise ValueError("modality must be text or multimodal")
    _require_nonempty(identity.runtime_family, "runtime_family")
    _require_nonempty(identity.dtype, "dtype")
    _require_optional_nonempty(identity.quantization, "quantization")
    _require_optional_nonempty(identity.kv_cache_dtype, "kv_cache_dtype")
    for name in (
        "tensor_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "max_loras",
        "max_cpu_loras",
        "max_lora_rank",
    ):
        _require_positive_int(getattr(identity, name), name)
    if identity.max_cpu_loras < identity.max_loras:
        raise ValueError("max_cpu_loras must be at least max_loras")
    _require_optional_positive_int(identity.max_num_batched_tokens, "max_num_batched_tokens")
    _require_optional_positive_int(identity.image_limit, "image_limit")
    if identity.modality == "text" and identity.image_limit is not None:
        raise ValueError("text engines cannot declare an image_limit")
    if identity.modality == "multimodal" and identity.image_limit is None:
        raise ValueError("multimodal engines require a positive image_limit")
    utilization = _canonical_float(identity.gpu_memory_utilization, "gpu_memory_utilization")
    if utilization <= 0 or utilization > 1:
        raise ValueError("gpu_memory_utilization must be greater than zero and at most one")
    object.__setattr__(identity, "gpu_memory_utilization", utilization)
    for name in ("cpu_offload_gb", "mm_processor_cache_gb"):
        object.__setattr__(
            identity,
            name,
            _require_nonnegative_number(getattr(identity, name), name),
        )
    _require_bool(identity.enable_tower_connector_lora, "enable_tower_connector_lora")
    if identity.modality == "text" and identity.enable_tower_connector_lora:
        raise ValueError("text engines cannot enable tower connector lora")
    _require_optional_nonempty(identity.reasoning_parser, "reasoning_parser")
    _require_bool(identity.trust_remote_code, "trust_remote_code")
    for name in (
        "engine_args_fingerprint",
        "tokenizer_kwargs_fingerprint",
        "processor_kwargs_fingerprint",
    ):
        _require_exact_digest(getattr(identity, name), name, _HEX_64_RE)


@dataclass(frozen=True, slots=True)
class ResolvedAdapter:
    """one authorized permanent checkpoint and its private exact artifact source."""

    run_id: str
    checkpoint_id: str
    artifact_repo_id: str
    artifact_repo_type: RepoType
    artifact_revision: str
    artifact_digest: str
    artifact_subfolder: str
    base_model: str
    base_model_revision: str
    lora_rank: int
    thinking_default: bool
    structured_outputs_default_json: str | None

    def __post_init__(self) -> None:
        validate_resolved_adapter(self)


def validate_resolved_adapter(adapter: ResolvedAdapter) -> None:
    """validate one immutable checkpoint and its logical base provenance."""

    if type(adapter) is not ResolvedAdapter:
        raise ValueError("adapters must contain exact ResolvedAdapter records")
    run_id = _require_identifier(adapter.run_id, "run_id")
    parsed = parse_checkpoint_ref(adapter.checkpoint_id)
    if parsed is None:
        raise ValueError("checkpoint_id must be `<run_id>/final` or `<run_id>/step-N`")
    if parsed[0] != run_id:
        raise ValueError("checkpoint_id does not belong to run_id")
    _validate_repo_id(adapter.artifact_repo_id)
    _validate_repo_type(adapter.artifact_repo_type)
    _require_exact_digest(adapter.artifact_revision, "artifact_revision", _HEX_40_RE)
    _require_exact_digest(adapter.artifact_digest, "artifact_digest", _HEX_64_RE)
    _validate_subfolder(adapter.artifact_subfolder)
    _require_nonempty(adapter.base_model, "base_model")
    _require_exact_digest(adapter.base_model_revision, "base_model_revision", _HEX_40_RE)
    _require_positive_int(adapter.lora_rank, "adapter lora_rank")
    _require_bool(adapter.thinking_default, "thinking_default")
    _validate_structured_default(adapter.structured_outputs_default_json)


@dataclass(frozen=True, slots=True)
class ModalPlacement:
    """modal-specific placement with explicit gpu type and count."""

    workspace_name: str
    environment: str
    gpu: str
    region: str | None
    gpu_count: int = 1
    # modal builds a web url as `<workspace>-<web_suffix>--<label>.modal.run`, where the suffix is
    # a per-environment field the operator sets -- NOT the environment name, and not derivable
    # from it. one environment per workspace may have an empty suffix, which is the `<workspace>--`
    # form. None means exactly that no-suffix environment.
    web_suffix: str | None = None

    def __post_init__(self) -> None:
        validate_modal_placement(self)

    @property
    def provider(self) -> Literal["modal"]:
        return "modal"


def validate_modal_placement(placement: ModalPlacement) -> None:
    if type(placement) is not ModalPlacement:
        raise ValueError("modal requests require ModalPlacement")
    _require_nonempty(placement.workspace_name, "modal workspace_name")
    _require_nonempty(placement.environment, "modal environment")
    _require_optional_nonempty(placement.web_suffix, "modal web_suffix")
    _require_nonempty(placement.gpu, "modal gpu")
    _require_positive_int(placement.gpu_count, "modal gpu_count")
    _require_optional_nonempty(placement.region, "modal region")


Placement: TypeAlias = ModalPlacement


def _validate_provider_placement(provider: object, placement: object) -> Placement:
    if provider != "modal":
        raise ValueError("provider must be modal")
    validate_modal_placement(placement)
    return placement


def _validate_deployment_components(
    *,
    deployment_id: object,
    generation: object,
    provider: object,
    placement: object,
    engine: object,
    adapters: object,
) -> None:
    _require_identifier(deployment_id, "deployment_id")
    _require_positive_int(generation, "generation")
    validated_placement = _validate_provider_placement(provider, placement)
    validate_engine_identity(engine)
    if engine.tensor_parallel_size != validated_placement.gpu_count:
        raise ValueError("placement gpu_count must equal engine tensor_parallel_size")
    if type(adapters) is not tuple or not adapters:
        raise ValueError("deployments require at least one adapter")

    checkpoint_ids: set[str] = set()
    expected_base: tuple[str, str] | None = None
    for adapter in adapters:
        validate_resolved_adapter(adapter)
        if adapter.checkpoint_id in checkpoint_ids:
            raise ValueError("deployment contains a duplicate checkpoint identity")
        checkpoint_ids.add(adapter.checkpoint_id)
        if adapter.lora_rank > engine.max_lora_rank:
            raise ValueError("adapter lora_rank exceeds engine max_lora_rank")
        base = (adapter.base_model, adapter.base_model_revision)
        if expected_base is None:
            expected_base = base
        elif base != expected_base:
            raise ValueError(
                "all adapters in one deployment must use the same logical base model and revision"
            )

    if len(adapters) > engine.adapter_capacity:
        raise ValueError("deployment adapter count exceeds the validated max_cpu_loras capacity")


@dataclass(frozen=True, slots=True)
class DeploymentRequest:
    """pure deployment input containing no provider or endpoint credentials."""

    deployment_id: str
    generation: int
    provider: Provider
    placement: Placement
    engine: EngineIdentity
    adapters: tuple[ResolvedAdapter, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapters", tuple(self.adapters))


@dataclass(frozen=True, slots=True)
class DeploymentSpec:
    """one immutable engine deployment with a deterministic adapter order."""

    deployment_id: str
    generation: int
    provider: Provider
    placement: Placement
    engine: EngineIdentity
    adapters: tuple[ResolvedAdapter, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapters", tuple(self.adapters))
        validate_deployment_spec(self)

    @property
    def spec_id(self) -> str:
        from ._serialization import spec_identity

        return spec_identity(self)


def validate_deployment_request(request: DeploymentRequest) -> None:
    """validate raw deployment input before deterministic ordering."""

    if type(request) is not DeploymentRequest:
        raise ValueError("request must be an exact DeploymentRequest")
    _validate_deployment_components(
        deployment_id=request.deployment_id,
        generation=request.generation,
        provider=request.provider,
        placement=request.placement,
        engine=request.engine,
        adapters=request.adapters,
    )


def validate_deployment_spec(spec: DeploymentSpec) -> None:
    """revalidate one exact canonically ordered deployment spec."""

    if type(spec) is not DeploymentSpec:
        raise ValueError("spec must be an exact DeploymentSpec")
    _validate_deployment_components(
        deployment_id=spec.deployment_id,
        generation=spec.generation,
        provider=spec.provider,
        placement=spec.placement,
        adapters=spec.adapters,
        engine=spec.engine,
    )
    from ._serialization import canonical_adapter_sort_key

    canonical = tuple(sorted(spec.adapters, key=canonical_adapter_sort_key))
    if spec.adapters != canonical:
        raise ValueError("deployment adapters must use canonical ordering")


@dataclass(frozen=True, slots=True)
class ModalProviderHandle:
    """sanitized exact modal resource identities and managed public url."""

    deployment_id: str
    generation: int
    engine_id: str
    workspace_name: str
    app_id: str
    app_name: str
    volume_id: str
    volume_name: str
    inference_secret_id: str
    inference_secret_name: str
    environment: str
    region: str | None
    image_digest: str
    public_url: str
    provider: Literal["modal"] = field(default="modal", init=False)

    def __post_init__(self) -> None:
        validate_modal_handle(self)


def validate_modal_provider_id(
    value: object,
    role: str,
    *,
    name: str | None = None,
) -> str:
    """validate one role-specific provider id from the pinned modal contract."""

    pattern = _MODAL_PROVIDER_ID_PATTERNS.get(role)
    if pattern is None:
        raise ValueError("modal provider id role is not allowlisted")
    selected = _require_nonempty(value, name or f"{role}_id")
    if pattern.fullmatch(selected) is None:
        raise ValueError(f"modal {role} id does not match the pinned provider contract")
    return selected


def validate_modal_handle(handle: ModalProviderHandle) -> None:
    if type(handle) is not ModalProviderHandle:
        raise ValueError("handle must be an exact ModalProviderHandle")
    for name in (
        "deployment_id",
        "workspace_name",
        "app_name",
        "volume_name",
        "inference_secret_name",
        "environment",
    ):
        _require_nonempty(getattr(handle, name), name)
    validate_modal_provider_id(handle.app_id, "app", name="app_id")
    validate_modal_provider_id(handle.volume_id, "volume", name="volume_id")
    validate_modal_provider_id(
        handle.inference_secret_id,
        "secret",
        name="inference_secret_id",
    )
    _require_exact_digest(handle.engine_id, "engine_id", _HEX_64_RE)
    _require_exact_digest(handle.image_digest, "image_digest", _IMAGE_DIGEST_RE)
    _require_positive_int(handle.generation, "generation")
    _require_optional_nonempty(handle.region, "region")
    validate_modal_public_url(handle.public_url)


ProviderHandle: TypeAlias = ModalProviderHandle


def _validate_handle_against_plan(
    *,
    deployment_id: str,
    generation: int,
    provider: Provider,
    placement: Placement,
    engine_id: str,
    image_digest: str,
    handle: ProviderHandle,
) -> None:
    validate_modal_handle(handle)
    if type(placement) is not ModalPlacement:
        raise ValueError("provider handle placement must be an exact ModalPlacement")
    if (
        handle.workspace_name != placement.workspace_name
        or handle.environment != placement.environment
        or handle.region != placement.region
    ):
        raise ValueError("provider handle placement does not match the planned deployment")
    if (
        handle.deployment_id != deployment_id
        or handle.generation != generation
        or handle.provider != provider
        or handle.engine_id != engine_id
        or handle.image_digest != image_digest
    ):
        raise ValueError("provider handle provenance does not match the planned deployment")


@dataclass(frozen=True, slots=True, init=False)
class DeploymentResult:
    """one deployment outcome bound to one complete exact deployment spec."""

    spec: DeploymentSpec = field(repr=False)
    status: DeploymentStatus
    handle: ProviderHandle | None
    error_code: DeploymentErrorCode | None
    error_reason: DeploymentErrorReason | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("deployment results must be constructed from an exact DeploymentSpec")

    @property
    def deployment_id(self) -> str:
        return self.spec.deployment_id

    @property
    def generation(self) -> int:
        return self.spec.generation

    @property
    def provider(self) -> Provider:
        return self.spec.provider

    @property
    def placement(self) -> Placement:
        return self.spec.placement

    @property
    def engine_id(self) -> str:
        return self.spec.engine.engine_id

    @property
    def image_digest(self) -> str:
        return self.spec.engine.image_digest

    @property
    def spec_id(self) -> str:
        return self.spec.spec_id

    @classmethod
    def from_spec(
        cls,
        spec: DeploymentSpec,
        *,
        status: DeploymentStatus,
        handle: ProviderHandle | None = None,
        error_code: DeploymentErrorCode | None = None,
        error_reason: DeploymentErrorReason | None = None,
    ) -> DeploymentResult:
        """construct one result bound to the exact planned deployment spec."""

        if cls is not DeploymentResult:
            raise TypeError("deployment results require the exact DeploymentResult factory")
        validate_deployment_spec(spec)
        result = object.__new__(DeploymentResult)
        values = {
            "spec": spec,
            "status": status,
            "handle": handle,
            "error_code": error_code,
            "error_reason": error_reason,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        validate_deployment_result(result)
        return result


def validate_deployment_result(result: DeploymentResult) -> None:
    """validate one result's complete structural contract."""

    if type(result) is not DeploymentResult:
        raise ValueError("result must be an exact DeploymentResult")
    validate_deployment_spec(result.spec)
    placement = result.placement
    if type(result.status) is not str or result.status not in _DEPLOYMENT_STATUSES:
        raise ValueError("status is not an allowlisted deployment status")
    if result.error_code is not None and (
        type(result.error_code) is not str or result.error_code not in _DEPLOYMENT_ERROR_CODES
    ):
        raise ValueError("error_code is not an allowlisted deployment error")
    if result.error_reason is not None and (
        type(result.error_reason) is not str or result.error_reason not in _DEPLOYMENT_ERROR_REASONS
    ):
        raise ValueError("error_reason is not an allowlisted deployment reason")

    if result.status == "ready" and result.handle is None:
        raise ValueError("ready deployment results require a sanitized provider handle")
    if result.status == "absent" and result.handle is not None:
        raise ValueError("absent deployment results cannot carry a provider handle")
    if result.status in {"failed", "outcome_unknown"}:
        if result.error_code is None:
            raise ValueError(
                f"{result.status} deployment results require an allowlisted error_code"
            )
    elif result.error_code is not None:
        raise ValueError(f"{result.status} deployment results cannot carry an error_code")
    if result.error_reason is not None and result.error_code is None:
        raise ValueError("deployment error_reason requires an error_code")

    if result.handle is not None:
        if type(result.handle) is not ModalProviderHandle:
            raise ValueError("provider handle must be an exact ModalProviderHandle")
        _validate_handle_against_plan(
            deployment_id=result.deployment_id,
            generation=result.generation,
            provider=result.provider,
            placement=placement,
            engine_id=result.engine_id,
            image_digest=result.image_digest,
            handle=result.handle,
        )
