"""explicit allowlisted serialization schemas for control records."""

from __future__ import annotations

import hashlib

from ._canonical import canonical_json
from .types import (
    AdapterAliasIntent,
    DeploymentResult,
    DeploymentSpec,
    EngineIdentity,
    ModalPlacement,
    ModalProviderHandle,
    ResolvedAdapter,
    RunPodPlacement,
    RunPodProviderHandle,
    validate_deployment_result,
    validate_deployment_spec,
    validate_engine_identity,
    validate_modal_handle,
    validate_modal_placement,
    validate_resolved_adapter,
    validate_runpod_handle,
    validate_runpod_placement,
)


def _require_exact(value: object, expected: type, name: str) -> None:
    if type(value) is not expected:
        raise TypeError(f"{name} must be an exact {expected.__name__}")


def _engine_payload(value: EngineIdentity) -> dict[str, object]:
    return {
        "served_model": value.served_model,
        "model_revision": value.model_revision,
        "tokenizer_model": value.tokenizer_model,
        "tokenizer_revision": value.tokenizer_revision,
        "image_digest": value.image_digest,
        "modality": value.modality,
        "runtime_family": value.runtime_family,
        "dtype": value.dtype,
        "quantization": value.quantization,
        "kv_cache_dtype": value.kv_cache_dtype,
        "tensor_parallel_size": value.tensor_parallel_size,
        "max_model_len": value.max_model_len,
        "max_num_seqs": value.max_num_seqs,
        "max_num_batched_tokens": value.max_num_batched_tokens,
        "max_loras": value.max_loras,
        "max_cpu_loras": value.max_cpu_loras,
        "max_lora_rank": value.max_lora_rank,
        "gpu_memory_utilization": value.gpu_memory_utilization,
        "swap_space_gb": value.swap_space_gb,
        "cpu_offload_gb": value.cpu_offload_gb,
        "image_limit": value.image_limit,
        "mm_processor_cache_gb": value.mm_processor_cache_gb,
        "enable_tower_connector_lora": value.enable_tower_connector_lora,
        "reasoning_parser": value.reasoning_parser,
        "trust_remote_code": value.trust_remote_code,
        "engine_args_fingerprint": value.engine_args_fingerprint,
        "tokenizer_kwargs_fingerprint": value.tokenizer_kwargs_fingerprint,
        "processor_kwargs_fingerprint": value.processor_kwargs_fingerprint,
    }


def serialize_engine(value: EngineIdentity) -> dict[str, object]:
    _require_exact(value, EngineIdentity, "engine")
    validate_engine_identity(value)
    return _engine_payload(value)


def _alias_payload(value: AdapterAliasIntent) -> dict[str, object]:
    return {
        "activate": value.activate,
        "expected_adapter_revision": value.expected_adapter_revision,
    }


def _adapter_payload(value: ResolvedAdapter) -> dict[str, object]:
    return {
        "run_id": value.run_id,
        "checkpoint": value.checkpoint,
        "adapter_revision": value.adapter_revision,
        "artifact_repo_id": value.artifact_repo_id,
        "artifact_repo_type": value.artifact_repo_type,
        "artifact_revision": value.artifact_revision,
        "artifact_digest": value.artifact_digest,
        "artifact_subfolder": value.artifact_subfolder,
        "base_model": value.base_model,
        "base_model_revision": value.base_model_revision,
        "lora_rank": value.lora_rank,
        "thinking_default": value.thinking_default,
        "structured_outputs_default_json": value.structured_outputs_default_json,
        "alias_intent": _alias_payload(value.alias_intent),
    }


def serialize_adapter(value: ResolvedAdapter) -> dict[str, object]:
    _require_exact(value, ResolvedAdapter, "adapter")
    validate_resolved_adapter(value)
    return _adapter_payload(value)


def canonical_adapter_sort_key(value: ResolvedAdapter) -> str:
    """return the one canonical ordering key for deployment adapters."""

    _require_exact(value, ResolvedAdapter, "adapter")
    validate_resolved_adapter(value)
    return canonical_json(_adapter_payload(value))


def _placement_payload(value: ModalPlacement | RunPodPlacement) -> dict[str, object]:
    if type(value) is ModalPlacement:
        return {
            "workspace_name": value.workspace_name,
            "environment": value.environment,
            "gpu": value.gpu,
            "region": value.region,
            "gpu_count": value.gpu_count,
            "provider": value.provider,
        }
    return {
        "account_id": value.account_id,
        "gpu_type_id": value.gpu_type_id,
        "gpu_count": value.gpu_count,
        "data_center_id": value.data_center_id,
        "container_disk_gb": value.container_disk_gb,
        "volume_size_gb": value.volume_size_gb,
        "provider": value.provider,
    }


def serialize_placement(value: object) -> dict[str, object]:
    if type(value) is ModalPlacement:
        validate_modal_placement(value)
        return _placement_payload(value)
    if type(value) is RunPodPlacement:
        validate_runpod_placement(value)
        return _placement_payload(value)
    raise TypeError("placement must be an exact ModalPlacement or RunPodPlacement")


def _spec_payload(value: DeploymentSpec) -> dict[str, object]:
    return {
        "deployment_id": value.deployment_id,
        "generation": value.generation,
        "provider": value.provider,
        "placement": _placement_payload(value.placement),
        "engine": _engine_payload(value.engine),
        "adapters": [_adapter_payload(adapter) for adapter in value.adapters],
    }


def _spec_identity_from_payload(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def spec_identity(value: DeploymentSpec) -> str:
    """return the deterministic sha-256 identity of one complete exact spec."""

    _require_exact(value, DeploymentSpec, "spec")
    validate_deployment_spec(value)
    return _spec_identity_from_payload(_spec_payload(value))


def serialize_spec(value: DeploymentSpec) -> dict[str, object]:
    _require_exact(value, DeploymentSpec, "spec")
    validate_deployment_spec(value)
    payload = _spec_payload(value)
    return {"spec_id": _spec_identity_from_payload(payload), **payload}


def _serialize_modal_handle(value: ModalProviderHandle) -> dict[str, object]:
    _require_exact(value, ModalProviderHandle, "handle")
    validate_modal_handle(value)
    return {
        "deployment_id": value.deployment_id,
        "generation": value.generation,
        "engine_id": value.engine_id,
        "workspace_name": value.workspace_name,
        "app_id": value.app_id,
        "app_name": value.app_name,
        "volume_id": value.volume_id,
        "volume_name": value.volume_name,
        "inference_secret_id": value.inference_secret_id,
        "inference_secret_name": value.inference_secret_name,
        "environment": value.environment,
        "region": value.region,
        "image_digest": value.image_digest,
        "public_url": value.public_url,
        "provider": value.provider,
    }


def _serialize_runpod_handle(value: RunPodProviderHandle) -> dict[str, object]:
    _require_exact(value, RunPodProviderHandle, "handle")
    validate_runpod_handle(value)
    return {
        "deployment_id": value.deployment_id,
        "generation": value.generation,
        "engine_id": value.engine_id,
        "account_id": value.account_id,
        "pod_id": value.pod_id,
        "pod_name": value.pod_name,
        "network_volume_id": value.network_volume_id,
        "network_volume_name": value.network_volume_name,
        "template_id": value.template_id,
        "template_name": value.template_name,
        "inference_secret_id": value.inference_secret_id,
        "inference_secret_name": value.inference_secret_name,
        "data_center_id": value.data_center_id,
        "image_digest": value.image_digest,
        "public_url": value.public_url,
        "provider": value.provider,
    }


def _serialize_handle(value: object) -> dict[str, object]:
    if type(value) is ModalProviderHandle:
        return _serialize_modal_handle(value)
    if type(value) is RunPodProviderHandle:
        return _serialize_runpod_handle(value)
    raise TypeError("handle must be an exact sanitized provider handle")


def _serialize_result(value: DeploymentResult) -> dict[str, object]:
    _require_exact(value, DeploymentResult, "result")
    validate_deployment_result(value)
    return {
        "deployment_id": value.deployment_id,
        "generation": value.generation,
        "provider": value.provider,
        "placement": _placement_payload(value.placement),
        "engine_id": value.engine_id,
        "image_digest": value.image_digest,
        "spec_id": value.spec_id,
        "status": value.status,
        "handle": None if value.handle is None else _serialize_handle(value.handle),
        "error_code": value.error_code,
    }


def serialize_control_record(value: object) -> dict[str, object]:
    """serialize only exact public records and their exact nested schemas."""

    if type(value) is DeploymentSpec:
        return serialize_spec(value)
    if type(value) is ModalProviderHandle:
        return _serialize_modal_handle(value)
    if type(value) is RunPodProviderHandle:
        return _serialize_runpod_handle(value)
    if type(value) is DeploymentResult:
        return _serialize_result(value)
    raise TypeError("only exact sanitized control records can be serialized")
