"""explicit allowlisted serialization schemas for control records."""

from __future__ import annotations

import hashlib

from ._canonical import canonical_json
from .types import (
    DeploymentSpec,
    EngineIdentity,
    ModalPlacement,
    ResolvedAdapter,
    validate_deployment_spec,
    validate_engine_identity,
    validate_resolved_adapter,
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


def _adapter_payload(value: ResolvedAdapter) -> dict[str, object]:
    return {
        "run_id": value.run_id,
        "checkpoint_id": value.checkpoint_id,
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
    }


def canonical_adapter_sort_key(value: ResolvedAdapter) -> str:
    """return the one canonical ordering key for deployment adapters."""

    _require_exact(value, ResolvedAdapter, "adapter")
    validate_resolved_adapter(value)
    return canonical_json(_adapter_payload(value))


def _placement_payload(value: ModalPlacement) -> dict[str, object]:
    _require_exact(value, ModalPlacement, "placement")
    return {
        "workspace_name": value.workspace_name,
        "environment": value.environment,
        # part of the identity, not decoration: the suffix is what makes the public url
        # `<workspace>-<suffix>--<label>.modal.run`, so two placements that differ only here
        # are two different endpoints. omitting it would collapse them to one `spec_id`.
        "web_suffix": value.web_suffix,
        "gpu": value.gpu,
        "region": value.region,
        "gpu_count": value.gpu_count,
        "provider": value.provider,
    }


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
