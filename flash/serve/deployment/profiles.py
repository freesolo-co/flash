"""complete immutable serving inputs for customer-owned modal deployments.

``provision_modal_deployment`` takes a ``DeploymentBundle``, which requires an exact
``EngineIdentity`` (26 fields), an exact ``ModalPlacement``, and a digest-qualified
``ServingImage``. This module is that producer.

Every value here is immutable serving identity: it feeds ``engine_id``, which is the sha-256 of
the canonical engine json. Two deployments agreeing on every field share an engine; changing any
field is a different engine and a different deployment. So the registry states values rather than
deriving them from mutable runtime state, and refuses to fill in a default for anything missing.

The engine kwargs are carried, not their fingerprints. ``build_serving_manifest`` recomputes the
fingerprints from the kwargs and rejects the bundle if they disagree, so storing a fingerprint
beside the mapping it describes would create a second source of truth that can drift. Deriving
them makes an inconsistent profile unconstructible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from flash.core.catalog import ModelInfo, get_model
from flash.serve.control import (
    EngineIdentity,
    Modality,
    ModalPlacement,
    Provider,
    canonical_mapping_fingerprint,
)
from flash.serve.provisioning import ServingImage

# the serving image installs `vllm==0.23.0` plus the exact pr42120 moe lora backport. runtime_family
# is part of the engine identity so a runtime repair or upgrade cannot reuse an engine id validated
# against different execution bytes.
SERVE_RUNTIME_FAMILY = "vllm-0.23.0-pr42120"
_CERTIFIED_MODAL_IMAGE_DIGEST = (
    "sha256:2bf27b51f6e4b7f0b2d805d96202579d94868e2c594b7c496777d350ad6936f6"
)


class ProfileError(ValueError):
    """the requested serving profile is unknown or its inputs are incomplete."""


@dataclass(frozen=True, slots=True)
class ServingProfile:
    """every immutable input one model needs to reach a supported provider."""

    model_id: str
    modality: Modality
    served_model: str
    served_model_revision: str | None
    tokenizer_model: str
    dtype: str
    quantization: str | None
    kv_cache_dtype: str | None
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
    engine_args: Mapping[str, Any]
    tokenizer_kwargs: Mapping[str, Any]
    processor_kwargs: Mapping[str, Any]
    modal_gpu: str
    modal_gpu_request: str
    modal_live_qualified: bool
    tensor_parallel_size: int = 1
    modal_certified_image_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine_args", MappingProxyType(dict(self.engine_args)))
        object.__setattr__(self, "tokenizer_kwargs", MappingProxyType(dict(self.tokenizer_kwargs)))
        object.__setattr__(self, "processor_kwargs", MappingProxyType(dict(self.processor_kwargs)))

    def engine(
        self,
        *,
        model_revision: str,
        tokenizer_revision: str,
        image: ServingImage,
        trust_remote_code: bool = False,
    ) -> EngineIdentity:
        """build the exact engine identity for one resolved model and image.

        the three revisions and the image digest are runtime-resolved immutables, not registry
        constants: a profile describes an engine shape, and pinning a commit here would go stale
        the moment the checkpoint moves. ``EngineIdentity`` validates each of them.
        """

        return EngineIdentity(
            served_model=self.served_model,
            model_revision=model_revision,
            tokenizer_model=self.tokenizer_model,
            tokenizer_revision=tokenizer_revision,
            image_digest=image.digest,
            modality=self.modality,
            runtime_family=SERVE_RUNTIME_FAMILY,
            dtype=self.dtype,
            quantization=self.quantization,
            kv_cache_dtype=self.kv_cache_dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_loras=self.max_loras,
            max_cpu_loras=self.max_cpu_loras,
            max_lora_rank=self.max_lora_rank,
            gpu_memory_utilization=self.gpu_memory_utilization,
            cpu_offload_gb=self.cpu_offload_gb,
            image_limit=self.image_limit,
            mm_processor_cache_gb=self.mm_processor_cache_gb,
            enable_tower_connector_lora=self.enable_tower_connector_lora,
            reasoning_parser=self.reasoning_parser,
            trust_remote_code=trust_remote_code,
            engine_args_fingerprint=canonical_mapping_fingerprint(self.engine_args),
            tokenizer_kwargs_fingerprint=canonical_mapping_fingerprint(self.tokenizer_kwargs),
            processor_kwargs_fingerprint=canonical_mapping_fingerprint(self.processor_kwargs),
        )

    def modal_placement(
        self, *, workspace_name: str, environment: str, region: str, web_suffix: str | None = None
    ) -> ModalPlacement:
        """build the exact modal placement for this profile's validated gpu.

        region is required rather than defaulted: `_validate_placement` rejects None, so a profile
        that hardcoded it produced a placement the planner could never provision. it is also a
        deployment decision like the workspace, not a property of the model, and modal prices a
        pinned region above an unpinned one, so the caller states it.
        """

        return ModalPlacement(
            workspace_name=workspace_name,
            environment=environment,
            gpu=self.modal_gpu_request,
            region=region,
            gpu_count=self.tensor_parallel_size,
            web_suffix=web_suffix,
        )


# every public catalog model has an explicit immutable profile. provider qualification remains a
# separate fact: provisional placement data may build an offline plan, but cannot allocate a gpu.
_PROFILES: dict[str, ServingProfile] = {
    "Qwen/Qwen3.5-9B": ServingProfile(
        model_id="Qwen/Qwen3.5-9B",
        modality="multimodal",
        served_model="Freesolo-Co/Qwen3.5-9B-FP8",
        served_model_revision=None,
        tokenizer_model="Freesolo-Co/Qwen3.5-9B-FP8",
        dtype="bfloat16",
        quantization=None,
        kv_cache_dtype="fp8",
        max_model_len=32768,
        max_num_seqs=8,
        max_num_batched_tokens=None,
        max_loras=16,
        max_cpu_loras=16,
        max_lora_rank=128,
        gpu_memory_utilization=0.90,
        cpu_offload_gb=0.0,
        image_limit=4,
        mm_processor_cache_gb=0.0,
        enable_tower_connector_lora=True,
        reasoning_parser="qwen3",
        engine_args={},
        tokenizer_kwargs={},
        processor_kwargs={},
        modal_gpu="L40S",
        modal_gpu_request="L40S",
        modal_live_qualified=True,
    ),
    "Qwen/Qwen3.8-27B": ServingProfile(
        model_id="Qwen/Qwen3.8-27B",
        modality="multimodal",
        served_model="Qwen/Qwen3.8-27B-FP8",
        served_model_revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        tokenizer_model="Qwen/Qwen3.8-27B",
        dtype="bfloat16",
        quantization=None,
        kv_cache_dtype="fp8",
        max_model_len=32768,
        max_num_seqs=8,
        max_num_batched_tokens=None,
        max_loras=16,
        max_cpu_loras=16,
        max_lora_rank=64,
        gpu_memory_utilization=0.90,
        cpu_offload_gb=0.0,
        image_limit=4,
        mm_processor_cache_gb=0.0,
        enable_tower_connector_lora=True,
        reasoning_parser="qwen3",
        engine_args={"enforce_eager": False},
        tokenizer_kwargs={},
        processor_kwargs={},
        modal_gpu="H100",
        # modal's trailing `!` forbids automatic h200 substitution for an h100 request.
        modal_gpu_request="H100!",
        modal_live_qualified=True,
        modal_certified_image_digest=_CERTIFIED_MODAL_IMAGE_DIGEST,
    ),
    "Qwen/Qwen3.6-35B-A3B": ServingProfile(
        model_id="Qwen/Qwen3.6-35B-A3B",
        modality="multimodal",
        served_model="Qwen/Qwen3.6-35B-A3B",
        served_model_revision=None,
        tokenizer_model="Qwen/Qwen3.6-35B-A3B",
        dtype="bfloat16",
        quantization=None,
        kv_cache_dtype="fp8",
        max_model_len=32768,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        max_loras=6,
        max_cpu_loras=6,
        max_lora_rank=64,
        gpu_memory_utilization=0.90,
        cpu_offload_gb=0.0,
        image_limit=4,
        mm_processor_cache_gb=0.0,
        enable_tower_connector_lora=True,
        reasoning_parser="qwen3",
        engine_args={"enforce_eager": False},
        tokenizer_kwargs={},
        processor_kwargs={},
        modal_gpu="H200",
        modal_gpu_request="H200",
        modal_live_qualified=True,
        modal_certified_image_digest=_CERTIFIED_MODAL_IMAGE_DIGEST,
    ),
}

_CATALOG_CHECKED_FIELDS = (
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_loras",
    "max_cpu_loras",
    "max_lora_rank",
    "tensor_parallel_size",
    "gpu_memory_utilization",
    "image_limit",
)


def _public_catalog_models() -> frozenset[str]:
    from flash.core.catalog import MODELS

    return frozenset(MODELS)


def _require_profile_string(value: object, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ProfileError(f"{name} must be a nonempty unpadded string")


def _require_certified_image_digest(value: object, name: str) -> None:
    if value is None:
        return
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ProfileError(f"{name} must be sha256: followed by 64 lowercase hex characters")


def _require_profile_structure(profile: ServingProfile) -> None:
    """validate registry-only facts before any artifact or provider access."""

    for name in (
        "model_id",
        "served_model",
        "tokenizer_model",
        "dtype",
        "modal_gpu",
        "modal_gpu_request",
    ):
        _require_profile_string(getattr(profile, name), f"{profile.model_id} {name}")
    if profile.modal_gpu_request not in {profile.modal_gpu, f"{profile.modal_gpu}!"}:
        raise ProfileError(
            f"{profile.model_id} modal_gpu_request must match modal_gpu with an optional exact pin"
        )
    if type(profile.modal_live_qualified) is not bool:
        raise ProfileError(f"{profile.model_id} modal_live_qualified must be an exact bool")
    _require_certified_image_digest(
        profile.modal_certified_image_digest,
        f"{profile.model_id} modal_certified_image_digest",
    )
    revision = profile.served_model_revision
    if revision is not None and (
        type(revision) is not str
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ProfileError(f"{profile.model_id} served_model_revision is not an immutable commit")
    try:
        profile.engine(
            model_revision=revision or "0" * 40,
            tokenizer_revision="1" * 40,
            image=ServingImage(
                reference="registry.example/flash/profile-validation@sha256:" + "2" * 64,
                digest="sha256:" + "2" * 64,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{profile.model_id} has invalid engine inputs: {exc}") from exc


def _require_registry_complete() -> None:
    expected = _public_catalog_models()
    actual = frozenset(_PROFILES)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        extra = ", ".join(sorted(actual - expected)) or "none"
        raise ProfileError(
            f"customer-owned serving profile registry disagrees with the public catalog: "
            f"missing={missing}; extra={extra}"
        )
    for key, profile in _PROFILES.items():
        if type(profile) is not ServingProfile:
            raise ProfileError(f"customer-owned serving profile {key!r} must use ServingProfile")
        if key != profile.model_id:
            raise ProfileError(
                f"customer-owned serving profile key {key!r} disagrees with "
                f"profile.model_id={profile.model_id!r}"
            )
        _require_profile_structure(profile)
        _require_catalog_agreement(profile)


def supported_models() -> tuple[str, ...]:
    """return every public catalog model after validating the whole registry."""

    _require_registry_complete()
    return tuple(sorted(_PROFILES))


def get_profile(model_id: str) -> ServingProfile:
    """return the complete profile for one model, or raise.

    fails closed on an unknown model AND on a profile that has drifted from the catalog. the
    catalog is the product's serving capacity contract, so a profile that quietly serves a longer
    context or a higher lora rank than the catalog advertises would deploy a shape no gate checked.
    """

    _require_registry_complete()
    profile = _PROFILES.get(model_id)
    if profile is None:
        known = ", ".join(sorted(_PROFILES)) or "none"
        raise ProfileError(
            f"no customer-owned serving profile for {model_id!r}. supported: {known}"
        )
    _require_catalog_agreement(profile)
    return profile


def _require_catalog_agreement(profile: ServingProfile) -> None:
    info: ModelInfo = get_model(profile.model_id)
    serving = info.serving
    if serving is None:
        raise ProfileError(
            f"{profile.model_id} has a serving profile but no catalog serving capacity"
        )
    for name in _CATALOG_CHECKED_FIELDS:
        expected = getattr(serving, name)
        actual = getattr(profile, name)
        if name == "max_num_batched_tokens":
            expected = expected or None
            actual = actual or None
        if expected != actual:
            raise ProfileError(
                f"{profile.model_id} serving profile {name}={actual!r} disagrees with the "
                f"catalog capacity {expected!r}"
            )
    # NB: modal_gpu is deliberately NOT checked against serving.gpu. They describe different
    # planes: serving.gpu is the card the freesolo-owned hosted plane runs on, while modal_gpu is
    # the card THIS customer-owned profile was live-qualified on (modal_live_qualified). The hosted
    # plane moved to B200; the customer profiles stay on their qualified cards until a customer-owned
    # B200 qualification runs. Engine SHAPE still has to agree -- see _CATALOG_CHECKED_FIELDS, which
    # is what stops a profile serving a longer context or higher lora rank than the catalog
    # advertises. The card is placement, not shape.
    for label, expected, actual in (
        ("served_model", serving.serve_model_id, profile.served_model),
    ):
        if expected != actual:
            raise ProfileError(
                f"{profile.model_id} serving profile {label}={actual!r} disagrees with the "
                f"catalog capacity {expected!r}"
            )


def require_live_qualification(
    profile: ServingProfile, provider: Provider, image_digest: str
) -> None:
    """reject modal allocation unless the requested live shape was certified."""

    if provider != "modal":
        raise ProfileError("provider must be modal")
    if not profile.modal_live_qualified:
        raise ProfileError(
            f"{profile.model_id} modal serving profile is pending exact live qualification; "
            "offline dry-run construction is available, but provider allocation is disabled"
        )
    certified_digest = profile.modal_certified_image_digest
    if certified_digest is not None and image_digest != certified_digest:
        raise ProfileError(
            f"{profile.model_id} modal serving profile is qualified only for certified image "
            f"digest {certified_digest}; requested {image_digest}"
        )


def placement_for(
    profile: ServingProfile,
    provider: Provider,
    *,
    workspace_name: str = "",
    environment: str = "",
    region: str = "",
    web_suffix: str | None = None,
) -> ModalPlacement:
    """build the exact modal placement from explicit operator inputs."""

    if provider != "modal":
        raise ProfileError("provider must be modal")
    _require_inputs(
        provider,
        (
            ("workspace_name", workspace_name),
            ("environment", environment),
            ("region", region),
        ),
    )
    return profile.modal_placement(
        workspace_name=workspace_name,
        environment=environment,
        region=region,
        web_suffix=web_suffix,
    )


def _require_inputs(provider: str, supplied: tuple[tuple[str, str], ...]) -> None:
    missing = [name for name, value in supplied if not value.strip()]
    if missing:
        raise ProfileError(f"{provider} placement requires {', '.join(sorted(missing))}")
