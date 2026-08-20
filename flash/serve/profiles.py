"""complete immutable serving inputs for one model on one supported provider.

``provision_modal_deployment`` and ``provision_runpod_deployment`` take a ``DeploymentBundle``,
which requires an exact ``EngineIdentity`` (27 fields), an exact provider ``Placement``, and a
digest-qualified ``ServingImage``. Nothing in flash produced those, so the provisioning code had
no caller outside its tests. This module is that producer.

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
    RunPodPlacement,
    canonical_mapping_fingerprint,
)
from flash.serve.provisioning import ServingImage

# the serving image installs `vllm==0.23.0` (pyproject `serve-runtime`). runtime_family is part of
# the engine identity so a vllm upgrade forces a new engine rather than silently reusing an id
# validated against a different engine build.
SERVE_RUNTIME_FAMILY = "vllm-0.23.0"


class ProfileError(ValueError):
    """the requested serving profile is unknown or its inputs are incomplete."""


@dataclass(frozen=True, slots=True)
class RunPodGpu:
    """one runpod gpu type id with the container and volume sizing it is validated for.

    ``gpu_type_id`` is runpod's own display id (the value its api returns as ``gpuTypeId``), not a
    flash GPU_CLASSES name. flash's runpod training path resolves cards through the runpod-flash
    SDK's ``GpuType`` enum, which the serving path deliberately does not import: the provisioning
    transport speaks the rest api directly, and the SDK is not in the serving install. The two are
    also not interchangeable strings, so this is stated rather than translated.
    """

    gpu_type_id: str
    container_disk_gb: int
    volume_size_gb: int


# runpod's L40S and L4 ids. catalog `serving.gpu` holds MODAL gpu names, and L4/L40S have no
# GPU_CLASSES row at all (that table covers training cards), so a runpod id cannot be derived
# from either and is stated per profile below.
# containerDiskInGb must hold the EXTRACTED image, not the registry download. the serving image is
# 13.7 GB compressed but 40.7 GB on disk (`docker system df -v`), and the container disk also holds
# the extraction scratch and the runtime's own writes. the previous 40 GB was read off the
# compressed number, so the image could not fit on the disk it was pulled onto at all.
#
# do not try to confirm this from the runpod api's `runtime` field: it reads null on pods that are
# serving fine. the pod proxy is the signal that discriminates -- a live pod answers
# `https://{podId}-8000.proxy.runpod.net/` with 200 and an exited one with 404.
_RUNPOD_L40S = RunPodGpu(gpu_type_id="NVIDIA L40S", container_disk_gb=100, volume_size_gb=120)
# 24 GB card, so the VOLUME (weights and adapters) is sized down relative to the 9B set. the
# container disk is NOT sized down: it holds the same image on either card. both ids are the exact
# `gpuTypes.id` strings the runpod api returns ("NVIDIA L4", "NVIDIA L40S"), not display names.
_RUNPOD_L4 = RunPodGpu(gpu_type_id="NVIDIA L4", container_disk_gb=100, volume_size_gb=60)


@dataclass(frozen=True, slots=True)
class ServingProfile:
    """every immutable input one model needs to reach a supported provider."""

    model_id: str
    modality: Modality
    served_model: str
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
    runpod_gpu: RunPodGpu
    tensor_parallel_size: int = 1

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
            gpu=self.modal_gpu,
            region=region,
            gpu_count=self.tensor_parallel_size,
            web_suffix=web_suffix,
        )

    def runpod_placement(self, *, account_id: str, data_center_id: str) -> RunPodPlacement:
        """build the exact persistent-pod placement for this profile's validated gpu."""

        return RunPodPlacement(
            account_id=account_id,
            gpu_type_id=self.runpod_gpu.gpu_type_id,
            gpu_count=self.tensor_parallel_size,
            data_center_id=data_center_id,
            container_disk_gb=self.runpod_gpu.container_disk_gb,
            volume_size_gb=self.runpod_gpu.volume_size_gb,
        )


# only models whose serving shape is validated on BOTH supported providers appear here. a catalog
# entry alone is not enough: `serving.gpu` names a modal card, and the runpod id, container disk,
# and volume size have no catalog source. an unlisted model raises rather than defaulting, because
# a guessed placement is a real gpu rental in the customer's account.
_PROFILES: dict[str, ServingProfile] = {
    "Qwen/Qwen3.5-9B": ServingProfile(
        model_id="Qwen/Qwen3.5-9B",
        modality="multimodal",
        # the engine loads the freesolo-owned fp8 checkpoint, while adapters declare the base model
        # they trained against. catalog `serving.serve_model_id` is the authority for that split.
        served_model="Freesolo-Co/Qwen3.5-9B-FP8",
        tokenizer_model="Freesolo-Co/Qwen3.5-9B-FP8",
        dtype="bfloat16",
        # none, not "fp8": these checkpoints declare quant_method "compressed-tensors" in
        # their own config, and vllm rejects a `quantization` argument that disagrees with
        # the checkpoint rather than treating it as a hint. the checkpoint is the authority,
        # so nothing is forced here.
        quantization=None,
        kv_cache_dtype=None,
        max_model_len=32768,
        max_num_seqs=8,
        max_num_batched_tokens=None,
        max_loras=16,
        max_cpu_loras=16,
        max_lora_rank=128,
        gpu_memory_utilization=0.90,
        cpu_offload_gb=0.0,
        # multimodal, not text: both served checkpoints are Qwen3_5ForConditionalGeneration with
        # a vision_config, and flash trains image loras on both (catalog supports_image_training is
        # true for each). declaring text here loaded no processor and passed no limit_mm_per_prompt,
        # so a customer who trained an image adapter got an engine that rejected every image request.
        #
        # 4 matches _MAX_IMAGES, the ceiling the runtime already clamps to, so the engine advertises
        # exactly what it will accept rather than a larger number it would silently trim.
        image_limit=4,
        mm_processor_cache_gb=0.0,
        # true, because flash's own image adapters contain vision-tower weights. training targets
        # "all-linear", which peft resolves to include model.visual.blocks.*.attn.{qkv,proj} and
        # .mlp.linear_fc{1,2}; real image runs publish 196 such tensors out of 692. with this false
        # vllm wraps no visual.* module, so those suffixes are missing from expected_lora_modules
        # and from_local_checkpoint RAISES on them ("expected target modules in ... but received").
        # a customer who trained on images would get a deployment that refuses to load the adapter.
        enable_tower_connector_lora=True,
        reasoning_parser=None,
        engine_args={},
        tokenizer_kwargs={},
        processor_kwargs={},
        modal_gpu="L40S",
        runpod_gpu=_RUNPOD_L40S,
    ),
    "Qwen/Qwen3.5-4B": ServingProfile(
        model_id="Qwen/Qwen3.5-4B",
        modality="multimodal",
        served_model="Freesolo-Co/Qwen3.5-4B-FP8",
        tokenizer_model="Freesolo-Co/Qwen3.5-4B-FP8",
        dtype="bfloat16",
        # none, not "fp8": these checkpoints declare quant_method "compressed-tensors" in
        # their own config, and vllm rejects a `quantization` argument that disagrees with
        # the checkpoint rather than treating it as a hint. the checkpoint is the authority,
        # so nothing is forced here.
        quantization=None,
        kv_cache_dtype=None,
        max_model_len=32768,
        max_num_seqs=8,
        max_loras=16,
        max_cpu_loras=16,
        max_lora_rank=128,
        max_num_batched_tokens=None,
        # 0.98, not the 9B's 0.90: the catalog states it per model and _require_catalog_agreement
        # rejects any drift from it, so this is copied from the catalog rather than chosen here.
        gpu_memory_utilization=0.98,
        cpu_offload_gb=0.0,
        # image-capable for the same reason as the 9B above.
        image_limit=4,
        mm_processor_cache_gb=0.0,
        # true for the same reason as the 9B above.
        enable_tower_connector_lora=True,
        reasoning_parser=None,
        engine_args={},
        tokenizer_kwargs={},
        processor_kwargs={},
        modal_gpu="L4",
        runpod_gpu=_RUNPOD_L4,
    ),
}

_CATALOG_CHECKED_FIELDS = (
    "max_model_len",
    "max_num_seqs",
    "max_loras",
    "max_lora_rank",
    "gpu_memory_utilization",
)


def supported_models() -> tuple[str, ...]:
    """return every model id with a complete profile, in catalog order."""

    return tuple(sorted(_PROFILES))


def get_profile(model_id: str) -> ServingProfile:
    """return the complete profile for one model, or raise.

    fails closed on an unknown model AND on a profile that has drifted from the catalog. the
    catalog is the product's serving capacity contract, so a profile that quietly serves a longer
    context or a higher lora rank than the catalog advertises would deploy a shape no gate checked.
    """

    profile = _PROFILES.get(model_id)
    if profile is None:
        known = ", ".join(supported_models()) or "none"
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
        if expected != actual:
            raise ProfileError(
                f"{profile.model_id} serving profile {name}={actual!r} disagrees with the "
                f"catalog capacity {expected!r}"
            )
    for label, expected, actual in (
        ("served_model", serving.serve_model_id, profile.served_model),
        ("modal gpu", serving.gpu, profile.modal_gpu),
    ):
        if expected != actual:
            raise ProfileError(
                f"{profile.model_id} serving profile {label}={actual!r} disagrees with the "
                f"catalog capacity {expected!r}"
            )


def placement_for(
    profile: ServingProfile,
    provider: Provider,
    *,
    workspace_name: str = "",
    environment: str = "",
    region: str = "",
    web_suffix: str | None = None,
    account_id: str = "",
    data_center_id: str = "",
) -> ModalPlacement | RunPodPlacement:
    """build the placement for one provider, requiring exactly that provider's inputs.

    the unused provider's arguments are rejected rather than ignored: passing a runpod data center
    to a modal deployment means the caller believes something untrue about where this will run.
    """

    if provider == "modal":
        _reject_foreign(provider, (("account_id", account_id), ("data_center_id", data_center_id)))
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
    if provider == "runpod":
        _reject_foreign(
            provider,
            (
                ("workspace_name", workspace_name),
                ("environment", environment),
                ("region", region),
            ),
        )
        _require_inputs(provider, (("account_id", account_id), ("data_center_id", data_center_id)))
        return profile.runpod_placement(account_id=account_id, data_center_id=data_center_id)
    raise ProfileError("provider must be modal or runpod")


def _require_inputs(provider: str, supplied: tuple[tuple[str, str], ...]) -> None:
    missing = [name for name, value in supplied if not value.strip()]
    if missing:
        raise ProfileError(f"{provider} placement requires {', '.join(sorted(missing))}")


def _reject_foreign(provider: str, supplied: tuple[tuple[str, str], ...]) -> None:
    present = [name for name, value in supplied if value.strip()]
    if present:
        raise ProfileError(f"{provider} placement does not accept {', '.join(sorted(present))}")
