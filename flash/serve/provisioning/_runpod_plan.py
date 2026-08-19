"""prevalidated immutable runpod create plan without secret values."""

from __future__ import annotations

import json
from dataclasses import dataclass

from flash.serve.control import RunPodPlacement

from ._common import (
    MAX_ENCODED_MANIFEST_BYTES,
    DeploymentBundle,
    ServingResourceNames,
    encode_manifest_environment,
    serving_resource_names,
)
from ._runpod_protocol import (
    LAUNCH_COMMAND_ARGV,
    NETWORK_VOLUME_MOUNT,
    PROXY_PORT,
    PROXY_PORT_SPEC,
    SERVING_CACHE_ROOT,
    secret_reference,
)

MAX_RUNPOD_REQUEST_BYTES = 128 * 1024
MAX_RUNPOD_ENVIRONMENT_BYTES = 96 * 1024


def _canonical_payload(value: dict[str, object], name: str) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > MAX_RUNPOD_REQUEST_BYTES:
        raise ValueError(f"{name} exceeds the runpod request byte limit")
    return encoded


def _payload(encoded: str) -> dict[str, object]:
    value = json.loads(encoded)
    if type(value) is not dict:
        raise AssertionError("prevalidated runpod payload is not an object")
    return value


def _environment(
    bundle: DeploymentBundle,
    names: ServingResourceNames,
    encoded_manifest: str,
    *,
    include_artifact_secret: bool,
) -> tuple[tuple[str, str], ...]:
    values = {
        "FLASH_INFERENCE_TOKEN": secret_reference(names.inference_secret),
        "FLASH_SERVING_CACHE_ROOT": SERVING_CACHE_ROOT,
        "FLASH_SERVING_HOST": "0.0.0.0",
        "FLASH_SERVING_IMAGE_DIGEST": bundle.image.digest,
        "FLASH_SERVING_MANIFEST": encoded_manifest,
        "FLASH_SERVING_MANIFEST_ID": bundle.manifest.manifest_id,
        "FLASH_SERVING_PORT": str(PROXY_PORT),
    }
    if include_artifact_secret:
        values["FLASH_ARTIFACT_TOKEN"] = secret_reference(names.artifact_secret)
    environment = tuple(sorted(values.items()))
    serialized = _canonical_payload(
        {"env": [{"key": key, "value": value} for key, value in environment]},
        "runpod template environment",
    )
    if len(serialized.encode("utf-8")) > MAX_RUNPOD_ENVIRONMENT_BYTES:
        raise ValueError("runpod template environment exceeds its byte limit")
    return environment


@dataclass(frozen=True, slots=True)
class RunPodCreatePlan:
    """one immutable fully serialized runpod resource plan without secret values."""

    bundle: DeploymentBundle
    placement: RunPodPlacement
    names: ServingResourceNames
    encoded_manifest: str
    environment_without_artifact: tuple[tuple[str, str], ...]
    environment_with_artifact: tuple[tuple[str, str], ...]
    volume_json: str
    template_without_artifact_json: str
    template_with_artifact_json: str
    pod_static_json: str

    def environment(self, include_artifact_secret: bool) -> tuple[tuple[str, str], ...]:
        return (
            self.environment_with_artifact
            if include_artifact_secret
            else self.environment_without_artifact
        )

    def volume_payload(self) -> dict[str, object]:
        return _payload(self.volume_json)

    def template_payload(self, include_artifact_secret: bool) -> dict[str, object]:
        return _payload(
            self.template_with_artifact_json
            if include_artifact_secret
            else self.template_without_artifact_json
        )

    def pod_payload(self, *, template_id: str, volume_id: str) -> dict[str, object]:
        payload = _payload(self.pod_static_json)
        payload["networkVolumeId"] = volume_id
        payload["templateId"] = template_id
        _canonical_payload(payload, "runpod pod payload")
        return payload


def build_runpod_create_plan(bundle: DeploymentBundle) -> RunPodCreatePlan:
    """validate every deterministic runpod create input before provider access."""

    if type(bundle) is not DeploymentBundle:
        raise ValueError("bundle must be an exact DeploymentBundle")
    bundle.__post_init__()
    if bundle.spec.provider != "runpod" or type(bundle.spec.placement) is not RunPodPlacement:
        raise ValueError("runpod provisioning requires a runpod deployment bundle")
    placement = bundle.spec.placement
    names = serving_resource_names(
        bundle.spec.deployment_id,
        bundle.spec.generation,
        bundle.spec.engine.engine_id,
        workload_role="pod",
    )
    encoded_manifest = encode_manifest_environment(bundle.manifest)
    if len(encoded_manifest.encode("ascii")) > MAX_ENCODED_MANIFEST_BYTES:
        raise ValueError("encoded serving manifest exceeds its environment limit")
    without_artifact = _environment(
        bundle,
        names,
        encoded_manifest,
        include_artifact_secret=False,
    )
    with_artifact = _environment(
        bundle,
        names,
        encoded_manifest,
        include_artifact_secret=True,
    )

    volume = {
        "dataCenterId": placement.data_center_id,
        "name": names.volume,
        "size": placement.volume_size_gb,
    }
    template_base: dict[str, object] = {
        "containerDiskInGb": placement.container_disk_gb,
        # runpod's rest schema types TemplateCreateInput.dockerStartCmd as array<string>, and
        # GET /templates returns it the same way. sending the bare string was accepted by the
        # transport and only failed later, when parse_templates read the list back.
        "dockerStartCmd": list(LAUNCH_COMMAND_ARGV),
        "imageName": bundle.image.reference,
        "isServerless": False,
        "name": names.template,
        "ports": [PROXY_PORT_SPEC],
        "volumeInGb": 0,
        "volumeMountPath": NETWORK_VOLUME_MOUNT,
    }

    def template(environment: tuple[tuple[str, str], ...]) -> dict[str, object]:
        # env is an object in runpod's schema, not a list of {key, value} rows. the pair-list form
        # round-trips through _environment on read, so this mismatch stayed invisible offline.
        return {**template_base, "env": dict(environment)}

    pod_static = {
        "containerDiskInGb": placement.container_disk_gb,
        "dataCenterIds": [placement.data_center_id],
        "gpuCount": placement.gpu_count,
        "gpuTypeIds": [placement.gpu_type_id],
        "imageName": bundle.image.reference,
        "name": names.app_or_pod,
        "ports": [PROXY_PORT_SPEC],
    }
    return RunPodCreatePlan(
        bundle=bundle,
        placement=placement,
        names=names,
        encoded_manifest=encoded_manifest,
        environment_without_artifact=without_artifact,
        environment_with_artifact=with_artifact,
        volume_json=_canonical_payload(volume, "runpod network volume payload"),
        template_without_artifact_json=_canonical_payload(
            template(without_artifact), "runpod template payload"
        ),
        template_with_artifact_json=_canonical_payload(
            template(with_artifact), "runpod template payload"
        ),
        pod_static_json=_canonical_payload(pod_static, "runpod pod static payload"),
    )
