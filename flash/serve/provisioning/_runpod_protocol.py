"""strict runpod resource records, parsing, and exact request payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypeAlias

from flash.serve.control import RunPodPlacement
from flash.serve.control._urls import validate_runpod_pod_id

from ._common import DeploymentBundle, ServingResourceNames, encode_manifest_environment

PROXY_PORT = 8000
PROXY_PORT_SPEC = "8000/http"
NETWORK_VOLUME_MOUNT = "/runpod-volume"
SERVING_CACHE_ROOT = "/runpod-volume/flash-serving"
LAUNCH_COMMAND = "python /app/serve_launch.py"

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,127}")

LIST_ACCOUNT_SECRETS = """
query FlashServingAccountSecrets {
  myself {
    id
    secrets {
      id
      name
    }
  }
}
""".strip()

CREATE_SECRET = """
mutation FlashServingCreateSecret($name: String!, $value: String!) {
  createSecret(input: {name: $name, value: $value}) {
    id
    name
  }
}
""".strip()

DELETE_SECRET = """
mutation FlashServingDeleteSecret($id: String!) {
  deleteSecret(input: {id: $id})
}
""".strip()


@dataclass(frozen=True, slots=True)
class RunPodSecretObservation:
    """one opaque runpod secret identity without its value."""

    id: str
    name: str


@dataclass(frozen=True, slots=True)
class RunPodTemplateObservation:
    """one parsed persistent pod template."""

    id: str
    name: str
    image_name: str
    docker_start_cmd: str
    container_disk_gb: int
    volume_gb: int
    volume_mount_path: str
    ports: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    is_serverless: bool


@dataclass(frozen=True, slots=True)
class RunPodVolumeObservation:
    """one parsed network volume."""

    id: str
    name: str
    size_gb: int
    data_center_id: str


@dataclass(frozen=True, slots=True)
class RunPodPodObservation:
    """one parsed persistent pod."""

    id: str
    name: str
    desired_status: str
    image_name: str
    gpu_type_id: str
    gpu_count: int
    data_center_id: str
    container_disk_gb: int
    network_volume_id: str
    template_id: str
    ports: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunPodObservation:
    """one authoritative account-scoped view of deterministic resources."""

    account_id: str
    inference_secrets: tuple[RunPodSecretObservation, ...]
    artifact_secrets: tuple[RunPodSecretObservation, ...]
    templates: tuple[RunPodTemplateObservation, ...]
    volumes: tuple[RunPodVolumeObservation, ...]
    pods: tuple[RunPodPodObservation, ...]

    @property
    def resource_count(self) -> int:
        return sum(
            len(values)
            for values in (
                self.inference_secrets,
                self.artifact_secrets,
                self.templates,
                self.volumes,
                self.pods,
            )
        )


ResourceObservation: TypeAlias = (
    RunPodSecretObservation
    | RunPodTemplateObservation
    | RunPodVolumeObservation
    | RunPodPodObservation
)


def _mapping(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty unpadded string")
    return value


def _provider_id(value: object, name: str) -> str:
    selected = _string(value, name)
    if _ID_RE.fullmatch(selected) is None:
        raise ValueError(f"{name} is malformed")
    return selected


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _ports(value: object, name: str) -> tuple[str, ...]:
    if type(value) is str:
        raw = value.split(",") if value else []
    elif type(value) is list:
        raw = value
    else:
        raise ValueError(f"{name} must be a string or list")
    parsed = tuple(_string(item, name) for item in raw)
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} contains duplicates")
    return parsed


def _environment(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is dict:
        items = list(value.items())
    elif type(value) is list:
        items = []
        for entry in value:
            row = _mapping(entry, "template env entry")
            items.append((row.get("key"), row.get("value")))
    else:
        raise ValueError("template env must be an object or list")
    parsed = tuple(
        sorted((_string(key, "env key"), _string(val, "env value")) for key, val in items)
    )
    if len(parsed) != len({key for key, _value in parsed}):
        raise ValueError("template env contains duplicate keys")
    return parsed


def _resource_rows(value: object, key: str) -> list[object]:
    if type(value) is list:
        return value
    root = _mapping(value, f"{key} response")
    rows = root.get(key)
    if type(rows) is not list:
        raise ValueError(f"{key} response must contain a list")
    return rows


def parse_account_secrets(value: object) -> tuple[str, tuple[RunPodSecretObservation, ...]]:
    root = _mapping(value, "graphql response")
    if "errors" in root:
        raise ValueError("graphql response contains errors")
    data = _mapping(root.get("data"), "graphql data")
    myself = _mapping(data.get("myself"), "graphql myself")
    account_id = _provider_id(myself.get("id"), "account id")
    rows = myself.get("secrets")
    if type(rows) is not list:
        raise ValueError("graphql secrets must be a list")
    secrets = []
    for entry in rows:
        row = _mapping(entry, "secret")
        secrets.append(
            RunPodSecretObservation(
                id=_provider_id(row.get("id"), "secret id"),
                name=_string(row.get("name"), "secret name"),
            )
        )
    return account_id, tuple(secrets)


def parse_created_secret(value: object) -> RunPodSecretObservation:
    root = _mapping(value, "graphql response")
    if "errors" in root:
        raise ValueError("graphql response contains errors")
    data = _mapping(root.get("data"), "graphql data")
    row = _mapping(data.get("createSecret"), "created secret")
    return RunPodSecretObservation(
        id=_provider_id(row.get("id"), "secret id"),
        name=_string(row.get("name"), "secret name"),
    )


def parse_deleted_secret(value: object) -> bool:
    root = _mapping(value, "graphql response")
    if "errors" in root:
        raise ValueError("graphql response contains errors")
    data = _mapping(root.get("data"), "graphql data")
    if data.get("deleteSecret") is not True:
        raise ValueError("deleted secret response is malformed")
    return True


def parse_templates(value: object) -> tuple[RunPodTemplateObservation, ...]:
    parsed = []
    for entry in _resource_rows(value, "templates"):
        row = _mapping(entry, "template")
        parsed.append(
            RunPodTemplateObservation(
                id=_provider_id(row.get("id"), "template id"),
                name=_string(row.get("name"), "template name"),
                image_name=_string(row.get("imageName"), "template imageName"),
                docker_start_cmd=_string(row.get("dockerStartCmd"), "template dockerStartCmd"),
                container_disk_gb=_positive_int(
                    row.get("containerDiskInGb"), "template containerDiskInGb"
                ),
                volume_gb=_nonnegative_int(row.get("volumeInGb"), "template volumeInGb"),
                volume_mount_path=_string(row.get("volumeMountPath"), "template volumeMountPath"),
                ports=_ports(row.get("ports"), "template ports"),
                environment=_environment(row.get("env")),
                is_serverless=row.get("isServerless")
                if type(row.get("isServerless")) is bool
                else _invalid_bool("template isServerless"),
            )
        )
    return tuple(parsed)


def _invalid_bool(name: str) -> bool:
    raise ValueError(f"{name} must be a boolean")


def parse_volumes(value: object) -> tuple[RunPodVolumeObservation, ...]:
    parsed = []
    for entry in _resource_rows(value, "networkVolumes"):
        row = _mapping(entry, "network volume")
        parsed.append(
            RunPodVolumeObservation(
                id=_provider_id(row.get("id"), "network volume id"),
                name=_string(row.get("name"), "network volume name"),
                size_gb=_positive_int(row.get("size"), "network volume size"),
                data_center_id=_string(row.get("dataCenterId"), "network volume dataCenterId"),
            )
        )
    return tuple(parsed)


def parse_pods(value: object) -> tuple[RunPodPodObservation, ...]:
    parsed = []
    for entry in _resource_rows(value, "pods"):
        row = _mapping(entry, "pod")
        gpu_type = row.get("gpuTypeId")
        if gpu_type is None and type(row.get("gpu")) is dict:
            gpu_type = row["gpu"].get("id")
        parsed.append(
            RunPodPodObservation(
                id=validate_runpod_pod_id(row.get("id")),
                name=_string(row.get("name"), "pod name"),
                desired_status=_string(row.get("desiredStatus"), "pod desiredStatus"),
                image_name=_string(row.get("imageName"), "pod imageName"),
                gpu_type_id=_string(gpu_type, "pod gpuTypeId"),
                gpu_count=_positive_int(row.get("gpuCount"), "pod gpuCount"),
                data_center_id=_string(row.get("dataCenterId"), "pod dataCenterId"),
                container_disk_gb=_positive_int(
                    row.get("containerDiskInGb"), "pod containerDiskInGb"
                ),
                network_volume_id=_provider_id(row.get("networkVolumeId"), "pod networkVolumeId"),
                template_id=_provider_id(row.get("templateId"), "pod templateId"),
                ports=_ports(row.get("ports"), "pod ports"),
            )
        )
    return tuple(parsed)


def parse_created_template(value: object) -> RunPodTemplateObservation:
    return _single_created(parse_templates([value]), "template")


def parse_created_volume(value: object) -> RunPodVolumeObservation:
    return _single_created(parse_volumes([value]), "network volume")


def parse_created_pod(value: object) -> RunPodPodObservation:
    return _single_created(parse_pods([value]), "pod")


def _single_created(values: tuple[ResourceObservation, ...], name: str):
    if len(values) != 1:
        raise ValueError(f"created {name} response is malformed")
    return values[0]


def secret_reference(name: str) -> str:
    return f"{{{{ RUNPOD_SECRET_{name} }}}}"


def expected_environment(
    bundle: DeploymentBundle,
    names: ServingResourceNames,
    *,
    include_artifact_secret: bool,
) -> tuple[tuple[str, str], ...]:
    environment = {
        "FLASH_INFERENCE_TOKEN": secret_reference(names.inference_secret),
        "FLASH_SERVING_CACHE_ROOT": SERVING_CACHE_ROOT,
        "FLASH_SERVING_HOST": "0.0.0.0",
        "FLASH_SERVING_IMAGE_DIGEST": bundle.image.digest,
        "FLASH_SERVING_MANIFEST": encode_manifest_environment(bundle.manifest),
        "FLASH_SERVING_MANIFEST_ID": bundle.manifest.manifest_id,
        "FLASH_SERVING_PORT": str(PROXY_PORT),
    }
    if include_artifact_secret:
        environment["FLASH_ARTIFACT_TOKEN"] = secret_reference(names.artifact_secret)
    return tuple(sorted(environment.items()))


def template_payload(
    bundle: DeploymentBundle,
    names: ServingResourceNames,
    *,
    include_artifact_secret: bool,
) -> dict[str, object]:
    placement = bundle.spec.placement
    assert type(placement) is RunPodPlacement
    return {
        "name": names.template,
        "imageName": bundle.image.reference,
        "dockerStartCmd": LAUNCH_COMMAND,
        "containerDiskInGb": placement.container_disk_gb,
        "volumeInGb": 0,
        "volumeMountPath": NETWORK_VOLUME_MOUNT,
        "ports": [PROXY_PORT_SPEC],
        "env": [
            {"key": key, "value": value}
            for key, value in expected_environment(
                bundle,
                names,
                include_artifact_secret=include_artifact_secret,
            )
        ],
        "isServerless": False,
    }


def volume_payload(bundle: DeploymentBundle, names: ServingResourceNames) -> dict[str, object]:
    placement = bundle.spec.placement
    assert type(placement) is RunPodPlacement
    return {
        "name": names.volume,
        "size": placement.volume_size_gb,
        "dataCenterId": placement.data_center_id,
    }


def pod_payload(
    bundle: DeploymentBundle,
    names: ServingResourceNames,
    *,
    template_id: str,
    volume_id: str,
) -> dict[str, object]:
    placement = bundle.spec.placement
    assert type(placement) is RunPodPlacement
    return {
        "name": names.app_or_pod,
        "imageName": bundle.image.reference,
        "gpuTypeIds": [placement.gpu_type_id],
        "gpuCount": placement.gpu_count,
        "dataCenterIds": [placement.data_center_id],
        "containerDiskInGb": placement.container_disk_gb,
        "networkVolumeId": volume_id,
        "templateId": template_id,
        "ports": [PROXY_PORT_SPEC],
    }
