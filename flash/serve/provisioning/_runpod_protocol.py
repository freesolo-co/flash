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
# runpod carries dockerStartCmd as argv (array<string> in its rest schema, and what GET /templates
# returns), so argv is the source of truth and LAUNCH_COMMAND is derived for display and for the
# adoption comparison in _runpod_resources. defining it the other way round would need shell
# splitting to recover the argv the api actually wants.
LAUNCH_COMMAND_ARGV = ("python", "/app/serve_launch.py")
LAUNCH_COMMAND = " ".join(LAUNCH_COMMAND_ARGV)
# the serving image's vllm ships a compiled extension linked against libcudart.so.13, so the host
# has to expose a CUDA 13 driver. runpod's PodCreateInput documents allowedCudaVersions as "if not
# set, any CUDA version is acceptable", so leaving it off let it place the pod on an L4 host
# reporting driver 12080 -- the container then died at engine init and restarted forever. asking a
# gpu type alone does not ask for a driver.
ALLOWED_CUDA_VERSIONS = ("13.0",)

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

# the mutations are `secretCreate` / `secretDelete`, not `createSecret` / `deleteSecret`:
# runpod's graphql schema rejects the latter outright with "Cannot query field ... on type
# Mutation", so every provisioning attempt died at the first secret. secretDelete takes a bare
# `id: ID!` rather than an input object and returns Void, which takes no selection set.
CREATE_SECRET = """
mutation FlashServingCreateSecret($name: String!, $value: String!) {
  secretCreate(input: {name: $name, value: $value}) {
    id
    name
  }
}
""".strip()

DELETE_SECRET = """
mutation FlashServingDeleteSecret($id: ID!) {
  secretDelete(id: $id)
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
    # None until runpod assigns a machine. a CREATED or PENDING pod has no `machine` object yet, so
    # placement is simply not decided. these are parsed account-wide like the two ids below, so
    # demanding them failed the whole pass on any foreign pod still waiting for capacity -- and on
    # our own, before it was placed. `pod_identity_matches` compares both against nonempty plan
    # values, so None never matches and an unplaced pod is never mistaken for a ready one.
    gpu_type_id: str | None
    gpu_count: int
    data_center_id: str | None
    container_disk_gb: int
    # None when the pod has no network volume attached. every pod in the account is parsed during
    # observation, including ones flash did not create, and demanding an id there failed the whole
    # pass on a foreign volume-less pod. our own pods always carry one, and `pod_matches` compares
    # this against the plan's volume id, so None simply never matches.
    network_volume_id: str | None
    # None when the pod was not created from a template. runpod returns "" rather than omitting
    # the key, and the same foreign-pod reasoning as network_volume_id applies.
    template_id: str | None
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
    """read runpod's ports field, which is absent for a resource that exposes none.

    the api omits `ports` entirely (json null) rather than sending an empty list, and it does that
    for resources flash did not create as well as its own. rejecting null failed the whole
    observation pass on the first foreign port-less pod in the account, so flash could never see
    its own pods -- the same failure `_docker_start_cmd` below already had to fix. an empty string
    was always accepted here and means exactly this, so null maps to the same empty result.
    """

    if value is None:
        raw: list[object] = []
    elif type(value) is str:
        raw = value.split(",") if value else []
    elif type(value) is list:
        raw = list(value)
    else:
        raise ValueError(f"{name} must be a string or list")
    parsed = tuple(_string(item, name) for item in raw)
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} contains duplicates")
    return parsed


def _docker_start_cmd(value: object) -> str:
    """read runpod's argv-shaped dockerStartCmd back as the joined command string.

    the api types this as array<string> and returns it that way for every template in the
    account, including ones flash did not create -- so demanding a string here failed the whole
    observation pass on a foreign template and never reached our own. the joined form is what
    `_runpod_resources` compares against LAUNCH_COMMAND. a plain string is still accepted because
    runpod's graphql surface answers with one.
    """

    if type(value) is str:
        return _string(value, "template dockerStartCmd")
    if type(value) is not list or not value:
        raise ValueError("template dockerStartCmd must be a string or nonempty list of strings")
    return " ".join(_string(part, "template dockerStartCmd entry") for part in value)


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
    row = _mapping(data.get("secretCreate"), "created secret")
    return RunPodSecretObservation(
        id=_provider_id(row.get("id"), "secret id"),
        name=_string(row.get("name"), "secret name"),
    )


def parse_deleted_secret(value: object) -> bool:
    """confirm a secret deletion from runpod's Void-returning secretDelete mutation.

    the field's type is Void, so a success carries `null` rather than `true`. absence of the key
    is not accepted: that is what a response for some other mutation would look like, and this
    result is what lets teardown report the secret gone. errors are still rejected first, so a
    failed delete can never read as a successful one.
    """

    root = _mapping(value, "graphql response")
    if "errors" in root:
        raise ValueError("graphql response contains errors")
    data = _mapping(root.get("data"), "graphql data")
    if "secretDelete" not in data or data["secretDelete"] is not None:
        raise ValueError("deleted secret response is malformed")
    return True


def parse_templates(
    value: object, *, keep_name: str | None = None
) -> tuple[RunPodTemplateObservation, ...]:
    """read the account's templates, strictly parsing only the ones flash owns.

    `keep_name` filters by template name before the per-field parsing below. the customer's
    account holds templates flash did not create, and those rows legitimately omit
    `dockerStartCmd` (they use the image's default command) or `env` (no overrides). parsing
    every row strictly meant one unrelated template failed the whole observation pass with
    `transport_failed`, blocking deployments that had no conflicting flash resource at all.
    name is the identity flash matches on, so filter first and validate only what is ours.
    """

    parsed = []
    for entry in _resource_rows(value, "templates"):
        row = _mapping(entry, "template")
        name = _string(row.get("name"), "template name")
        if keep_name is not None and name != keep_name:
            continue
        parsed.append(
            RunPodTemplateObservation(
                id=_provider_id(row.get("id"), "template id"),
                name=name,
                image_name=_string(row.get("imageName"), "template imageName"),
                docker_start_cmd=_docker_start_cmd(row.get("dockerStartCmd")),
                container_disk_gb=_positive_int(
                    row.get("containerDiskInGb"), "template containerDiskInGb"
                ),
                # runpod omits volumeInGb and isServerless from GET /templates when they hold their
                # defaults rather than returning 0/false, so a missing key is the documented
                # default, not a malformed row. a present key is still type-checked; only absence
                # is defaulted, so a wrong-typed value still fails.
                volume_gb=_nonnegative_int(row.get("volumeInGb", 0), "template volumeInGb"),
                volume_mount_path=_string(row.get("volumeMountPath"), "template volumeMountPath"),
                ports=_ports(row.get("ports"), "template ports"),
                environment=_environment(row.get("env")),
                is_serverless=row.get("isServerless", False)
                if type(row.get("isServerless", False)) is bool
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


def _first_present(*sources: tuple[dict[str, object], str]) -> object | None:
    """first value whose key is actually present, or None when no source carries it."""

    for mapping, key in sources:
        if key in mapping:
            return mapping[key]
    return None


def parse_pods(value: object) -> tuple[RunPodPodObservation, ...]:
    parsed = []
    for entry in _resource_rows(value, "pods"):
        row = _mapping(entry, "pod")
        # runpod's rest Pod schema carries no flat gpuTypeId / dataCenterId / networkVolumeId.
        # they live in the nested `machine`, `gpu`, and `networkVolume` objects, which the api
        # only returns when includeMachine / includeNetworkVolume are set. the flat keys are read
        # first because runpod's graphql surface does return them at the top level.
        machine = row.get("machine") if type(row.get("machine")) is dict else {}
        gpu = row.get("gpu") if type(row.get("gpu")) is dict else {}
        volume = row.get("networkVolume") if type(row.get("networkVolume")) is dict else {}
        # `or` would conflate "runpod has not assigned placement yet" with "it sent a malformed
        # value": an empty string is falsy, so it would fall through to None and read as unplaced.
        # presence of the key is what distinguishes them, so absence stays absence and anything
        # actually sent is still validated.
        gpu_type = _first_present((row, "gpuTypeId"), (machine, "gpuTypeId"), (gpu, "id"))
        data_center = _first_present((row, "dataCenterId"), (machine, "dataCenterId"))
        network_volume = row.get("networkVolumeId") or volume.get("id")
        parsed.append(
            RunPodPodObservation(
                id=validate_runpod_pod_id(row.get("id")),
                name=_string(row.get("name"), "pod name"),
                desired_status=_string(row.get("desiredStatus"), "pod desiredStatus"),
                image_name=_string(row.get("imageName"), "pod imageName"),
                gpu_type_id=(None if gpu_type is None else _string(gpu_type, "pod gpuTypeId")),
                gpu_count=_positive_int(row.get("gpuCount"), "pod gpuCount"),
                data_center_id=(
                    None if data_center is None else _string(data_center, "pod dataCenterId")
                ),
                container_disk_gb=_positive_int(
                    row.get("containerDiskInGb"), "pod containerDiskInGb"
                ),
                network_volume_id=(
                    None
                    if network_volume is None
                    else _provider_id(network_volume, "pod networkVolumeId")
                ),
                template_id=(
                    _provider_id(row.get("templateId"), "pod templateId")
                    if row.get("templateId")
                    else None
                ),
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
        "allowedCudaVersions": list(ALLOWED_CUDA_VERSIONS),
        "dataCenterIds": [placement.data_center_id],
        "containerDiskInGb": placement.container_disk_gb,
        "networkVolumeId": volume_id,
        "templateId": template_id,
        "ports": [PROXY_PORT_SPEC],
    }
