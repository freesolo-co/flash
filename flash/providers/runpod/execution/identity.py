"""RunPod Pod handle phases and immutable request identity."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from typing import ClassVar

from flash.providers._lifecycle.instances.instance import (
    InstanceJobHandle,
    instance_label,
    run_label_prefix,
)
from flash.providers._lifecycle.net.worker import worker_image_for_gpu
from flash.providers.core.base import min_cuda_modern
from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.client import pods as runpod_pods
from flash.providers.runpod.client.gpus import gpu_type_id

PAYLOAD_ENV = "FLASH_INSTANCE_PAYLOAD"
POD_LAUNCH_COMMAND = ("python", "/opt/flash/runpod_pod_launcher.py")
NETWORK_VOLUME_MOUNT = "/runpod-volume"
SECRET_CREATE_PENDING = "secret_create_pending"
PRE_POD_CREATE = "pre_pod_create"
POD_CREATE_PENDING = "pod_create_pending"
EXACT = "exact"
PHASES = frozenset({SECRET_CREATE_PENDING, PRE_POD_CREATE, POD_CREATE_PENDING, EXACT})
PHASE_ORDER = {
    SECRET_CREATE_PENDING: 0,
    PRE_POD_CREATE: 1,
    POD_CREATE_PENDING: 2,
    EXACT: 3,
}
_SECRET_PREFIX = "FLASH_PAYLOAD_"
_NONCE_HEX_LENGTH = 16


class RunpodCreateAbsent(RuntimeError):
    """A durable pre-create phase was proven unable to own a Pod."""


@dataclass
class RunpodPodHandle(InstanceJobHandle):
    """Durable identity for one explicit non-idempotent RunPod creation phase."""

    phase: str
    label: str
    key_fingerprint: str
    account_id: str
    payload_secret_id: str | None
    payload_secret_name: str
    data_center_id: str | None
    network_volume_id: str | None
    container_disk_gb: int
    gpu_count: int
    container_registry_auth_id: str | None = None
    image_name: str | None = None
    gpu_type_id_override: str | None = None
    allowed_cuda_versions: tuple[str, ...] | None = None
    docker_start_cmd: tuple[str, ...] | None = None

    provider: ClassVar[str] = "runpod"

    @staticmethod
    def _coerce_instance_id(raw) -> str:
        if type(raw) is not str or not raw or raw != raw.strip():
            raise ValueError("invalid runpod Pod identity")
        return raw

    @property
    def pod_id(self) -> str | None:
        return str(self.instance_id) if self.phase == EXACT else None

    @property
    def pending(self) -> bool:
        return self.phase != EXACT

    def _extra_to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "label": self.label,
            "key_fingerprint": self.key_fingerprint,
            "account_id": self.account_id,
            "payload_secret_id": self.payload_secret_id,
            "payload_secret_name": self.payload_secret_name,
            "data_center_id": self.data_center_id,
            "network_volume_id": self.network_volume_id,
            "container_disk_gb": self.container_disk_gb,
            "container_registry_auth_id": self.container_registry_auth_id,
            "gpu_count": self.gpu_count,
            "image_name": self.image_name,
            "gpu_type_id_override": self.gpu_type_id_override,
            "allowed_cuda_versions": (
                None if self.allowed_cuda_versions is None else list(self.allowed_cuda_versions)
            ),
            "docker_start_cmd": list(self.docker_start_cmd or ()),
        }

    @staticmethod
    def _extra_from_dict(d: dict) -> dict:
        phase = d.get("phase")
        label = d.get("label")
        fingerprint = d.get("key_fingerprint")
        account_id = d.get("account_id")
        secret_id = d.get("payload_secret_id")
        secret_name = d.get("payload_secret_name")
        data_center_id = d.get("data_center_id")
        volume_id = d.get("network_volume_id")
        disk_gb = d.get("container_disk_gb")
        registry_id = d.get("container_registry_auth_id")
        gpu_count = d.get("gpu_count")
        image_name = d.get("image_name")
        gpu_type_override = d.get("gpu_type_id_override")
        allowed_cuda_raw = d.get("allowed_cuda_versions")
        command_raw = d.get("docker_start_cmd", [])
        if phase not in PHASES:
            raise ValueError("persisted RunPod creation phase is invalid")
        if type(label) is not str or not label:
            raise ValueError("persisted RunPod Pod label is invalid")
        if not runpod_api._is_valid_key_fingerprint(fingerprint):
            raise ValueError("persisted RunPod key fingerprint is invalid")
        for value, field in (
            (account_id, "account"),
            (secret_name, "payload secret name"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"persisted RunPod {field} identity is invalid")
        if secret_id is not None and (type(secret_id) is not str or not secret_id):
            raise ValueError("persisted RunPod payload secret identity is invalid")
        if phase == SECRET_CREATE_PENDING and secret_id is not None:
            raise ValueError("secret-create-pending RunPod handle cannot have a secret id")
        if phase != SECRET_CREATE_PENDING and secret_id is None:
            raise ValueError("persisted RunPod phase requires an exact payload secret id")
        for value, field in (
            (data_center_id, "data center"),
            (volume_id, "network volume"),
            (registry_id, "registry credential"),
        ):
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"persisted RunPod {field} identity is invalid")
        if type(disk_gb) is not int or disk_gb <= 0:
            raise ValueError("persisted RunPod container disk is invalid")
        if type(gpu_count) is not int or gpu_count <= 0:
            raise ValueError("persisted RunPod GPU count is invalid")
        for value, field in (
            (image_name, "image"),
            (gpu_type_override, "GPU type override"),
        ):
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"persisted RunPod {field} is invalid")
        if allowed_cuda_raw is not None and (
            type(allowed_cuda_raw) is not list
            or any(type(item) is not str or not item for item in allowed_cuda_raw)
        ):
            raise ValueError("persisted RunPod CUDA version identity is invalid")
        if type(command_raw) is not list or any(
            type(item) is not str or not item for item in command_raw
        ):
            raise ValueError("persisted RunPod command identity is invalid")
        return {
            "phase": phase,
            "label": label,
            "key_fingerprint": fingerprint,
            "account_id": account_id,
            "payload_secret_id": secret_id,
            "payload_secret_name": secret_name,
            "data_center_id": data_center_id,
            "network_volume_id": volume_id,
            "container_disk_gb": disk_gb,
            "container_registry_auth_id": registry_id,
            "gpu_count": gpu_count,
            "image_name": image_name,
            "gpu_type_id_override": gpu_type_override,
            "allowed_cuda_versions": (
                None if allowed_cuda_raw is None else tuple(allowed_cuda_raw)
            ),
            "docker_start_cmd": tuple(command_raw) or None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunpodPodHandle:
        handle = super().from_dict(d)
        if handle.phase == EXACT:
            if handle.instance_id == handle.label or handle.instance_id.startswith("flash-"):
                raise ValueError("persisted RunPod exact Pod identity is invalid")
        elif handle.instance_id != handle.label:
            raise ValueError("persisted RunPod pending identity is invalid")
        return handle


def fresh_payload_secret_name() -> str:
    """Return a fresh non-secret random identity for one secret mutation."""
    return f"{_SECRET_PREFIX}{secrets.token_hex(_NONCE_HEX_LENGTH // 2)}"


def _secret_nonce(name: str) -> str:
    nonce = name.removeprefix(_SECRET_PREFIX)
    if (
        not name.startswith(_SECRET_PREFIX)
        or len(nonce) != _NONCE_HEX_LENGTH
        or any(char not in "0123456789abcdef" for char in nonce)
    ):
        raise ValueError("RunPod payload secret name is not nonce-backed")
    return nonce


def payload_secret_name_from_pod_label(label: str) -> str:
    parts = label.rsplit("-", 2)
    if len(parts) != 3:
        raise ValueError("RunPod Pod label has no payload-secret nonce")
    nonce = parts[1]
    _secret_nonce(f"{_SECRET_PREFIX}{nonce}")
    return f"{_SECRET_PREFIX}{nonce}"


def secret_reference(name: str) -> str:
    return f"{{{{ RUNPOD_SECRET_{name} }}}}"


def build_pod_payload(
    spec,
    *,
    label: str,
    secret_name: str,
    data_center_id: str | None,
    network_volume_id: str | None,
    container_registry_auth_id: str | None = None,
) -> dict:
    from flash.core.spec import gpu_count_of

    payload: dict[str, object] = {
        "allowedCudaVersions": [min_cuda_modern(spec.gpu.type)],
        "cloudType": "SECURE",
        "containerDiskInGb": int(spec.gpu.disk_gb),
        "dockerStartCmd": list(POD_LAUNCH_COMMAND),
        "env": {PAYLOAD_ENV: secret_reference(secret_name)},
        "gpuCount": gpu_count_of(spec),
        "gpuTypeIds": [gpu_type_id(spec.gpu.type)],
        "imageName": worker_image_for_gpu(spec.gpu.type),
        "interruptible": False,
        "name": label,
        "supportPublicIp": False,
        "volumeInGb": 0,
        "volumeMountPath": NETWORK_VOLUME_MOUNT,
    }
    if network_volume_id is not None:
        payload["networkVolumeId"] = network_volume_id
    elif data_center_id is not None:
        payload["dataCenterIds"] = [data_center_id]
    registry_id = container_registry_auth_id
    if registry_id is None:
        registry_id = (os.environ.get("RUNPOD_CONTAINER_REGISTRY_AUTH_ID") or "").strip() or None
    if registry_id:
        payload["containerRegistryAuthId"] = registry_id
    return payload


def pod_run_prefix(run_id: str) -> str:
    bounded = run_label_prefix(run_id)
    if bounded.startswith("flash-preload-d"):
        return bounded
    digest = hashlib.sha256(bounded.encode("utf-8")).hexdigest()[:12]
    return f"flash-{digest}"


def pod_attempt_label_base(run_id: str, seed: int, attempt: int) -> str:
    generic = instance_label(run_id, seed, attempt)
    suffix = generic[len(run_label_prefix(run_id)) :]
    return f"{pod_run_prefix(run_id)}{suffix}"


def pod_label_from_payload(base: str, secret_name: str, payload: dict) -> str:
    """Bind a label to every immutable request field except the label itself."""
    identity = dict(payload)
    identity.pop("name", None)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    suffix = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:8]
    nonce = _secret_nonce(secret_name)
    return f"{base}-{nonce}-{suffix}"


def pod_label(
    spec,
    seed: int,
    attempt: int,
    *,
    secret_name: str,
    data_center_id: str | None,
    network_volume_id: str | None,
    container_registry_auth_id: str | None,
) -> str:
    base = pod_attempt_label_base(spec.run_id, seed, attempt)
    identity = build_pod_payload(
        spec,
        label=base,
        secret_name=secret_name,
        data_center_id=data_center_id,
        network_volume_id=network_volume_id,
        container_registry_auth_id=container_registry_auth_id,
    )
    return pod_label_from_payload(base, secret_name, identity)


def payload_for_handle(handle: RunpodPodHandle) -> dict:
    payload: dict[str, object] = {
        "allowedCudaVersions": (
            list(handle.allowed_cuda_versions)
            if handle.allowed_cuda_versions is not None
            else [min_cuda_modern(handle.gpu)]
        ),
        "cloudType": "SECURE",
        "containerDiskInGb": handle.container_disk_gb,
        "dockerStartCmd": list(handle.docker_start_cmd or POD_LAUNCH_COMMAND),
        "env": {PAYLOAD_ENV: secret_reference(handle.payload_secret_name)},
        "gpuCount": handle.gpu_count,
        "gpuTypeIds": [handle.gpu_type_id_override or gpu_type_id(handle.gpu)],
        "imageName": handle.image_name or worker_image_for_gpu(handle.gpu),
        "interruptible": False,
        "name": handle.label,
        "supportPublicIp": False,
        "volumeInGb": 0,
        "volumeMountPath": NETWORK_VOLUME_MOUNT,
    }
    if handle.network_volume_id is not None:
        payload["networkVolumeId"] = handle.network_volume_id
    elif handle.data_center_id is not None:
        payload["dataCenterIds"] = [handle.data_center_id]
    if handle.container_registry_auth_id is not None:
        payload["containerRegistryAuthId"] = handle.container_registry_auth_id
    return payload


def handle_label_digest_is_valid(handle: RunpodPodHandle) -> bool:
    try:
        if payload_secret_name_from_pod_label(handle.label) != handle.payload_secret_name:
            return False
    except ValueError:
        return False
    _base, separator, suffix = handle.label.rpartition("-")
    if not separator or len(suffix) != 8 or any(char not in "0123456789abcdef" for char in suffix):
        return False
    identity = payload_for_handle(handle)
    identity.pop("name")
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:8]
    return suffix == expected


def pod_identity_is_incomplete(
    pod: runpod_pods.RunpodPod,
    payload: dict,
    *,
    network_volume_id: str | None,
    data_center_id: str | None,
    allow_preplacement: bool = False,
) -> bool:
    """Return whether missing realized fields prevent a conclusive identity decision."""
    required = (
        pod.docker_start_cmd,
        pod.payload_env_sha256,
        pod.payload_secret_name,
        pod.secure_cloud,
        pod.public_ip_assigned,
        pod.volume_mount_path,
    )
    if any(value is None for value in required):
        return True
    if (
        payload.get("containerRegistryAuthId") is not None
        and pod.container_registry_auth_id is None
    ):
        return True
    if pod.gpu_type_id is None:
        return True
    if data_center_id is not None and network_volume_id is None and pod.data_center_id is None:
        return True
    return bool(network_volume_id is not None and pod.network_volume_id is None)


def pod_matches(
    pod: runpod_pods.RunpodPod,
    payload: dict,
    *,
    network_volume_id: str | None,
    data_center_id: str | None,
    allow_preplacement: bool = False,
) -> bool:
    expected_gpu = payload["gpuTypeIds"][0]
    expected_env = payload["env"][PAYLOAD_ENV]
    expected_env_sha256 = hashlib.sha256(expected_env.encode("utf-8")).hexdigest()
    expected_registry_id = payload.get("containerRegistryAuthId")
    gpu_matches = pod.gpu_type_id == expected_gpu
    if data_center_id is None:
        placement_matches = True
    elif network_volume_id is not None:
        placement_matches = pod.data_center_id in {None, data_center_id}
    else:
        placement_matches = pod.data_center_id == data_center_id
    volume_matches = pod.network_volume_id == network_volume_id
    return bool(
        pod.name == payload["name"]
        and pod.image_name == payload["imageName"]
        and pod.gpu_count == payload["gpuCount"]
        and pod.container_disk_gb == payload["containerDiskInGb"]
        and gpu_matches
        and placement_matches
        and volume_matches
        and pod.docker_start_cmd == tuple(payload["dockerStartCmd"])
        and pod.payload_env_sha256 == expected_env_sha256
        and pod.payload_secret_name == payload_secret_name_from_pod_label(payload["name"])
        and pod.secure_cloud is True
        and (pod.interruptible is None or pod.interruptible is False)
        and (pod.support_public_ip is None or pod.support_public_ip is False)
        and pod.public_ip_assigned is False
        and pod.volume_mount_path == payload["volumeMountPath"]
        and pod.container_registry_auth_id == expected_registry_id
    )
