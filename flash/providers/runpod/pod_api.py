"""Strict account-scoped RunPod Pod, volume, catalog, and opaque-secret API."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlencode

from flash.providers._lifecycle.deadline import remaining_seconds
from flash.providers._lifecycle.http import is_not_found
from flash.providers.runpod.api import (
    _CLIENT,
    _NO_REDIRECT_OPENER,
    CATALOG_BASE,
    REST_BASE,
    RunpodApiError,
    _key_for_fingerprint,
    _keys,
    grow_network_volumes_for_key,
    key_fingerprint,
)

# persistent Pod training API -------------------------------------------------

_GRAPHQL_URL = "https://api.runpod.io/graphql"
_POD_QUERY = {"includeMachine": "true", "includeNetworkVolume": "true"}
_POD_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,127}")
_MANAGED_TRAINING_POD_RE = re.compile(r"flash-[0-9a-f]{12}(?:-|$)")
_MANAGED_PRELOAD_POD_RE = re.compile(r"flash-preload-d[0-9]{10,}(?:-|$)")
_SECRET_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,127}")
_SECRET_REFERENCE_RE = re.compile(r"\{\{ RUNPOD_SECRET_([A-Za-z][A-Za-z0-9_]{2,127}) \}\}")
_CAPACITY_PATTERNS = (
    "no instances currently available",
    "could not find any pods with required specifications",
    "insufficient capacity",
)
_DELETE_CONFIRM_POLLS = 4
_DELETE_CONFIRM_WAIT_S = 0.25

_OBSERVE_SECRETS = """
query FlashTrainingSecrets {
  myself {
    id
    secrets {
      id
      name
    }
  }
}
""".strip()

_CREATE_SECRET = """
mutation FlashTrainingCreateSecret($name: String!, $value: String!) {
  secretCreate(input: {name: $name, value: $value}) {
    id
    name
  }
}
""".strip()

_DELETE_SECRET = """
mutation FlashTrainingDeleteSecret($id: ID!) {
  secretDelete(id: $id)
}
""".strip()


class RunpodCapacityError(RunpodApiError):
    """A definite no-capacity refusal that created no Pod."""


class RunpodMutationAmbiguous(RunpodApiError):
    """A non-idempotent mutation may have taken effect."""


class RunpodRequestError(RunpodApiError):
    """A definite provider rejection with its HTTP status preserved."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class RunpodPod:
    id: str
    name: str
    desired_status: str
    image_name: str
    gpu_type_id: str | None
    gpu_count: int
    data_center_id: str | None
    container_disk_gb: int
    network_volume_id: str | None
    cost_per_hr: float | None
    docker_start_cmd: tuple[str, ...] | None = None
    payload_env_sha256: str | None = None
    payload_secret_name: str | None = None
    secure_cloud: bool | None = None
    interruptible: bool | None = None
    support_public_ip: bool | None = None
    volume_mount_path: str | None = None
    container_registry_auth_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunpodSecret:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class RunpodNetworkVolume:
    id: str
    name: str
    size_gb: int
    data_center_id: str


@dataclass(frozen=True, slots=True)
class RunpodDataCenter:
    id: str
    network_volume_types: tuple[str, ...]


def _strict_string(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise RunpodApiError(f"runpod {field} is invalid")
    return value


def _strict_id(value: object, field: str) -> str:
    selected = _strict_string(value, field)
    if _POD_ID_RE.fullmatch(selected) is None:
        raise RunpodApiError(f"runpod {field} is invalid")
    return selected


def _strict_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise RunpodApiError(f"runpod {field} is invalid")
    return value


def _optional_string(value: object, field: str) -> str | None:
    return None if value is None else _strict_string(value, field)


def _optional_rate(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunpodApiError("runpod Pod hourly rate is invalid")
    rate = float(value)
    if not math.isfinite(rate) or rate < 0:
        raise RunpodApiError("runpod Pod hourly rate is invalid")
    return rate


def _optional_bool(value: object, field: str) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise RunpodApiError(f"runpod {field} is invalid")
    return value


def _optional_string_tuple(value: object, field: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if type(value) is not list or not value:
        raise RunpodApiError(f"runpod {field} is invalid")
    return tuple(_strict_string(item, field) for item in value)


def _payload_env_identity(value: object) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if type(value) is not dict or set(value) != {"FLASH_INSTANCE_PAYLOAD"}:
        raise RunpodApiError("runpod Pod environment identity is invalid")
    payload_reference = value["FLASH_INSTANCE_PAYLOAD"]
    if type(payload_reference) is not str or not payload_reference:
        raise RunpodApiError("runpod Pod environment identity is invalid")
    match = _SECRET_REFERENCE_RE.fullmatch(payload_reference)
    if match is None:
        raise RunpodApiError("runpod Pod environment identity is invalid")
    return hashlib.sha256(payload_reference.encode("utf-8")).hexdigest(), match.group(1)


def _parse_pod(row: object) -> RunpodPod:
    if type(row) is not dict:
        raise RunpodApiError("runpod Pod response row is invalid")
    machine = row.get("machine") if type(row.get("machine")) is dict else {}
    gpu = row.get("gpu") if type(row.get("gpu")) is dict else {}
    volume = row.get("networkVolume") if type(row.get("networkVolume")) is dict else {}
    gpu_type = machine.get("gpuTypeId") if "gpuTypeId" in machine else gpu.get("id")
    gpu_count = row.get("gpuCount") if "gpuCount" in row else gpu.get("count")
    data_center = machine.get("dataCenterId")
    volume_id = row.get("networkVolumeId")
    if volume_id is None:
        volume_id = volume.get("id")
    desired_status = row.get("desiredStatus", row.get("status"))
    image_name = row.get("imageName") if "imageName" in row else row.get("image")
    support_public_ip = (
        row.get("supportPublicIp") if "supportPublicIp" in row else machine.get("supportPublicIp")
    )
    payload_env_sha256, payload_secret_name = _payload_env_identity(row.get("env"))
    return RunpodPod(
        id=_strict_id(row.get("id"), "Pod id"),
        name=_strict_string(row.get("name"), "Pod name"),
        desired_status=_strict_string(desired_status, "Pod status").upper(),
        image_name=_strict_string(image_name, "Pod image"),
        gpu_type_id=_optional_string(gpu_type, "Pod GPU type"),
        gpu_count=_strict_positive_int(gpu_count, "Pod GPU count"),
        data_center_id=_optional_string(data_center, "Pod data center"),
        container_disk_gb=_strict_positive_int(row.get("containerDiskInGb"), "Pod container disk"),
        network_volume_id=(
            None if volume_id is None else _strict_id(volume_id, "Pod network volume id")
        ),
        cost_per_hr=_optional_rate(row.get("costPerHr", row.get("costPerHour"))),
        docker_start_cmd=_optional_string_tuple(row.get("dockerStartCmd"), "Pod command"),
        payload_env_sha256=payload_env_sha256,
        payload_secret_name=payload_secret_name,
        secure_cloud=_optional_bool(machine.get("secureCloud"), "Pod secure cloud flag"),
        interruptible=_optional_bool(row.get("interruptible"), "Pod interruptible flag"),
        support_public_ip=_optional_bool(support_public_ip, "Pod public IP flag"),
        volume_mount_path=_optional_string(row.get("volumeMountPath"), "Pod volume mount path"),
        container_registry_auth_id=_optional_string(
            row.get("containerRegistryAuthId"), "Pod registry credential id"
        ),
    )


def _managed_pod_name(name: object) -> bool:
    if type(name) is not str:
        return False
    return bool(_MANAGED_TRAINING_POD_RE.match(name) or _MANAGED_PRELOAD_POD_RE.match(name))


def _pod_rows(value: object, *, keep_name: str | None = None) -> list[RunpodPod]:
    if type(value) is not list:
        raise RunpodApiError("runpod /pods response must be a list")
    pods = []
    for raw in value:
        if type(raw) is not dict:
            raise RunpodApiError("runpod /pods response row is invalid")
        name = raw.get("name")
        if keep_name is not None and name != keep_name:
            continue
        if keep_name is None and not _managed_pod_name(name):
            continue
        pods.append(_parse_pod(raw))
    return pods


def _pod_url(path: str, query: dict[str, str] | None = None) -> str:
    url = f"{REST_BASE}{path}"
    return url if not query else f"{url}?{urlencode(sorted(query.items()))}"


def list_pods_for_key(
    key: str,
    *,
    keep_name: str | None = None,
    deadline_at: float | None = None,
) -> list[RunpodPod]:
    out = _CLIENT.request_with_retries_for_key(
        key,
        _pod_url("/pods", _POD_QUERY),
        retries=2,
        deadline_at=deadline_at,
    )
    return _pod_rows(out, keep_name=keep_name)


def list_pods_by_key(
    *, deadline_at: float | None = None
) -> tuple[dict[str, list[RunpodPod]], list[str]]:
    pool = _keys.keys()
    if not pool:
        raise RunpodApiError("RUNPOD_API_KEY is not set; refusing to report an empty Pod fleet")
    by_fingerprint: dict[str, list[RunpodPod]] = {}
    failed: list[str] = []
    for key in pool:
        fingerprint = key_fingerprint(key)
        try:
            by_fingerprint[fingerprint] = list_pods_for_key(key, deadline_at=deadline_at)
        except Exception:
            failed.append(fingerprint)
    return by_fingerprint, failed


def get_pod_for_fingerprint(
    pod_id: str,
    fingerprint: str,
    *,
    deadline_at: float | None = None,
) -> RunpodPod | None:
    key = _key_for_fingerprint(fingerprint)
    try:
        out = _CLIENT.request_with_retries_for_key(
            key,
            _pod_url(f"/pods/{_strict_id(pod_id, 'Pod id')}", _POD_QUERY),
            retries=2,
            deadline_at=deadline_at,
        )
    except Exception as exc:
        if is_not_found(exc):
            return None
        raise RunpodApiError(f"runpod Pod lookup failed for {pod_id}") from None
    return _parse_pod(out)


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read(4096).decode("utf-8", "replace")
    except Exception:
        return ""


def _mutation_once(
    key: str,
    url: str,
    *,
    method: str,
    body: dict | None,
    deadline_at: float,
) -> object:
    remaining = remaining_seconds(deadline_at)
    if remaining <= 0:
        raise RunpodMutationAmbiguous("runpod mutation outcome is unknown")
    encoded = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        method=method,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "flash-training",
        },
    )
    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=min(30.0, remaining)) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            body_text = _read_error_body(exc)
            lowered = body_text.lower()
            if exc.code in {402, 429} or any(pattern in lowered for pattern in _CAPACITY_PATTERNS):
                raise RunpodCapacityError("runpod has no matching Pod capacity") from None
            if exc.code in {400, 401, 403, 404, 409, 422}:
                raise RunpodRequestError(
                    f"runpod mutation was rejected with HTTP {exc.code}",
                    status_code=exc.code,
                ) from exc
            raise RunpodMutationAmbiguous("runpod mutation outcome is unknown") from None
        finally:
            exc.close()
    except (TimeoutError, urllib.error.URLError, OSError):
        raise RunpodMutationAmbiguous("runpod mutation outcome is unknown") from None
    try:
        return None if not raw else json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        raise RunpodMutationAmbiguous("runpod mutation outcome is unknown") from None


def create_pod_for_fingerprint(
    fingerprint: str,
    payload: dict,
    *,
    deadline_at: float,
) -> RunpodPod:
    out = _mutation_once(
        _key_for_fingerprint(fingerprint),
        f"{REST_BASE}/pods",
        method="POST",
        body=payload,
        deadline_at=deadline_at,
    )
    return _parse_pod(out)


def delete_pod_for_fingerprint(
    pod_id: str,
    fingerprint: str,
    *,
    deadline_at: float,
) -> None:
    key = _key_for_fingerprint(fingerprint)
    try:
        _mutation_once(
            key,
            f"{REST_BASE}/pods/{_strict_id(pod_id, 'Pod id')}",
            method="DELETE",
            body=None,
            deadline_at=deadline_at,
        )
    except RunpodApiError as exc:
        if isinstance(exc, RunpodMutationAmbiguous):
            pass
        elif "HTTP 404" not in str(exc):
            raise
    for poll in range(_DELETE_CONFIRM_POLLS):
        if get_pod_for_fingerprint(pod_id, fingerprint, deadline_at=deadline_at) is None:
            return
        if poll + 1 < _DELETE_CONFIRM_POLLS:
            time.sleep(_DELETE_CONFIRM_WAIT_S)
    raise RunpodApiError(f"runpod Pod {pod_id} deletion is unconfirmed")


def _parse_secret_account(value: object) -> tuple[str, list[RunpodSecret]]:
    if type(value) is not dict or "errors" in value:
        raise RunpodApiError("runpod secret observation is invalid")
    data = value.get("data")
    myself = data.get("myself") if type(data) is dict else None
    if type(myself) is not dict or type(myself.get("secrets")) is not list:
        raise RunpodApiError("runpod secret observation is invalid")
    account_id = _strict_id(myself.get("id"), "account id")
    secrets = []
    for raw in myself["secrets"]:
        if type(raw) is not dict:
            raise RunpodApiError("runpod secret observation is invalid")
        name = _strict_string(raw.get("name"), "secret name")
        secrets.append(RunpodSecret(_strict_id(raw.get("id"), "secret id"), name))
    return account_id, secrets


def _graphql_read(key: str, document: str, variables: dict, *, deadline_at: float) -> object:
    remaining = remaining_seconds(deadline_at)
    if remaining <= 0:
        raise RunpodApiError("runpod secret observation deadline exceeded")
    encoded = json.dumps(
        {"query": document, "variables": variables},
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        _GRAPHQL_URL,
        data=encoded,
        method="POST",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "flash-training",
        },
    )
    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=min(30.0, remaining)) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        exc.close()
        raise RunpodRequestError(
            "runpod secret observation failed", status_code=status_code
        ) from exc
    except (TimeoutError, urllib.error.URLError, OSError):
        raise RunpodApiError("runpod secret observation failed") from None
    try:
        return None if not raw else json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        raise RunpodApiError("runpod secret observation failed") from None


def list_secrets_for_fingerprint(
    fingerprint: str,
    *,
    name: str | None = None,
    deadline_at: float,
) -> tuple[str, list[RunpodSecret]]:
    key = _key_for_fingerprint(fingerprint)
    account_id, secrets = _parse_secret_account(
        _graphql_read(key, _OBSERVE_SECRETS, {}, deadline_at=deadline_at)
    )
    if name is not None:
        secrets = [secret for secret in secrets if secret.name == name]
    return account_id, secrets


def _graphql_error_status(value: object) -> int | None:
    if type(value) is not dict or type(value.get("errors")) is not list:
        return None
    statuses = set()
    for raw in value["errors"]:
        if type(raw) is not dict:
            continue
        extensions = raw.get("extensions") if type(raw.get("extensions")) is dict else {}
        status = extensions.get("statusCode", extensions.get("status"))
        if type(status) is int:
            statuses.add(status)
    return next(iter(statuses)) if len(statuses) == 1 else None


def _raise_graphql_mutation_error(value: object, operation: str) -> None:
    status = _graphql_error_status(value)
    if status in {400, 401, 403, 404, 409, 422}:
        raise RunpodRequestError(
            f"runpod {operation} was rejected with HTTP {status}", status_code=status
        )
    if status in {402, 429}:
        raise RunpodCapacityError(f"runpod {operation} capacity is unavailable")
    raise RunpodMutationAmbiguous(f"runpod {operation} outcome is unknown")


def create_secret_for_fingerprint(
    fingerprint: str,
    name: str,
    value: str,
    *,
    deadline_at: float,
) -> RunpodSecret:
    if _SECRET_NAME_RE.fullmatch(name) is None:
        raise ValueError("runpod payload secret name is invalid")
    if type(value) is not str or not value:
        raise ValueError("runpod payload secret value is invalid")
    out = _mutation_once(
        _key_for_fingerprint(fingerprint),
        _GRAPHQL_URL,
        method="POST",
        body={"query": _CREATE_SECRET, "variables": {"name": name, "value": value}},
        deadline_at=deadline_at,
    )
    if type(out) is not dict or "errors" in out:
        _raise_graphql_mutation_error(out, "secret creation")
    data = out.get("data")
    row = data.get("secretCreate") if type(data) is dict else None
    if type(row) is not dict:
        raise RunpodMutationAmbiguous("runpod secret creation outcome is unknown")
    secret = RunpodSecret(
        _strict_id(row.get("id"), "secret id"),
        _strict_string(row.get("name"), "secret name"),
    )
    if secret.name != name:
        raise RunpodMutationAmbiguous("runpod secret creation outcome is unknown")
    return secret


def delete_secret_for_fingerprint(
    fingerprint: str,
    secret_id: str,
    secret_name: str,
    *,
    deadline_at: float,
) -> None:
    key = _key_for_fingerprint(fingerprint)
    try:
        out = _mutation_once(
            key,
            _GRAPHQL_URL,
            method="POST",
            body={"query": _DELETE_SECRET, "variables": {"id": secret_id}},
            deadline_at=deadline_at,
        )
        if type(out) is not dict or "errors" in out:
            _raise_graphql_mutation_error(out, "secret deletion")
        data = out.get("data")
        if type(data) is not dict or "secretDelete" not in data or data["secretDelete"] is not None:
            raise RunpodMutationAmbiguous("runpod secret deletion outcome is unknown")
    except RunpodRequestError as exc:
        if exc.status_code != 404:
            raise
    except RunpodMutationAmbiguous:
        pass
    for poll in range(_DELETE_CONFIRM_POLLS):
        _account, remaining = list_secrets_for_fingerprint(
            fingerprint, name=secret_name, deadline_at=deadline_at
        )
        if not remaining:
            return
        if poll + 1 < _DELETE_CONFIRM_POLLS:
            time.sleep(_DELETE_CONFIRM_WAIT_S)
    raise RunpodApiError(f"runpod payload secret {secret_id} deletion is unconfirmed")


def _parse_volume(row: object) -> RunpodNetworkVolume:
    if type(row) is not dict:
        raise RunpodApiError("runpod network volume response row is invalid")
    return RunpodNetworkVolume(
        id=_strict_id(row.get("id"), "network volume id"),
        name=_strict_string(row.get("name"), "network volume name"),
        size_gb=_strict_positive_int(row.get("size"), "network volume size"),
        data_center_id=_strict_string(row.get("dataCenterId"), "network volume data center"),
    )


def list_network_volumes_for_fingerprint(
    fingerprint: str, *, deadline_at: float
) -> list[RunpodNetworkVolume]:
    out = _CLIENT.request_with_retries_for_key(
        _key_for_fingerprint(fingerprint),
        f"{REST_BASE}/networkvolumes",
        retries=2,
        deadline_at=deadline_at,
    )
    rows = out if type(out) is list else out.get("networkVolumes") if type(out) is dict else None
    if type(rows) is not list:
        raise RunpodApiError("runpod network volume response must be a list")
    return [_parse_volume(row) for row in rows]


def create_network_volume_for_fingerprint(
    fingerprint: str,
    *,
    name: str,
    size_gb: int,
    data_center_id: str,
    deadline_at: float,
) -> RunpodNetworkVolume:
    out = _mutation_once(
        _key_for_fingerprint(fingerprint),
        f"{REST_BASE}/networkvolumes",
        method="POST",
        body={"name": name, "size": size_gb, "dataCenterId": data_center_id},
        deadline_at=deadline_at,
    )
    return _parse_volume(out)


def delete_network_volume_for_fingerprint(
    fingerprint: str,
    volume_id: str,
    *,
    deadline_at: float,
) -> None:
    """Delete one exact account-owned volume and confirm authoritative absence."""
    try:
        _mutation_once(
            _key_for_fingerprint(fingerprint),
            f"{REST_BASE}/networkvolumes/{_strict_id(volume_id, 'network volume id')}",
            method="DELETE",
            body=None,
            deadline_at=deadline_at,
        )
    except RunpodRequestError as exc:
        if exc.status_code != 404:
            raise
    except RunpodMutationAmbiguous:
        pass
    for poll in range(_DELETE_CONFIRM_POLLS):
        remaining = list_network_volumes_for_fingerprint(fingerprint, deadline_at=deadline_at)
        if all(volume.id != volume_id for volume in remaining):
            return
        if poll + 1 < _DELETE_CONFIRM_POLLS:
            time.sleep(_DELETE_CONFIRM_WAIT_S)
    raise RunpodApiError(f"runpod network volume {volume_id} deletion is unconfirmed")


def grow_network_volumes_for_fingerprint(
    fingerprint: str,
    wanted: dict[str, int],
    *,
    deadline_at: float,
) -> dict[str, int]:
    return grow_network_volumes_for_key(
        _key_for_fingerprint(fingerprint), wanted, deadline_at=deadline_at
    )


def _parse_data_centers(value: object) -> list[RunpodDataCenter]:
    rows = value.get("dataCenters") if type(value) is dict else value
    if type(rows) is not list:
        raise RunpodApiError("runpod data center response must be a list")
    parsed = []
    seen = set()
    for raw in rows:
        if type(raw) is not dict:
            raise RunpodApiError("runpod data center response row is invalid")
        data_center_id = _strict_id(raw.get("id"), "data center id")
        volume_types = raw.get("networkVolumeTypes")
        if type(volume_types) is not list:
            raise RunpodApiError("runpod data center volume types are invalid")
        types = tuple(
            sorted({_strict_string(item, "network volume type") for item in volume_types})
        )
        if data_center_id in seen:
            raise RunpodApiError("runpod data center response contains duplicate ids")
        seen.add(data_center_id)
        parsed.append(RunpodDataCenter(data_center_id, types))
    return parsed


def list_storage_datacenters_for_fingerprint(fingerprint: str, *, deadline_at: float) -> list[str]:
    out = _CLIENT.request_with_retries_for_key(
        _key_for_fingerprint(fingerprint),
        f"{CATALOG_BASE}/datacenters",
        retries=2,
        deadline_at=deadline_at,
    )
    return [item.id for item in _parse_data_centers(out) if item.network_volume_types]
