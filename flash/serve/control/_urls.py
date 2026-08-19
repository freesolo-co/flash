"""canonical provider-managed public url validation."""

from __future__ import annotations

import re
from urllib.parse import SplitResult, urlsplit

# runpod does not publish a fixed pod id length: its own rest schema gives a 14-character example
# ("xedezhzb9la3ye") and live accounts return 14, while 13 also occurs. pinning an exact length
# rejected every real pod at parse time, so this bounds the charset and a sane range instead. the
# length is not what makes the proxy origin safe -- validate_runpod_public_url requires the
# hostname to equal f"{pod_id}-8000.proxy.runpod.net" exactly, which is the actual control.
_RUNPOD_POD_ID_RE = re.compile(r"[a-z0-9]{10,32}")
_MODAL_HOST_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.modal\.run")


def validate_runpod_pod_id(value: object) -> str:
    if type(value) is not str or _RUNPOD_POD_ID_RE.fullmatch(value) is None:
        raise ValueError("pod_id must be 10 to 32 lowercase alphanumeric characters")
    return value


def _split_canonical_origin(value: object, name: str) -> tuple[str, SplitResult]:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty unpadded string")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical provider-managed https origin") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(f"{name} must be a canonical provider-managed https origin")
    hostname = parsed.hostname
    if parsed.netloc != hostname:
        raise ValueError(f"{name} must use a canonical lowercase hostname")
    return hostname, parsed


def validate_modal_public_url(value: object) -> str:
    hostname, parsed = _split_canonical_origin(value, "public_url")
    if _MODAL_HOST_RE.fullmatch(hostname) is None:
        raise ValueError("modal public_url must use an exact modal.run hostname")
    canonical = f"https://{hostname}"
    if parsed.path == "/":
        canonical += "/"
    if value != canonical:
        raise ValueError("modal public_url must be a canonical provider-managed https origin")
    return value


def validate_runpod_public_url(value: object, pod_id: object) -> str:
    validated_pod_id = validate_runpod_pod_id(pod_id)
    hostname, parsed = _split_canonical_origin(value, "public_url")
    expected_hostname = f"{validated_pod_id}-8000.proxy.runpod.net"
    if hostname != expected_hostname:
        raise ValueError("runpod public_url must match the exact persistent pod proxy origin")
    canonical = f"https://{expected_hostname}"
    if parsed.path == "/":
        canonical += "/"
    if value != canonical:
        raise ValueError("runpod public_url must be a canonical persistent pod proxy origin")
    return value
