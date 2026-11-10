"""Explicit project scope for Nebius CLI authentication."""

import hashlib
import os
import shutil

_KEYS = (
    "FLASH_NEBIUS_PROJECT_ID",
    "FLASH_NEBIUS_REGION",
    "FLASH_NEBIUS_SUBNET_ID",
    "FLASH_NEBIUS_IMAGE_ID",
)


def settings() -> dict[str, str]:
    return {
        key.removeprefix("FLASH_NEBIUS_").lower(): os.environ.get(key, "").strip() for key in _KEYS
    }


def preemptible_vm_limit() -> int | None:
    """The project's preemptible-VM quota (FLASH_NEBIUS_PREEMPTIBLE_VM_LIMIT), or None if unset.

    Nebius counts every preemptible VM in the project against `compute.instance.preemptible.count`
    and refuses creates beyond it. Set this to that quota so Spot jobs wait for a free slot.
    """
    raw = os.environ.get("FLASH_NEBIUS_PREEMPTIBLE_VM_LIMIT", "").strip()
    if not raw:
        return None
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError("FLASH_NEBIUS_PREEMPTIBLE_VM_LIMIT must be a positive integer")
    return int(raw)


def configured() -> bool:
    return all(settings().values()) and shutil.which("nebius") is not None


def scope_fingerprint() -> str:
    config = settings()
    if not all(config.values()):
        raise RuntimeError(
            "Nebius project, region, subnet and worker host image must be configured"
        )
    return hashlib.sha256(f"{config['project_id']}:{config['region']}".encode()).hexdigest()
