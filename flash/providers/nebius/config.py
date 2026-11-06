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


def configured() -> bool:
    return all(settings().values()) and shutil.which("nebius") is not None


def scope_fingerprint() -> str:
    config = settings()
    if not all(config.values()):
        raise RuntimeError(
            "Nebius project, region, subnet and worker host image must be configured"
        )
    return hashlib.sha256(f"{config['project_id']}:{config['region']}".encode()).hexdigest()
