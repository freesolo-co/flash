"""Control-plane-only Mithril credentials and explicit project/region ownership."""

import hashlib
import os


def settings() -> dict[str, str]:
    return {
        key: os.environ.get(f"FLASH_MITHRIL_{key.upper()}", "").strip()
        for key in ("api_key", "project_id", "region", "image_version", "ssh_key_ids")
    }


def configured() -> bool:
    config = settings()
    return all(config[key] for key in ("api_key", "project_id", "region"))


def scope_fingerprint() -> str:
    config = settings()
    if not configured():
        raise RuntimeError("Configure FLASH_MITHRIL_API_KEY, PROJECT_ID and REGION")
    # Namespaced so a persisted legacy provider identity can never be adopted by Mithril.
    return hashlib.sha256(f"mithril:{config['project_id']}:{config['region']}".encode()).hexdigest()
