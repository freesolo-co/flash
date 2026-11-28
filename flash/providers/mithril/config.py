"""Control-plane-only Mithril credentials and project ownership.

There is no region setting: Flash admits Spot offers from every region and bids in the region of
the cheapest qualifying offer. Each bid records its own region.
"""

import hashlib
import os


def settings() -> dict[str, str]:
    return {
        key: os.environ.get(f"FLASH_MITHRIL_{key.upper()}", "").strip()
        for key in ("api_key", "project_id", "image_version", "ssh_key_ids")
    }


def configured() -> bool:
    config = settings()
    return all(config[key] for key in ("api_key", "project_id"))


def scope_fingerprint() -> str:
    config = settings()
    if not configured():
        raise RuntimeError("Configure FLASH_MITHRIL_API_KEY and PROJECT_ID")
    # Ownership is the project; each bid records its own region.
    return hashlib.sha256(f"mithril:{config['project_id']}:*".encode()).hexdigest()
