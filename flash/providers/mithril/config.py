"""Control-plane-only Mithril credentials and project (optionally region) ownership.

FLASH_MITHRIL_REGION is optional. Unset, Flash admits Spot offers from every region and bids in
the region of the cheapest qualifying offer; set, it keeps the original single-region behavior.
"""

import hashlib
import os


def settings() -> dict[str, str]:
    return {
        key: os.environ.get(f"FLASH_MITHRIL_{key.upper()}", "").strip()
        for key in ("api_key", "project_id", "region", "image_version", "ssh_key_ids")
    }


def configured() -> bool:
    config = settings()
    return all(config[key] for key in ("api_key", "project_id"))


def region_scope_fingerprint(project_id: str, region: str) -> str:
    """The original project+region ownership scope (also what a pinned-region plane issues)."""
    # Namespaced so a persisted legacy provider identity can never be adopted by Mithril.
    return hashlib.sha256(f"mithril:{project_id}:{region}".encode()).hexdigest()


def scope_fingerprint() -> str:
    config = settings()
    if not configured():
        raise RuntimeError("Configure FLASH_MITHRIL_API_KEY and PROJECT_ID")
    if config["region"]:
        return region_scope_fingerprint(config["project_id"], config["region"])
    # Any-region plane: ownership is the project; each bid records its own region.
    return hashlib.sha256(f"mithril:{config['project_id']}:*".encode()).hexdigest()


def region_allowed(region: str) -> bool:
    configured_region = settings()["region"]
    return bool(region) and (not configured_region or region == configured_region)
