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


def region_scope_fingerprint(project_id: str, region: str) -> str:
    """The retired project+region scope that attempts from region-pinned releases carry."""
    # Namespaced so a persisted legacy provider identity can never be adopted by Mithril.
    return hashlib.sha256(f"mithril:{project_id}:{region}".encode()).hexdigest()


def scope_fingerprint() -> str:
    config = settings()
    if not configured():
        raise RuntimeError("Configure FLASH_MITHRIL_API_KEY and PROJECT_ID")
    # Ownership is the project; each bid records its own region.
    return hashlib.sha256(f"mithril:{config['project_id']}:*".encode()).hexdigest()


def recorded_scope_is_ours(fingerprint: str) -> bool:
    """Whether a run-level recorded Spot scope belongs to this control plane's project.

    Runs submitted by region-pinned releases recorded the retired project+region scope. Without a
    handle there is no bid whose region could be checked, so accept that scope for any region
    Mithril lists; the fingerprint embeds the project, so a match still proves the owner. A failed
    region listing raises (ownership unverified, not disproven), so recovery fails closed and retries.
    """
    current = scope_fingerprint()
    if fingerprint == current:
        return True
    from flash.providers.mithril.client import call

    project = settings()["project_id"]
    auctions = call("GET", "/spot/availability")
    if not isinstance(auctions, list):
        return False
    regions = {
        auction["region"]
        for auction in auctions
        if isinstance(auction, dict) and isinstance(auction.get("region"), str)
    }
    return any(fingerprint == region_scope_fingerprint(project, region) for region in regions)
