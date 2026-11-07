"""Environment-scoped Lambda firewall and persistent model-cache resources."""

from .config import PORT


async def firewall(api, environment, region):
    name = f"flash-serving-{environment}-{region}"
    rules = [
        {
            "protocol": "tcp",
            "port_range": [PORT, PORT],
            "source_network": "0.0.0.0/0",
            "description": "Per-replica authenticated TLS serving",
        }
    ]
    existing = [
        item
        for item in await api.call("GET", "firewall-rulesets")
        if item["name"] == name and item["region"]["name"] == region
    ]
    if existing:
        if len(existing) != 1 or existing[0]["rules"] != rules:
            raise RuntimeError("Lambda serving firewall differs from its declared rules")
        return existing[0]["id"]
    result = await api.call(
        "POST", "firewall-rulesets", {"name": name, "region": region, "rules": rules}
    )
    return result["id"]


async def cache_filesystem(api, environment, region):
    name = f"flash-serving-cache-{environment}-{region}"
    existing = [
        item
        for item in await api.call("GET", "filesystems")
        if item["name"] == name and item["region"]["name"] == region
    ]
    if len(existing) > 1:
        raise RuntimeError("duplicate Lambda serving cache filesystem")
    if existing:
        return existing[0]["id"]
    result = await api.call("POST", "filesystems", {"name": name, "region": region})
    return result["id"]
