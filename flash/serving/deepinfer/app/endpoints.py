"""Resolve the active replica directory; legacy deployments remain readable."""

from urllib.parse import urlsplit

FLEET_NAME = "freesolo-deepseek-fleet"
FLEET_STATE = "freesolo-deepseek-fleet-state"


def validate_endpoint(endpoint):
    url = urlsplit(endpoint or "")
    if (
        url.scheme != "https"
        or not url.hostname
        or not url.hostname.endswith((".modal.direct", ".modal.run"))
        or url.username
        or url.password
        or url.port not in {None, 443}
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("Invalid private Modal replica endpoint")
    return endpoint.rstrip("/")


def replica_endpoints():
    import modal

    from flash.serving.deepinfer.app.client import endpoint_url

    try:
        routes = modal.Dict.from_name(FLEET_STATE).get("routes", None)
    except modal.exception.NotFoundError:
        routes = None
    if routes is None:
        return [endpoint_url()]
    # An explicitly empty directory is not permission to resurrect retired code.
    return [validate_endpoint(url) for url in routes["endpoints"]]
