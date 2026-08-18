"""deadline-bounded direct endpoint proof for persistent runpod pods."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ._common import DeploymentBundle
from ._runpod_transport import build_no_redirect_opener

_MAX_PROBE_RESPONSE_BYTES = 2 * 1024 * 1024


class RunPodEndpointProbe:
    """one no-redirect authenticated proof of exact serving provenance."""

    __slots__ = ("_opener",)

    def __init__(self, *, opener: object | None = None) -> None:
        self._opener = build_no_redirect_opener() if opener is None else opener

    def __call__(
        self,
        public_url: str,
        inference_token: str,
        bundle: DeploymentBundle,
        timeout_seconds: float,
    ) -> bool:
        if type(timeout_seconds) not in {int, float} or timeout_seconds <= 0:
            return False
        request = urllib.request.Request(
            public_url + "/v1/models",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {inference_token}",
            },
            method="GET",
        )
        try:
            with self._open(request, float(timeout_seconds)) as response:
                if getattr(response, "status", None) != 200:
                    return False
                raw = response.read(_MAX_PROBE_RESPONSE_BYTES + 1)
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError, OSError):
            return False
        except Exception:
            return False
        if len(raw) > _MAX_PROBE_RESPONSE_BYTES:
            return False
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            return False
        return _provenance_matches(payload, bundle)

    def _open(self, request: urllib.request.Request, timeout: float):
        method = getattr(self._opener, "open", None)
        if callable(method):
            return method(request, timeout=timeout)
        if callable(self._opener):
            return self._opener(request, timeout=timeout)
        raise OSError("probe opener is invalid")


def _provenance_matches(payload: object, bundle: DeploymentBundle) -> bool:
    if type(payload) is not dict or type(payload.get("data")) is not list:
        return False
    expected = {adapter.adapter_revision for adapter in bundle.spec.adapters}
    observed: set[str] = set()
    for entry in payload["data"]:
        if type(entry) is not dict or type(entry.get("id")) is not str:
            return False
        provenance = entry.get("flash_provenance")
        if type(provenance) is not dict:
            return False
        if (
            provenance.get("deployment_id") != bundle.spec.deployment_id
            or provenance.get("spec_id") != bundle.spec.spec_id
            or provenance.get("manifest_id") != bundle.manifest.manifest_id
            or provenance.get("engine_id") != bundle.spec.engine.engine_id
            or provenance.get("image_digest") != bundle.image.digest
        ):
            return False
        observed.add(entry["id"])
    return expected <= observed
