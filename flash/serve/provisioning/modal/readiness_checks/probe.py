"""deadline-bounded direct endpoint proof for modal deployments."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from flash.serve.app.manifest import ManifestAdapter
from flash.serve.control._urls import validate_modal_public_url
from flash.serve.provisioning.common.records import DeploymentBundle

_MAX_PROBE_RESPONSE_BYTES = 2 * 1024 * 1024


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ModalEndpointProbe:
    """one no-redirect authenticated proof of exact serving provenance."""

    __slots__ = ("_opener",)

    def __init__(self, *, opener: object | None = None) -> None:
        self._opener = (
            urllib.request.build_opener(_NoRedirectHandler()) if opener is None else opener
        )

    def __call__(
        self,
        public_url: str,
        inference_token: str,
        bundle: DeploymentBundle,
        timeout_seconds: float,
    ) -> bool:
        if type(timeout_seconds) not in {int, float} or timeout_seconds <= 0:
            return False
        try:
            origin = validate_modal_public_url(public_url).rstrip("/")
        except ValueError:
            return False
        request = urllib.request.Request(
            origin + "/v1/models",
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


def _expected_provenance(
    bundle: DeploymentBundle,
    requested_model: str,
    adapter: ManifestAdapter,
) -> dict[str, object]:
    manifest = bundle.manifest
    return {
        "deployment_id": bundle.spec.deployment_id,
        "spec_id": bundle.spec.spec_id,
        "manifest_id": manifest.manifest_id,
        "engine_id": bundle.spec.engine.engine_id,
        "image_digest": bundle.image.digest,
        "logical_base_model": manifest.logical_base_model,
        "logical_base_revision": manifest.logical_base_revision,
        "served_checkpoint": manifest.engine.served_model,
        "served_checkpoint_revision": manifest.engine.model_revision,
        "tokenizer_model": manifest.engine.tokenizer_model,
        "tokenizer_revision": manifest.engine.tokenizer_revision,
        "requested_model": requested_model,
        "checkpoint_id": adapter.checkpoint_id,
    }


def _expected_models(bundle: DeploymentBundle) -> dict[str, dict[str, object]]:
    return {
        adapter.checkpoint_id: _expected_provenance(bundle, adapter.checkpoint_id, adapter)
        for adapter in bundle.manifest.adapters
    }


def _provenance_matches(payload: object, bundle: DeploymentBundle) -> bool:
    if type(payload) is not dict or type(payload.get("data")) is not list:
        return False
    expected = _expected_models(bundle)
    entries = payload["data"]
    if not expected or len(entries) != len(expected):
        return False
    observed: set[str] = set()
    for entry in entries:
        if type(entry) is not dict or type(entry.get("id")) is not str:
            return False
        model_id = entry["id"]
        if model_id in observed or model_id not in expected:
            return False
        if entry.get("flash_provenance") != expected[model_id]:
            return False
        observed.add(model_id)
    return observed == set(expected)
