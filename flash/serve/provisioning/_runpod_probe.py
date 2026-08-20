"""deadline-bounded direct endpoint proof for persistent runpod pods."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ._common import DeploymentBundle
from ._modal_probe import _provenance_matches as _modal_provenance_matches
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
    """the same exact comparison the modal probe applies, not a weaker one.

    readiness means the same thing on both providers, so this reuses `_modal_probe`'s check rather
    than keeping a second implementation that can drift from it. the check this replaced compared
    only five deployment-wide fields and accepted any superset of the revision ids, so a pod that
    omitted the run alias, served extra models, or reported wrong per-adapter provenance still
    passed -- and the customer got a "ready" deployment whose documented alias request could 404.
    """

    return _modal_provenance_matches(payload, bundle)
