"""deadline-bounded direct endpoint proof for persistent runpod pods."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from flash.serve.provisioning.common.records import DeploymentBundle

# runpod deliberately reuses modal's exact provenance check so the providers cannot drift.
# a weaker check let pods pass while omitting the run alias or serving extra models.
from flash.serve.provisioning.modal.readiness_checks.probe import _provenance_matches
from flash.serve.provisioning.runpod.transport import USER_AGENT, build_no_redirect_opener

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
                # cloudflare fronts `*.proxy.runpod.net` and answers urllib's default
                # "Python-urllib/x.y" agent with a 403 error 1010 (browser_signature_banned) while
                # serving the identical request under any other agent. without this the probe can
                # never succeed against a real pod: it burns the whole readiness deadline against a
                # pod that is serving correctly and reports the deployment as outcome_unknown.
                # shared with the lifecycle transport so the two clients cannot drift.
                "User-Agent": USER_AGENT,
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
