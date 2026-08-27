"""Charge completed Flash training runs to the freesolo backend."""

from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request
from enum import Enum

from flash._internal.diagnostics import sanitize_diagnostic
from flash.core.spec import attributed_gpu_type
from flash.cost.currency import usd_cents as _cents
from flash.server.platform.internal_client import (
    DEFAULT_TIMEOUT_S,
    build_internal_request,
    org_id_of,
)

_COMPLETION_CHARGE_PATH = "/api/billing/training-usage/internal"
_PRECHECK_PATH = "/api/billing/training-usage/precheck"


class BillingError(Exception):
    """A run charge that didn't succeed."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class PrecheckFailureSource(Enum):
    HTTP = "http"
    TRANSPORT = "transport"
    PROTOCOL = "protocol"


class PrecheckHttpDisposition(Enum):
    SUCCESS = "success"
    AFFORDABILITY = "affordability"
    FAIL_OPEN = "fail_open"
    BLOCK = "block"


class PrecheckRetryDisposition(Enum):
    FAIL_OPEN = "fail_open"
    BLOCK = "block"


class PrecheckError(Exception):
    """A typed affordability-precheck failure for route classification."""

    def __init__(
        self,
        *,
        source: PrecheckFailureSource,
        retry: PrecheckRetryDisposition = PrecheckRetryDisposition.BLOCK,
        status_code: int | None = None,
        public_detail: str | None = None,
        private_detail: object = "",
    ) -> None:
        self.source = source
        self.retry = retry
        self.status_code = status_code
        self.public_detail = (
            sanitize_diagnostic(public_detail, limit=500) if public_detail is not None else None
        )
        self.private_detail = sanitize_diagnostic(private_detail, limit=500)
        super().__init__(self.public_detail or self.private_detail or source.value)


def _http_reason(exc: urllib.error.HTTPError) -> str:
    """Return a human-readable reason string from an HTTPError."""
    reason = getattr(exc, "reason", None) or getattr(exc, "msg", None)
    return str(reason).strip() if reason else str(exc.code)


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Extract a clean error message from a backend HTTPError response."""

    def _fallback() -> str:
        return f"billing failed ({exc.code} {_http_reason(exc)})"

    try:
        body = json.loads(exc.read() or b"{}")
    except (ValueError, OSError):
        return _fallback()
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return str(detail.get("error") or detail.get("code") or _fallback())
    if isinstance(detail, str) and detail:
        return detail
    return _fallback()


def _post_billing(*, token: str, path: str, body: dict) -> dict:
    """POST a JSON body to the backend billing path; raises BillingError on failure."""
    req = build_internal_request(path, body, token=token)
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise BillingError(exc.code, _http_error_detail(exc)) from exc
    except OSError as exc:
        raise BillingError(503, f"billing service unavailable: {exc}") from exc
    try:
        return json.loads(raw or b"{}")
    except ValueError as exc:
        # The backend responded but the body isn't JSON -- a bad gateway, not an outage.
        raise BillingError(502, f"billing service returned an invalid response: {exc}") from exc


_TRANSIENT_PRECHECK_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def classify_precheck_http_status(status_code: int) -> PrecheckHttpDisposition:
    """Classify an exposed HTTP status without transport or payload context."""
    if status_code == 200:
        return PrecheckHttpDisposition.SUCCESS
    if status_code == 402:
        return PrecheckHttpDisposition.AFFORDABILITY
    if status_code in _TRANSIENT_PRECHECK_HTTP_STATUSES:
        return PrecheckHttpDisposition.FAIL_OPEN
    return PrecheckHttpDisposition.BLOCK


def _is_transient_precheck_transport(exc: BaseException) -> bool:
    """Return whether a transport failure is safe to fail open."""
    current: object = exc
    seen: set[int] = set()
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, http.client.RemoteDisconnected):
            return False
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if isinstance(current, socket.gaierror):
            return current.errno == socket.EAI_AGAIN
        if not isinstance(current, urllib.error.URLError):
            return False
        current = current.reason
    return False


def _precheck_error_from_exception(exc: Exception) -> PrecheckError:
    source = (
        PrecheckFailureSource.TRANSPORT
        if isinstance(exc, (OSError, urllib.error.URLError))
        else PrecheckFailureSource.PROTOCOL
    )
    return PrecheckError(
        source=source,
        retry=(
            PrecheckRetryDisposition.FAIL_OPEN
            if source is PrecheckFailureSource.TRANSPORT and _is_transient_precheck_transport(exc)
            else PrecheckRetryDisposition.BLOCK
        ),
        private_detail=exc,
    )


def _precheck_response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if status is None and callable(getattr(response, "getcode", None)):
        status = response.getcode()
    if isinstance(status, int):
        return status
    raise PrecheckError(
        source=PrecheckFailureSource.PROTOCOL,
        private_detail="billing precheck response omitted an HTTP status",
    )


def _parse_precheck_response(raw: bytes) -> dict:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise PrecheckError(
            source=PrecheckFailureSource.PROTOCOL,
            private_detail=exc,
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"ok"} or payload["ok"] is not True:
        raise PrecheckError(
            source=PrecheckFailureSource.PROTOCOL,
            private_detail=f"unexpected billing precheck payload type {type(payload).__name__}",
        )
    return payload


def precheck_training_run(*, internal_key: str, org_id: str, estimate_usd: float) -> dict:
    """Verify affordability before a hosted billable run can create durable state."""
    body = {"orgId": org_id, "estimateCents": _cents(estimate_usd)}
    req = build_internal_request(_PRECHECK_PATH, body, token=internal_key)
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as response:
            status_code = _precheck_response_status(response)
            disposition = classify_precheck_http_status(status_code)
            if disposition is not PrecheckHttpDisposition.SUCCESS:
                raise PrecheckError(
                    source=(
                        PrecheckFailureSource.PROTOCOL
                        if 200 <= status_code < 300
                        else PrecheckFailureSource.HTTP
                    ),
                    retry=(
                        PrecheckRetryDisposition.FAIL_OPEN
                        if disposition is PrecheckHttpDisposition.FAIL_OPEN
                        else PrecheckRetryDisposition.BLOCK
                    ),
                    status_code=status_code,
                    private_detail=f"billing precheck returned HTTP {status_code}",
                )
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = _http_error_detail(exc)
        disposition = classify_precheck_http_status(exc.code)
        raise PrecheckError(
            source=PrecheckFailureSource.HTTP,
            retry=(
                PrecheckRetryDisposition.FAIL_OPEN
                if disposition is PrecheckHttpDisposition.FAIL_OPEN
                else PrecheckRetryDisposition.BLOCK
            ),
            status_code=exc.code,
            public_detail=(sanitize_diagnostic(detail, limit=500) if exc.code == 402 else None),
            private_detail=detail,
        ) from exc
    except PrecheckError:
        raise
    except Exception as exc:
        raise _precheck_error_from_exception(exc) from exc
    return _parse_precheck_response(raw)


def _charge_run(
    *, internal_key: str, status, total_usd: float, cost_basis: str, cost_source: str
) -> dict:
    """POST a customer charge for one run; backend route is idempotent by runId.

    ``cost_basis``/``cost_source`` ride on the (free-form) estimate metadata for audit -- they say
    where ``total_usd`` came from."""
    context = status.billing_context if isinstance(status.billing_context, dict) else {}
    org_id = org_id_of(context)
    if not org_id:
        raise BillingError(400, "missing billing org id for training run")
    spec = status.spec if isinstance(status.spec, dict) else {}
    remote = status.remote or status.realized_cost_remote
    remote = remote if isinstance(remote, dict) else {}
    allocated_gpu = remote.get("allocated_gpu")
    gpu = (
        allocated_gpu
        if isinstance(allocated_gpu, str) and allocated_gpu
        else attributed_gpu_type(status)
    )
    provider = remote.get("provider")
    total_usd = float(total_usd or 0.0)
    body = {
        "orgId": org_id,
        "runId": status.run_id,
        "costCents": _cents(total_usd),
        "gpu": gpu,
        "provider": provider,
        "method": spec.get("algorithm"),
        "model": spec.get("model"),
        "estimate": {
            "totalUsd": total_usd,
            "costBasis": cost_basis,
            "costSource": cost_source,
        },
    }
    return _post_billing(token=internal_key, path=_COMPLETION_CHARGE_PATH, body=body)


def charge_completed_run(*, internal_key: str, status) -> dict:
    """Charge one run its ``cost_usd`` -- the flash.cost estimate we quoted it at.

    ``cost_usd`` is the quote (planned steps) for a completed run, or the estimate at the steps
    actually run for a run cancelled mid-training (set by deploy.cancel_run). Backend route is
    idempotent by runId."""
    return _charge_run(
        internal_key=internal_key,
        status=status,
        total_usd=float(status.cost_usd or 0.0),
        cost_basis="estimate",
        cost_source="run_status.cost_usd",
    )
