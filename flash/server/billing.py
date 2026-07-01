"""Charge completed Flash training runs to the freesolo backend."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from decimal import ROUND_HALF_UP, Decimal

from ._internal_client import DEFAULT_TIMEOUT_S, build_internal_request, org_id_of

_COMPLETION_CHARGE_PATH = "/api/billing/training-usage/internal"
_PRECHECK_PATH = "/api/billing/training-usage/precheck"


class BillingError(Exception):
    """A run charge that didn't succeed."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _cents(usd: float) -> int:
    """Whole cents, ROUND_HALF_UP (not banker's rounding), never negative."""
    cents = Decimal(str(usd)).scaleb(2).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return max(0, int(cents))


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
    except (urllib.error.URLError, OSError) as exc:
        raise BillingError(503, f"billing service unavailable: {exc}") from exc
    try:
        return json.loads(raw or b"{}")
    except ValueError as exc:
        # The backend responded but the body isn't JSON -- a bad gateway, not an outage.
        raise BillingError(502, f"billing service returned an invalid response: {exc}") from exc


def precheck_training_run(*, internal_key: str, org_id: str, estimate_usd: float) -> dict:
    """Verify-only budget pre-flight before a run is submitted.

    Asks the backend whether ``org_id`` could cover the run's flash.cost estimate. Raises
    ``BillingError`` (status_code 402) when it can't, so the caller can reject the run before any
    GPU is allocated. Moves no money -- the real charge still happens at completion. Backend
    unreachable / 5xx surface as ``BillingError`` with a non-402 status_code so the caller can
    fail open (a billing-infra blip must not block all training)."""
    body = {"orgId": org_id, "estimateCents": _cents(estimate_usd)}
    return _post_billing(token=internal_key, path=_PRECHECK_PATH, body=body)


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
    spec = status.spec or {}
    remote = status.remote or {}
    gpu = remote.get("allocated_gpu") or (spec.get("gpu") or {}).get("type")
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
