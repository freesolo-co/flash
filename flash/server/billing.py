"""Charge a run's pre-flight cost to the freesolo backend at submit time.

The control plane computes the estimate from the spec and POSTs it (authenticated with the
submitting user's freesolo key) to the billing endpoint, which debits the org. The charge GATES
the run: a non-2xx (e.g. 402) or an unreachable backend raises ``BillingError`` so the run never
starts unpaid. Tests stub the network boundary directly."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from decimal import ROUND_HALF_UP, Decimal

from flash.cost.spec import estimate_for_spec

from .auth import freesolo_base_url

# A charge does more than a verify; allow a bit more than auth's 5s but stay bounded.
_CHARGE_TIMEOUT_S = 10.0
# The backend route family the charge (and its reversal) POST to.
_CHARGE_PATH = "/api/billing/training-usage"
_REVERSE_PATH = "/api/billing/training-usage/reverse"


class BillingError(Exception):
    """A run charge that didn't succeed. ``status_code`` (402 insufficient balance, 503 backend
    unreachable) is surfaced to the client with ``detail``."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _cents(usd: float) -> int:
    """Whole cents for a USD amount, round-HALF-UP (not Python's banker's rounding, which would
    undercharge a half-cent tie), never negative."""
    cents = Decimal(str(usd)).scaleb(2).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return max(0, int(cents))


def _http_reason(exc: urllib.error.HTTPError) -> str:
    """The backend's status reason phrase (``.reason``/``.msg``), else the bare code."""
    reason = getattr(exc, "reason", None) or getattr(exc, "msg", None)
    return str(reason).strip() if reason else str(exc.code)


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Clean message from the backend's ``{"detail": {"error", "code"}}`` JSON error body, else
    the status reason (never a bare code)."""

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
    """POST a JSON body to the backend billing ``path`` and return the parsed response.

    Raises ``BillingError`` (the route's status + a clean detail) on a non-2xx, and ``503``
    when the service is unreachable -- the same translation the charge and its reversal share.
    """
    url = f"{freesolo_base_url()}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_CHARGE_TIMEOUT_S) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise BillingError(exc.code, _http_error_detail(exc)) from exc
    except (urllib.error.URLError, OSError) as exc:
        # Unreachable billing service: block the run (503, retry later) rather than run free.
        raise BillingError(503, f"billing service unavailable: {exc}") from exc
    try:
        return json.loads(raw or b"{}")
    except ValueError as exc:
        # The backend responded but the body isn't JSON -- a bad gateway, not an outage.
        raise BillingError(502, f"billing service returned an invalid response: {exc}") from exc


def charge_run_estimate(*, token: str, spec) -> dict:
    """Compute ``spec``'s estimate and charge it to the submitting user's org; return the backend
    response. Raises ``BillingError`` on a non-2xx or unreachable backend. The charge equals the
    ``flash train --cost`` quote (same catalog-only, cheapest-fit basis)."""
    estimate = estimate_for_spec(spec)
    body = {
        "runId": spec.run_id,
        "costCents": _cents(estimate.total_usd),
        "gpu": estimate.gpu,
        "provider": estimate.provider,
        "method": estimate.method,
        "model": estimate.model_id,
        "estimate": {
            "totalUsd": estimate.total_usd,
            "gpuHourlyUsd": estimate.gpu_hourly_usd,
            "wallClockHours": estimate.wall_clock_hours,
            "steps": estimate.steps,
        },
    }
    return _post_billing(token=token, path=_CHARGE_PATH, body=body)


def reverse_run_charge(*, token: str, run_id: str, charge: dict | None = None) -> dict:
    """Refund the charge for a run whose submit failed (create_run charges before handing off).
    Best-effort + idempotent by ``runId`` (a retried submit re-charges cleanly); ``charge`` is
    forwarded so the backend can match the exact debit."""
    body: dict = {"runId": run_id, "reverse": True}
    if isinstance(charge, dict) and charge.get("amountCents") is not None:
        body["costCents"] = int(charge["amountCents"])
    return _post_billing(token=token, path=_REVERSE_PATH, body=body)
