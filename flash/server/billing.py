"""Charge completed Flash training runs to the freesolo backend.

The control plane records the submitting org id when a run is accepted, then POSTs the final
``RunStatus.cost_usd`` after the run reaches ``done``. The call is authenticated with the
operator internal key, so Flash never persists a user's freesolo API key while training runs.
Tests stub the network boundary directly."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from decimal import ROUND_HALF_UP, Decimal

from .auth import freesolo_base_url

# A charge does more than a verify; allow a bit more than auth's 5s but stay bounded.
_CHARGE_TIMEOUT_S = 10.0
_COMPLETION_CHARGE_PATH = "/api/billing/training-usage/internal"


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
        raise BillingError(503, f"billing service unavailable: {exc}") from exc
    try:
        return json.loads(raw or b"{}")
    except ValueError as exc:
        # The backend responded but the body isn't JSON -- a bad gateway, not an outage.
        raise BillingError(502, f"billing service returned an invalid response: {exc}") from exc


def charge_completed_run(*, internal_key: str, status) -> dict:
    """Charge one completed external run using its persisted non-secret billing context.

    The backend route is idempotent by ``runId``. Raises ``BillingError`` on a non-2xx or
    unreachable backend; callers should record that billing failed without changing the run's
    terminal training state.
    """
    context = status.billing_context if isinstance(status.billing_context, dict) else {}
    org_id = str(context.get("org_id") or "").strip()
    if not org_id:
        raise BillingError(400, "missing billing org id for completed training run")
    spec = status.spec or {}
    remote = status.remote or {}
    gpu = remote.get("allocated_gpu") or (spec.get("gpu") or {}).get("type")
    provider = remote.get("provider")
    total_usd = float(status.cost_usd or 0.0)
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
            "costBasis": "final",
            "costSource": "run_status.cost_usd",
        },
    }
    return _post_billing(token=internal_key, path=_COMPLETION_CHARGE_PATH, body=body)
