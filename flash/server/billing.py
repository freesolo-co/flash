"""Charge a run's pre-flight cost estimate to the freesolo backend at submit time.

When a user submits a run (`slm train`), the control plane computes the run's pre-flight
estimate (``flash.cost``: wall-clock hours x market $/hr) from the validated spec and POSTs
it to the freesolo billing endpoint, authenticated with the SUBMITTING USER'S freesolo API
key (the same key the control plane already verifies for auth). The backend resolves the org
from that key and debits its prepaid balance for the estimate, returning the charge.

The charge GATES the run: a non-2xx (e.g. 402 insufficient balance) raises ``BillingError``,
which ``create_run`` turns into the same status so the user must top up before the run
consumes GPU. A network/backend outage also blocks (we never run for free). ``FLASH_SKIP_NET``
disables the call entirely (offline / tests), matching ``auth._freesolo_verify``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from flash.cost.spec import estimate_for_spec

from .auth import DEFAULT_FREESOLO_BASE_URL, FREESOLO_BASE_URL_ENV

# Pre-flight estimate + a debit is more involved than a verify; give the backend a little
# more room than auth's 5s, but still bounded so a hung backend doesn't wedge a submit.
_CHARGE_TIMEOUT_S = 10.0


class BillingError(Exception):
    """A run charge that did not succeed (insufficient balance, bad key, backend down).

    Carries the HTTP ``status_code`` to surface to the client (402 for insufficient balance,
    503 when the billing service is unreachable) and a human-readable ``detail``.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _cents(usd: float) -> int:
    """Whole cents for a USD amount (never negative)."""
    return max(0, round(float(usd) * 100))


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Pull a clean message out of the backend's JSON error body, falling back to the reason.

    The freesolo billing route returns ``{"detail": {"error": ..., "code": ...}}`` on a
    BillingError; surface that ``error`` so the user sees "Insufficient balance. Top up ..."
    rather than a bare status line.
    """
    try:
        body = json.loads(exc.read() or b"{}")
    except (ValueError, OSError):
        return f"billing failed ({exc.code})"
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return str(detail.get("error") or detail.get("code") or f"billing failed ({exc.code})")
    if isinstance(detail, str) and detail:
        return detail
    return f"billing failed ({exc.code})"


def charge_run_estimate(*, token: str, spec) -> dict:
    """Compute ``spec``'s estimate and charge it to the submitting user's org.

    Returns the backend's charge response (``{amountCents, balanceCents, ...}``) on success;
    a no-op ``{}`` when ``FLASH_SKIP_NET`` is set. Raises ``BillingError`` on any non-2xx
    response or when the billing service can't be reached.
    """
    # Offline bypass: no network, no charge (tests / air-gapped). Mirrors auth.
    if os.environ.get("FLASH_SKIP_NET"):
        return {}

    estimate = estimate_for_spec(spec)
    payload = json.dumps(
        {
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
    ).encode("utf-8")

    base = os.environ.get(FREESOLO_BASE_URL_ENV) or DEFAULT_FREESOLO_BASE_URL
    url = f"{base.rstrip('/')}/api/billing/training-usage"
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_CHARGE_TIMEOUT_S) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raise BillingError(exc.code, _http_error_detail(exc)) from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # The billing service is unreachable: block the run rather than run for free. 503 so
        # the user retries once billing is back, instead of treating it as a bad request.
        raise BillingError(503, f"billing service unavailable: {exc}") from exc
