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
from decimal import ROUND_HALF_UP, Decimal

from flash.cost.spec import estimate_for_spec, offline_estimate_for_spec

from .auth import DEFAULT_FREESOLO_BASE_URL, FREESOLO_BASE_URL_ENV

# Pre-flight estimate + a debit is more involved than a verify; give the backend a little
# more room than auth's 5s, but still bounded so a hung backend doesn't wedge a submit.
_CHARGE_TIMEOUT_S = 10.0
# The backend route family the charge (and its reversal) POST to.
_CHARGE_PATH = "/api/billing/training-usage"
_REVERSE_PATH = "/api/billing/training-usage/reverse"


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
    """Whole cents for a USD amount, round-HALF-UP, never negative.

    Money rounding must be explicit: Python's built-in ``round`` is banker's rounding (ties to
    even), which would turn e.g. $0.005 into 0 cents and silently undercharge on a half-cent tie
    (PR #3 review). Convert via ``Decimal`` with ``ROUND_HALF_UP`` so a tie always rounds up to
    the next cent, then clamp at zero.
    """
    cents = Decimal(str(float(usd))).scaleb(2).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return max(0, int(cents))


def _http_reason(exc: urllib.error.HTTPError) -> str:
    """The backend's status reason phrase (``exc.reason``/``.msg``), for the fallback detail.

    ``urllib`` exposes the reason on ``.reason`` (and historically ``.msg``); fall back across
    both and finally to the bare code so the detail is always informative.
    """
    reason = getattr(exc, "reason", None) or getattr(exc, "msg", None)
    return str(reason).strip() if reason else str(exc.code)


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    """Pull a clean message out of the backend's JSON error body, falling back to the reason.

    The freesolo billing route returns ``{"detail": {"error": ..., "code": ...}}`` on a
    BillingError; surface that ``error`` so the user sees "Insufficient balance. Top up ..."
    rather than a bare status line. When the body is missing/unparseable or carries no usable
    message, fall back to the backend's status REASON (e.g. "Payment Required") rather than
    dropping it -- a bare ``billing failed (<code>)`` is harder to diagnose (PR #3 review).
    """

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
    base = os.environ.get(FREESOLO_BASE_URL_ENV) or DEFAULT_FREESOLO_BASE_URL
    url = f"{base.rstrip('/')}{path}"
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
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raise BillingError(exc.code, _http_error_detail(exc)) from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # The billing service is unreachable: block the run rather than run for free. 503 so
        # the user retries once billing is back, instead of treating it as a bad request.
        raise BillingError(503, f"billing service unavailable: {exc}") from exc


def charge_run_estimate(*, token: str, spec) -> dict:
    """Compute ``spec``'s estimate and charge it to the submitting user's org.

    Returns the backend's charge response (``{amountCents, balanceCents, ...}``) on success;
    a no-op ``{}`` when ``FLASH_SKIP_NET`` is set. Raises ``BillingError`` on any non-2xx
    response or when the billing service can't be reached.

    CHARGE <= QUOTE (PR #3 review): the user is NEVER billed more than ``slm train --cost``
    quoted. The CLI prices the spec fully offline; the control plane parses the SAME spec
    without that guard, so for a ``model_policy = "allow"`` unlisted model the server-resolved
    GPU/sizing (a live HF probe) could otherwise diverge from the quote. We bill ``min`` of the
    OFFLINE-consistent estimate (the quote the user saw) and the spec-as-parsed estimate: in the
    normal case these are equal so the charge EQUALS the quote, and in the only case they differ
    (a divergent unlisted/open model) the charge is the smaller of the two -- so it can be lower
    than the quote but never higher.
    """
    # Offline bypass: no network, no charge (tests / air-gapped). Mirrors auth.
    if os.environ.get("FLASH_SKIP_NET"):
        return {}

    # The amount the user saw at --cost time (offline-consistent), and the amount the
    # server's HF-aware parse would imply; charge the smaller so the debit never exceeds the
    # quote even if the two ever diverge for an unlisted/open model.
    quote = offline_estimate_for_spec(spec)
    parsed = estimate_for_spec(spec)
    estimate = quote if quote.total_usd <= parsed.total_usd else parsed
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
    """Refund a charge that ``charge_run_estimate`` made for a run that then failed to start.

    ``create_run`` charges the org BEFORE handing the run to the runner; if submission then
    raises, the org has paid for a run that never started, so the debit must be reversed
    (PR #3 review: "Debit not reversed on submit failure"). Best-effort and idempotent: the
    backend dedupes by ``runId`` (reversing the same run twice is a no-op), so a retried
    submit can re-charge cleanly. A ``FLASH_SKIP_NET`` no-op mirrors the charge. ``charge`` is
    the original charge response, forwarded so the backend can match the exact debit when it
    wants to (otherwise it reverses by ``runId``).
    """
    if os.environ.get("FLASH_SKIP_NET"):
        return {}
    body: dict = {"runId": run_id, "reverse": True}
    if isinstance(charge, dict) and charge.get("amountCents") is not None:
        body["costCents"] = int(charge["amountCents"])
    return _post_billing(token=token, path=_REVERSE_PATH, body=body)
