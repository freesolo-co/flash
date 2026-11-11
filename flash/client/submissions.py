"""Idempotent paid-run submission over PUT /v1/run-submissions/{id}.

A paid submit can outlive any one HTTP request: the control plane reads and profiles the whole
dataset before admitting the run (minutes for a large image dataset). Over the legacy
non-idempotent POST /v1/runs, a client that timed out reported a failure while the server kept
going and created the run anyway, so the natural retry created a second paid run (and started a
second profile beside the first, slowing both). Here one random submission ID is bound to the
request before the first send and reused on every retry: while the first request is still being
admitted the plane answers 409 "in progress" without starting new work, and once it has
finished the same PUT returns that run. At most one run is ever created per call.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable

from flash.client.http import ApiClient, ApiError, RequestTimeoutError, ServiceUnreachableError

# Total time to keep re-attaching to one submission before giving up on waiting for it.
SUBMIT_DEADLINE_SECONDS = 60 * 60.0
# Pause between re-attaches while the plane reports the submission still in progress.
IN_PROGRESS_WAIT_SECONDS = 10.0
# Pause after a dropped request or gateway error; consecutive transport failures are capped so a
# misconfigured URL fails in about a minute instead of retrying for the whole deadline.
TRANSPORT_WAIT_SECONDS = 5.0
MAX_CONSECUTIVE_TRANSPORT_FAILURES = 12
_GATEWAY_STATUSES = frozenset({502, 503, 504})


def new_submission_id() -> str:
    """A fresh idempotency key (32 URL-safe characters, inside the plane's 16-128 limit)."""
    return secrets.token_urlsafe(24)


def _route_missing(exc: ApiError) -> bool:
    """A plane older than run-submissions: the route itself is absent, so nothing ran."""
    return exc.status in (404, 405) and str(exc.detail) in ("Not Found", "Method Not Allowed")


def _in_progress(exc: ApiError) -> bool:
    return exc.status == 409 and "in progress" in str(exc.detail)


def submit_run(
    client: ApiClient,
    body: dict,
    *,
    submission_id: str | None = None,
    deadline_seconds: float = SUBMIT_DEADLINE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict | None:
    """Submit a paid run exactly once; returns None when the plane predates run-submissions."""
    submission_id = submission_id or new_submission_id()
    path = f"/v1/run-submissions/{submission_id}"
    give_up = clock() + deadline_seconds
    transport_failures = 0
    while True:
        try:
            return client._request("PUT", path, body=body, require={"run_id": str})
        except ApiError as exc:
            if _route_missing(exc):
                return None
            if _in_progress(exc):
                transport_failures, wait = 0, IN_PROGRESS_WAIT_SECONDS
            elif exc.status in _GATEWAY_STATUSES:
                transport_failures, wait = transport_failures + 1, TRANSPORT_WAIT_SECONDS
            else:
                raise
            last: Exception = exc
        except (RequestTimeoutError, ServiceUnreachableError) as exc:
            # the request may still be running server-side; re-sending the same ID attaches to it.
            transport_failures, wait, last = transport_failures + 1, TRANSPORT_WAIT_SECONDS, exc
        if transport_failures > MAX_CONSECUTIVE_TRANSPORT_FAILURES or clock() + wait > give_up:
            raise RequestTimeoutError(
                f"submission {submission_id} was not confirmed ({last}). It may still create a "
                "run: check `flash runs list` before submitting again"
            ) from last
        sleep(wait)
