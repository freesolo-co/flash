"""Idempotent paid-run submission over PUT /v1/run-submissions/{id}.

A paid submit can outlive any one HTTP request: the control plane reads and profiles the whole
dataset before admitting the run (minutes for a large image dataset). Over the legacy
non-idempotent POST /v1/runs, a client that timed out reported a failure while the server kept
going and created the run anyway, so the natural retry created a second paid run (and started a
second profile beside the first, slowing both). Here one random submission ID is bound to the
request before the first send and reused on every retry: while the first request is still being
admitted the plane answers 409 "in progress" without starting new work, and once it has
finished the same PUT returns that run. At most one run is ever created per call.

The ID also outlives the process: it is recorded under the Flash data dir before the first send
(see ``pending_submissions``), so rerunning the same command after an unconfirmed submit attaches
to the run the first attempt may have created instead of paying for a second one.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from flash.client import pending_submissions
from flash.client.http import ApiClient, ApiError, RequestTimeoutError, ServiceUnreachableError
from flash.client.pending_submissions import new_submission_id

# Total time to keep re-attaching to one submission before giving up on waiting for it.
SUBMIT_DEADLINE_SECONDS = 60 * 60.0
# Pause between re-attaches while the plane reports the submission still in progress.
IN_PROGRESS_WAIT_SECONDS = 10.0
# Pause after a dropped request or gateway error; consecutive transport failures are capped so a
# misconfigured URL fails in about a minute instead of retrying for the whole deadline.
TRANSPORT_WAIT_SECONDS = 5.0
MAX_CONSECUTIVE_TRANSPORT_FAILURES = 12
_GATEWAY_STATUSES = frozenset({502, 503, 504})


def _route_missing(exc: ApiError) -> bool:
    """A plane older than run-submissions: the route itself is absent, so nothing ran."""
    return exc.status in (404, 405) and str(exc.detail) in ("Not Found", "Method Not Allowed")


def _in_progress(exc: ApiError) -> bool:
    return exc.status == 409 and "in progress" in str(exc.detail)


class _LegacyPlane(Exception):
    """The plane predates run-submissions and nothing was sent that could create a run."""


def _definitive(exc: BaseException) -> bool:
    """A 4xx rejection: the plane created nothing for this ID, so a rerun must not reuse it.

    In-progress 409s never escape ``_attach`` (they end as an uncertain timeout), and a 5xx or
    a malformed success may follow a created run, so both keep the record.
    """
    return isinstance(exc, ApiError) and 400 <= exc.status < 500


def submit_run(
    client: ApiClient,
    body: dict,
    *,
    submission_id: str | None = None,
    deadline_seconds: float = SUBMIT_DEADLINE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict | None:
    """Submit a paid run exactly once; returns None when the plane predates run-submissions.

    Without an explicit ``submission_id``, the ID comes from (and is recorded in) the local
    pending-submission store, so an identical rerun after an uncertain outcome reuses it.
    """
    key = recorded = None
    if submission_id is None:
        try:
            key = pending_submissions.request_key(client, body)
            submission_id, recorded = pending_submissions.claim(key)
        except Exception:
            key, submission_id = None, new_submission_id()
    confirmed = False
    try:
        created = _attach(
            client, body, submission_id, bool(recorded), deadline_seconds, sleep, clock
        )
        confirmed = True
        return created
    except _LegacyPlane:
        confirmed = True
        return None
    except BaseException as exc:
        confirmed = _definitive(exc)
        raise
    finally:
        if key is not None:
            if confirmed:
                pending_submissions.forget(key, submission_id)
            else:
                pending_submissions.release(key, submission_id)


def _uncertain_detail(submission_id: str, recorded: bool) -> str:
    if recorded:
        return (
            f"submission {submission_id} was not confirmed. It may still create a run: "
            "re-running the same command attaches to this submission instead of creating a "
            "duplicate. Check `flash runs list` before submitting a changed config"
        )
    return (
        f"submission {submission_id} was not confirmed. It may still create a "
        "run: check `flash runs list` before submitting again"
    )


def _attach(
    client: ApiClient,
    body: dict,
    submission_id: str,
    recorded: bool,
    deadline_seconds: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> dict:
    """PUT one submission ID until the plane confirms a run, rejects it, or the deadline passes."""
    path = f"/v1/run-submissions/{submission_id}"
    give_up = clock() + deadline_seconds
    transport_failures = 0
    may_have_submitted = False
    uncertain_detail = _uncertain_detail(submission_id, recorded)
    while True:
        remaining = give_up - clock()
        if remaining <= 0:
            raise RequestTimeoutError(uncertain_detail)
        try:
            return client._request(
                "PUT",
                path,
                body=body,
                require={"run_id": str},
                timeout=min(client.timeout, remaining),
                body_deadline=remaining,
            )
        except ApiError as exc:
            if _route_missing(exc):
                # A rollback can remove the route while an earlier PUT still creates its run.
                if may_have_submitted:
                    raise RequestTimeoutError(uncertain_detail) from exc
                raise _LegacyPlane from exc
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
        may_have_submitted = True
        if transport_failures > MAX_CONSECUTIVE_TRANSPORT_FAILURES or clock() + wait >= give_up:
            raise RequestTimeoutError(uncertain_detail) from last
        sleep(wait)
