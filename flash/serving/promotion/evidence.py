"""What a promotion must prove, as pure data and total predicates.

Every function here is pure: no network, no environment, no clock. The gate does the I/O and hands
the results in, so each rule is directly testable against a hand-built payload, including the
malformed ones a live deployment actually produces.

The rules fail CLOSED. Absent, malformed, ambiguous, or unparseable evidence is a failure, never a
pass, because the failure mode being defended against is a broken release that LOOKS healthy: a
router that booted while its engines did not, a stream that opened and closed without generating,
or usage captured in memory that never settled durably.

Reasons are stable snake_case codes. They are printed into a public build log, so they must never
carry a credential, a response body, or any part of a request payload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

# an undelivered row older than this is not "in flight", it is wedged.
#
# the threshold has to clear the outbox's OWN retry budget, or a transient billing 5xx reads as a
# stall and rolls back a healthy release. `DurableUsageOutbox._retry_or_quarantine` retries 8 times
# with `min(2 ** (attempt - 1), 300)` seconds of backoff plus up to 20% jitter, each attempt runs
# against a 10s client timeout, and the worker wakes on a 2s poll. that is ~154s of backoff, ~80s of
# attempts and ~16s of polling: a row still retrying legitimately can be ~250s old. past the budget
# the row is quarantined, so it leaves `oldest_undelivered_age_seconds` on its own.
#
# 360 sits above that worst case with margin and still well under the settlement horizon, so a row
# older than this is not retrying -- no healthy path leaves one undelivered that long.
_STALL_AGE_SECONDS = 360

_MISSING = object()


def _number(value: Any) -> float | None:
    """A JSON number from the snapshot, or None when the field is absent or not a number.

    Accepts floats as well as ints because these fields come from postgres numerics: the canonical
    reader (`usage_outbox.DurableUsageOutbox.snapshot`) coerces every one of them with `int(...)`
    for exactly that reason. Rejecting floats here would make the ORDINARY live shape unparseable,
    and an unparseable snapshot fails the gate -- which rolls a healthy release back.

    `bool` is excluded ahead of `int` because it is a subclass of it, so `True` would otherwise read
    as the number 1 and let a wrong-typed field pass as a plausible counter.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


HEALTH_MALFORMED = "health_malformed"
HEALTH_NOT_OK = "health_not_ok"
HEALTH_SHA_MISMATCH = "health_sha_mismatch"
HEALTH_DEPLOYMENT_ID_MISMATCH = "health_deployment_id_mismatch"
HEALTH_NO_ENGINES = "health_no_engines"

STREAM_CONTENT_TYPE = "stream_content_type_not_sse"
STREAM_NO_CONTENT = "stream_no_content_delta"
STREAM_NO_FINISH_REASON = "stream_no_terminal_finish_reason"
STREAM_NO_USAGE = "stream_no_terminal_usage"
STREAM_NO_DONE = "stream_no_done_sentinel"

ACCOUNTING_MALFORMED = "accounting_malformed"
ACCOUNTING_STALLED = "accounting_stalled"


@dataclass(frozen=True)
class PromotionVerdict:
    ok: bool
    reason: str = ""


PASS = PromotionVerdict(ok=True)


def _fail(reason: str) -> PromotionVerdict:
    return PromotionVerdict(ok=False, reason=reason)


@dataclass(frozen=True)
class HealthEvidence:
    ok: bool
    deployment_sha: str
    deployment_id: str
    gpus: int


def parse_health(payload: Any) -> HealthEvidence | None:
    """Read a `/healthz` body, or None when it is not one.

    Returning None rather than raising keeps the caller's failure path uniform: a transport error, a
    JSON parse error, and a 200 carrying an HTML error page all have to reach the same verdict.
    """
    if not isinstance(payload, dict):
        return None
    sha = payload.get("deployment_sha")
    deployment_id = payload.get("deployment_id")
    gpus = payload.get("gpus")
    if not isinstance(sha, str) or not isinstance(deployment_id, str):
        return None
    # bool is a subclass of int, so `isinstance(True, int)` is True: a body reporting `gpus: true`
    # would otherwise be read as one engine.
    if isinstance(gpus, bool) or not isinstance(gpus, int):
        return None
    return HealthEvidence(
        ok=payload.get("ok") is True,
        deployment_sha=sha,
        deployment_id=deployment_id,
        gpus=gpus,
    )


def verify_health(
    evidence: HealthEvidence | None,
    *,
    expected_sha: str,
    expected_deployment_id: str,
) -> PromotionVerdict:
    """The live router must be THIS release, and must have engines to route to.

    The sha comparison is full-length equality on a validated 40-hex string. A prefix or substring
    check would accept an unrelated release whose sha happens to share a prefix, which is exactly
    what an abbreviated sha copied from a log looks like.
    """
    if evidence is None:
        return _fail(HEALTH_MALFORMED)
    if not evidence.ok:
        return _fail(HEALTH_NOT_OK)
    reported = evidence.deployment_sha.lower()
    expected = expected_sha.lower()
    if _SHA_PATTERN.match(reported) is None or _SHA_PATTERN.match(expected) is None:
        return _fail(HEALTH_SHA_MISMATCH)
    if reported != expected:
        return _fail(HEALTH_SHA_MISMATCH)
    if evidence.deployment_id != expected_deployment_id:
        # a matching sha with a stale attempt id is a PREVIOUS deploy of the same commit, so the
        # readiness check would pass against a router this run never replaced.
        return _fail(HEALTH_DEPLOYMENT_ID_MISMATCH)
    if evidence.gpus < 1:
        return _fail(HEALTH_NO_ENGINES)
    return PASS


@dataclass(frozen=True)
class StreamEvidence:
    content_type_ok: bool
    content_delta_count: int
    finish_reason: str | None
    completion_tokens: int | None
    saw_done_sentinel: bool


def verify_stream(evidence: StreamEvidence) -> PromotionVerdict:
    """A real generation reached a real client over a real stream.

    Each clause rules out a distinct way a stream can look successful while proving nothing:
    a 200 that is not SSE at all, frames that carry no generated text, a stream cut off before its
    terminal chunk, a terminal chunk whose usage says zero tokens were produced, and a body that
    never reaches `[DONE]` because a proxy buffered it to EOF.
    """
    if not evidence.content_type_ok:
        return _fail(STREAM_CONTENT_TYPE)
    if evidence.content_delta_count < 1:
        return _fail(STREAM_NO_CONTENT)
    if not evidence.finish_reason:
        return _fail(STREAM_NO_FINISH_REASON)
    if evidence.completion_tokens is None or evidence.completion_tokens < 1:
        return _fail(STREAM_NO_USAGE)
    if not evidence.saw_done_sentinel:
        return _fail(STREAM_NO_DONE)
    return PASS


@dataclass(frozen=True)
class AccountingEvidence:
    """The backlog counters this release's outbox reports.

    This is a WHOLE-DEPLOYMENT view, not a per-request one. `serving_usage_backlog_snapshot` is the
    only durable accounting surface the router exposes, and it aggregates across every generation in
    flight. There is no RPC that reads back one row by correlation id, so the canary's own row is not
    individually observable from here.
    """

    # Only the two STALL signals are held. `pending` and `leased` were carried by the first
    # revision, which asserted a drained backlog; review narrowed the claim to what a
    # deployment-wide snapshot can actually support, and those two counters became unread. Keeping
    # them would be worse than useless: a field that is parsed and required but never consulted
    # cannot be verified by any test, so it silently exempts itself from the suite.
    #
    # `float` rather than `int` because postgres numerics arrive as JSON floats and `float` already
    # admits `int`. `oldest_undelivered_age_seconds` in particular is derived with
    # `EXTRACT(EPOCH FROM ...)`, so a fractional value is the NORMAL shape, not a malformed one.
    # Neither reading needs integrality: both are threshold comparisons.
    expired_leases: float
    oldest_undelivered_age_seconds: float | None

    @property
    def stalled(self) -> bool:
        """Delivery is not making progress, as opposed to merely having work in flight.

        A nonzero backlog is normal on a live deployment: concurrent traffic is always producing
        rows. What is never normal is an expired lease (a worker took a row and died holding it) or
        an undelivered row older than the stall threshold. Those are properties of the DELIVERY
        LOOP, so they hold regardless of how much unrelated traffic shares the counters.
        """
        if self.expired_leases > 0:
            return True
        age = self.oldest_undelivered_age_seconds
        return age is not None and age >= _STALL_AGE_SECONDS


def parse_accounting(payload: Any) -> AccountingEvidence | None:
    """Read a `serving_usage_backlog_snapshot` body, or None when it is not one.

    Parsing the RPC body directly -- rather than an `OutboxSnapshot` built from it -- is deliberate.
    `DurableUsageOutbox.snapshot` coerces every absent counter with `int(... or 0)`, which is right
    for a delivery worker deciding whether to wake up and wrong for a gate: zero is precisely the
    value that means "healthy", so a body that lost its fields to a rename, a schema drift, or a
    permission-shaped empty result would read as a perfectly drained backlog and PASS. Absence has
    to survive parsing to be rejected here.
    """
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    if not isinstance(payload, dict):
        return None
    # `states` is still required even though no field is read out of it. Its absence is the exact
    # shape a renamed schema or a permission-shaped empty result takes, and this parser's whole job
    # is to make that unreadable rather than healthy.
    if not isinstance(payload.get("states"), dict):
        return None
    expired_leases = _number(payload.get("expired_leases", _MISSING))
    if expired_leases is None:
        return None
    # None is a real reading here ("nothing undelivered"); an ABSENT field is not, because it would
    # silently read as healthy while carrying no stall signal at all.
    raw_age = payload.get("oldest_undelivered_age_seconds", _MISSING)
    age = None if raw_age is None else _number(raw_age)
    if raw_age is not None and age is None:
        return None
    return AccountingEvidence(expired_leases=expired_leases, oldest_undelivered_age_seconds=age)


def verify_accounting(evidence: AccountingEvidence | None) -> PromotionVerdict:
    """This release's durable delivery loop is running, not wedged.

    Deliberately NOT "the canary's own row settled": the snapshot cannot express that. Asserting a
    zero backlog would be both fail-open and flaky on a live deployment -- unrelated traffic can
    drain the counters to zero while the canary's own row is stuck, and can equally hold them
    nonzero while everything is healthy. Stall signals are the part of this snapshot that means the
    same thing no matter whose rows are in it.
    """
    if evidence is None:
        return _fail(ACCOUNTING_MALFORMED)
    if evidence.stalled:
        return _fail(ACCOUNTING_STALLED)
    return PASS
