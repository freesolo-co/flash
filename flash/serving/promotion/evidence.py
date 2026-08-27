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
ACCOUNTING_NOT_SETTLED = "accounting_not_settled"


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
    pending: int
    leased: int
    due_pending: int
    expired_leases: int

    @property
    def settled(self) -> bool:
        return (
            self.pending == 0
            and self.leased == 0
            and self.due_pending == 0
            and self.expired_leases == 0
        )


def accounting_from_snapshot(snapshot: Any) -> AccountingEvidence | None:
    """Read the four backlog counters off an `OutboxSnapshot`, or None when it is not one.

    Missing counters must NOT default to zero. Zero is precisely the value that means "settled", so
    a defaulted read would turn an unreadable snapshot into a pass.
    """
    values: list[int] = []
    for field in ("pending", "leased", "due_pending", "expired_leases"):
        value = getattr(snapshot, field, None)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        values.append(value)
    return AccountingEvidence(*values)


def verify_accounting(evidence: AccountingEvidence | None) -> PromotionVerdict:
    """The canary's own usage settled durably, rather than only being captured in memory.

    A captured-but-undelivered event still bills nothing and still loses the request on restart, so
    an in-memory capture is not evidence that accounting works on this release.
    """
    if evidence is None:
        return _fail(ACCOUNTING_MALFORMED)
    if not evidence.settled:
        return _fail(ACCOUNTING_NOT_SETTLED)
    return PASS
