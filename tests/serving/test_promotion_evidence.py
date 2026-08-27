"""The promotion rules, against the payloads a broken release actually produces.

Every case here is a way a deployment can look healthy while being unable to serve. The readiness
poll in `deploy-modal.yml` passes all of them, which is why these rules exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from flash.serving.promotion.evidence import (
    ACCOUNTING_MALFORMED,
    ACCOUNTING_NOT_SETTLED,
    HEALTH_DEPLOYMENT_ID_MISMATCH,
    HEALTH_MALFORMED,
    HEALTH_NO_ENGINES,
    HEALTH_NOT_OK,
    HEALTH_SHA_MISMATCH,
    STREAM_CONTENT_TYPE,
    STREAM_NO_CONTENT,
    STREAM_NO_DONE,
    STREAM_NO_FINISH_REASON,
    STREAM_NO_USAGE,
    AccountingEvidence,
    StreamEvidence,
    accounting_from_snapshot,
    parse_health,
    verify_accounting,
    verify_health,
    verify_stream,
)

SHA = "94210a323f9beaa713241e305f178b364848446d"
DEPLOYMENT_ID = "12345-1"


def _health(**overrides):
    body = {
        "ok": True,
        "deployment_sha": SHA,
        "deployment_id": DEPLOYMENT_ID,
        "gpus": 2,
    }
    body.update(overrides)
    return body


def _check_health(payload):
    return verify_health(
        parse_health(payload), expected_sha=SHA, expected_deployment_id=DEPLOYMENT_ID
    )


def _stream(**overrides):
    fields = {
        "content_type_ok": True,
        "content_delta_count": 3,
        "finish_reason": "stop",
        "completion_tokens": 7,
        "saw_done_sentinel": True,
    }
    fields.update(overrides)
    return StreamEvidence(**fields)


def test_a_sha_differing_by_one_character_is_not_this_release():
    """Substring or prefix matching would accept a different release entirely.

    An abbreviated sha is a prefix of the full one, so a check that is not full-length equality
    passes for any release sharing those leading characters.
    """
    other = SHA[:-1] + ("0" if SHA[-1] != "0" else "1")
    assert _check_health(_health(deployment_sha=other)).reason == HEALTH_SHA_MISMATCH


def test_a_matching_sha_from_an_earlier_attempt_is_a_stale_router():
    """Redeploying the same commit produces the same sha under a new attempt id.

    Without the attempt check, a run whose deploy silently failed would read the PREVIOUS run's
    router, see its own sha, and promote a release it never actually deployed.
    """
    verdict = _check_health(_health(deployment_id="12345-0"))
    assert verdict.reason == HEALTH_DEPLOYMENT_ID_MISMATCH


def test_a_router_reporting_no_engines_cannot_serve():
    """`ok: true` describes the router process, not the fleet behind it."""
    assert _check_health(_health(gpus=0)).reason == HEALTH_NO_ENGINES


def test_an_unhealthy_router_fails_even_with_the_right_identity():
    assert _check_health(_health(ok=False)).reason == HEALTH_NOT_OK


def test_a_health_body_that_is_not_json_object_fails_rather_than_raising():
    """A 200 carrying an HTML error page, or a truncated body, must reach a verdict.

    Raising here would crash the gate step instead of failing it, and a crashed step cannot trigger
    the rollback that a failed one does.
    """
    for payload in (None, [], "ok", 7, {"ok": True}, _health(gpus="2"), _health(gpus=True)):
        assert _check_health(payload).reason == HEALTH_MALFORMED


def test_a_complete_health_body_for_this_exact_release_passes():
    assert _check_health(_health()).ok


def test_a_stream_that_emitted_no_content_proves_no_generation():
    """Frames arrived and the stream closed cleanly, but nothing was generated.

    This is what a broken engine looks like from the client side: a well-formed, empty stream.
    """
    assert verify_stream(_stream(content_delta_count=0)).reason == STREAM_NO_CONTENT


def test_a_stream_without_a_terminal_finish_reason_was_cut_off():
    for reason in (None, ""):
        assert verify_stream(_stream(finish_reason=reason)).reason == STREAM_NO_FINISH_REASON


def test_a_stream_that_never_reaches_done_was_buffered_or_truncated():
    """A proxy that buffers SSE to EOF delivers every frame except the ending."""
    assert verify_stream(_stream(saw_done_sentinel=False)).reason == STREAM_NO_DONE


def test_terminal_usage_reporting_no_completion_tokens_is_not_a_generation():
    for tokens in (0, None):
        assert verify_stream(_stream(completion_tokens=tokens)).reason == STREAM_NO_USAGE


def test_a_non_sse_response_is_rejected_before_its_frames_are_trusted():
    assert verify_stream(_stream(content_type_ok=False)).reason == STREAM_CONTENT_TYPE


def test_a_real_stream_passes():
    assert verify_stream(_stream()).ok


def test_each_backlog_counter_independently_blocks_promotion():
    """Any one non-zero counter means this release's usage did not settle.

    Checking only `pending` would miss a row stuck in a lease, which is the failure that loses
    billing on restart.
    """
    for field in ("pending", "leased", "due_pending", "expired_leases"):
        counters = {"pending": 0, "leased": 0, "due_pending": 0, "expired_leases": 0}
        counters[field] = 1
        verdict = verify_accounting(AccountingEvidence(**counters))
        assert verdict.reason == ACCOUNTING_NOT_SETTLED, field


def test_a_snapshot_missing_a_counter_is_unreadable_not_settled():
    """Defaulting an absent counter to zero would turn an unreadable snapshot into a pass."""

    @dataclass
    class Partial:
        pending: int = 0
        leased: int = 0
        due_pending: int = 0

    assert accounting_from_snapshot(Partial()) is None
    assert verify_accounting(accounting_from_snapshot(Partial())).reason == ACCOUNTING_MALFORMED


def test_a_fully_drained_snapshot_passes():
    @dataclass
    class Snapshot:
        pending: int = 0
        leased: int = 0
        due_pending: int = 0
        expired_leases: int = 0
        quarantined: int = 4

    # quarantined rows are a pre-existing backlog condition, not something this release caused.
    assert verify_accounting(accounting_from_snapshot(Snapshot())).ok
