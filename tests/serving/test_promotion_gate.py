"""Gate ordering, bounded accounting, and what reaches the build log on failure."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from flash.serving.promotion import gate as gate_module
from flash.serving.promotion.canary import CANARY_TIMEOUT, CanaryError
from flash.serving.promotion.evidence import (
    ACCOUNTING_MALFORMED,
    ACCOUNTING_NOT_SETTLED,
    HEALTH_SHA_MISMATCH,
    STREAM_NO_CONTENT,
    StreamEvidence,
)
from flash.serving.promotion.gate import (
    GATE_CONFIG_INCOMPLETE,
    HEALTH_UNREACHABLE,
    evaluate_promotion,
)

SHA = "94210a323f9beaa713241e305f178b364848446d"
DEPLOYMENT_ID = "12345-1"


@dataclass
class _Snapshot:
    pending: int = 0
    leased: int = 0
    due_pending: int = 0
    expired_leases: int = 0


def _health(**overrides):
    body = {"ok": True, "deployment_sha": SHA, "deployment_id": DEPLOYMENT_ID, "gpus": 2}
    body.update(overrides)
    return body


def _good_stream() -> StreamEvidence:
    return StreamEvidence(
        content_type_ok=True,
        content_delta_count=2,
        finish_reason="stop",
        completion_tokens=5,
        saw_done_sentinel=True,
    )


def _evaluate(*, health, stream, accounting, **kwargs):
    calls: list[str] = []

    async def health_loader():
        calls.append("health")
        if isinstance(health, BaseException):
            raise health
        return health

    async def stream_runner():
        calls.append("stream")
        if isinstance(stream, BaseException):
            raise stream
        return stream

    async def accounting_loader():
        calls.append("accounting")
        if isinstance(accounting, BaseException):
            raise accounting
        if callable(accounting):
            return accounting()
        return accounting

    async def no_sleep(_seconds: float) -> None:
        return None

    verdict = asyncio.run(
        evaluate_promotion(
            health_loader=health_loader,
            stream_runner=stream_runner,
            accounting_loader=accounting_loader,
            expected_sha=SHA,
            expected_deployment_id=DEPLOYMENT_ID,
            sleep=no_sleep,
            **kwargs,
        )
    )
    return verdict, calls


def test_a_wrong_release_is_never_streamed_against():
    """Streaming against a router that is not this release would prove nothing about it.

    It would also bill a real generation and settle real usage under the WRONG release id, so the
    accounting evidence would be actively misleading rather than merely useless.
    """
    verdict, calls = _evaluate(
        health=_health(deployment_sha="0" * 40), stream=_good_stream(), accounting=_Snapshot()
    )
    assert verdict.reason == HEALTH_SHA_MISMATCH
    assert calls == ["health"]


def test_an_unreachable_router_fails_the_gate_instead_of_crashing_it():
    """A crashed step does not run the rollback that a failed step does."""
    verdict, calls = _evaluate(
        health=RuntimeError("connection refused"), stream=_good_stream(), accounting=_Snapshot()
    )
    assert verdict.reason == HEALTH_UNREACHABLE
    assert calls == ["health"]


def test_a_failed_stream_does_not_wait_on_an_accounting_backlog():
    """There is no usage row to wait for, so polling would only burn the deadline."""
    verdict, calls = _evaluate(
        health=_health(),
        stream=StreamEvidence(
            content_type_ok=True,
            content_delta_count=0,
            finish_reason="stop",
            completion_tokens=5,
            saw_done_sentinel=True,
        ),
        accounting=_Snapshot(),
    )
    assert verdict.reason == STREAM_NO_CONTENT
    assert calls == ["health", "stream"]


def test_a_canary_error_becomes_its_reason_code_not_an_exception():
    verdict, _ = _evaluate(
        health=_health(), stream=CanaryError(CANARY_TIMEOUT), accounting=_Snapshot()
    )
    assert verdict.reason == CANARY_TIMEOUT


def test_a_backlog_that_never_drains_fails_within_the_deadline():
    """Delivery is asynchronous, so the gate retries; it must still stop."""
    verdict, calls = _evaluate(
        health=_health(),
        stream=_good_stream(),
        accounting=_Snapshot(pending=1),
        accounting_deadline_seconds=10,
        poll_seconds=5,
    )
    assert verdict.reason == ACCOUNTING_NOT_SETTLED
    # bounded: three reads at t=0, 5, 10, then the deadline stops it.
    assert calls.count("accounting") == 3


def test_a_backlog_still_draining_on_the_first_read_is_given_time_to_settle():
    """Usage in flight immediately after a generation is normal, not a failure."""
    snapshots = iter([_Snapshot(leased=1), _Snapshot(pending=1), _Snapshot()])
    verdict, calls = _evaluate(
        health=_health(),
        stream=_good_stream(),
        accounting=lambda: next(snapshots),
        accounting_deadline_seconds=60,
        poll_seconds=5,
    )
    assert verdict.ok
    assert calls.count("accounting") == 3


def test_an_unreadable_snapshot_never_passes_as_settled():
    verdict, _ = _evaluate(
        health=_health(),
        stream=_good_stream(),
        accounting=RuntimeError("supabase_rpc_500"),
        accounting_deadline_seconds=0,
    )
    assert verdict.reason == ACCOUNTING_MALFORMED


def test_promotion_passes_only_when_all_three_layers_prove_themselves():
    verdict, calls = _evaluate(health=_health(), stream=_good_stream(), accounting=_Snapshot())
    assert verdict.ok
    assert verdict.reason == ""
    assert calls == ["health", "stream", "accounting"]


def test_incomplete_configuration_fails_closed_without_naming_the_value(monkeypatch, capsys):
    """A missing secret must not be diagnosed by echoing anything about it."""
    monkeypatch.setenv("SERVING_BASE_URL", "https://serve.freesolo.co")
    for name in gate_module._REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    assert gate_module.main([]) == 1
    captured = capsys.readouterr()
    assert GATE_CONFIG_INCOMPLETE in captured.err
    assert captured.out == ""


def test_a_missing_base_url_does_not_deploy_a_gate_against_nothing(monkeypatch, capsys):
    monkeypatch.delenv("SERVING_BASE_URL", raising=False)
    assert gate_module.main([]) == 1
    assert GATE_CONFIG_INCOMPLETE in capsys.readouterr().err


def test_the_entrypoint_is_importable_as_a_module(monkeypatch, capsys):
    """`python -m flash.serving.promotion.gate` must reach main().

    Defining the entrypoint below a `__main__` guard makes it dead for `python -m` while every
    in-process test still passes.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "flash.serving.promotion.gate"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(gate_module.__file__).split("/flash/")[0]},
    )
    assert result.returncode == 1
    assert GATE_CONFIG_INCOMPLETE in result.stderr
