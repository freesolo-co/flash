from __future__ import annotations

import logging
import time

import pytest

from flash.providers.artifacts.hf import hf_call


def test_hf_call_returns_real_success_after_deadline() -> None:
    mutations: list[str] = []

    def finish_late() -> str:
        time.sleep(0.03)
        mutations.append("done")
        return "uploaded"

    result = hf_call(
        finish_late,
        "upload",
        logger=logging.getLogger(__name__),
        deadline_at=time.time() + 0.01,
    )

    assert result == "uploaded"
    assert mutations == ["done"]


def test_hf_call_returns_real_failure_after_deadline_without_retry() -> None:
    attempts = 0

    class TransientError(RuntimeError):
        response = type("Response", (), {"status_code": 503, "headers": {}})()

    def fail_late() -> None:
        nonlocal attempts
        attempts += 1
        time.sleep(0.03)
        raise TransientError("provider unavailable")

    with pytest.raises(TransientError, match="provider unavailable"):
        hf_call(
            fail_late,
            "download",
            logger=logging.getLogger(__name__),
            deadline_at=time.time() + 0.01,
            retry_delays=(0.0,),
        )

    assert attempts == 1


def test_hf_call_starts_no_attempt_after_deadline() -> None:
    called = False

    def call() -> None:
        nonlocal called
        called = True

    with pytest.raises(TimeoutError, match="upload exceeded the run wall deadline"):
        hf_call(
            call,
            "upload",
            logger=logging.getLogger(__name__),
            deadline_at=time.time() - 1.0,
        )

    assert called is False
