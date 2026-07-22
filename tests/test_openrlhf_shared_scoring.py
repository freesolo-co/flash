"""cpu-only tests for shared OpenRLHF scoring futures."""

from __future__ import annotations

import threading

import pytest

from flash.engine.worker.openrlhf_shared_scoring import (
    ScoringBatchIdentity,
    ScoringCapacityError,
    ScoringIdentityError,
    ScoringKind,
    SharedScoringPool,
    bind_scoring_bridge,
)


def test_submit_returns_before_reward_scoring_finishes():
    started = threading.Event()
    release = threading.Event()
    training_thread = threading.get_ident()
    scoring_threads = []

    def score(payload):
        scoring_threads.append(threading.get_ident())
        started.set()
        assert release.wait(timeout=2)
        return {"rewards": [payload["reward"]]}

    with SharedScoringPool(pool_size=1) as pool:
        pool.register_run("run-a", kind=ScoringKind.REWARD, bridge=score)
        identity = ScoringBatchIdentity("run-a", 3, "batch-7")
        future = pool.submit(identity, {"reward": 1.25})

        assert started.wait(timeout=1)
        assert future.done() is False
        assert pool.outstanding_count == 1

        release.set()
        result = pool.consume(identity, future, timeout=1)

    assert len(scoring_threads) == 1
    assert scoring_threads[0] != training_thread
    assert result.identity == identity
    assert result.kind is ScoringKind.REWARD
    assert result.value == {"rewards": [1.25]}


def test_future_requires_the_exact_originating_run_step_and_batch():
    with SharedScoringPool(pool_size=1) as pool:
        pool.register_run(
            "run-a",
            kind=ScoringKind.REWARD,
            bridge=lambda payload: {"value": payload["value"]},
        )
        identity = ScoringBatchIdentity("run-a", 4, "batch-a")
        future = pool.submit(identity, {"value": 9})

        mismatches = (
            ScoringBatchIdentity("run-b", 4, "batch-a"),
            ScoringBatchIdentity("run-a", 5, "batch-a"),
            ScoringBatchIdentity("run-a", 4, "batch-b"),
        )
        for mismatch in mismatches:
            with pytest.raises(ScoringIdentityError, match="requested run"):
                future.result_for(mismatch, timeout=1)
            with pytest.raises(ScoringIdentityError, match="no outstanding"):
                pool.consume(mismatch, future, timeout=1)
        with pytest.raises(AttributeError):
            future.identity = mismatches[0]  # type: ignore[misc]

        result = pool.consume(identity, future, timeout=1)

    assert result.identity == identity
    assert result.value == {"value": 9}


def test_each_run_uses_only_its_own_bound_bridge_endpoint():
    calls = []
    calls_lock = threading.Lock()

    def post_request(url, payload):
        with calls_lock:
            calls.append((url, dict(payload)))
        return {"endpoint": url, "label": payload["label"]}

    with SharedScoringPool(pool_size=2) as pool:
        pool.register_run(
            "run-a",
            kind=ScoringKind.REWARD,
            bridge=bind_scoring_bridge("http://127.0.0.1:1/reward/key-a", post_request),
        )
        pool.register_run(
            "run-b",
            kind=ScoringKind.TEACHER,
            bridge=bind_scoring_bridge("http://127.0.0.1:2/teacher/key-b", post_request),
        )
        identity_a = ScoringBatchIdentity("run-a", 1, "a")
        identity_b = ScoringBatchIdentity("run-b", 8, "b")
        future_a = pool.submit(identity_a, {"label": "from-a"})
        future_b = pool.submit(identity_b, {"label": "from-b"})

        result_a = pool.consume(identity_a, future_a, timeout=1)
        result_b = pool.consume(identity_b, future_b, timeout=1)

    assert result_a.value == {
        "endpoint": "http://127.0.0.1:1/reward/key-a",
        "label": "from-a",
    }
    assert result_b.value == {
        "endpoint": "http://127.0.0.1:2/teacher/key-b",
        "label": "from-b",
    }
    assert sorted(calls, key=lambda item: item[0]) == [
        ("http://127.0.0.1:1/reward/key-a", {"label": "from-a"}),
        ("http://127.0.0.1:2/teacher/key-b", {"label": "from-b"}),
    ]


def test_bridge_retry_and_fail_closed_exceptions_are_preserved():
    class _TeacherBridgeFailure(RuntimeError):
        def __init__(self, classification):
            super().__init__(f"teacher {classification}")
            self.classification = classification

    class _RetryingRequest:
        def __init__(self):
            self.attempts = 0

        def __call__(self, _url, _payload):
            while True:
                self.attempts += 1
                try:
                    if self.attempts < 3:
                        raise _TeacherBridgeFailure("transient")
                    return {"teacher_logsums": [-0.4], "attempts": self.attempts}
                except _TeacherBridgeFailure as exc:
                    if exc.classification != "transient" or self.attempts >= 3:
                        raise

    retrying_request = _RetryingRequest()
    expected_failure = _TeacherBridgeFailure("permanent")

    def failing_request(_url, _payload):
        raise expected_failure

    with SharedScoringPool(pool_size=2) as pool:
        pool.register_run(
            "retry",
            kind=ScoringKind.TEACHER,
            bridge=bind_scoring_bridge("http://127.0.0.1:1/teacher/retry", retrying_request),
        )
        pool.register_run(
            "fail",
            kind=ScoringKind.TEACHER,
            bridge=bind_scoring_bridge("http://127.0.0.1:2/teacher/fail", failing_request),
        )
        retry_identity = ScoringBatchIdentity("retry", 2, "retry-batch")
        fail_identity = ScoringBatchIdentity("fail", 2, "fail-batch")
        retry_future = pool.submit(retry_identity, {"sequence_ids": [1, 2]})
        fail_future = pool.submit(fail_identity, {"sequence_ids": [3, 4]})

        retry_result = pool.consume(retry_identity, retry_future, timeout=1)
        with pytest.raises(_TeacherBridgeFailure) as error:
            pool.consume(fail_identity, fail_future, timeout=1)

    assert retrying_request.attempts == 3
    assert retry_result.value == {"teacher_logsums": [-0.4], "attempts": 3}
    assert error.value is expected_failure
    assert error.value.classification == "permanent"


def test_pool_rejects_submission_at_capacity_until_a_future_is_consumed():
    started = threading.Event()
    release = threading.Event()

    def blocking_bridge(payload):
        started.set()
        assert release.wait(timeout=2)
        return payload["value"]

    with SharedScoringPool(pool_size=1) as pool:
        pool.register_run("run-a", kind=ScoringKind.REWARD, bridge=blocking_bridge)
        first_identity = ScoringBatchIdentity("run-a", 0, "first")
        second_identity = ScoringBatchIdentity("run-a", 1, "second")
        first = pool.submit(first_identity, {"value": 1})
        assert started.wait(timeout=1)

        with pytest.raises(ScoringCapacityError, match="pool is full"):
            pool.submit(second_identity, {"value": 2})

        release.set()
        assert pool.consume(first_identity, first, timeout=1).value == 1
        second = pool.submit(second_identity, {"value": 2})
        assert pool.consume(second_identity, second, timeout=1).value == 2


def test_submit_snapshots_payload_before_worker_execution():
    first_started = threading.Event()
    release_first = threading.Event()

    def bridge(payload):
        first_started.set()
        assert release_first.wait(timeout=2)
        return payload

    payload = {"query": ["original"], "nested": {"label": 4}}
    with SharedScoringPool(pool_size=1) as pool:
        pool.register_run("run-a", kind=ScoringKind.REWARD, bridge=bridge)
        identity = ScoringBatchIdentity("run-a", 1, "snapshot")
        future = pool.submit(identity, payload)
        assert first_started.wait(timeout=1)
        payload["query"][0] = "mutated"
        payload["nested"]["label"] = 8
        release_first.set()
        result = pool.consume(identity, future, timeout=1)

    assert result.value == {"query": ["original"], "nested": {"label": 4}}


def test_cancelled_run_rejects_a_late_result():
    started = threading.Event()
    release = threading.Event()

    def bridge(payload):
        started.set()
        assert release.wait(timeout=2)
        return payload

    pool = SharedScoringPool(pool_size=1)
    pool.register_run("run-a", kind=ScoringKind.TEACHER, bridge=bridge)
    identity = ScoringBatchIdentity("run-a", 6, "late")
    future = pool.submit(identity, {"value": "late"})
    assert started.wait(timeout=1)
    assert pool.cancel_run("run-a") == 1
    release.set()

    try:
        with pytest.raises(ScoringIdentityError, match="no outstanding"):
            pool.consume(identity, future, timeout=1)
    finally:
        pool.shutdown()
