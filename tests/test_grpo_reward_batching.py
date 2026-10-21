from __future__ import annotations

import concurrent.futures
import math
import sys
import threading
import time

import pytest

import flash.engine.worker.model.decoding as worker_decoding
import flash.engine.worker.train.rl.rollout.multi_turn as rl_multi_turn
import flash.engine.worker.train.rl.rollout.reward_module as rl_reward_module
import flash.engine.worker.train.rl.rollout.single_turn as rl_single_turn
from flash.engine.worker.train.entry import score_batcher
from flash.engine.worker.train.entry.score_batcher import ScoreBatcher
from flash.engine.worker.train.rl.child import multiturn as grpo_multiturn


@pytest.fixture
def _identity_graded(monkeypatch):
    monkeypatch.setattr(
        worker_decoding, "graded_text", lambda text, prompt_opened_thinking=False: text
    )
    monkeypatch.setattr(
        worker_decoding, "thinking_text", lambda text, prompt_opened_thinking=False: ""
    )
    monkeypatch.setattr(
        worker_decoding, "think_token_count", lambda text, tok, prompt_opened_thinking=False: 3
    )


class _ConditionContentionProbe(threading.Condition):
    """report the target thread's first failed nonblocking acquisition."""

    def __init__(self, contender_name):
        super().__init__()
        self._contender_name = contender_name
        self._probed = False
        self.contention_observed = threading.Event()

    def __enter__(self):
        if threading.current_thread().name == self._contender_name and not self._probed:
            self._probed = True
            if self.acquire(blocking=False):
                self.release()
                raise AssertionError("contender acquired _condition while consumer start was gated")
            self.contention_observed.set()
        return super().__enter__()


def _make_test_score_batcher(score_batch, thread_name):
    return ScoreBatcher(
        score_batch,
        max_batch_size=1,
        flush_wait_s=0.01,
        label="test",
        thread_name=thread_name,
    )


def _assert_batcher_empty(batcher):
    with batcher._condition:
        assert batcher._thread is None
        assert batcher._pending == []
        assert batcher._in_flight == []


@pytest.mark.parametrize("failure_mode", ["score_base_exception", "error_wrapper_exception"])
def test_score_batcher_abnormal_consumer_exit_settles_claimed_waiter(failure_mode):
    score_error = SystemExit("scorer exited")
    wrapper_error = RuntimeError("error wrapper failed")

    def score_batch(_requests):
        if failure_mode == "score_base_exception":
            raise score_error
        raise ValueError("scoring failed")

    def wrap_batch_error(_error):
        raise wrapper_error

    batcher = ScoreBatcher(
        score_batch,
        max_batch_size=1,
        flush_wait_s=0.01,
        label="test",
        thread_name="test-abnormal-exit-score-batcher",
        wrap_batch_error=wrap_batch_error if failure_mode == "error_wrapper_exception" else None,
    )
    waiter = score_batcher._Waiter("request", enqueued_at=0.0, label="test")
    batcher._pending.append(waiter)
    escaped = score_error if failure_mode == "score_base_exception" else wrapper_error

    with pytest.raises(type(escaped), match=str(escaped)) as raised:
        batcher._run()

    assert raised.value is escaped
    assert waiter.done.is_set(), "consumer exit stranded its claimed waiter"
    assert waiter.result is None
    assert isinstance(waiter.error, RuntimeError)
    assert str(waiter.error) == "test stopped"


def test_score_batcher_claim_is_atomic_against_close():
    consumer_name = "test-atomic-claim-score-batcher"
    closer_name = "test-atomic-claim-closer"
    slice_entered = threading.Event()
    release_slice = threading.Event()
    dispatched = threading.Event()
    outcomes = []

    class GatedPending(list):
        def __getitem__(self, key):
            if threading.current_thread().name == consumer_name and isinstance(key, slice):
                slice_entered.set()
                assert release_slice.wait(timeout=2.0)
            return super().__getitem__(key)

    def score_batch(requests):
        dispatched.set()
        return [f"scored:{request}" for request in requests]

    batcher = ScoreBatcher(
        score_batch,
        max_batch_size=1,
        flush_wait_s=0.01,
        label="test",
        thread_name=consumer_name,
        cancel_undispatched_on_close=True,
    )
    condition = _ConditionContentionProbe(closer_name)
    batcher._condition = condition
    batcher._pending = GatedPending()

    def submit():
        try:
            outcomes.append(("result", batcher.submit("request")))
        except Exception as error:
            outcomes.append(("error", str(error)))

    submitter = threading.Thread(target=submit, name="test-atomic-claim-submitter")
    closer = threading.Thread(target=lambda: batcher.close(1.0), name=closer_name)
    submitter.start()
    assert slice_entered.wait(timeout=2.0)
    closer.start()
    try:
        assert condition.contention_observed.wait(timeout=2.0)
    finally:
        release_slice.set()
    submitter.join(timeout=2.0)
    closer.join(timeout=2.0)
    batcher.close(0.1)

    assert not submitter.is_alive()
    assert not closer.is_alive()
    assert dispatched.is_set()
    assert outcomes == [("result", "scored:request")]


def test_waiter_complete_preserves_first_outcome():
    shutdown_error = RuntimeError("shut down")
    result_first = score_batcher._Waiter("request", enqueued_at=0.0, label="test")
    result_first.complete(result="provider result")
    result_first.complete(error=shutdown_error)

    assert result_first.done.is_set()
    assert result_first.result == "provider result"
    assert result_first.error is None

    error_first = score_batcher._Waiter("request", enqueued_at=0.0, label="test")
    error_first.complete(error=shutdown_error)
    error_first.complete(result="late provider result")

    assert error_first.done.is_set()
    assert error_first.result is None
    assert error_first.error is shutdown_error


def test_concurrent_single_turn_requests_are_batched_and_scattered_in_order(monkeypatch):
    calls = []
    live = 0
    peak = 0
    lock = threading.Lock()

    def score_batch(requests):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
            calls.append(list(requests))
        time.sleep(0.02)
        with lock:
            live -= 1
        return [float(index) + len(solution) for index, solution in requests]

    server, url = rl_multi_turn.start_reward_server(
        lambda index, solution: pytest.fail("the scalar scorer should not run"),
        example_count=8,
        score_batch=score_batch,
    )
    try:
        monkeypatch.setitem(sys.modules, "flash_grpo_multiturn", grpo_multiturn)
        namespace: dict = {}
        exec(
            compile(rl_reward_module.render_reward_module("TEST_URL"), "<reward>", "exec"),
            namespace,
        )
        namespace["_URL"] = url
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda i: namespace["compute_score"](
                        "env",
                        f"a{i}",
                        "unused",
                        extra_info={
                            "index": i,
                            "flash_rollout_identity": {
                                "optimizer_step": 1,
                                "sample_index": i,
                                "rollout_ordinal": 0,
                                "validate": False,
                            },
                        },
                    ),
                    range(8),
                )
            )

        assert results == [float(i) + 2 for i in range(8)]
        assert sum(len(call) for call in calls) == 8
        assert max(len(call) for call in calls) > 1, f"no batch formed: {calls}"
        assert peak == 1, f"the env saw {peak} concurrent top-level batch calls"
    finally:
        server.shutdown()


def test_score_batcher_wrong_length_fails_every_waiter_without_partial_scatter():
    batch_sizes = []

    def short_batch(requests):
        batch_sizes.append(len(requests))
        return [1.0] * max(0, len(requests) - 1)

    batcher = ScoreBatcher(
        short_batch,
        max_batch_size=8,
        flush_wait_s=0.1,
        label="test",
        thread_name="test-score-batcher",
    )
    outcomes = []
    lock = threading.Lock()

    def score(request):
        try:
            batcher.submit(request)
        except Exception as exc:
            outcome = str(exc)
        else:
            outcome = "scored"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=score, args=(i,), daemon=True) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    batcher.close(1.0)

    assert not [thread for thread in threads if thread.is_alive()]
    assert max(batch_sizes) > 1, f"no shared batch formed: {batch_sizes}"
    assert len(outcomes) == 4
    assert all("zip() argument" in outcome for outcome in outcomes)


def test_score_batcher_close_claims_undispatched_grpo_work_within_bound():
    class FlushWaitCondition(threading.Condition):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()

        def wait(self, timeout=None):
            if threading.current_thread().name == "test-grpo-final-batch" and timeout is not None:
                self.entered.set()
            return super().wait(timeout)

    scored = threading.Event()

    def score_batch(requests):
        scored.set()
        return [f"scored:{request}" for request in requests]

    batcher = ScoreBatcher(
        score_batch,
        max_batch_size=8,
        flush_wait_s=30.0,
        label="test",
        thread_name="test-grpo-final-batch",
    )
    condition = FlushWaitCondition()
    batcher._condition = condition
    outcomes = []

    def score():
        outcomes.append(batcher.submit("request"))

    waiter = threading.Thread(target=score)
    waiter.start()
    try:
        assert condition.entered.wait(timeout=2.0)
        batcher.close(1.0)
        waiter.join(timeout=2.0)
    finally:
        batcher.close(0.1)

    assert not waiter.is_alive()
    assert scored.is_set()
    assert outcomes == ["scored:request"]


def test_score_batcher_close_releases_an_inflight_waiter_after_the_join_bound():
    entered = threading.Event()
    release = threading.Event()

    def blocked_batch(requests):
        entered.set()
        release.wait(timeout=30)
        return [1.0 for _ in requests]

    batcher = ScoreBatcher(
        blocked_batch,
        max_batch_size=8,
        flush_wait_s=0.01,
        label="test",
        thread_name="test-blocked-score-batcher",
    )
    outcomes = []

    def score():
        try:
            batcher.submit("request")
        except Exception as exc:
            outcomes.append(str(exc))

    waiter = threading.Thread(target=score, daemon=True)
    waiter.start()
    assert entered.wait(timeout=10), "the scorer never entered the env call"
    batcher.close(0.01)
    waiter.join(timeout=2)
    release.set()
    if batcher._thread is not None:
        batcher._thread.join(timeout=2)

    assert not waiter.is_alive(), "shutdown left an in-flight waiter blocked"
    assert outcomes == ["test shut down"]


def test_score_batcher_start_failure_is_transactional_and_retry_is_idempotent(monkeypatch):
    consumer_name = "test-transactional-score-batcher"
    original_start = threading.Thread.start
    start_error = RuntimeError("consumer start failed")
    consumer_start_calls = 0
    scored = []

    def controlled_start(thread):
        nonlocal consumer_start_calls
        if thread.name != consumer_name:
            return original_start(thread)
        consumer_start_calls += 1
        if consumer_start_calls == 1:
            raise start_error
        return original_start(thread)

    def score_batch(requests):
        scored.append(list(requests))
        return [f"scored:{request}" for request in requests]

    monkeypatch.setattr(threading.Thread, "start", controlled_start)
    batcher = _make_test_score_batcher(score_batch, consumer_name)

    with pytest.raises(RuntimeError) as raised:
        batcher.start()
    assert raised.value is start_error
    _assert_batcher_empty(batcher)

    try:
        assert batcher.submit("request") == "scored:request"
        started_thread = batcher._thread
        batcher.start()
    finally:
        batcher.close(1.0)

    assert started_thread is not None
    assert consumer_start_calls == 2
    assert scored == [["request"]]


def test_score_batcher_racing_submitter_retries_after_start_failure(monkeypatch):
    consumer_name = "test-racing-score-batcher"
    contender_name = "second-submitter"
    original_start = threading.Thread.start
    start_error = RuntimeError("first consumer start failed")
    first_start_entered = threading.Event()
    release_first_start = threading.Event()
    consumer_start_calls = 0
    scored = []
    outcomes = {}

    def controlled_start(thread):
        nonlocal consumer_start_calls
        if thread.name != consumer_name:
            return original_start(thread)
        consumer_start_calls += 1
        if consumer_start_calls == 1:
            first_start_entered.set()
            assert release_first_start.wait(timeout=2.0)
            raise start_error
        return original_start(thread)

    def score_batch(requests):
        scored.append(list(requests))
        return [f"result:{request}" for request in requests]

    monkeypatch.setattr(threading.Thread, "start", controlled_start)
    batcher = _make_test_score_batcher(score_batch, consumer_name)
    condition = _ConditionContentionProbe(contender_name)
    batcher._condition = condition

    def submit(key, request):
        try:
            outcomes[key] = ("result", batcher.submit(request))
        except Exception as error:
            outcomes[key] = ("error", error)

    first = threading.Thread(target=submit, args=("first", "first"), name="first-submitter")
    second = threading.Thread(target=submit, args=("second", "second"), name=contender_name)
    first.start()
    assert first_start_entered.wait(timeout=2.0)
    second.start()
    try:
        assert condition.contention_observed.wait(timeout=2.0)
        assert outcomes == {}
        assert first.is_alive()
        assert second.is_alive()
    finally:
        release_first_start.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)
    batcher.close(1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert outcomes["first"] == ("error", start_error)
    assert outcomes["second"] == ("result", "result:second")
    assert consumer_start_calls == 2
    assert scored == [["second"]]
    with batcher._condition:
        assert batcher._pending == []
        assert batcher._in_flight == []


def test_score_batcher_close_returns_when_never_started_or_after_start_failure(monkeypatch):
    never_started = _make_test_score_batcher(
        lambda requests: requests, "test-never-started-score-batcher"
    )
    never_started.close(0.0)
    assert never_started._thread is None

    consumer_name = "test-failed-start-score-batcher"
    original_start = threading.Thread.start
    start_error = RuntimeError("consumer start failed")

    def fail_consumer_start(thread):
        if thread.name == consumer_name:
            raise start_error
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_consumer_start)
    failed_start = _make_test_score_batcher(lambda requests: requests, consumer_name)
    with pytest.raises(RuntimeError) as raised:
        failed_start.start()
    assert raised.value is start_error
    failed_start.close(0.0)
    assert failed_start._thread is None


def test_score_batcher_close_racing_start_failure_never_joins_unstarted_thread(monkeypatch):
    consumer_name = "test-gated-failure-score-batcher"
    contender_name = "gated-failure-closer"
    original_start = threading.Thread.start
    original_join = threading.Thread.join
    start_error = RuntimeError("gated consumer start failed")
    start_entered = threading.Event()
    release_start = threading.Event()
    joined_unstarted = []
    outcomes = {}

    def gated_start(thread):
        if thread.name != consumer_name:
            return original_start(thread)
        start_entered.set()
        assert release_start.wait(timeout=2.0)
        raise start_error

    def guarded_join(thread, timeout=None):
        if thread.name == consumer_name:
            joined_unstarted.append(thread)
            raise AssertionError("close joined an unstarted consumer")
        return original_join(thread, timeout)

    monkeypatch.setattr(threading.Thread, "start", gated_start)
    monkeypatch.setattr(threading.Thread, "join", guarded_join)
    batcher = _make_test_score_batcher(lambda requests: requests, consumer_name)
    condition = _ConditionContentionProbe(contender_name)
    batcher._condition = condition

    def submit():
        try:
            batcher.submit("request")
        except Exception as error:
            outcomes["submit"] = error

    def close():
        try:
            batcher.close(0.0)
        except Exception as error:
            outcomes["close"] = error

    submitter = threading.Thread(target=submit, name="gated-failure-submitter")
    closer = threading.Thread(target=close, name=contender_name)
    submitter.start()
    assert start_entered.wait(timeout=2.0)
    closer.start()
    try:
        assert condition.contention_observed.wait(timeout=2.0)
        assert outcomes == {}
        assert submitter.is_alive()
        assert closer.is_alive()
    finally:
        release_start.set()
    submitter.join(timeout=2.0)
    closer.join(timeout=2.0)

    assert not submitter.is_alive()
    assert not closer.is_alive()
    assert outcomes == {"submit": start_error}
    assert joined_unstarted == []
    _assert_batcher_empty(batcher)


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_batch_preserves_breakdowns_penalties_and_nonfinite_masking():
    class _BatchEnv:
        def __init__(self):
            self.calls = 0

        def scores_breakdown_many(self, items):
            self.calls += 1
            return [
                {"quality": 0.5, "total": 1.0},
                {"quality": 0.25, "total": float("nan")},
            ]

    env = _BatchEnv()
    results = rl_single_turn.score_single_turn_batch(
        env,
        [("good", {"gt": "good"}), ("bad", {"gt": "bad"})],
        tok=object(),
        thinking=True,
        prompt_opened_thinking=True,
        think_penalty=0.1,
    )

    assert env.calls == 1
    assert results[0][0] == pytest.approx(0.7)
    assert results[0][1] == [{"quality": 0.5, "total": 1.0}]
    assert results[1][0] == 0.0
    assert results[1][1][0]["quality"] == 0.25
    assert math.isnan(results[1][1][0]["total"])


@pytest.mark.usefixtures("_identity_graded")
def test_score_single_turn_batch_falls_back_per_item_without_failing_neighbors():
    class _BatchThenScalarEnv:
        def __init__(self):
            self.batch_calls = 0
            self.scalar_calls = []

        def scores_breakdown_many(self, items):
            self.batch_calls += 1
            raise RuntimeError("batch judge unavailable")

        def scores_breakdown(self, graded, ex, state):
            self.scalar_calls.append(graded)
            if graded == "bad":
                raise ValueError("one malformed completion")
            return {"success": 1.0, "total": 1.0}

    env = _BatchThenScalarEnv()
    results = rl_single_turn.score_single_turn_batch(
        env,
        [("good-a", {}), ("bad", {}), ("good-b", {})],
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )

    assert env.batch_calls == 1
    assert env.scalar_calls == ["good-a", "bad", "good-b"]
    assert [score for score, _ in results] == [1.0, 0.0, 1.0]
    assert [breakdowns for _, breakdowns in results] == [
        [{"success": 1.0, "total": 1.0}],
        [None],
        [{"success": 1.0, "total": 1.0}],
    ]


def test_score_single_turn_batch_isolates_a_preprocessing_failure(monkeypatch):
    def graded_text(text, prompt_opened_thinking=False):
        if text == "bad":
            raise ValueError("malformed reasoning tags")
        return text

    monkeypatch.setattr(worker_decoding, "graded_text", graded_text)
    monkeypatch.setattr(
        worker_decoding, "thinking_text", lambda text, prompt_opened_thinking=False: ""
    )

    class _Env:
        def scores_breakdown_many(self, items):
            pytest.fail("failed preprocessing must fall back before batch scoring")

        def scores_breakdown(self, graded, ex, state):
            return {"success": 1.0, "total": 1.0}

    results = rl_single_turn.score_single_turn_batch(
        _Env(),
        [("good-a", {}), ("bad", {}), ("good-b", {})],
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )

    assert [score for score, _ in results] == [1.0, 0.0, 1.0]
    assert [breakdowns for _, breakdowns in results] == [
        [{"success": 1.0, "total": 1.0}],
        [None],
        [{"success": 1.0, "total": 1.0}],
    ]


def test_score_single_turn_batch_repairs_only_the_unusable_rows():
    """A batch that RETURNED has already spent its env calls; only bad rows may be re-scored.

    Re-running the whole batch to recover one malformed row double-charges a paid or
    side-effecting grader for every completion that came back fine.
    """

    class _PartiallyBadBatchEnv:
        def __init__(self):
            self.scalar_calls = []

        def scores_breakdown_many(self, items):
            # row 1 is unusable (no parseable `total`); the neighbours are well-formed.
            return [
                {"success": 1.0, "total": 1.0},
                {"success": 0.0, "total": "not-a-number"},
                {"success": 1.0, "total": 3.0},
            ]

        def scores_breakdown(self, graded, ex, state):
            self.scalar_calls.append(graded)
            return {"success": 1.0, "total": 9.0}

    env = _PartiallyBadBatchEnv()
    results = rl_single_turn.score_single_turn_batch(
        env,
        [("good-a", {}), ("bad", {}), ("good-b", {})],
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )

    # only the malformed row is re-scored: the good rows keep their batch values.
    assert env.scalar_calls == ["bad"]
    assert [score for score, _ in results] == [1.0, 9.0, 3.0]


def test_score_single_turn_batch_repairs_every_row_on_a_wrong_length_return():
    """A wrong-length payload cannot be trusted to line up, so no row keeps its batch value."""

    class _ShortBatchEnv:
        def __init__(self):
            self.scalar_calls = []

        def scores_breakdown_many(self, items):
            return [{"success": 1.0, "total": 1.0}]

        def scores_breakdown(self, graded, ex, state):
            self.scalar_calls.append(graded)
            return {"success": 1.0, "total": 5.0}

    env = _ShortBatchEnv()
    results = rl_single_turn.score_single_turn_batch(
        env,
        [("good-a", {}), ("good-b", {}), ("good-c", {})],
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )

    assert env.scalar_calls == ["good-a", "good-b", "good-c"]
    assert [score for score, _ in results] == [5.0, 5.0, 5.0]
