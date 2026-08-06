from __future__ import annotations

import concurrent.futures
import math
import threading
import time

import pytest

from flash.engine.worker import rl_train
from flash.engine.worker._pkg import W


@pytest.fixture
def _identity_graded(monkeypatch):
    monkeypatch.setattr(W, "graded_text", lambda text, prompt_opened_thinking=False: text)
    monkeypatch.setattr(W, "thinking_text", lambda text, prompt_opened_thinking=False: "")
    monkeypatch.setattr(W, "think_token_count", lambda text, tok, prompt_opened_thinking=False: 3)


def test_concurrent_single_turn_requests_are_batched_and_scattered_in_order():
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

    server, url = rl_train.start_reward_server(
        lambda index, solution: pytest.fail("the scalar scorer should not run"),
        example_count=8,
        score_batch=score_batch,
    )
    try:
        namespace: dict = {}
        exec(
            compile(rl_train.render_reward_module("TEST_URL"), "<reward>", "exec"),
            namespace,
        )
        namespace["_URL"] = url
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda i: namespace["compute_score"](
                        "env", f"a{i}", "unused", extra_info={"index": i}
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

    batcher = rl_train._ScoreBatcher(
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
            batcher.score(request)
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


def test_score_batcher_close_releases_an_inflight_waiter_after_the_join_bound():
    entered = threading.Event()
    release = threading.Event()

    def blocked_batch(requests):
        entered.set()
        release.wait(timeout=30)
        return [1.0 for _ in requests]

    batcher = rl_train._ScoreBatcher(
        blocked_batch,
        max_batch_size=8,
        flush_wait_s=0.01,
        label="test",
        thread_name="test-blocked-score-batcher",
    )
    outcomes = []

    def score():
        try:
            batcher.score("request")
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
    assert outcomes == ["test score batcher shut down"]


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
    results = rl_train.score_single_turn_batch(
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
    results = rl_train.score_single_turn_batch(
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

    monkeypatch.setattr(W, "graded_text", graded_text)
    monkeypatch.setattr(W, "thinking_text", lambda text, prompt_opened_thinking=False: "")

    class _Env:
        def scores_breakdown_many(self, items):
            pytest.fail("failed preprocessing must fall back before batch scoring")

        def scores_breakdown(self, graded, ex, state):
            return {"success": 1.0, "total": 1.0}

    results = rl_train.score_single_turn_batch(
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
    results = rl_train.score_single_turn_batch(
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
    results = rl_train.score_single_turn_batch(
        env,
        [("good-a", {}), ("good-b", {}), ("good-c", {})],
        tok=None,
        thinking=False,
        prompt_opened_thinking=False,
        think_penalty=0.0,
    )

    assert env.scalar_calls == ["good-a", "good-b", "good-c"]
    assert [score for score, _ in results] == [5.0, 5.0, 5.0]
