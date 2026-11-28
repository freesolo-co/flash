"""Score batches share the plane-wide in-flight cap fairly and chunk one-token logprob reads."""

import threading
import time

import pytest
from fastapi import HTTPException

from flash.server.domain.ops import score_batches
from flash.server.platform import db

MODEL = "Qwen/Qwen3.5-9B-Base"
INTERNAL = {"id": 7, "auth_kind": "internal"}


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setattr(score_batches, "PROBE_RETRY_SECONDS", 0)
    monkeypatch.setattr(score_batches, "ITEM_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(score_batches, "SLOT_POLL_SECONDS", 0.01)


def _logprob_body(count, **extra):
    requests = [
        {
            "id": f"case-{i}",
            "messages": [{"role": "user", "content": f"q{i}"}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 5,
        }
        for i in range(count)
    ]
    return {"model": MODEL, "requests": requests, **extra}


def _drain(runner, timeout=20.0):
    deadline = time.monotonic() + timeout
    runner.tick()
    while time.monotonic() < deadline:
        for _, thread in list(runner.active.values()):
            thread.join(max(0.0, deadline - time.monotonic()))
        runner.tick()
        if not runner.active:
            return
    raise AssertionError("score batch did not finish")


def _results(batch_id):
    return score_batches.public(score_batches.get_row(batch_id), include_results=True)["results"]


def _ok(request):
    return {"choices": [{"message": {"content": request["messages"][0]["content"]}}]}


# fair share ------------------------------------------------------------------------------------


def _acquire_later(share, owner, weight=1):
    granted = threading.Event()

    def run():
        share.acquire(owner, weight)
        granted.set()

    threading.Thread(target=run, daemon=True).start()
    return granted


def test_a_batch_alone_may_use_the_whole_cap():
    share = score_batches.FairShare(4)
    for _ in range(4):
        share.acquire("big")
    assert share.held("big") == 4


def test_a_freed_slot_goes_to_the_waiting_batch_holding_the_fewest():
    share = score_batches.FairShare(4)
    for _ in range(4):
        share.acquire("big")
    big_more = _acquire_later(share, "big")
    time.sleep(0.05)  # big queued first: FIFO alone would serve it before small
    small = _acquire_later(share, "small")
    time.sleep(0.05)
    assert not small.is_set()
    assert not big_more.is_set()
    share.release("big", 1)
    assert small.wait(2)
    assert not big_more.is_set()
    share.release("big", 1)
    assert big_more.wait(2)
    assert share.held("small") == 1
    assert share.held("big") == 3


def test_shares_converge_to_cap_over_n_while_every_batch_is_backlogged():
    share = score_batches.FairShare(6)
    for _ in range(6):
        share.acquire("a")
    waiting = [_acquire_later(share, owner) for owner in ("b", "b", "b", "c", "c", "c")]
    time.sleep(0.05)
    for _ in range(4):
        share.release("a", 1)
    deadline = time.monotonic() + 2
    while sum(event.is_set() for event in waiting) < 4 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (share.held("a"), share.held("b"), share.held("c")) == (2, 2, 2)


def test_a_heavy_request_at_the_head_is_not_overtaken_by_lighter_ones():
    share = score_batches.FairShare(4)
    for _ in range(3):
        share.acquire("a")
    chunk = _acquire_later(share, "chunk", weight=4)
    time.sleep(0.05)
    light = _acquire_later(share, "light")
    time.sleep(0.05)
    # One slot is free, but the older chunk is ahead of the light request.
    assert not chunk.is_set()
    assert not light.is_set()
    share.release("a", 3)
    assert chunk.wait(2)
    assert not light.is_set()
    share.release("chunk", 4)
    assert light.wait(2)


def test_an_abandoned_wait_raises_and_gives_its_place_back():
    share = score_batches.FairShare(1)
    share.acquire("a")
    stop = threading.Event()
    errors = []

    def run():
        try:
            share.acquire("b", abandon=stop.is_set)
        except score_batches._Cancelled as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    stop.set()
    thread.join(2)
    assert errors
    share.release("a", 1)
    share.acquire("c")
    assert share.held("c") == 1


def test_a_huge_batch_does_not_starve_a_small_one_on_the_plane(monkeypatch):
    monkeypatch.setenv(score_batches.MAX_INFLIGHT_ENV, "2")
    finished = {}
    lock = threading.Lock()

    def send(row, key, request):
        time.sleep(0.005)
        with lock:
            finished.setdefault(row["batch_id"], []).append(time.monotonic())
        return _ok(request)

    body = {
        "model": MODEL,
        "requests": [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(200)],
        "concurrency": 64,
    }
    big = score_batches.submit(body, INTERNAL, "org-test", None)["batch_id"]
    small_body = {**body, "requests": body["requests"][:3], "concurrency": 4}
    runner = score_batches.ScoreBatchRunner(send=send)
    runner.tick()  # the huge batch is running and holds the whole cap
    time.sleep(0.05)
    small = score_batches.submit(small_body, INTERNAL, "org-test", None)["batch_id"]
    _drain(runner)
    small_done = max(finished[small])
    big_before = sum(1 for at in finished[big] if at <= small_done)
    # Under first-come-first-served the 3 small requests queue behind ~190 big ones.
    assert big_before < 100
    assert all("response" in r for r in _results(small) + _results(big))


# prefill mode ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_fields", "eligible"),
    [
        ({"max_tokens": 1, "logprobs": True}, True),
        ({"max_tokens": 1, "top_logprobs": 3}, True),
        ({"max_tokens": 1, "logprob_token_ids": [10, 11]}, True),
        ({"max_tokens": 1, "logprobs": True, "n": 1}, True),
        ({"max_tokens": 2, "logprobs": True}, False),
        ({"max_tokens": 1}, False),
        ({"max_tokens": 1, "logprobs": True, "n": 2}, False),
        ({"max_tokens": 1, "logprobs": True, "tools": [{"type": "function"}]}, False),
        ({"max_tokens": 1, "logprobs": True, "tool_choice": "none"}, False),
        ({"max_tokens": True, "logprobs": True}, False),
        ({"logprobs": True}, False),
    ],
)
def test_prefill_eligibility(request_fields, eligible):
    request = {"messages": [{"role": "user", "content": "x"}], **request_fields}
    assert score_batches.prefill_eligible(request) is eligible


def test_one_token_logprob_batches_are_sent_in_chunks_with_one_result_each():
    chunks = []

    def send_chunk(row, key, requests):
        chunks.append(len(requests))
        return [_ok(request) for request in requests]

    batch = score_batches.submit(_logprob_body(70), INTERNAL, "org-test", None)["batch_id"]
    runner = score_batches.ScoreBatchRunner(send=lambda row, key, request: _ok(request))
    runner.send_chunk = send_chunk
    _drain(runner)
    # The first request confirms capacity on its own, then 69 go as 32 + 32 + 5.
    assert chunks == [32, 32, 5]
    results = _results(batch)
    assert [r["id"] for r in results] == [f"case-{i}" for i in range(70)]
    assert [r["response"] for r in results] == [
        _ok({"messages": [{"content": f"q{i}"}]}) for i in range(70)
    ]


def test_a_batch_with_one_generating_request_keeps_the_per_request_path():
    body = _logprob_body(5)
    body["requests"][3]["max_tokens"] = 4
    batch = score_batches.submit(body, INTERNAL, "org-test", None)["batch_id"]
    runner = score_batches.ScoreBatchRunner(send=lambda row, key, request: _ok(request))
    runner.send_chunk = lambda *a: pytest.fail("must not chunk")
    _drain(runner)
    assert all("response" in r for r in _results(batch))


def test_chunk_items_keep_their_own_errors_and_retryable_ones_are_retried_together():
    chunks = []
    busy = {"left": 2}

    def send_chunk(row, key, requests):
        chunks.append([request["messages"][0]["content"] for request in requests])
        out = []
        for request in requests:
            content = request["messages"][0]["content"]
            if content == "q2":
                out.append(HTTPException(status_code=400, detail="bad request 2"))
            elif content in ("q3", "q4") and busy["left"]:
                busy["left"] -= 1
                out.append(HTTPException(status_code=503, detail="busy"))
            else:
                out.append(_ok(request))
        return out

    batch = score_batches.submit(_logprob_body(6), INTERNAL, "org-test", None)["batch_id"]
    runner = score_batches.ScoreBatchRunner(send=lambda row, key, request: _ok(request))
    runner.send_chunk = send_chunk
    _drain(runner)
    results = _results(batch)
    assert results[2] == {"id": "case-2", "error": "bad request 2", "status_code": 400}
    assert results[3]["response"] == _ok({"messages": [{"content": "q3"}]})
    assert results[4]["response"] == _ok({"messages": [{"content": "q4"}]})
    # q0 confirmed capacity alone; the two retryable failures went again as one chunk.
    assert chunks == [["q1", "q2", "q3", "q4", "q5"], ["q3", "q4"]]


def test_a_chunk_that_keeps_failing_records_each_requests_last_error():
    calls = []

    def send_chunk(row, key, requests):
        calls.append(len(requests))
        raise ConnectionError("serving connection reset")

    batch = score_batches.submit(_logprob_body(4), INTERNAL, "org-test", None)["batch_id"]
    runner = score_batches.ScoreBatchRunner(send=lambda row, key, request: _ok(request))
    runner.send_chunk = send_chunk
    _drain(runner)
    assert calls == [3] * score_batches.ITEM_ATTEMPTS
    results = _results(batch)
    assert "response" in results[0]
    assert all(r["error"] == "serving connection reset" for r in results[1:])


def test_a_chunk_past_the_deadline_is_unfinished_not_sent():
    batch = score_batches.submit(_logprob_body(3), INTERNAL, "org-test", None)["batch_id"]
    runner = score_batches.ScoreBatchRunner(send=lambda *a: pytest.fail("must not send"))
    runner.send_chunk = lambda *a: pytest.fail("must not send")
    work = score_batches._Batch(runner, runner._claim(time.time()))
    work.stop_at = time.time() - 1
    requests = [{"id": f"case-{i}", "request": {}} for i in range(3)]
    work._run_chunk(requests, [0, 1, 2])
    assert all(r.get("unfinished") for r in score_batches._results(batch).values())
