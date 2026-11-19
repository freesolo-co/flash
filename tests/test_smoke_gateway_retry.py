"""Replay of the 2026-11-18 deployment smoke 504s.

Flash run flash-1795016904-01c3b6ad/step-175 kept the error of its failed deployment: the
fixed-prompt smoke started 18 s after the deploy request and serve.freesolo.co answered
`504 Gateway Timeout` 22 s later, so the deployment went to `failed`. tau-fable's step-116 and
step-174 deployments failed the same way 18 s and 152 s after their requests, and a redeploy of the
same checkpoint then became ready. Research callers saw only "did not become ready (state=failed)".
"""

import httpx
import pytest

import flash.server.routes.serving_smoke as smoke

URL = "https://serve.freesolo.co/v1/chat/completions"
CHECKPOINT = "flash-1795016904-01c3b6ad/step-175"


def gateway_error(status):
    response = httpx.Response(status, request=httpx.Request("POST", URL))
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise AssertionError("expected an error status")


class ServingEdge:
    """serve.freesolo.co as observed: some smoke calls end in a gateway error after ~22 s."""

    def __init__(self, replies, clock):
        self.replies = list(replies)
        self.clock = clock
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        self.clock.now += 22.0
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(smoke.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(smoke.time, "sleep", clock.sleep)
    # run the trusted call inline: the replay clock, not a thread join, bounds it.
    monkeypatch.setattr(smoke, "_bounded_call", lambda fn, *, deadline, budget_s: fn())
    return clock


def smoke_chat(edge, clock, monkeypatch, budget_s=smoke._SMOKE_BUDGET_SECONDS):
    monkeypatch.setattr(smoke.serving_deploy, "chat", edge.chat)
    return smoke._bounded_smoke_chat(
        serving_model=CHECKPOINT,
        thinking=False,
        expected_checkpoint=CHECKPOINT,
        org_id="org-1",
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        max_tokens=16,
        stop_sequences=None,
        deadline=clock.monotonic() + budget_s,
        budget_s=budget_s,
    )


ANSWER = {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}


@pytest.mark.parametrize("status", [504, 502])
def test_a_transient_gateway_error_is_retried_inside_the_smoke(clock, monkeypatch, status):
    edge = ServingEdge([gateway_error(status), ANSWER], clock)
    assert smoke_chat(edge, clock, monkeypatch) == ANSWER
    assert edge.calls == 2
    assert clock.sleeps == [smoke._SMOKE_GATEWAY_RETRY_DELAY_SECONDS]


def test_repeated_gateway_errors_within_the_retry_budget_pass(clock, monkeypatch):
    edge = ServingEdge([gateway_error(504)] * 3 + [ANSWER], clock)
    assert smoke_chat(edge, clock, monkeypatch) == ANSWER
    assert edge.calls == 4


def test_a_persistently_failing_gateway_still_fails_the_smoke_with_its_error(clock, monkeypatch):
    edge = ServingEdge([gateway_error(504)] * 10, clock)
    with pytest.raises(httpx.HTTPStatusError, match="504 Gateway Timeout"):
        smoke_chat(edge, clock, monkeypatch)
    assert edge.calls == 1 + smoke._SMOKE_GATEWAY_RETRIES


@pytest.mark.parametrize("status", [400, 404, 409, 500])
def test_other_errors_are_not_retried(clock, monkeypatch, status):
    edge = ServingEdge([gateway_error(status), ANSWER], clock)
    with pytest.raises(httpx.HTTPStatusError):
        smoke_chat(edge, clock, monkeypatch)
    assert edge.calls == 1


def test_a_retry_never_outlives_the_smoke_deadline(clock, monkeypatch):
    edge = ServingEdge([gateway_error(504), ANSWER], clock)
    # 22 s per call leaves 3 s, less than the 5 s retry delay.
    with pytest.raises(httpx.HTTPStatusError):
        smoke_chat(edge, clock, monkeypatch, budget_s=25.0)
    assert edge.calls == 1
    assert clock.sleeps == []
