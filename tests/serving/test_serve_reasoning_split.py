"""OpenAI-surface reasoning split for thinking adapters.

The Qwen chat template opens the reasoning block in the PROMPT, so a thinking completion starts
already inside the block and its text is ``<reasoning></think><answer>`` -- a closing tag with no
opener. This router is a custom OpenAI-compatible front door (not vLLM's own OpenAI server), so
nothing split that for us: callers got a stray ``</think>`` glued to the answer and an empty
``reasoning_content``. These tests pin the split on both the streaming and non-streaming paths,
and pin that a NON-thinking completion is never torn apart on the strength of a quoted tag.
"""

from __future__ import annotations

import json

import orjson
import pytest
from fastapi.testclient import TestClient

from flash.serve.app.openai import ReasoningDeltaSplitter, split_reasoning
from flash.serving.src.accounting.usage import AuthorizedTraffic, principal_for_external_org
from flash.serving.src.http.router import AdapterRouter
from flash.serving.src.http.router import build_offline_serving_app as build_serving_app
from flash.serving.src.io.schemas import AdapterRecord
from tests.serving.checkpoint_fixtures import checkpoint_record
from tests.serving.conftest import attest

QWEN = "Qwen/Qwen3.5-9B"


async def _allow(_token: str, _adapter_id: str, _scope: dict | None = None) -> AuthorizedTraffic:
    return AuthorizedTraffic(principal_for_external_org("org-1"), "generation-a")


def _rec(run_id: str, *, thinking: bool = True) -> AdapterRecord:
    return checkpoint_record(run_id, QWEN, thinking=thinking)


class _NoReadyPool:
    """engine pool whose first event is an attested content delta."""

    def __init__(self) -> None:
        self.first_event = None

    async def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
        self.first_event = attest(
            record,
            {
                "type": "delta",
                "text": "write </think> verbatim",
                "checkpoint": record.model_id,
            },
        )
        yield self.first_event
        yield {"type": "final", "finish_reason": "stop"}

    async def generate(self, base_model, payload, record, *, expected_checkpoint=None):
        raise AssertionError("stream test must not use buffered generation")

    async def register(self, base_model, record) -> None:
        pass

    async def unregister(self, base_model, org_id, model_id, expected_generation=None) -> None:
        pass


class _Pool:
    """Engine pool that reports the rendered thinking mode, like the real engine does."""

    def __init__(self, text: str, *, thinking: bool, deltas: list[str] | None = None) -> None:
        self._text = text
        self._thinking = thinking
        self._deltas = deltas if deltas is not None else [text]

    async def generate(self, base_model, payload, record, *, expected_checkpoint=None):
        return attest(
            record,
            {
                "ok": True,
                "text": self._text,
                "finish_reason": "stop",
                "thinking": self._thinking,
                "checkpoint": record.model_id,
            },
        )

    async def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
        yield attest(
            record,
            {"type": "ready", "checkpoint": record.model_id, "thinking": self._thinking},
        )
        for delta in self._deltas:
            yield {"type": "delta", "text": delta}
        yield {"type": "final", "finish_reason": "stop"}

    async def register(self, base_model, record) -> None:
        pass

    async def unregister(self, base_model, org_id, model_id, expected_generation=None) -> None:
        pass


def _client(pool: _Pool, *, thinking: bool = True) -> TestClient:
    revision = _rec("qa", thinking=thinking)
    app = build_serving_app(pool, AdapterRouter([revision]), chat_authorizer=_allow)
    return TestClient(
        app,
        headers={"Authorization": "Bearer t", "X-Freesolo-Org-Id": "org-1"},
    )


def _chat(client: TestClient) -> dict:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qa/final", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["choices"][0]["message"]


def _stream_deltas(client: TestClient) -> list[dict]:
    """Every non-empty ``delta`` object from an SSE chat stream, in order."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "qa/final", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        body = resp.read().decode("utf-8")
    deltas = []
    for line in body.split("\n"):
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            continue
        delta = json.loads(payload)["choices"][0]["delta"]
        if delta:
            deltas.append(delta)
    return deltas


# ------------------------------------------------------------------------------------------
# split_reasoning: the pure non-streaming split
# ------------------------------------------------------------------------------------------


def test_split_reasoning_separates_reasoning_from_answer():
    assert split_reasoning("weighing it up</think>the answer", thinking=True) == (
        "weighing it up",
        "the answer",
    )


def test_split_reasoning_unclosed_block_is_all_reasoning():
    """Hitting max_tokens mid-reasoning leaves no answer; an empty content is the honest report."""
    assert split_reasoning("still thinking", thinking=True) == ("still thinking", "")


def test_split_reasoning_leaves_non_thinking_text_verbatim():
    """A non-thinking completion that merely QUOTES the tag must not be torn in half, and gets no
    reasoning_content at all."""
    assert split_reasoning("write </think> to close it", thinking=False) == (
        None,
        "write </think> to close it",
    )


def test_split_reasoning_splits_on_the_first_close_only():
    """A quoted tag inside the answer stays in the answer."""
    assert split_reasoning("think</think>say </think> to close", thinking=True) == (
        "think",
        "say </think> to close",
    )


# ------------------------------------------------------------------------------------------
# ReasoningDeltaSplitter: the incremental split
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize(
    "text",
    [
        "weighing it up</think>the answer",
        "<think>weighing it up</think>the answer",
        "thinking</thin",
        "think</think>say </think> to close",
    ],
)
def test_shared_stream_splitter_matches_complete_response_at_every_boundary(thinking, text):
    expected_reasoning, expected_content = split_reasoning(text, thinking=thinking)
    for cut in range(len(text) + 1):
        splitter = ReasoningDeltaSplitter(thinking=thinking)
        events = splitter.feed(text[:cut]) + splitter.feed(text[cut:]) + splitter.finish()
        reasoning = "".join(value for kind, value in events if kind == "reasoning_content")
        content = "".join(value for kind, value in events if kind == "content")
        assert (reasoning, content) == (expected_reasoning or "", expected_content)
        assert splitter.finish() == []


# ------------------------------------------------------------------------------------------
# end to end: /v1/chat/completions
# ------------------------------------------------------------------------------------------


def test_chat_completion_splits_reasoning_out_of_content():
    message = _chat(_client(_Pool("weighing it up</think>the answer", thinking=True)))
    assert message["reasoning_content"] == "weighing it up"
    assert message["content"] == "the answer"
    assert "</think>" not in message["content"]


def test_chat_completion_without_thinking_has_no_reasoning_content():
    message = _chat(_client(_Pool("plain answer", thinking=False), thinking=False))
    assert message == {"role": "assistant", "content": "plain answer"}


def test_chat_stream_splits_reasoning_deltas_from_content_deltas():
    client = _client(_Pool("", thinking=True, deltas=["weigh", "ing</thi", "nk>the ", "answer"]))
    deltas = _stream_deltas(client)
    assert deltas[0] == {"role": "assistant"}
    reasoning = "".join(d["reasoning_content"] for d in deltas if "reasoning_content" in d)
    content = "".join(d["content"] for d in deltas if "content" in d)
    assert (reasoning, content) == ("weighing", "the answer")
    # the tag itself is never emitted on either channel.
    assert all("</think>" not in orjson.dumps(d).decode() for d in deltas)


def test_chat_stream_without_thinking_emits_only_content_deltas():
    client = _client(
        _Pool("", thinking=False, deltas=["write </thi", "nk> to close"]), thinking=False
    )
    deltas = _stream_deltas(client)
    assert all("reasoning_content" not in d for d in deltas)
    content = "".join(d["content"] for d in deltas if "content" in d)
    assert content == "write </think> to close"


def test_chat_stream_accepts_attested_delta_without_ready_event() -> None:
    pool = _NoReadyPool()
    client = _client(pool, thinking=False)

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "qa/final", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["X-Freesolo-Checkpoint"] == "qa/final"
        assert response.headers["X-Freesolo-LoRA-Request-Adapter"] == "qa/final"
        body = response.read().decode("utf-8")

    assert pool.first_event["checkpoint"] == "qa/final"
    assert pool.first_event["lora_request_adapter"] == "qa/final"
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    deltas = [payload["choices"][0]["delta"] for payload in payloads]
    assert all("reasoning_content" not in delta for delta in deltas)
    assert "".join(delta.get("content", "") for delta in deltas) == "write </think> verbatim"
