"""OpenAI-surface reasoning split for thinking adapters.

The Qwen chat template opens the reasoning block in the PROMPT, so a thinking completion starts
already inside the block and its text is ``<reasoning></think><answer>`` -- a closing tag with no
opener. This router is a custom OpenAI-compatible front door (not vLLM's own OpenAI server), so
nothing split that for us: callers got a stray ``</think>`` glued to the answer and an empty
``reasoning_content``. These tests pin the split on both the streaming and non-streaming paths,
and pin that a NON-thinking completion is never torn apart on the strength of a quoted tag.
"""

from __future__ import annotations

import hashlib
import json

import orjson
import pytest
from fastapi.testclient import TestClient

from flash.serving.src.router import (
    AdapterRouter,
    _ReasoningStreamSplitter,
    _split_reasoning,
    build_serving_app,
)
from flash.serving.src.schemas import AdapterRecord

QWEN = "Qwen/Qwen3.5-0.8B"


async def _allow(_token: str, _adapter_id: str) -> None:
    return None


def _rec(run_id: str, *, thinking: bool = True) -> AdapterRecord:
    sha = hashlib.sha1(run_id.encode()).hexdigest()
    return AdapterRecord.model_validate(
        {
            "adapter_id": f"{run_id}@final.{sha}",
            "repo_id": f"org/{run_id}",
            "org_id": "org-1",
            "base_model": QWEN,
            "checkpoint": run_id,
            "status": "ready",
            "thinking": thinking,
            "metadata": {
                "record_type": "revision",
                "run_id": run_id,
                "checkpoint_step": None,
                "hf_revision": sha,
            },
        }
    )


def _alias(revision: AdapterRecord) -> AdapterRecord:
    run_id = str(revision.metadata["run_id"])
    return revision.model_copy(
        update={
            "adapter_id": run_id,
            "checkpoint": None,
            "metadata": {
                "record_type": "alias",
                "run_id": run_id,
                "alias_of": revision.adapter_id,
            },
        }
    )


class _Pool:
    """Engine pool that reports the rendered thinking mode, like the real engine does."""

    def __init__(self, text: str, *, thinking: bool, deltas: list[str] | None = None) -> None:
        self._text = text
        self._thinking = thinking
        self._deltas = deltas if deltas is not None else [text]

    async def generate(self, base_model, payload, record, *, expected_checkpoint=None):
        return {
            "ok": True,
            "text": self._text,
            "finish_reason": "stop",
            "thinking": self._thinking,
        }

    async def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
        yield {"type": "ready", "checkpoint": "", "thinking": self._thinking}
        for delta in self._deltas:
            yield {"type": "delta", "text": delta}
        yield {"type": "final", "finish_reason": "stop"}

    async def register(self, base_model, record) -> None:
        pass

    async def unregister(self, base_model, adapter_id, expected_generation=None) -> None:
        pass


class _NoReadyPool(_Pool):
    """Streams without a leading ``ready`` event, so the router takes the replay path."""

    async def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
        for delta in self._deltas:
            yield {"type": "delta", "text": delta}
        yield {"type": "final", "finish_reason": "stop"}


def _client(pool: _Pool, *, thinking: bool = True) -> TestClient:
    revision = _rec("qa", thinking=thinking)
    app = build_serving_app(
        pool, AdapterRouter([revision, _alias(revision)]), chat_authorizer=_allow
    )
    return TestClient(app, headers={"Authorization": "Bearer t"})


def _chat(client: TestClient) -> dict:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qa", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["choices"][0]["message"]


def _stream_deltas(client: TestClient) -> list[dict]:
    """Every non-empty ``delta`` object from an SSE chat stream, in order."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "qa", "messages": [{"role": "user", "content": "hi"}], "stream": True},
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
# _split_reasoning: the pure non-streaming split
# ------------------------------------------------------------------------------------------


def test_split_reasoning_separates_reasoning_from_answer():
    assert _split_reasoning("weighing it up</think>the answer", True) == (
        "weighing it up",
        "the answer",
    )


def test_split_reasoning_unclosed_block_is_all_reasoning():
    """Hitting max_tokens mid-reasoning leaves no answer; an empty content is the honest report."""
    assert _split_reasoning("still thinking", True) == ("still thinking", "")


def test_split_reasoning_leaves_non_thinking_text_verbatim():
    """A non-thinking completion that merely QUOTES the tag must not be torn in half, and gets no
    reasoning_content at all."""
    assert _split_reasoning("write </think> to close it", False) == (
        None,
        "write </think> to close it",
    )


def test_split_reasoning_splits_on_the_first_close_only():
    """A quoted tag inside the answer stays in the answer."""
    assert _split_reasoning("think</think>say </think> to close", True) == (
        "think",
        "say </think> to close",
    )


# ------------------------------------------------------------------------------------------
# _ReasoningStreamSplitter: the incremental split
# ------------------------------------------------------------------------------------------


def test_stream_splitter_routes_reasoning_then_content():
    splitter = _ReasoningStreamSplitter(True)
    assert splitter.feed("weighing") == ("weighing", "")
    assert splitter.feed(" it up</think>the ") == (" it up", "the ")
    assert splitter.feed("answer") == ("", "answer")
    assert splitter.flush() == ""


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5, 6, 7])
def test_stream_splitter_handles_a_tag_straddling_chunks(cut):
    """The closing tag arrives token by token and can land across any chunk boundary. No matter
    where it is cut, the tag itself must never leak into either side of the split."""
    splitter = _ReasoningStreamSplitter(True)
    reasoning = ""
    content = ""
    for chunk in ("think", "</think>"[:cut], "</think>"[cut:], "answer"):
        head, tail = splitter.feed(chunk)
        reasoning += head
        content += tail
    reasoning += splitter.flush()
    assert (reasoning, content) == ("think", "answer")


def test_stream_splitter_flushes_an_unclosed_block_as_reasoning():
    """The block never closed, so the held-back partial match is reasoning, not a lost fragment."""
    splitter = _ReasoningStreamSplitter(True)
    assert splitter.feed("thinking</thin") == ("thinking", "")
    assert splitter.flush() == "</thin"
    # flush drains: a second call must not replay the same text.
    assert splitter.flush() == ""


def test_stream_splitter_passes_non_thinking_text_straight_through():
    splitter = _ReasoningStreamSplitter(False)
    assert splitter.feed("write </think> to close") == ("", "write </think> to close")
    assert splitter.flush() == ""


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


def test_chat_stream_without_a_ready_event_does_not_split():
    """No ready event means the rendered mode is unknown. Reporting non-thinking keeps the answer
    intact rather than guessing from the record and risking a torn completion."""
    pool = _NoReadyPool("", thinking=True, deltas=["weighing</think>the answer"])
    deltas = _stream_deltas(_client(pool))
    assert all("reasoning_content" not in d for d in deltas)
    content = "".join(d["content"] for d in deltas if "content" in d)
    assert content == "weighing</think>the answer"
