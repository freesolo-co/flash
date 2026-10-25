"""Local reservation parity and startup readiness must preserve native contracts."""

from types import SimpleNamespace

import pytest

from flash.serving.deepinfer.app import token_budget as budget
from flash.serving.deepinfer.contract.config import MODEL_ID
from flash.serving.deepinfer.contract.errors import DeepInferError
from flash.serving.deepinfer.contract.protocol import parse_request
from flash.serving.deepinfer.contract.tokenizer import encode_request
from flash.serving.deepinfer.runtime.warmup import warmup
from tests.serving.test_deepinfer import body


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("thinking", [False, True])
async def test_local_admission_uses_native_encoding_and_exact_output_reservation(
    monkeypatch, thinking
):
    seen = []

    def encode(messages, *, thinking_mode):
        seen.append((messages, thinking_mode))
        return "encoded prompt"

    class Tokenizer:
        def encode(self, prompt, *, add_special_tokens):
            assert prompt == "encoded prompt"
            assert not add_special_tokens
            return SimpleNamespace(ids=[3, 4, 5])

    monkeypatch.setattr(budget, "prepare_tokenizer", lambda: (Tokenizer(), encode))
    data = body(max_tokens=77, chat_template_kwargs={"enable_thinking": thinking})
    parsed = parse_request(data)
    payload = SimpleNamespace(model_dump=lambda: {**data, "model_id": MODEL_ID})
    assert await budget.token_budget(payload) == 80
    assert encode_request(Tokenizer(), encode, parsed) == [3, 4, 5]
    assert all(mode == ("thinking" if thinking else "chat") for _, mode in seen)


@pytest.mark.anyio
async def test_local_admission_rejects_over_context_before_gpu_dispatch(monkeypatch):
    tokenizer = SimpleNamespace(encode=lambda *a, **k: SimpleNamespace(ids=[1] * 32700))
    monkeypatch.setattr(budget, "prepare_tokenizer", lambda: (tokenizer, lambda *a, **k: "prompt"))
    payload = SimpleNamespace(model_dump=lambda: {**body(max_tokens=64), "model_id": MODEL_ID})
    with pytest.raises(DeepInferError, match="context") as exc:
        await budget.token_budget(payload)
    assert exc.value.status_code == 422


@pytest.mark.anyio
async def test_startup_warmup_drains_and_closes_native_requests():
    calls, closed = [], []

    class Runtime:
        async def prepare_tokens(self, request):
            return [2] * 8192

        async def prefix_length(self, request, tokens):
            return 198

        async def generate(self, request, tokens):
            calls.append(request)
            try:
                yield {"type": "tokens", "tokens": [3], "finish_reason": "length"}
            finally:
                closed.append(request)

    await warmup(Runtime())
    assert len(calls) == len(closed) == 585
    assert all(r.prefix_tokens == 198 and r.cache_namespace == "0" * 64 for r in calls)


@pytest.mark.anyio
async def test_startup_warmup_fails_readiness_on_native_error():
    class Runtime:
        async def prepare_tokens(self, request):
            return [2] * 8192

        async def prefix_length(self, request, tokens):
            return 0

        async def generate(self, request, tokens):
            yield {"type": "error"}

    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await warmup(Runtime())
