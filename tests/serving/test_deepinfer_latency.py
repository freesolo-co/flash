"""Local reservation parity and startup readiness must preserve native contracts."""

from types import SimpleNamespace

import pytest

from flash.serving.deepinfer.app import token_budget as budget
from flash.serving.deepinfer.contract.config import MAX_MODEL_LEN, MODEL_ID
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
        def encode_batch(self, prompts, *, add_special_tokens):
            assert prompts == ["encoded prompt"]
            assert not add_special_tokens
            return [SimpleNamespace(ids=[3, 4, 5])]

    monkeypatch.setattr(budget, "prepare_tokenizer", lambda: (Tokenizer(), encode, None))
    data = body(max_tokens=77, chat_template_kwargs={"enable_thinking": thinking})
    parsed = parse_request(data)
    payload = SimpleNamespace(model_dump=lambda: {**data, "model_id": MODEL_ID})
    assert await budget.token_budget(payload) == 80
    assert encode_request(Tokenizer(), encode, parsed) == [3, 4, 5]
    assert all(mode == ("thinking" if thinking else "chat") for _, mode in seen)


@pytest.mark.anyio
async def test_local_admission_rejects_over_context_before_gpu_dispatch(monkeypatch):
    tokenizer = SimpleNamespace(
        encode_batch=lambda *a, **k: [SimpleNamespace(ids=[1] * (MAX_MODEL_LEN - 68))]
    )
    monkeypatch.setattr(
        budget, "prepare_tokenizer", lambda: (tokenizer, lambda *a, **k: "prompt", None)
    )
    payload = SimpleNamespace(model_dump=lambda: {**body(max_tokens=64), "model_id": MODEL_ID})
    with pytest.raises(DeepInferError, match="context") as exc:
        await budget.token_budget(payload)
    assert exc.value.status_code == 422


@pytest.mark.anyio
async def test_startup_warmup_drains_and_closes_native_requests():
    calls, closed = [], []

    class Runtime:
        async def prepare_tokens(self, request):
            return [2] * 16384

        async def prefix_length(self, request, tokens):
            return 198

        async def generate(self, request, tokens):
            calls.append(request)
            try:
                yield {"type": "tokens", "tokens": [3], "finish_reason": "length"}
            finally:
                closed.append(request)

    await warmup(Runtime())
    assert len(calls) == len(closed) == 2401
    assert all(r.prefix_tokens == 198 for r in calls)
    assert all(r.cache_namespace is None for r in calls[:-6])
    assert all(r.cache_namespace == "0" * 64 for r in calls[-6:])
    assert calls[-6:-4] == calls[-4:-2] == calls[-2:]


@pytest.mark.anyio
async def test_startup_warmup_fails_readiness_on_native_error():
    class Runtime:
        async def prepare_tokens(self, request):
            return [2] * 16384

        async def prefix_length(self, request, tokens):
            return 0

        async def generate(self, request, tokens):
            yield {"type": "error"}

    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await warmup(Runtime())


@pytest.mark.parametrize(
    "content", ["Hello world!", "世界 🙂 café\n\n", "<end> special [UNK]", "a" * 20000]
)
def test_thread_friendly_tokenizer_preserves_exact_single_sequence_ids(content):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.train_from_iterator(
        ["Hello world!", "世界 🙂 café", "special text", "a" * 100],
        trainer=trainers.BpeTrainer(vocab_size=128, special_tokens=["[UNK]", "<end>"]),
    )
    request = parse_request(body(messages=[{"role": "user", "content": content}]))
    actual = encode_request(tokenizer, lambda *a, **k: content, request)
    assert actual == tokenizer.encode(content, add_special_tokens=False).ids


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "valid"),
    [
        (698512, 350000, True),
        (698513, 350000, False),
        (1048511, 1, True),
        (1048512, 1, False),
    ],
)
def test_million_token_capacity_reserves_space_for_requested_output(
    input_tokens, output_tokens, valid
):
    from flash.serving.deepinfer.contract.tokenizer import validate_length

    request = parse_request(body(max_tokens=output_tokens))
    tokens = range(input_tokens)
    if valid:
        validate_length(tokens, request.max_tokens)
    else:
        with pytest.raises(ValueError, match="serving context"):
            validate_length(tokens, request.max_tokens)


@pytest.mark.anyio
async def test_deepinfer_admission_binds_token_prefix_cache_to_authenticated_namespace(monkeypatch):
    seen = []

    class Cache:
        def encode(self, prompt, namespace):
            seen.append((prompt, namespace))
            return [1, 2, 3]

    monkeypatch.setattr(
        budget, "prepare_tokenizer", lambda: (None, lambda *a, **k: "rendered", Cache())
    )
    payload = SimpleNamespace(
        model_dump=lambda: {**body(max_tokens=77), "model_id": MODEL_ID},
        _cache_namespace="a" * 64,
    )
    assert await budget.token_budget(payload) == 80
    assert seen == [("rendered", "a" * 64)]
