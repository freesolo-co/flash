"""Generation tokenization and startup readiness preserve native contracts."""

import pytest

from flash.serving.deepinfer.contract.protocol import parse_request
from flash.serving.deepinfer.contract.tokenizer import encode_request
from flash.serving.deepinfer.runtime.warmup import warmup
from tests.serving.test_deepinfer import body


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("verified", [False, True])
async def test_startup_warmup_drains_and_closes_native_requests(monkeypatch, verified):
    monkeypatch.setattr(
        "flash.serving.deepinfer.runtime.warmup.verified_compiler_cache", lambda _: verified
    )
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
    assert len(calls) == len(closed) == (363 if verified else 2401)
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


@pytest.mark.parametrize("thinking", [False, True])
def test_generation_uses_official_formatter_and_scoped_prefix_cache(thinking):
    from dataclasses import replace

    seen = []

    class Cache:
        def encode(self, prompt, namespace):
            seen.append((prompt, namespace))
            return [1, 2, 3]

    def encode(messages, *, thinking_mode):
        assert thinking_mode == ("thinking" if thinking else "chat")
        return "rendered"

    request = replace(
        parse_request(body(chat_template_kwargs={"enable_thinking": thinking})),
        cache_namespace="a" * 64,
    )
    assert encode_request(None, encode, request, prefix_cache=Cache()) == [1, 2, 3]
    assert seen == [("rendered", "a" * 64)]
