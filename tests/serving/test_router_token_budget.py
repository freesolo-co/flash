"""CPU admission shares Qwen rendering and retains conservative image reservations."""

import pytest

from flash.serving.src.engine.router_token_budget import RouterTokenBudgets
from flash.serving.src.engine.text_tokenizer import TextPromptTokenizer
from flash.serving.src.io.openai_request import OpenAIGenerateRequest
from flash.serving.src.io.schemas import AdapterRecord
from tests.serving.checkpoint_fixtures import checkpoint_record

BASE = "Qwen/Qwen3.5-9B"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_router_reserves_exact_prompt_and_all_choices_with_authoritative_thinking():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(kwargs)
            return [1, 2, 3]

    owner = TextPromptTokenizer()
    owner.base_model = BASE
    owner.tokenizer = Tokenizer()
    budgets = RouterTokenBudgets()
    budgets._owners[BASE] = owner
    payload = OpenAIGenerateRequest(
        model_id=BASE,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=10,
        n=4,
        temperature=0.8,
        chat_template_kwargs={"enable_thinking": True},
    )
    record = AdapterRecord(
        model_id=BASE, repo_id=BASE, base_model=BASE, serve_base_model=True, thinking=False
    )
    assert await budgets.count(BASE, payload, record) == 43
    assert calls[0]["enable_thinking"] is True
    assert calls[0]["preserve_thinking"] is False
    record = checkpoint_record("qa", BASE, org_id="org-a", thinking=False)
    payload.model_id = record.model_id
    assert await budgets.count(BASE, payload, record) == 43
    assert calls[1]["enable_thinking"] is False


@pytest.mark.anyio
async def test_router_image_reservation_never_fetches_or_tokenizes_pixels():
    owner = TextPromptTokenizer()
    owner.base_model = BASE
    # No tokenizer: image expansion must reserve the context without reading the URL.
    budgets = RouterTokenBudgets()
    budgets._owners[BASE] = owner
    payload = OpenAIGenerateRequest(
        model_id=BASE,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://unreachable.invalid/image.png"},
                    }
                ],
            }
        ],
        max_tokens=10,
        n=2,
        temperature=0.8,
    )
    record = AdapterRecord(
        model_id=BASE, repo_id=BASE, base_model=BASE, serve_base_model=True, thinking=False
    )
    assert await budgets.count(BASE, payload, record) == 32768 * 2


@pytest.mark.anyio
async def test_cached_qwen_text_reservation_matches_generation_and_keeps_tenant_scope():
    from tokenizers import AddedToken

    from tests.serving.test_token_prefix_cache import tokenizer

    raw = tokenizer()
    raw.add_tokens([AddedToken("<|im_start|>", normalized=False, special=True)])

    class ChatTokenizer:
        backend_tokenizer = raw

        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            rendered = (
                "".join(
                    "<|im_start|>" + msg["role"] + "\n" + msg["content"] + "\n" for msg in messages
                )
                + "<|im_start|>assistant\n"
            )
            if not tokenize:
                return rendered
            ids = raw.encode(rendered, add_special_tokens=False).ids
            maximum = kwargs.get("tokenizer_kwargs", {}).get("max_length")
            return ids[:maximum] if maximum is not None else ids

    owner = TextPromptTokenizer()
    owner.base_model = BASE
    owner.tokenizer = ChatTokenizer()
    budgets = RouterTokenBudgets()
    budgets._owners[BASE] = owner
    record = AdapterRecord(
        model_id=BASE, repo_id=BASE, base_model=BASE, serve_base_model=True, thinking=False
    )
    payload = OpenAIGenerateRequest(
        model_id=BASE,
        messages=[
            {"role": "system", "content": "hello world " * 200},
            {"role": "user", "content": "first question"},
        ],
        max_tokens=10,
        n=2,
        temperature=0.8,
    )
    for namespace, question in [("a" * 64, "first"), ("a" * 64, "next"), ("b" * 64, "next")]:
        payload._cache_namespace = namespace
        payload.messages[-1]["content"] = question
        expected = owner.tokenizer.apply_chat_template(payload.messages, tokenize=True)
        assert await budgets.count(BASE, payload, record) == len(expected) + 20
        assert (await owner._prompt_input(payload, False))["prompt_token_ids"] == expected
    assert {key[0] for key in owner._text_prefix_cache.entries} == {"a" * 64, "b" * 64}

    payload.chat_template_kwargs = {"tokenizer_kwargs": {"max_length": 2}}
    assert await budgets.count(BASE, payload, record) == 22
    assert len((await owner._prompt_input(payload, False))["prompt_token_ids"]) == 2
    assert owner._text_prefix_cache.tokenizer is not raw
