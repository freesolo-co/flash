"""Count exact DeepSeek reservations on the router without a GPU network trip."""

import asyncio
from functools import lru_cache
from threading import Lock

from flash.serving.deepinfer.app.events import chat_body
from flash.serving.deepinfer.contract.config import MODEL_ID, MODEL_REVISION
from flash.serving.deepinfer.contract.errors import DeepInferError
from flash.serving.deepinfer.contract.protocol import parse_request
from flash.serving.deepinfer.contract.tokenizer import encode_request, load_tokenizer

_load_lock = Lock()


@lru_cache(maxsize=1)
def _load_tokenizer():
    from huggingface_hub import snapshot_download

    # Never download model weights onto CPU routers; execute only the pinned encoder.
    path = snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, allow_patterns=["tokenizer.json", "encoding/*.py"]
    )
    return load_tokenizer(path)


def prepare_tokenizer():
    # Concurrent cold requests must not each allocate a tokenizer. Call off-loop.
    with _load_lock:
        return _load_tokenizer()


def _count(request):
    tokenizer, encode_messages = prepare_tokenizer()
    try:
        return len(encode_request(tokenizer, encode_messages, request)) + request.max_tokens
    except ValueError as exc:
        raise DeepInferError(422, str(exc)) from exc


async def token_budget(payload):
    request = parse_request(chat_body(payload.model_dump()))
    return await asyncio.to_thread(_count, request)
