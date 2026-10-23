"""Native token events in the hosted router's durable-accounting protocol."""

import asyncio
import time
import uuid

from flash.serving.deepinfer.config import MODEL_ID, MODEL_REVISION
from flash.serving.deepinfer.protocol import parse_request
from flash.serving.deepinfer.render import Decoder

REPLICA_ID = "deepinfer-" + uuid.uuid4().hex


def chat_body(payload):
    """Reject unsupported generation controls before dispatching to a GPU replica."""
    if payload.get("model_id") != MODEL_ID:
        raise ValueError("DeepInfer only serves its pinned base model")
    if payload.get("prompt") is not None:
        raise ValueError("DeepInfer requires messages; raw prompt serving is unsupported")
    defaults = {
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "logprobs": False,
        "top_logprobs": 0,
        "tools": None,
        "tool_choice": None,
        "parallel_tool_calls": None,
        "structured_outputs": None,
        "stop": None,
    }
    for key, default in defaults.items():
        if payload.get(key, default) != default:
            raise ValueError(f"DeepInfer does not support {key}")
    result = {
        "model": MODEL_ID,
        "messages": payload.get("messages"),
        **{
            key: payload[key]
            for key in ("temperature", "top_p", "max_tokens", "seed", "n")
            if key in payload
        },
        "chat_template_kwargs": payload.get("chat_template_kwargs") or {},
    }
    parse_request(result)
    return result


async def engine_events(runtime, parsed, tokens, request_id):
    decoder = Decoder(runtime.tokenizer, thinking=parsed.thinking, prompt_tokens=len(tokens))
    started = time.monotonic()
    native = runtime.generate(parsed, tokens)
    text = ""
    boundary_sent = not parsed.thinking

    def facts():
        return {
            "request_id": request_id,
            "model_id": MODEL_ID,
            "thinking": parsed.thinking,
            "prompt_tokens": len(tokens),
            "completion_tokens": decoder.completion_tokens,
            "reasoning_tokens": decoder.reasoning_tokens,
            # This native release exposes no per-request prefix-cache hit counter.
            "cached_tokens": 0,
            "cached_tokens_reported": False,
            "engine_replica_id": REPLICA_ID,
            "tokenizer_identity": MODEL_ID,
            "tokenizer_version": MODEL_REVISION,
            "inference_time_seconds": time.monotonic() - started,
        }

    try:
        # Admission and the first native output must succeed before the router captures usage.
        current = await anext(native)
        first = True
        while True:
            deltas = decoder.consume(current)
            if first:
                yield {"type": "ready", **facts()}
                first = False
            raw = ""
            for delta in deltas:
                if "content" in delta and not boundary_sent:
                    raw += "</think>"
                    boundary_sent = True
                raw += delta.get("reasoning_content", delta.get("content", ""))
            if not decoder.thinking and not boundary_sent:
                raw += "</think>"
                boundary_sent = True
            text += raw
            yield {"type": "delta" if raw else "usage_progress", "index": 0, "text": raw, **facts()}
            if reason := current.get("finish_reason"):
                yield {"type": "choice_finished", "index": 0, "finish_reason": reason, **facts()}
                yield {"type": "final", "text": text, "finish_reason": reason, **facts()}
                return
            current = await anext(native)
    finally:
        await native.aclose()


async def prepare(runtime, body):
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 512:
        raise ValueError("request_id is required")
    parsed = parse_request(body.get("chat", {}))
    tokens = await asyncio.to_thread(runtime.tokenize, parsed)
    return engine_events(runtime, parsed, tokens, request_id)
