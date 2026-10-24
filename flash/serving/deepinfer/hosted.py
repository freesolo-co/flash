"""Native token events in the hosted router's durable-accounting protocol."""

import re
import time
import uuid
from dataclasses import replace

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
    decoder = Decoder(runtime.tokenizer, thinking=parsed.thinking)
    started = time.monotonic()
    native = runtime.generate(parsed, tokens)
    text = ""
    boundary_sent = not parsed.thinking
    cached_tokens = 0
    cached_reported = False

    def facts():
        return {
            "request_id": request_id,
            "model_id": MODEL_ID,
            "thinking": parsed.thinking,
            "prompt_tokens": len(tokens),
            "completion_tokens": decoder.completion_tokens,
            "reasoning_tokens": decoder.reasoning_tokens,
            "cached_tokens": cached_tokens,
            "cached_tokens_reported": cached_reported,
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
            count = current.get("cached_tokens", 0)
            reported = current.get("cached_tokens_reported", False)
            if type(count) is not int or not 0 <= count <= len(tokens):
                raise RuntimeError("Invalid native cache usage")
            if not first and (count != cached_tokens or reported != cached_reported):
                raise RuntimeError("Native cache usage changed during generation")
            cached_tokens, cached_reported = count, reported
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
    namespace = body.get("cache_namespace")
    if namespace is not None and (
        not isinstance(namespace, str) or not re.fullmatch(r"[0-9a-f]{64}", namespace)
    ):
        raise ValueError("invalid cache namespace")
    parsed = replace(parsed, cache_namespace=namespace)
    tokens = await runtime.prepare_tokens(parsed)
    parsed = replace(parsed, prefix_tokens=await runtime.prefix_length(parsed, tokens))
    return engine_events(runtime, parsed, tokens, request_id)
