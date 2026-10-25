"""Shared text rendering/tokenization for CPU admission and GPU generation."""

import hashlib
import json
from typing import Any

from flash.content.thinking import messages_for_chat_template
from flash.serve.request.tool_calls import (
    detached_template_messages,
    normalize_tools,
    tools_active,
    tools_wire,
)
from flash.serve.runtime.prompt_cache import cached_prompt_tokens
from flash.serving.src.engine.support import _safe_chat_template_kwargs


class TextPromptTokenizer:
    """Tokenizer-only methods; owners supply tokenizer and optional loop-owned cache."""

    def _thinking_default(self, record: Any, payload: Any) -> bool:
        """The ``enable_thinking`` to render with. A trained LoRA forces its own value; a base-model
        serve has none, so it honors the caller's chat_template_kwargs enable_thinking when given."""
        if getattr(record, "serve_base_model", False):
            caller = _safe_chat_template_kwargs(getattr(payload, "chat_template_kwargs", None))
            override = caller.get("enable_thinking")
            if isinstance(override, bool):
                return override
        return record.thinking

    def _effective_chat_template_kwargs(
        self, payload: Any, thinking_default: bool | None = None
    ) -> dict[str, Any]:
        """sanitized template kwargs with one already-resolved authoritative thinking value.

        immutable adapters always supply their persisted training mode. base-model requests may
        supply the validated per-call override resolved by ``_thinking_default``. injecting that
        resolved value here keeps prompt rendering, parser state, and accounting aligned.

        for an immutable adapter, ``thinking_default`` is resolved alongside the lora weights under
        the adapter lock, so prompt rendering stays bound to the same record during redeploys.
        """
        ctk = _safe_chat_template_kwargs(getattr(payload, "chat_template_kwargs", None))
        if not isinstance(thinking_default, bool):
            raise ValueError("adapter thinking default is required")
        effective = {
            **ctk,
            "enable_thinking": thinking_default,
            "preserve_thinking": False,
        }
        if tools_active(getattr(payload, "tools", None), getattr(payload, "tool_choice", None)):
            effective["tools"] = tools_wire(normalize_tools(payload.tools))
        return effective

    def _prompt_cache_key(
        self, payload: Any, thinking_default: bool | None = None
    ) -> tuple[str, str] | None:
        # Key on a fixed-size hash of the prompt/messages, not the raw text: with up to
        # _prompt_cache_size entries, storing full prompt strings as keys could retain a lot of
        # duplicated text (and risk OOM) for long prompts. blake2b is fast; 128-bit is collision-safe
        # for a cache key.
        if getattr(self, "_prompt_cache_size", 0) <= 0:
            return None
        if payload.messages:
            try:
                raw = json.dumps(
                    payload.messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
                # chat_template_kwargs changes the rendered prompt, so it MUST be part of the key
                # — otherwise the same messages with thinking on vs off collide and the wrong cached
                # token IDs are reused. Use the SAME *effective* view _tokenize_prompt splats
                # (caller kwargs sanitized, then enable_thinking resolved per-adapter), so two
                # adapters on the same base model with different thinking defaults don't collide,
                # while
                # reserved/ignored keys still don't split identical prompts (unnecessary misses).
                ctk = self._effective_chat_template_kwargs(payload, thinking_default)
                if ctk:
                    raw += "\x00" + json.dumps(
                        ctk,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                        default=str,
                    )
            except TypeError:
                return None
            kind = "messages"
        elif payload.prompt:
            raw, kind = payload.prompt, "prompt"
        else:
            return None
        return (kind, hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest())

    async def _prompt_input(
        self, payload: Any, thinking_default: bool | None = None
    ) -> dict[str, Any]:
        # TTFT: an LRU hit is microseconds, so it is served inline - a thread hop would cost more
        # than the work. A MISS runs the chat template + tokenizer, which is 10 ms at 6k tokens and
        # ~51 ms at 24k (measured, Qwen3.5 fast tokenizer). Every tier serves 32k context and
        # modal.concurrent packs up to max_inputs requests onto ONE container, so tokenizing on the
        # event loop head-of-line blocks EVERY co-resident request's TTFT, not just this one. HF
        # fast tokenizers are Rust and release the GIL (measured 5.6x on 8 threads), so the offload
        # buys real parallelism instead of just moving the stall.
        # Only _tokenize_prompt moves to the worker thread, so the cache is still read and written
        # solely from the loop thread and needs no lock. The await does mean two identical
        # concurrent misses can both tokenize; that is idempotent (same messages + same effective
        # kwargs -> same ids), and the duplicate now costs a worker thread rather than the loop.
        key = self._prompt_cache_key(payload, thinking_default)
        cache = getattr(self, "_prompt_token_cache", None)
        prompt_token_ids = await cached_prompt_tokens(
            cache,
            key,
            lambda: self._tokenize_prompt(payload, thinking_default),
            max_entries=getattr(self, "_prompt_cache_size", 0),
        )
        return {"prompt_token_ids": prompt_token_ids}

    def _tokenize_prompt(self, payload: Any, thinking_default: bool | None = None) -> list[int]:
        if payload.messages:
            # Forward sanitized chat_template_kwargs, but force enable_thinking from the adapter's
            # trained ``thinking`` value so every caller renders in the mode this LoRA was trained
            # with. Reserved/return-shape kwargs are dropped before rendering to avoid 500s.
            ctk = self._effective_chat_template_kwargs(payload, thinking_default)
            try:
                prompt_token_ids = self.tokenizer.apply_chat_template(
                    messages_for_chat_template(detached_template_messages(payload.messages)),
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                    **ctk,
                )
            except Exception as exc:
                raise ValueError(f"chat template rejected the request messages: {exc}") from exc
            return list(prompt_token_ids)
        if payload.prompt:
            prompt_token_ids = self.tokenizer.encode(payload.prompt, add_special_tokens=False)
            return list(prompt_token_ids)
        raise ValueError("prompt or messages is required")
