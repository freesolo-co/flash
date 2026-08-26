"""text and multimodal prompt preparation for one initialized runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from flash.content.reasoning_normalization import messages_for_chat_template
from flash.serve.request.tool_calls import detached_template_messages, tools_active, tools_wire

from .errors import MultimodalRequestError, PromptError, ServingRuntimeError
from .multimodal import has_image_blocks, normalize_text_messages, prepare_multimodal_request
from .types import AdapterSpec, EngineConfig, GenerationRequest

_RESERVED_CHAT_TEMPLATE_KWARGS = frozenset(
    {
        "tokenize",
        "add_generation_prompt",
        "return_dict",
        "return_tensors",
        "return_assistant_tokens_mask",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "enable_thinking",
        "conversation",
        "documents",
        "chat_template",
        "padding",
        "truncation",
        "max_length",
    }
)


@dataclass(slots=True)
class PreparedPrompt:
    """vllm prompt input plus resources that must be released after generation."""

    value: dict[str, Any]
    images: tuple[Any, ...] = ()

    def close(self) -> None:
        for image in self.images:
            close = getattr(image, "close", None)
            if close is not None:
                close()


def safe_chat_template_kwargs(raw: Any) -> dict[str, Any]:
    """remove arguments whose values or return shapes are runtime-owned."""
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if key not in _RESERVED_CHAT_TEMPLATE_KWARGS}


def resolve_thinking(request: GenerationRequest, adapter: AdapterSpec | None) -> bool:
    """bind a known thinking mode to the same request and adapter state as generation."""
    if adapter is not None:
        return adapter.thinking
    if request.thinking is not None:
        return request.thinking
    return False


def effective_chat_template_kwargs(
    request: GenerationRequest,
    thinking: bool | None,
) -> dict[str, Any]:
    """return sanitized template kwargs with an authoritative thinking value when known."""
    kwargs = safe_chat_template_kwargs(request.chat_template_kwargs)
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    if tools_active(request.tools, request.tool_choice):
        kwargs["tools"] = tools_wire(request.tools)
    return kwargs


class PromptPreparer:
    """prepare vllm prompt inputs and retain a bounded text-token lru cache."""

    def __init__(self, config: EngineConfig, tokenizer: Any, processor: Any | None) -> None:
        self._config = config
        self._tokenizer = tokenizer
        self._processor = processor
        self._cache: OrderedDict[tuple[str, str], tuple[int, ...]] = OrderedDict()

    @property
    def cache_entries(self) -> int:
        return len(self._cache)

    async def prepare(
        self,
        request: GenerationRequest,
        thinking: bool | None,
    ) -> PreparedPrompt:
        if request.messages is not None and has_image_blocks(request.messages):
            return await self._prepare_multimodal(request, thinking)
        return PreparedPrompt(await self._prepare_text(request, thinking))

    async def _prepare_text(
        self,
        request: GenerationRequest,
        thinking: bool | None,
    ) -> dict[str, Any]:
        # normalizing here rather than at the http boundary is what keeps the image path intact:
        # normalization strips image payloads out into a separate channel, so only this branch --
        # already past the `has_image_blocks` dispatch -- can safely adopt the rewritten messages.
        # the key is derived from them too, so `input_text` and `text` spellings of one prompt
        # share the entry they already share a rendering of, instead of tokenizing twice.
        messages = None if request.messages is None else normalize_text_messages(request.messages)
        key = self._cache_key(request, messages, thinking)
        if key is not None:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return {"prompt_token_ids": list(cached)}
        token_ids = await asyncio.to_thread(self._tokenize, request, messages, thinking)
        if key is not None:
            self._cache[key] = tuple(token_ids)
            self._cache.move_to_end(key)
            while len(self._cache) > self._config.prompt_cache_size:
                self._cache.popitem(last=False)
        return {"prompt_token_ids": token_ids}

    def _tokenize(
        self,
        request: GenerationRequest,
        messages: list[dict[str, Any]] | None,
        thinking: bool | None,
    ) -> list[int]:
        if messages is not None:
            # a template rejects shapes its own vocabulary cannot render, and only the request
            # decides that shape -- Qwen3.5 raises `TypeError` on a tool call whose `arguments` is
            # the string the OpenAI schema specifies. Structural validation upstream deliberately
            # stops short of any one template's rules, so these reach here. Unclassified, they
            # escape before `_rejection_as_prompt_error` and answer 503, telling the caller to
            # retry a request that must fail identically while the engine is perfectly healthy.
            try:
                template_messages = messages_for_chat_template(detached_template_messages(messages))
                token_ids = self._tokenizer.apply_chat_template(
                    template_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                    **effective_chat_template_kwargs(request, thinking),
                )
            except ServingRuntimeError:
                # an engine fault surfacing through the template keeps its own meaning.
                raise
            except Exception as exc:
                # jinja raises its own error types, so this cannot enumerate them. `PromptError`
                # names the caller as the cause, which is what a rendering failure over
                # caller-supplied messages means.
                raise PromptError(f"chat template rejected the request messages: {exc}") from exc
        else:
            token_ids = self._tokenizer.encode(request.prompt, add_special_tokens=False)
        try:
            return [int(token_id) for token_id in token_ids]
        except (TypeError, ValueError) as exc:
            raise PromptError("tokenizer did not return a flat token-id sequence") from exc

    def _cache_key(
        self,
        request: GenerationRequest,
        messages: list[dict[str, Any]] | None,
        thinking: bool | None,
    ) -> tuple[str, str] | None:
        if self._config.prompt_cache_size <= 0:
            return None
        if messages is not None:
            try:
                raw = json.dumps(
                    messages,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                kwargs = effective_chat_template_kwargs(request, thinking)
                raw += "\0" + json.dumps(
                    kwargs,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                )
            except TypeError:
                return None
            kind = "messages"
        else:
            raw = request.prompt or ""
            kind = "prompt"
        digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()
        return kind, digest

    async def _prepare_multimodal(
        self,
        request: GenerationRequest,
        thinking: bool | None,
    ) -> PreparedPrompt:
        if self._processor is None or self._config.image_limit is None:
            raise MultimodalRequestError(
                "image input requires a processor on an image-capable runtime"
            )
        template_messages, images = await asyncio.to_thread(
            prepare_multimodal_request,
            request.messages,
            image_limit=self._config.image_limit,
        )
        try:
            kwargs = effective_chat_template_kwargs(request, thinking)
            try:
                # the same request-caused rejection the text branch translates. the handler below
                # is about releasing decoded images, not classifying failures, so without this an
                # unrenderable multimodal request escapes unclassified and answers 503 exactly as
                # the text path used to.
                template_messages = messages_for_chat_template(
                    detached_template_messages(template_messages)
                )
                rendered = await asyncio.to_thread(
                    self._processor.apply_chat_template,
                    template_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            except ServingRuntimeError:
                raise
            except Exception as exc:
                raise PromptError(f"chat template rejected the request messages: {exc}") from exc
            if not isinstance(rendered, str):
                raise MultimodalRequestError("processor chat template did not return text")
            image_data: Any = images[0] if len(images) == 1 else images
            value = {"prompt": rendered, "multi_modal_data": {"image": image_data}}
            return PreparedPrompt(value=value, images=tuple(images))
        except BaseException:
            PreparedPrompt(value={}, images=tuple(images)).close()
            raise
