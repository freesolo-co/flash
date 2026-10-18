"""lazy vllm engine lifecycle, generation, streaming, and health."""

from __future__ import annotations

import asyncio
import inspect
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

from flash.serve.request.runtime_support import reasoning_compatibility_guard
from flash.serve.request.tool_calls import tools_active

from .adapters import AdapterBinding, AdapterManager
from .errors import (
    EngineDeadError,
    PromptError,
    RuntimeNotReadyError,
    ServingRuntimeError,
)
from .prompt import (
    PreparedPrompt,
    PromptPreparer,
    effective_chat_template_kwargs,
    resolve_thinking,
)
from .sampling import complete_indexed_outputs, indexed_outputs, normalize_token_logprobs
from .tool_calls import ParsedToolCall, ToolCallStreamParser, parse_qwen3_coder_output
from .types import (
    AdapterSpec,
    EngineConfig,
    GenerationChoice,
    GenerationRequest,
    GenerationResult,
    RuntimeHealth,
    StreamChoiceFinished,
    StreamDelta,
    StreamEvent,
    StreamFinished,
    StreamReady,
    thaw_mapping,
)

EngineDeathCallback = Callable[[RuntimeHealth], Awaitable[None] | None]


@dataclass(slots=True)
class _StructuredState:
    params: Any = None
    reasoning_ended: bool | None = None
    parser_kwargs: dict[str, Any] | None = None


@dataclass(slots=True)
class _ChoiceStreamState:
    token_ids: list[int]
    logprobs: list[dict[str, Any]]
    tool_parser: ToolCallStreamParser | None = None
    text: str = ""
    finish_reason: str | None = None
    tool_calls: tuple[Any, ...] = ()
    terminal_emitted: bool = False


@dataclass(slots=True)
class _StreamState:
    n: int
    top_logprobs: int
    choices: dict[int, _ChoiceStreamState]
    prompt_tokens: int | None = None
    cached_tokens: int = 0
    cached_reported: bool = False

    @classmethod
    def create(cls, n: int, top_logprobs: int, tools: Any = None) -> _StreamState:
        return cls(
            n=n,
            top_logprobs=top_logprobs,
            choices={
                index: _ChoiceStreamState(
                    token_ids=[],
                    logprobs=[],
                    tool_parser=ToolCallStreamParser(tools) if tools is not None else None,
                )
                for index in range(n)
            },
        )

    def consume(
        self, request_output: Any
    ) -> list[tuple[StreamDelta | None, StreamChoiceFinished | None]]:
        if self.prompt_tokens is None:
            self.prompt_tokens = _optional_prompt_tokens(request_output)
        if not self.cached_reported:
            self.cached_tokens, self.cached_reported = _cached_token_state(request_output)
        events: list[tuple[StreamDelta | None, StreamChoiceFinished | None]] = []
        for index, output in sorted(indexed_outputs(request_output, n=self.n).items()):
            choice = self.choices[index]
            if choice.finish_reason is not None:
                raise RuntimeNotReadyError("vllm emitted data after a choice terminal")
            chunk_ids = [int(value) for value in (getattr(output, "token_ids", None) or [])]
            chunk_logprobs = normalize_token_logprobs(
                chunk_ids,
                getattr(output, "logprobs", None),
                top_logprobs=self.top_logprobs,
            )
            choice.token_ids.extend(chunk_ids)
            if chunk_logprobs is not None:
                choice.logprobs.extend(chunk_logprobs)
            chunk_text = str(getattr(output, "text", "") or "")
            visible = choice.tool_parser.feed(chunk_text) if choice.tool_parser else chunk_text
            choice.text += visible
            if visible or chunk_logprobs:
                events.append(
                    (StreamDelta(index=index, text=visible, logprobs=chunk_logprobs), None)
                )
            finish_reason = getattr(output, "finish_reason", None)
            if finish_reason is None:
                continue
            if not isinstance(finish_reason, str) or not finish_reason:
                raise RuntimeNotReadyError("vllm returned an invalid finish reason")
            tool_calls: tuple[ParsedToolCall, ...] = ()
            if choice.tool_parser is not None:
                parsed = choice.tool_parser.finish()
                if parsed.tools_called:
                    tool_calls = parsed.calls
                    finish_reason = "tool_calls"
                    events.append((StreamDelta(index=index, text="", tool_calls=tool_calls), None))
                elif parsed.content:
                    choice.text += parsed.content
                    events.append((StreamDelta(index=index, text=parsed.content), None))
            choice.finish_reason = finish_reason
            choice.tool_calls = tool_calls
            choice.terminal_emitted = True
            events.append(
                (
                    None,
                    StreamChoiceFinished(
                        index=index,
                        text=choice.text,
                        finish_reason=finish_reason,
                        token_ids=tuple(choice.token_ids),
                    ),
                )
            )
        return events

    def validate_complete(self) -> None:
        if self.prompt_tokens is None:
            raise RuntimeNotReadyError("vllm did not report the expanded prompt token count")
        if any(choice.finish_reason is None for choice in self.choices.values()):
            raise RuntimeNotReadyError("vllm ended with unterminated output choices")


def _generation_choice(
    index: int, output: Any, *, top_logprobs: int, tools: Any = None
) -> GenerationChoice:
    token_ids = tuple(int(value) for value in (getattr(output, "token_ids", None) or []))
    finish_reason = getattr(output, "finish_reason", None)
    if not isinstance(finish_reason, str) or not finish_reason:
        raise RuntimeNotReadyError("vllm generation ended without a finish reason")
    text = str(getattr(output, "text", "") or "")
    tool_calls: tuple[ParsedToolCall, ...] = ()
    if tools is not None:
        parsed = parse_qwen3_coder_output(text, tools)
        if parsed.tools_called:
            text = parsed.content or ""
            tool_calls = parsed.calls
            finish_reason = "tool_calls"
    return GenerationChoice(
        index=index,
        text=text,
        finish_reason=finish_reason,
        token_ids=token_ids,
        logprobs=normalize_token_logprobs(
            token_ids,
            getattr(output, "logprobs", None),
            top_logprobs=top_logprobs,
        ),
        tool_calls=tool_calls,
    )


def _optional_prompt_tokens(request_output: Any) -> int | None:
    """prompt token count when this output reports one, else none.

    a delta that omits prompt metadata is expected rather than invalid, but a delta that reports
    an unusable value is still an error.
    """
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    if prompt_token_ids is None and getattr(request_output, "num_prompt_tokens", None) is None:
        return None
    return _num_prompt_tokens(request_output)


def _num_prompt_tokens(request_output: Any) -> int:
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return len(prompt_token_ids)
    value = getattr(request_output, "num_prompt_tokens", None)
    if value is None:
        raise RuntimeNotReadyError("vllm did not report the expanded prompt token count")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeNotReadyError("vllm reported an invalid prompt token count") from exc
    if count < 0:
        raise RuntimeNotReadyError("vllm reported a negative prompt token count")
    return count


def _cached_token_state(request_output: Any) -> tuple[int, bool]:
    value = getattr(request_output, "num_cached_tokens", None)
    if value is None:
        return 0, False
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0, False
    if count < 0:
        return 0, False
    return count, True


_require_reasoning_compatibility = reasoning_compatibility_guard(
    RuntimeNotReadyError, "vllm reasoning api is missing "
)


def _has_hub_credential(configured_token: str | None) -> bool:
    """report whether this process can authenticate to the hub at all.

    an explicit token settles it. otherwise the hub is asked directly instead of reading
    HF_TOKEN here, because its own resolution order also covers HUGGING_FACE_HUB_TOKEN, the
    cached login file, and OIDC exchange -- reimplementing that would drift and would wrongly
    report "no credential" for a container that can in fact download.
    """

    if configured_token:
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        # if the hub cannot even be asked, assume no credential and stay on the cache. that is
        # the safe direction: a needless local-only load fails loudly on a missing file rather
        # than silently reaching the network from a container that has no way to authenticate.
        return False


@asynccontextmanager
async def _rejection_as_prompt_error() -> AsyncIterator[None]:
    """re-raise vllm's request-rejection `ValueError` as a `PromptError`.

    vllm signals an intrinsically invalid request -- most commonly a prompt longer than
    `max_model_len` -- by raising a plain `ValueError` from `generate`. Without this it reaches the
    http layer as an unclassified exception and is answered 503, which tells the client the service
    is temporarily unavailable and invites a retry that must fail identically, re-tokenizing and
    re-dispatching to the gpu every time. `PromptError` maps to 400, matching what the router this
    runtime replaces already returned for the same condition.

    Deliberately narrow: only `ValueError` is rewritten. A `TypeError` or an engine crash keeps its
    own meaning rather than being blamed on the caller's prompt. Errors that are already runtime
    errors pass through untouched, so `PromptError` (itself a `ValueError`) is not re-wrapped.
    """

    try:
        yield
    except ServingRuntimeError:
        raise
    except ValueError as exc:
        raise PromptError(str(exc)) from exc


class VllmLoraRuntime:
    """one lazy, process-local vllm engine with exact adapter incarnations."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        on_engine_death: EngineDeathCallback | None = None,
    ) -> None:
        self.config = config
        self.runtime_id = uuid.uuid4().hex
        self._on_engine_death = on_engine_death
        self._engine: Any | None = None
        self._tokenizer: Any | None = None
        self._processor: Any | None = None
        self._prompts: PromptPreparer | None = None
        self._adapters: AdapterManager | None = None
        self._start_lock = asyncio.Lock()
        self._liveness_task: asyncio.Task[None] | None = None
        self._death_notification_lock = asyncio.Lock()
        self._death_notified = False
        self._closed = False

    async def start(self) -> None:
        """construct tokenizer, processor, and engine exactly once on first use."""
        async with self._start_lock:
            if self._closed:
                raise RuntimeNotReadyError("runtime is closed")
            if self._engine is not None:
                if self._engine_is_dead():
                    await self._notify_engine_death()
                    raise EngineDeadError("vllm engine core is dead")
                return
            tokenizer, processor = await asyncio.to_thread(self._load_tokenizer_processor)
            engine = self._build_engine()
            self._tokenizer = tokenizer
            self._processor = processor
            self._engine = engine
            self._prompts = PromptPreparer(self.config, tokenizer, processor)
            self._adapters = AdapterManager(engine, self.config)
            self._liveness_task = asyncio.create_task(self._liveness_monitor())

    async def register_adapter(self, spec: AdapterSpec) -> bool:
        """register or replace an adapter; false means the exact incarnation was already loaded."""
        await self._ensure_started()
        assert self._adapters is not None
        return await self._adapters.register(spec)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """run one non-streaming generation with final-only vllm output."""
        await self._ensure_started()
        request_id = uuid.uuid4().hex
        async with self._binding(request) as binding:
            adapter = binding.spec if binding is not None else None
            lora_request = binding.lora_request if binding is not None else None
            thinking = resolve_thinking(request, adapter)
            if thinking and request.logprobs:
                raise PromptError("logprobs are not supported for thinking-enabled generation")
            self._validate_tools(request, thinking)
            structured = self._structured_state(request, adapter, thinking)
            self._reject_tools_with_structured_outputs(request, structured)
            sampling = self._sampling_params(request, structured, streaming=False)
            prompt = await self._prepare_prompt(request, thinking)
            final_output = None
            try:
                async with _rejection_as_prompt_error():
                    async for output in self._generate_stream(
                        prompt,
                        sampling,
                        request_id,
                        lora_request,
                        structured,
                    ):
                        final_output = output
            except Exception:
                await self._notify_if_dead()
                raise
            finally:
                prompt.close()
        if final_output is None:
            raise RuntimeNotReadyError("vllm returned no output")
        choices = tuple(
            _generation_choice(
                index,
                output,
                top_logprobs=request.top_logprobs,
                tools=request.tools if tools_active(request.tools, request.tool_choice) else None,
            )
            for index, output in sorted(complete_indexed_outputs(final_output, n=request.n).items())
        )
        cached_tokens, cached_reported = _cached_token_state(final_output)
        return GenerationResult(
            request_id=request_id,
            adapter_id=request.adapter_id,
            incarnation=adapter.incarnation if adapter is not None else None,
            choices=choices,
            prompt_tokens=_num_prompt_tokens(final_output),
            completion_tokens=sum(len(choice.token_ids) for choice in choices),
            cached_tokens=cached_tokens,
            cached_tokens_reported=cached_reported,
            thinking=thinking,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamEvent]:
        """yield ready, normalized delta, and terminal accounting events."""
        await self._ensure_started()
        request_id = uuid.uuid4().hex
        async with self._binding(request) as binding:
            adapter = binding.spec if binding is not None else None
            lora_request = binding.lora_request if binding is not None else None
            thinking = resolve_thinking(request, adapter)
            if thinking and request.logprobs:
                raise PromptError("logprobs are not supported for thinking-enabled generation")
            self._validate_tools(request, thinking)
            structured = self._structured_state(request, adapter, thinking)
            self._reject_tools_with_structured_outputs(request, structured)
            sampling = self._sampling_params(request, structured, streaming=True)
            prompt = await self._prepare_prompt(request, thinking)
            output_stream = self._generate_stream(
                prompt,
                sampling,
                request_id,
                lora_request,
                structured,
            )
            try:
                async with _rejection_as_prompt_error():
                    first_output = await self._first_stream_output(output_stream)
                yield StreamReady(
                    request_id=request_id,
                    runtime_id=self.runtime_id,
                    adapter_id=request.adapter_id,
                    incarnation=adapter.incarnation if adapter is not None else None,
                    thinking=thinking,
                )
                state = _StreamState.create(
                    request.n,
                    request.top_logprobs,
                    request.tools if tools_active(request.tools, request.tool_choice) else None,
                )
                output = first_output
                while True:
                    for delta, terminal in state.consume(output):
                        if delta is not None:
                            yield delta
                        if terminal is not None:
                            yield terminal
                    try:
                        output = await anext(output_stream)
                    except StopAsyncIteration:
                        break
                yield self._finished_event(request, adapter, thinking, request_id, state)
            except Exception:
                await self._notify_if_dead()
                raise
            finally:
                await self._close_output_stream(output_stream)
                prompt.close()

    def health(self) -> RuntimeHealth:
        """return process-local identity, state counts, and engine liveness."""
        engine_dead = self._engine_is_dead()
        adapters = self._adapters
        prompts = self._prompts
        return RuntimeHealth(
            ok=self._engine is not None and not engine_dead and not self._closed,
            started=self._engine is not None,
            engine_dead=engine_dead,
            runtime_id=self.runtime_id,
            model=self.config.model,
            served_model=self.config.effective_served_model,
            registered_adapters=adapters.registered_count if adapters is not None else 0,
            loaded_adapters=adapters.loaded_count if adapters is not None else 0,
            prompt_cache_entries=prompts.cache_entries if prompts is not None else 0,
        )

    async def close(self) -> None:
        """stop monitoring, unload adapters, and shut down the engine."""
        async with self._start_lock:
            if self._closed:
                return
            self._closed = True
            task = self._liveness_task
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            try:
                if self._adapters is not None and not self._engine_is_dead():
                    await self._adapters.unload_all()
            finally:
                shutdown = getattr(self._engine, "shutdown", None)
                if shutdown is not None:
                    result = shutdown()
                    if inspect.isawaitable(result):
                        await result

    def _load_tokenizer_processor(self) -> tuple[Any, Any | None]:
        from transformers import AutoProcessor, AutoTokenizer

        common = {
            "token": self.config.hf_token,
            "trust_remote_code": self.config.trust_remote_code,
            # a start with no credential at all must not reach the hub. transformers enumerates
            # the repo's additional_chat_templates/ over the network, and for a private served
            # model that 401 fails closed rather than falling back: list_repo_templates re-raises
            # RepositoryNotFoundError instead of reading local files, so even a fully hydrated
            # cache does not save it.
            #
            # having a credential is exactly what separates the two callers. the packaged
            # launcher hydrates the cache during bootstrap and then deletes the token, so every
            # later start has none and must resolve locally. the generated app keeps HF_TOKEN in
            # its environment and starts against an empty volume, so it must still be allowed to
            # download. the ambient variable counts because huggingface_hub falls back to it when
            # token is None, which is how that app has always authenticated.
            "local_files_only": not _has_hub_credential(self.config.hf_token),
        }
        if self.config.tokenizer_revision is not None:
            common["revision"] = self.config.tokenizer_revision
        if self.config.image_limit is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.effective_tokenizer_model,
                **common,
                **thaw_mapping(self.config.processor_kwargs, "processor_kwargs"),
            )
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeNotReadyError("multimodal processor has no tokenizer")
        else:
            processor = None
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.effective_tokenizer_model,
                **common,
                **thaw_mapping(self.config.tokenizer_kwargs, "tokenizer_kwargs"),
            )
        if (
            getattr(tokenizer, "pad_token", None) is None
            and getattr(tokenizer, "eos_token", None) is not None
        ):
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer, processor

    def _build_engine(self) -> Any:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        _require_reasoning_compatibility(
            AsyncEngineArgs,
            AsyncLLMEngine.generate,
            self.config.reasoning_parser,
        )
        kwargs: dict[str, Any] = {
            "model": self.config.effective_served_model,
            "tokenizer": self.config.effective_tokenizer_model,
            "trust_remote_code": self.config.trust_remote_code,
            "enable_lora": True,
            "max_loras": self.config.max_loras,
            "max_lora_rank": self.config.max_lora_rank,
            "max_cpu_loras": self.config.max_cpu_loras,
            **thaw_mapping(self.config.engine_args, "engine_args"),
        }
        if self.config.model_revision is not None:
            kwargs["revision"] = self.config.model_revision
        if self.config.tokenizer_revision is not None:
            kwargs["tokenizer_revision"] = self.config.tokenizer_revision
        if self.config.reasoning_parser is not None:
            kwargs["reasoning_parser"] = self.config.reasoning_parser
        if self.config.image_limit is not None:
            kwargs["limit_mm_per_prompt"] = {"image": self.config.image_limit}
        kwargs["mm_processor_cache_gb"] = self.config.mm_processor_cache_gb
        kwargs["enable_tower_connector_lora"] = self.config.enable_tower_connector_lora
        return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))

    async def _ensure_started(self) -> None:
        await self.start()
        if self._engine_is_dead():
            await self._notify_engine_death()
            raise EngineDeadError("vllm engine core is dead")

    @asynccontextmanager
    async def _binding(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[AdapterBinding | None]:
        if request.adapter_id is None:
            yield None
            return
        assert self._adapters is not None
        async with self._adapters.acquire(
            request.adapter_id,
            request.expected_incarnation,
        ) as binding:
            yield binding

    def _validate_tools(self, request: GenerationRequest, thinking: bool | None) -> None:
        if not tools_active(request.tools, request.tool_choice):
            return
        if thinking:
            raise PromptError("tools are not supported for thinking-enabled generation")
        if self.config.tool_parser != "qwen3_coder":
            raise PromptError("this serving engine is not qualified for tool calling")

    @staticmethod
    def _reject_tools_with_structured_outputs(
        request: GenerationRequest,
        structured: _StructuredState,
    ) -> None:
        if tools_active(request.tools, request.tool_choice) and structured.params is not None:
            raise PromptError("tools cannot be combined with logprobs or structured outputs")

    def _structured_state(
        self,
        request: GenerationRequest,
        adapter: AdapterSpec | None,
        thinking: bool | None,
    ) -> _StructuredState:
        request_spec = request.structured_outputs
        spec = (
            request_spec
            if request_spec is not None
            else getattr(adapter, "structured_outputs", None)
        )
        if not spec:
            return _StructuredState()
        if thinking:
            if self.config.reasoning_parser is None:
                raise PromptError(
                    "structured outputs with thinking require a configured reasoning parser"
                )
            if request.messages is None:
                raise PromptError("structured outputs with thinking require chat messages")
        from vllm.sampling_params import StructuredOutputsParams

        try:
            params = StructuredOutputsParams(**spec)
        except (TypeError, ValueError) as exc:
            raise PromptError(f"invalid structured outputs spec {spec!r}: {exc}") from exc
        parser_kwargs = None
        if self.config.reasoning_parser is not None:
            parser_kwargs = {
                "chat_template_kwargs": effective_chat_template_kwargs(request, thinking)
            }
        return _StructuredState(
            params=params,
            reasoning_ended=not bool(thinking),
            parser_kwargs=parser_kwargs,
        )

    @staticmethod
    def _sampling_params(
        request: GenerationRequest,
        structured: _StructuredState,
        *,
        streaming: bool,
    ) -> Any:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        output_kind = RequestOutputKind.DELTA if streaming else RequestOutputKind.FINAL_ONLY
        return SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            n=request.n,
            seed=request.seed,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logprobs=request.top_logprobs if request.logprobs else None,
            output_kind=output_kind,
            structured_outputs=structured.params,
            # pass none rather than an empty list so vllm keeps its own default.
            stop=list(request.stop) or None,
        )

    async def _prepare_prompt(
        self,
        request: GenerationRequest,
        thinking: bool | None,
    ) -> PreparedPrompt:
        assert self._prompts is not None
        return await self._prompts.prepare(request, thinking)

    def _generate_stream(
        self,
        prompt: PreparedPrompt,
        sampling: Any,
        request_id: str,
        lora_request: Any,
        structured: _StructuredState,
    ) -> AsyncIterator[Any]:
        assert self._engine is not None
        return self._engine.generate(
            prompt.value,
            sampling,
            request_id,
            lora_request=lora_request,
            reasoning_ended=structured.reasoning_ended,
            reasoning_parser_kwargs=structured.parser_kwargs,
        )

    async def _first_stream_output(self, output_stream: AsyncIterator[Any]) -> Any:
        try:
            return await anext(output_stream)
        except StopAsyncIteration as exc:
            raise RuntimeNotReadyError("vllm returned no output") from exc

    def _finished_event(
        self,
        request: GenerationRequest,
        adapter: AdapterSpec | None,
        thinking: bool | None,
        request_id: str,
        state: _StreamState,
    ) -> StreamFinished:
        state.validate_complete()
        assert state.prompt_tokens is not None
        return StreamFinished(
            request_id=request_id,
            runtime_id=self.runtime_id,
            adapter_id=request.adapter_id,
            incarnation=adapter.incarnation if adapter is not None else None,
            choices=tuple(
                GenerationChoice(
                    index=index,
                    text=choice.text,
                    finish_reason=choice.finish_reason,
                    token_ids=tuple(choice.token_ids),
                    logprobs=choice.logprobs or None,
                    tool_calls=choice.tool_calls,
                )
                for index, choice in sorted(state.choices.items())
            ),
            prompt_tokens=state.prompt_tokens,
            completion_tokens=sum(len(choice.token_ids) for choice in state.choices.values()),
            cached_tokens=state.cached_tokens,
            cached_tokens_reported=state.cached_reported,
            thinking=thinking,
        )

    @staticmethod
    async def _close_output_stream(output_stream: AsyncIterator[Any]) -> None:
        close = getattr(output_stream, "aclose", None)
        if close is None:
            return
        active_exception = sys.exc_info()[0] is not None
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            if not active_exception:
                raise

    def _engine_is_dead(self) -> bool:
        return bool(self._engine is not None and getattr(self._engine, "errored", False))

    async def _notify_if_dead(self) -> None:
        if self._engine_is_dead():
            await self._notify_engine_death()

    async def _notify_engine_death(self) -> bool:
        if self._death_notified or not self._engine_is_dead():
            return self._death_notified
        async with self._death_notification_lock:
            if self._death_notified or not self._engine_is_dead():
                return self._death_notified
            callback = self._on_engine_death
            if callback is None:
                self._death_notified = True
                return True
            try:
                result = callback(self.health())
                if inspect.isawaitable(result):
                    await result
            except Exception:
                return False
            self._death_notified = True
            return True

    async def _liveness_monitor(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.config.liveness_interval_seconds)
                if self._engine_is_dead() and await self._notify_engine_death():
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            return
