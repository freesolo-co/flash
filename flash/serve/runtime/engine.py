"""lazy vllm engine lifecycle, generation, streaming, and health."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

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
from .types import (
    AdapterSpec,
    EngineConfig,
    GenerationRequest,
    GenerationResult,
    RuntimeHealth,
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
class _StreamState:
    token_ids: list[int]
    text: str = ""
    final_output: Any = None
    prompt_tokens: int | None = None
    cached_tokens: int = 0
    cached_reported: bool = False

    def consume(self, request_output: Any) -> str:
        self.final_output = request_output
        # under delta output vllm reports prompt and cache metadata on the first delta only, so
        # keep the first values seen instead of reading them off the last one.
        if self.prompt_tokens is None:
            self.prompt_tokens = _optional_prompt_tokens(request_output)
        if not self.cached_reported:
            self.cached_tokens, self.cached_reported = _cached_token_state(request_output)
        output = _single_sequence(request_output)
        chunk_ids = [int(value) for value in (getattr(output, "token_ids", None) or [])]
        self.token_ids.extend(chunk_ids)
        chunk_text = str(getattr(output, "text", "") or "")
        self.text += chunk_text
        return chunk_text


def _single_sequence(request_output: Any) -> Any:
    outputs = getattr(request_output, "outputs", None)
    if not isinstance(outputs, list | tuple) or len(outputs) != 1:
        raise RuntimeNotReadyError("vllm must return exactly one output sequence")
    return outputs[0]


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


def _argument_names(argument_type: Any) -> set[str]:
    try:
        return {field.name for field in dataclasses.fields(argument_type)}
    except TypeError:
        try:
            return set(inspect.signature(argument_type).parameters)
        except (TypeError, ValueError):
            return set()


def _require_reasoning_compatibility(
    async_engine_args_type: Any,
    generate: Any,
    reasoning_parser: str | None,
) -> None:
    if reasoning_parser is None:
        return
    engine_args = _argument_names(async_engine_args_type)
    try:
        generate_args = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        generate_args = {}
    missing = [
        name
        for name, available in (
            ("reasoning_parser", "reasoning_parser" in engine_args),
            ("reasoning_ended", "reasoning_ended" in generate_args),
            ("reasoning_parser_kwargs", "reasoning_parser_kwargs" in generate_args),
        )
        if not available
    ]
    if missing:
        raise RuntimeNotReadyError("vllm reasoning api is missing " + ", ".join(missing))


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
        started = time.perf_counter()
        async with self._binding(request) as binding:
            adapter = binding.spec if binding is not None else None
            lora_request = binding.lora_request if binding is not None else None
            thinking = resolve_thinking(request, adapter)
            structured = self._structured_state(request, adapter, thinking)
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
        sequence = _single_sequence(final_output)
        token_ids = tuple(int(value) for value in (getattr(sequence, "token_ids", None) or []))
        cached_tokens, cached_reported = _cached_token_state(final_output)
        return GenerationResult(
            request_id=request_id,
            runtime_id=self.runtime_id,
            adapter_id=request.adapter_id,
            incarnation=adapter.incarnation if adapter is not None else None,
            text=str(getattr(sequence, "text", "") or ""),
            finish_reason=getattr(sequence, "finish_reason", None),
            token_ids=token_ids,
            prompt_tokens=_num_prompt_tokens(final_output),
            completion_tokens=len(token_ids),
            cached_tokens=cached_tokens,
            cached_tokens_reported=cached_reported,
            duration_seconds=time.perf_counter() - started,
            thinking=thinking,
        )

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamEvent]:
        """yield ready, normalized delta, and terminal accounting events."""
        await self._ensure_started()
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        async with self._binding(request) as binding:
            adapter = binding.spec if binding is not None else None
            lora_request = binding.lora_request if binding is not None else None
            thinking = resolve_thinking(request, adapter)
            structured = self._structured_state(request, adapter, thinking)
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
                state = _StreamState(token_ids=[])
                output = first_output
                while True:
                    delta = state.consume(output)
                    if delta:
                        yield StreamDelta(text=delta)
                    try:
                        output = await anext(output_stream)
                    except StopAsyncIteration:
                        break
                yield self._finished_event(request, adapter, thinking, request_id, started, state)
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
        started: float,
        state: _StreamState,
    ) -> StreamFinished:
        if state.final_output is None:
            raise RuntimeNotReadyError("vllm returned no output")
        sequence = _single_sequence(state.final_output)
        if state.prompt_tokens is None:
            raise RuntimeNotReadyError("vllm did not report the expanded prompt token count")
        return StreamFinished(
            request_id=request_id,
            runtime_id=self.runtime_id,
            adapter_id=request.adapter_id,
            incarnation=adapter.incarnation if adapter is not None else None,
            text=state.text,
            finish_reason=getattr(sequence, "finish_reason", None),
            prompt_tokens=state.prompt_tokens,
            completion_tokens=len(state.token_ids),
            cached_tokens=state.cached_tokens,
            cached_tokens_reported=state.cached_reported,
            duration_seconds=time.perf_counter() - started,
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
