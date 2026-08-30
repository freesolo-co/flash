"""The vLLM multi-LoRA engine implementation for the Modal serving app, plus its pure helpers.

``_LoraEngineImpl`` is the non-Modal base that ``modal_app._build_engine`` wraps per GPU tier.
This stays import-light: importing it triggers no Modal registration, and vllm/transformers remain
lazy inside engine methods; only pure serving support modules are imported at module scope.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

# Adapter-cache paths and token accounting live in engine_support.py, alongside
# _RESERVED_CHAT_TEMPLATE_KWARGS (the apply_chat_template args a caller must never re-supply) and
# the vllm build probes engine_boot uses.
from flash.content.thinking import messages_for_chat_template
from flash.serve.contract.provenance import CheckpointKey, engine_adapter_name, record_key
from flash.serving.src.engine.lora_lifecycle import (
    LoraLifecycleMixin,
    ReplicaSourceCache,
    _LoraEntry,
)
from flash.serving.src.engine.model_config import (
    engine_overrides_for,
    gpu_for,
    image_limit_for,
    immutable_serving_revisions,
    supports_image_input,
    tokenizer_model_for,
)
from flash.serving.src.engine.support import (
    _engine_is_dead,
    _load_adapters_for_base,
    _replica_adapter_cache_dir,
    _safe_chat_template_kwargs,
    enforce_expected_checkpoint,
)


class _LoraEngineImpl(LoraLifecycleMixin):
    """Implementation of one vLLM multi-LoRA engine for a base model.

    Plain base class: the GPU is chosen per base model (see ``model_config.gpu_for``), and Modal
    fixes a class's GPU at decoration time, so ``_build_engine`` wraps this in one ``@app.cls`` per
    distinct GPU tier and the router dispatches each base model to its tier's class. The Modal
    entrypoints (load/register/generate/stream_generate/unregister/health) live on the thin per-tier
    subclass and forward to the ``_``-prefixed methods here."""

    base_model: str  # set by the per-GPU Modal subclass via modal.parameter()

    async def _exit(self) -> None:
        task = getattr(self, "_liveness_task", None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._source_cache_manager().close()

    async def _load(self) -> None:
        # async so the cached-LoRA preload (and the engine's first async use) run on the SAME event
        # loop that serves requests. A one-off asyncio.run() here would bind vLLM's AsyncLLMEngine
        # to a loop that's then closed, breaking generate() on the real serving loop.
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        from flash.serving.src.engine.boot import load_engine_config, pin_loras_default
        from flash.serving.src.store import settings as cfg
        from flash.serving.src.store.registry import AdapterRegistry
        from flash.serving.src.store.settings import (
            ADAPTER_CACHE_DIR,
            ADAPTER_CACHE_MAX_BYTES,
            Settings,
        )

        self._replica_id = uuid.uuid4().hex
        self._adapter_cache_dir = _replica_adapter_cache_dir(ADAPTER_CACHE_DIR, self._replica_id)
        self.settings = Settings()
        self.registry = AdapterRegistry()
        self._adapter_locks: dict[CheckpointKey, asyncio.Lock] = {}
        self._adapter_locks_guard = asyncio.Lock()
        self._source_locks: dict[tuple[str, str, str, str, str | None], asyncio.Lock] = {}
        self._source_locks_guard = asyncio.Lock()
        self._source_paths: dict[tuple[str, str, str, str, str | None], Path] = {}
        self._source_cache = ReplicaSourceCache(
            self._adapter_cache_dir,
            ADAPTER_CACHE_MAX_BYTES,
        )
        self._lora_entries: dict[CheckpointKey, _LoraEntry] = {}
        self._prompt_token_cache: OrderedDict[tuple[str, str], tuple[int, ...]] = OrderedDict()
        self._prompt_cache_size = cfg.PROMPT_TOKEN_CACHE_SIZE
        base_model_adapters = _load_adapters_for_base(self.settings, self.base_model)
        self.registry.hydrate(base_model_adapters)
        self._adapter_cache_dir.mkdir(parents=True, exist_ok=True)

        self.processor, self.tokenizer, overrides, kwargs = load_engine_config(
            self.base_model,
            self.settings,
            cfg,
        )
        self._pin_loras = pin_loras_default(overrides, cfg)
        self.reasoning_parser = kwargs.get("reasoning_parser")
        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
        # No in-engine kernel-patching hook runs here (2026-07-05 35B outage post-mortem): under
        # vLLM V1 the model executes in a SEPARATE EngineCore process, so patching this process's
        # vLLM classes never reaches the running model — and a prior attempt's extra CUDA context +
        # GPU self-tests in this process stole the post-init headroom EngineCore needs for
        # FlashInfer's first-request decode-workspace allocation, OOM-killing the 35B engine on its
        # first request. Any future in-engine kernel work must patch INSIDE EngineCore (a vLLM
        # plugin/worker extension) and be gated behind a real-GPU canary that runs a real generation,
        # not just engine warmup.
        #
        # Self-healing liveness. vLLM V1 runs the model in a SEPARATE EngineCore worker process; if
        # that process dies (e.g. an OOM under load, or a bad build that first-request-OOMs), the
        # AsyncLLM in THIS process is permanently unusable — every later request raises
        # EngineDeadError. Modal has no per-method liveness probe, so a container with a dead engine
        # would otherwise stay in rotation serving fast 500s indefinitely (exactly the 2026-07-06
        # regression: a dead 35B container looped EngineDeadError 500s until a human redeployed). A
        # background monitor detects the death and drains this container. subsequent demand starts a
        # fresh engine instead of leaving a permanently dead container in rotation.
        self._self_heal_triggered = False
        self._liveness_task = asyncio.create_task(self._liveness_monitor())
        if cfg.PRELOAD_CACHED_LORAS:
            await self._preload_cached_loras()

    def _engine_dead(self) -> bool:
        """Instance-bound view of :func:`_engine_is_dead` for the live engine."""
        return _engine_is_dead(getattr(self, "engine", None))

    def _self_heal_if_dead(self, reason: str) -> None:
        """Drain a dead engine container so later demand can start a fresh one.

        The first caller, either a failing request or the liveness monitor, wins. Later calls are
        no-ops.
        """
        if getattr(self, "_self_heal_triggered", False) or not self._engine_dead():
            return
        self._self_heal_triggered = True
        print(
            f"serving: EngineCore dead on {self.base_model} ({reason}); draining container so "
            "subsequent demand starts a healthy engine",
            flush=True,
        )
        try:
            # stop pulling new inputs and let in-flight ones return cleanly before the container exits.
            # subsequent demand starts the replacement without severing active connections.
            from modal.experimental import stop_fetching_inputs

            stop_fetching_inputs()
        except Exception:  # fall back to a hard exit if the graceful API is unavailable
            os._exit(1)

    async def _liveness_monitor(self) -> None:
        """Poll EngineCore liveness so a dead container is recycled even when no request arrives to
        surface the death. Cheap: a boolean property check every few seconds."""
        try:
            while True:
                await asyncio.sleep(5.0)
                if self._engine_dead():
                    self._self_heal_if_dead("liveness monitor")
                    return
        except asyncio.CancelledError:  # container shutting down — let it stop
            raise
        except Exception:  # a monitor bug must never take serving down
            return

    def _thinking_default(self, record: Any, payload: Any) -> bool:
        """The ``enable_thinking`` to render with. A trained LoRA forces its own value; a base-model
        serve has none, so it honors the caller's chat_template_kwargs enable_thinking when given."""
        if getattr(record, "serve_base_model", False):
            caller = _safe_chat_template_kwargs(getattr(payload, "chat_template_kwargs", None))
            override = caller.get("enable_thinking")
            if isinstance(override, bool):
                return override
        return record.thinking

    def _structured_outputs_state(
        self, payload: Any, record: Any, enable_thinking: bool
    ) -> tuple[Any, bool | None, dict[str, Any] | None]:
        """Resolve fresh request-local grammar params and reasoning state."""
        request_spec = getattr(payload, "structured_outputs", None)
        spec = request_spec if request_spec is not None else record.structured_outputs
        if not spec:
            return None, None, None

        if enable_thinking:
            if self.reasoning_parser is None:
                raise ValueError(
                    "structured outputs with thinking enabled require a parser-enabled base model"
                )
            if not getattr(payload, "messages", None):
                raise ValueError(
                    "structured outputs with thinking enabled require messages; raw prompt reasoning "
                    "state is ambiguous"
                )

        from vllm.sampling_params import StructuredOutputsParams

        try:
            params = StructuredOutputsParams(**spec)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid structured outputs spec {spec!r}: {exc}") from exc

        parser_kwargs = None
        if self.reasoning_parser is not None:
            parser_kwargs = {
                "chat_template_kwargs": self._effective_chat_template_kwargs(
                    payload, enable_thinking
                )
            }
        return params, not enable_thinking, parser_kwargs

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
        return {
            **ctk,
            "enable_thinking": thinking_default,
            "preserve_thinking": False,
        }

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
        if key is not None and cache is not None:
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                return {"prompt_token_ids": list(cached)}

        prompt_token_ids = await asyncio.to_thread(self._tokenize_prompt, payload, thinking_default)
        if key is not None and cache is not None:
            cache[key] = tuple(prompt_token_ids)
            cache.move_to_end(key)
            max_size = getattr(self, "_prompt_cache_size", 0)
            while len(cache) > max_size:
                cache.popitem(last=False)
        return {"prompt_token_ids": prompt_token_ids}

    def _tokenize_prompt(self, payload: Any, thinking_default: bool | None = None) -> list[int]:
        if payload.messages:
            # Forward sanitized chat_template_kwargs, but force enable_thinking from the adapter's
            # trained ``thinking`` value so every caller renders in the mode this LoRA was trained
            # with. Reserved/return-shape kwargs are dropped before rendering to avoid 500s.
            ctk = self._effective_chat_template_kwargs(payload, thinking_default)
            prompt_token_ids = self.tokenizer.apply_chat_template(
                messages_for_chat_template(payload.messages),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **ctk,
            )
            return list(prompt_token_ids)
        if payload.prompt:
            prompt_token_ids = self.tokenizer.encode(payload.prompt, add_special_tokens=False)
            return list(prompt_token_ids)
        raise ValueError("prompt or messages is required")

    async def _prepare_prompt_input(
        self, payload: Any, thinking_default: bool | None = None
    ) -> dict[str, Any]:
        # imported lazily so `modal deploy` (which imports this module on a pillow-free
        # runner) does not require pillow; the remote image installs pillow from pyproject.
        from flash.serving.src.io.multimodal import (
            MultimodalRequestError,
            has_image_blocks,
            prepare_multimodal_request,
        )

        if not has_image_blocks(getattr(payload, "messages", None)):
            return await self._prompt_input(payload, thinking_default)
        processor = getattr(self, "processor", None)
        if not supports_image_input(self.base_model) or processor is None:
            raise MultimodalRequestError(
                "image input requires an initialized processor on an image-capable engine"
            )
        template_messages, images = await asyncio.to_thread(
            prepare_multimodal_request,
            payload.messages,
            image_limit=image_limit_for(self.base_model),
        )
        try:
            # Stays INSIDE the try: _effective_chat_template_kwargs raises on a missing thinking
            # default, and the decoded images above must still be closed on that path.
            ctk = self._effective_chat_template_kwargs(payload, thinking_default)
            # Same head-of-line reason as the text path: rendering an image chat template is jinja
            # over the message list, and its sibling image decode above already runs off-loop.
            rendered = await asyncio.to_thread(
                lambda: processor.apply_chat_template(
                    messages_for_chat_template(template_messages),
                    tokenize=False,
                    add_generation_prompt=True,
                    **ctk,
                )
            )
        except BaseException:
            for image in images:
                image.close()
            raise
        if not isinstance(rendered, str):
            for image in images:
                image.close()
            raise MultimodalRequestError("image processor chat template did not return text")
        image_data: Any = images[0] if len(images) == 1 else images
        return {"prompt": rendered, "multi_modal_data": {"image": image_data}}

    @staticmethod
    def _close_prompt_images(prompt_input: dict[str, Any]) -> None:
        image_data = (prompt_input.get("multi_modal_data") or {}).get("image")
        images = image_data if isinstance(image_data, list) else [image_data]
        for image in images:
            close = getattr(image, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _lora_request_attestation(record: Any, lora_request: Any) -> str | None:
        """attest the exact tenant-scoped permanent checkpoint loaded by the engine.

        The opaque engine name binds the owning organization and checkpoint id. Returning the
        checkpoint lets the router verify that generation used the authorized binding rather than
        trusting only the request payload.
        """
        if not record.is_checkpoint:
            return None
        if lora_request is None:
            raise RuntimeError("immutable adapter resolved without a LoRARequest")
        if lora_request.lora_name != engine_adapter_name(*record_key(record)):
            raise RuntimeError("immutable adapter resolved to a mismatched LoRARequest")
        return record.adapter_id

    def _enforce_expected_checkpoint(self, record: Any, expected_checkpoint: str | None) -> str:
        return enforce_expected_checkpoint(record, expected_checkpoint)

    async def _generate(
        self,
        payload_dict: dict[str, Any],
        record_dict: dict[str, Any] | None = None,
        expected_checkpoint: str | None = None,
        generation_id: str | None = None,
        pre_header_dispatch_deadline: float | None = None,
        admission_queue_id: str | None = None,
        invocation_nonce: str | None = None,
    ) -> dict[str, Any]:
        from flash.serving.src.engine.dispatch import publish_admission_acknowledgement
        from flash.serving.src.engine.generation import generate

        admit = None
        if admission_queue_id is not None:
            if not generation_id or not invocation_nonce:
                raise ValueError("non-streaming admission identity is incomplete")

            async def admit() -> None:
                if pre_header_dispatch_deadline is None:
                    raise ValueError("non-streaming admission deadline is missing")
                await publish_admission_acknowledgement(
                    admission_queue_id,
                    generation_id=generation_id,
                    invocation_nonce=invocation_nonce,
                    deadline=pre_header_dispatch_deadline,
                )

        return await generate(
            self,
            payload_dict,
            record_dict,
            expected_checkpoint,
            generation_id,
            pre_header_dispatch_deadline,
            admit=admit,
        )

    async def _stream_generate(
        self,
        payload_dict: dict[str, Any],
        record_dict: dict[str, Any] | None = None,
        expected_checkpoint: str | None = None,
        generation_id: str | None = None,
        pre_header_dispatch_deadline: float | None = None,
        *,
        pre_generate_check: Callable[[], Awaitable[None]] | None = None,
    ):
        from flash.serving.src.engine.generation import stream_generate

        stream = stream_generate(
            self,
            payload_dict,
            record_dict,
            expected_checkpoint,
            generation_id,
            pre_header_dispatch_deadline,
            pre_generate_check=pre_generate_check,
        )
        try:
            async for event in stream:
                yield event
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    def _health(self) -> dict[str, Any]:
        cuda_available: bool | None = None
        device_name: str | None = None
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
        except Exception:  # health should still return if torch probing fails
            pass

        from flash.serving.src.store import settings as cfg

        _ov = engine_overrides_for(self.base_model)
        _served_model = _ov.get("serve_model_id") or self.base_model
        _revisions = immutable_serving_revisions(self.base_model)
        _tokenizer_model = tokenizer_model_for(self.base_model)
        _processor_model = _tokenizer_model if supports_image_input(self.base_model) else None
        _immutable_identity = (
            {
                "model": {
                    "repo": _served_model,
                    "revision": _revisions.get("model_revision"),
                },
                "tokenizer": {
                    "repo": _tokenizer_model,
                    "revision": _revisions.get("tokenizer_revision"),
                },
                "processor": (
                    {
                        "repo": _processor_model,
                        "revision": _revisions.get("processor_revision"),
                    }
                    if _processor_model is not None
                    else None
                ),
            }
            if _revisions
            else None
        )
        # Report real EngineCore liveness, not a static True: a V1 engine whose worker process died
        # still runs this container, so an unconditional ok:True masked the dead 35B tier during the
        # 2026-07-06 outage. ``start_all``/monitoring can now see the death, and the liveness monitor
        # recycles the container regardless. Read the engine directly (not via ``self._engine_dead``)
        # so ``_health`` stays callable on a degraded/partial ``self``.
        engine_dead = _engine_is_dead(getattr(self, "engine", None))
        return {
            "ok": not engine_dead,
            "engine_dead": engine_dead,
            "base_model": self.base_model,
            # The checkpoint this engine actually loaded: the pre-quantized serve_model_id (owned FP8,
            # or the official 35B FP8). Surfaced so ops can confirm which weights came up.
            "served_model": _served_model,
            "immutable_identity": _immutable_identity,
            # effective weight quantization of the served checkpoint. every base now serves a
            # pre-quantized FP8 checkpoint (vLLM auto-detects, so the engine arg is None).
            "quantization": (
                "fp8" if _ov.get("serve_model_id") else _ov.get("quantization", cfg.QUANTIZATION)
            ),
            # KV-cache dtype this engine was built with (FP8 for every base; a model may override).
            "kv_cache_dtype": _ov.get("kv_cache_dtype", cfg.KV_CACHE_DTYPE),
            "adapters": len(self.registry.list_ready()),
            # The GPU tier this engine class was ACTUALLY pinned to in _build_engine (a class
            # attribute on the per-tier subclass), preferred over re-deriving it from the base model
            # name so misrouting onto the wrong tier's class shows up here. Falls back to the
            # expected tier for the bare _LoraEngineImpl (no pinned class), which has no fixed GPU.
            "configured_gpu": getattr(self, "pinned_gpu", None) or gpu_for(self.base_model),
            "cuda_available": cuda_available,
            "device_name": device_name,
            "enable_prefix_caching": cfg.ENABLE_PREFIX_CACHING,
            "prompt_token_cache_size": getattr(
                self, "_prompt_cache_size", cfg.PROMPT_TOKEN_CACHE_SIZE
            ),
            "prompt_token_cache_entries": len(getattr(self, "_prompt_token_cache", {})),
            # Report the EFFECTIVE context limit this engine was built with, not the global default:
            # a per-model override (every tier now pins 32k) must be reflected here, or
            # health/monitoring misreports the limit vLLM actually serves.
            "max_model_len": _ov.get("max_model_len", cfg.MAX_MODEL_LEN),
        }
