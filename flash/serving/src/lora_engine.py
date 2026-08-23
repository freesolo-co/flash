"""The vLLM multi-LoRA engine implementation for the Modal serving app, plus its pure helpers.

``_LoraEngineImpl`` is the non-Modal base that ``modal_app._build_engine`` wraps per GPU tier.
This stays import-light: importing it triggers no Modal registration, and vllm/transformers remain
lazy inside engine methods; only pure serving support modules are imported at module scope.
"""

import asyncio
import hashlib
import inspect
import json
import os
import shutil
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Adapter-cache paths and token accounting live in engine_support.py, alongside
# _RESERVED_CHAT_TEMPLATE_KWARGS (the apply_chat_template args a caller must never re-supply) and
# the vllm build probes engine_boot uses.
from flash.serving.src.engine_support import (
    _adapter_cache_path,
    _adapter_cache_ready,
    _adapter_source_cache_dir,
    _adapter_source_ident,
    _assert_source_cache_containment,
    _cached_tokens_reported,
    _engine_is_dead,
    _load_adapters_for_base,
    _num_cached_tokens,
    _num_prompt_tokens,
    _safe_chat_template_kwargs,
    _stream_text_delta,
    active_checkpoint_ref,
    enforce_expected_checkpoint,
)
from flash.serving.src.lora_entries import _LoraEntry, cached_lora_request, entries_for
from flash.serving.src.model_config import (
    engine_overrides_for,
    gpu_for,
    image_limit_for,
    supports_image_input,
)


def _stream_usage_fields(
    request_output: Any,
    completion_tokens: int,
    *,
    start: float,
    request_id: str,
    engine_replica_id: str,
    checkpoint: str,
) -> dict[str, Any]:
    return {
        "prompt_tokens": _num_prompt_tokens(request_output),
        "completion_tokens": completion_tokens,
        "cached_tokens": _num_cached_tokens(request_output),
        "cached_tokens_reported": _cached_tokens_reported(request_output),
        "inference_time_seconds": time.time() - start,
        "request_id": request_id,
        "engine_replica_id": engine_replica_id,
        "checkpoint": checkpoint,
    }


class _LoraEngineImpl:
    """Implementation of one vLLM multi-LoRA engine for a base model.

    Plain base class: the GPU is chosen per base model (see ``model_config.gpu_for``), and Modal
    fixes a class's GPU at decoration time, so ``_build_engine`` wraps this in one ``@app.cls`` per
    distinct GPU tier and the router dispatches each base model to its tier's class. The Modal
    entrypoints (load/register/generate/stream_generate/unregister/health) live on the thin per-tier
    subclass and forward to the ``_``-prefixed methods here."""

    base_model: str  # set by the per-GPU Modal subclass via modal.parameter()

    def _replica_identifier(self) -> str:
        replica_id = getattr(self, "_replica_id", None)
        if replica_id is None:
            replica_id = uuid.uuid4().hex
            self._replica_id = replica_id
        return replica_id

    async def _load(self) -> None:
        # async so the cached-LoRA preload (and the engine's first async use) run on the SAME event
        # loop that serves requests. A one-off asyncio.run() here would bind vLLM's AsyncLLMEngine
        # to a loop that's then closed, breaking generate() on the real serving loop.
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        from flash.serving.src import settings as cfg
        from flash.serving.src.engine_boot import (
            engine_args_for,
            load_tokenizer,
            pin_loras_default,
        )
        from flash.serving.src.model_config import engine_overrides_for
        from flash.serving.src.registry import AdapterRegistry
        from flash.serving.src.settings import ADAPTER_CACHE_DIR, Settings

        self._replica_id = uuid.uuid4().hex
        self.settings = Settings()
        self.registry = AdapterRegistry()
        self._adapter_locks: dict[str, asyncio.Lock] = {}
        self._adapter_locks_guard = asyncio.Lock()
        self._source_locks: dict[tuple[str, str, str, str | None], asyncio.Lock] = {}
        self._source_locks_guard = asyncio.Lock()
        self._source_paths: dict[tuple[str, str, str, str | None], Path] = {}
        self._lora_entries: dict[str, _LoraEntry] = {}
        self._prompt_token_cache: OrderedDict[tuple[str, str], tuple[int, ...]] = OrderedDict()
        self._prompt_cache_size = cfg.PROMPT_TOKEN_CACHE_SIZE
        base_model_adapters = _load_adapters_for_base(self.settings, self.base_model)
        self.registry.hydrate(base_model_adapters)
        ADAPTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self.processor, self.tokenizer = load_tokenizer(self.base_model, self.settings, cfg)

        # Per-base-model engine-arg overrides. Larger pre-quantized tiers use real-GPU-validated
        # rank-32 LoRA sizing and, where needed, tighter scheduler/memory caps.
        overrides = engine_overrides_for(self.base_model)
        self._pin_loras = pin_loras_default(overrides, cfg)
        kwargs = engine_args_for(self.base_model, overrides, cfg)
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

    async def _adapter_lock(self, adapter_id: str) -> asyncio.Lock:
        async with self._adapter_locks_guard:
            lock = self._adapter_locks.get(adapter_id)
            if lock is None:
                lock = asyncio.Lock()
                self._adapter_locks[adapter_id] = lock
            return lock

    async def _source_lock(self, record: Any) -> asyncio.Lock:
        ident = _adapter_source_ident(record)
        async with self._source_locks_guard:
            lock = self._source_locks.get(ident)
            if lock is None:
                lock = asyncio.Lock()
                self._source_locks[ident] = lock
            return lock

    def _entries(self) -> dict[str, _LoraEntry]:
        return entries_for(self)

    async def _evict_loaded_lora(self, adapter_id: str) -> None:
        entries = self._entries()
        if (entry := entries.get(adapter_id)) is None:
            return
        if entry.state == "reserved":
            entries.pop(adapter_id)
            return
        entries[adapter_id] = _LoraEntry(entry.source_ident, entry.lora_request, "unconfirmed")
        remove = getattr(self.engine, "remove_lora", None)
        if remove is None:
            raise RuntimeError("vLLM cannot confirm LoRA removal")
        result = remove(entry.lora_request.lora_int_id)
        if inspect.isawaitable(result):
            result = await result
        if result is False:
            raise RuntimeError("vLLM rejected LoRA removal")
        entries.pop(adapter_id)

    async def _pin_lora(self, lora_request: Any) -> None:
        pin = getattr(self.engine, "pin_lora", None)
        if pin is None:
            return
        result = pin(lora_request.lora_int_id)
        if inspect.isawaitable(result):
            await result

    async def _add_lora_locked(self, record: Any, path: Path) -> None:
        adapter_id = record.adapter_id
        entries = self._entries()
        lora_request = self._cached_lora_request_locked(record, path)
        was_reserved = entries[adapter_id].state == "reserved"
        try:
            added = await self.engine.add_lora(lora_request)
        except Exception:
            if was_reserved:
                entries.pop(adapter_id)
            raise
        if added is False and was_reserved:
            entries.pop(adapter_id)
            raise RuntimeError("vLLM rejected a new LoRA registration")
        entry = entries[adapter_id]
        entries[adapter_id] = _LoraEntry(entry.source_ident, lora_request, "loaded")
        if self._pin_loras:  # capped pools stay unpinned so surplus adapters can lru-swap
            await self._pin_lora(lora_request)

    async def _preload_cached_loras(self) -> None:
        from flash.serving.src.settings import ADAPTER_CACHE_DIR

        for record in self.registry.list_ready():
            source_ident = _adapter_source_ident(record)
            local_dir = _adapter_source_cache_dir(ADAPTER_CACHE_DIR, record)
            subfolder = getattr(record, "subfolder", None)
            path = _adapter_cache_path(local_dir, subfolder)
            if not _adapter_cache_ready(path):
                continue
            lock = await self._adapter_lock(record.adapter_id)
            async with lock:
                self._source_paths[source_ident] = path
                self.registry.set_local_path(record, path)
                try:
                    await self._add_lora_locked(record, path)
                except Exception as exc:  # a bad cached LoRA must not kill startup
                    print(
                        f"cached LoRA preload skipped for {record.adapter_id}: {exc!r}",
                        flush=True,
                    )

    async def _ensure_adapter_local_locked(self, record: Any) -> Path:
        # Download body; caller must already hold self._adapter_lock(record.adapter_id).
        import anyio
        from huggingface_hub import snapshot_download

        from flash.serving.src.settings import ADAPTER_CACHE_DIR

        adapter_id = record.adapter_id
        local_dir = _adapter_source_cache_dir(ADAPTER_CACHE_DIR, record)
        subfolder = getattr(record, "subfolder", None)
        cached_path = _adapter_cache_path(local_dir, subfolder)
        # Stale cached path (source changed) -> evict the old LoRA before re-downloading; check
        # before local_path(), which clears the stale entry.
        if self.registry.local_path_is_stale(record):
            # eviction releases this adapter's single lifecycle entry after confirmed removal.
            await self._evict_loaded_lora(adapter_id)
        path = self.registry.local_path(record)
        if path is not None:
            _assert_source_cache_containment(local_dir, path)
            # TTFT: this is the STEADY-STATE hit for an already-downloaded adapter, on the path of
            # every generation. _adapter_cache_ready stats + iterdirs the adapter directory, and here
            # that directory lives on a NETWORK-backed Modal Volume, so the syscalls can block far
            # longer than the local-disk floor. Off the event loop it can't stall co-resident requests.
            if await asyncio.to_thread(_adapter_cache_ready, path):
                return path

        source_ident = _adapter_source_ident(record)
        source_lock = await self._source_lock(record)
        async with source_lock:
            path = self.registry.local_path(record)
            if path is not None:
                _assert_source_cache_containment(local_dir, path)
                if _adapter_cache_ready(path):
                    return path
            if cached := self._source_paths.get(source_ident):
                _assert_source_cache_containment(local_dir, cached)
                if _adapter_cache_ready(cached):
                    self.registry.set_local_path(record, cached)
                    return cached
                self._source_paths.pop(source_ident, None)

            local_dir.parent.mkdir(parents=True, exist_ok=True)
            repo_type = getattr(record, "repo_type", "model") or "model"
            allow = [f"{subfolder}/**", f"{subfolder}/*"] if subfolder else None
            if _adapter_cache_ready(cached_path):
                self._source_paths[source_ident] = cached_path
                self.registry.set_local_path(record, cached_path)
                return cached_path
            if (cached_path / "adapter_config.json").exists():
                shutil.rmtree(local_dir, ignore_errors=True)

            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    downloaded = await anyio.to_thread.run_sync(
                        lambda: snapshot_download(
                            repo_id=record.repo_id,
                            repo_type=repo_type,
                            revision=record.hf_revision,
                            local_dir=str(local_dir),
                            token=self.settings.hf_api_key,
                            allow_patterns=allow,
                        )
                    )
                    downloaded_root = Path(downloaded)
                    # Cheap local stat guarding a path escape, on a path the download above
                    # already materialized; not worth an extra thread hop.
                    if downloaded_root.resolve() != local_dir.resolve():  # noqa: ASYNC240
                        raise RuntimeError("snapshot download escaped its exact-SHA source cache")
                    path = _adapter_cache_path(local_dir, subfolder)
                    if not _adapter_cache_ready(path):
                        raise RuntimeError(
                            f"downloaded adapter cache is incomplete: {path} has no "
                            "non-empty adapter_model tensor file"
                        )
                    self._source_paths[source_ident] = path
                    self.registry.set_local_path(record, path)
                    return path
                except Exception as exc:  # Hub/network errors are often transient
                    last_exc = exc
                    if attempt == 2:
                        break
                    await asyncio.sleep(0.5 * (2**attempt))

            assert last_exc is not None
            raise last_exc

    def _cached_lora_request_locked(self, record: Any, path: Path) -> Any:
        return cached_lora_request(self, record, path)

    async def _lora_request(
        self, adapter_id: str, record_dict: dict[str, Any] | None = None
    ) -> tuple[Any, Any]:
        """Resolve (LoRARequest, record) for ``adapter_id`` under the adapter lock.

        Returns the RESOLVED record alongside the request so the caller can bind the prompt's
        thinking default to the SAME record the weights came from — a later registry re-read could
        observe a different record after a concurrent same-id redeploy (see
        ``_effective_chat_template_kwargs``).
        """
        from flash.serving.src.schemas import AdapterRecord

        # Lock across read + download so a concurrent unregister can't slip in between.
        lock = await self._adapter_lock(adapter_id)
        async with lock:
            if record_dict is not None:
                self.registry.upsert(AdapterRecord.model_validate(record_dict))
            record = self.registry.get(adapter_id)
            if record is None or record.status != "ready":
                raise ValueError(f"Unknown adapter id on {self.base_model}: {adapter_id}")
            if not record.serve_base_model and not record.is_revision:
                raise ValueError(f"Unknown adapter id on {self.base_model}: {adapter_id}")
            if record.serve_base_model:
                # No LoRA to resolve: generate against the base weights the engine already has.
                return None, record
            path = await self._ensure_adapter_local_locked(record)
            return self._cached_lora_request_locked(record, path), record

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
        """Sanitized caller chat_template_kwargs, with ``enable_thinking`` fixed per-adapter.

        ``enable_thinking`` is not a caller knob: always inject the adapter's trained value even if
        the caller tried to supply an override. This is the parity fix: without it the template can
        run in a mode other than the adapter's training mode, so a thinking=false adapter may emit a
        reasoning preamble ("…</think>{json}") for callers that omit or override the flag.

        ``thinking_default`` is the value RESOLVED alongside the LoRA weights (carried out of
        ``_lora_request`` under its adapter lock), so prompt rendering stays bound to the same
        record as the weights during same-id redeploys.
        """
        ctk = _safe_chat_template_kwargs(getattr(payload, "chat_template_kwargs", None))
        if not isinstance(thinking_default, bool):
            raise ValueError("adapter thinking default is required")
        return {**ctk, "enable_thinking": thinking_default}

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
                payload.messages,
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
        from flash.serving.src.multimodal import (
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
                    template_messages,
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

    async def _register(
        self,
        record_dict: dict[str, Any],
        deployment_generation: str | None = None,
    ) -> dict[str, Any]:
        """Download + register an adapter into this engine's cache."""
        from flash.serving.src.schemas import AdapterRecord

        record = AdapterRecord.model_validate(record_dict).model_copy(
            update={"deployment_generation": deployment_generation}
        )
        if not record.serve_base_model and not record.is_revision:
            raise ValueError("only immutable adapter revisions can be registered")
        lock = await self._adapter_lock(record.adapter_id)
        async with lock:  # _locked variant: we hold the lock (the public one would deadlock)
            if record.serve_base_model:
                # No LoRA to download or add — the base weights are already loaded; just track the id.
                self.registry.upsert(record, revive=True)
                return {"ok": True, "adapter_id": record.adapter_id, "base_model": self.base_model}
            path = await self._ensure_adapter_local_locked(record)
            await self._add_lora_locked(record, path)
            self.registry.upsert(record, revive=True)
        return {"ok": True, "adapter_id": record.adapter_id, "base_model": self.base_model}

    @staticmethod
    def _lora_request_attestation(record: Any, lora_request: Any) -> str | None:
        """Attest that the engine resolved the exact immutable adapter that was asked for.

        A mutable alias may legitimately resolve to whatever it currently points at, so only a
        revision is attested. Returning the resolved name lets the router prove the weights it
        billed for came from the requested revision rather than trusting the id it sent.
        """
        if not record.is_revision:
            return None
        if lora_request is None:
            raise RuntimeError("immutable adapter resolved without a LoRARequest")
        if lora_request.lora_name != record.adapter_id:
            raise RuntimeError("immutable adapter resolved to a mismatched LoRARequest")
        return lora_request.lora_name

    def _active_checkpoint_ref(self, record: Any) -> str:
        return active_checkpoint_ref(record)

    def _enforce_expected_checkpoint(self, record: Any, expected_checkpoint: str | None) -> str:
        return enforce_expected_checkpoint(record, expected_checkpoint)

    async def _generate(
        self,
        payload_dict: dict[str, Any],
        record_dict: dict[str, Any] | None = None,
        expected_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        from flash.serving.src.schemas import GenerateRequest

        payload = GenerateRequest.model_validate(payload_dict)
        # Modal may route here to a container that never saw the registration -> adopt the forwarded
        # record (revive=False, so it can't resurrect an id just undeployed here). Carry the resolved
        # record so the prompt's thinking default binds to the SAME record the weights came from.
        lora_request, record = await self._lora_request(payload.adapter_id, record_dict)
        lora_request_attestation = self._lora_request_attestation(record, lora_request)
        active_checkpoint = self._enforce_expected_checkpoint(record, expected_checkpoint)
        thinking_default = self._thinking_default(record, payload)
        structured_outputs, reasoning_ended, reasoning_parser_kwargs = (
            self._structured_outputs_state(payload, record, thinking_default)
        )
        # FINAL_ONLY: the non-streaming caller only wants the completed text, so tell vLLM to emit a
        # single terminal RequestOutput instead of one cumulative object per decoded token. Saves N
        # Python-object allocations + detokenizations per request on the hot path.
        sampling_params = SamplingParams(
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            top_p=payload.top_p,
            output_kind=RequestOutputKind.FINAL_ONLY,
            structured_outputs=structured_outputs,
            stop=payload.stop,
        )
        request_id = str(uuid.uuid4())
        start = time.time()
        final_output = None
        prompt_input = await self._prepare_prompt_input(payload, thinking_default)
        try:
            async for out in self.engine.generate(
                prompt_input,
                sampling_params,
                request_id,
                lora_request=lora_request,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=reasoning_parser_kwargs,
            ):
                final_output = out
        except Exception:
            # A dead EngineCore raises here (EngineDeadError). Recycle this container immediately so
            # the NEXT request lands on a fresh engine instead of looping 500s on a corpse (the
            # background liveness monitor is the backstop; this makes the common path instant).
            self._self_heal_if_dead("generate")
            raise
        finally:
            self._close_prompt_images(prompt_input)
        if final_output is None:
            raise RuntimeError("vLLM returned no output")
        output = final_output.outputs[0]
        finish_reason = getattr(output, "finish_reason", None)
        if finish_reason is None:
            raise RuntimeError("vLLM generation ended without a finish reason")
        completion_token_ids = list(getattr(output, "token_ids", []) or [])
        prompt_tokens = _num_prompt_tokens(final_output)
        return {
            "ok": True,
            "adapter_id": payload.adapter_id,
            **(
                {"lora_request_adapter": lora_request_attestation}
                if lora_request_attestation is not None
                else {}
            ),
            "text": output.text,
            "finish_reason": finish_reason,
            "token_ids": completion_token_ids,
            # token counts for per-token billing; the router forwards these to the backend.
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(completion_token_ids),
            # Prefix-cached prompt tokens (served from the KV cache, prefill skipped). With
            # prefix caching ON this is non-zero whenever a prompt shares a prefix with an
            # earlier one; the backend bills these at a discount. 0 on a build that omits it.
            "cached_tokens": _num_cached_tokens(final_output),
            "cached_tokens_reported": _cached_tokens_reported(final_output),
            "inference_time_seconds": time.time() - start,
            # stable per-generation id: the router uses it as the usage-report idempotency key so
            # a future report retry can't double-bill the same generation.
            "request_id": request_id,
            "engine_replica_id": self._replica_identifier(),
            "checkpoint": active_checkpoint,
            # the thinking mode this generation was RENDERED with. the chat template opens the
            # reasoning block in the prompt, so a thinking completion carries only the closing
            # </think> and cannot be classified from its own text. the openai layer needs that
            # distinction to split reasoning from content; the raw /generate contract keeps
            # ``text`` verbatim.
            "thinking": thinking_default,
        }

    async def _stream_generate(
        self,
        payload_dict: dict[str, Any],
        record_dict: dict[str, Any] | None = None,
        expected_checkpoint: str | None = None,
    ):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        from flash.serving.src.schemas import GenerateRequest

        payload = GenerateRequest.model_validate(payload_dict)
        lora_request, record = await self._lora_request(payload.adapter_id, record_dict)
        # streaming sends its headers before the first token, so there is no response field left to
        # carry an attestation. the check still runs for its raise: a mismatched immutable adapter
        # fails here, before any token is emitted, rather than streaming the wrong weights.
        self._lora_request_attestation(record, lora_request)
        active_checkpoint = self._enforce_expected_checkpoint(record, expected_checkpoint)
        # resolve structured outputs and advance vllm before the ready event so validation failures
        # remain clean responses instead of surfacing after streaming has started.
        thinking_default = self._thinking_default(record, payload)
        structured_outputs, reasoning_ended, reasoning_parser_kwargs = (
            self._structured_outputs_state(payload, record, thinking_default)
        )
        sampling_params = SamplingParams(
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            top_p=payload.top_p,
            output_kind=RequestOutputKind.DELTA,
            structured_outputs=structured_outputs,
            stop=payload.stop,
        )
        request_id = str(uuid.uuid4())
        start = time.time()
        prompt_input = await self._prepare_prompt_input(payload, thinking_default)
        output_stream = None
        try:
            try:
                output_stream = self.engine.generate(
                    prompt_input,
                    sampling_params,
                    request_id,
                    lora_request=lora_request,
                    reasoning_ended=reasoning_ended,
                    reasoning_parser_kwargs=reasoning_parser_kwargs,
                )
                first_output = await anext(output_stream)
            except StopAsyncIteration as exc:
                raise RuntimeError("vLLM returned no output") from exc
            except Exception:
                self._self_heal_if_dead("stream_generate")
                raise

            completion_token_ids: list[int] = []
            first_completion_tokens = len(
                list(getattr(first_output.outputs[0], "token_ids", []) or [])
            )
            usage_context = {
                "start": start,
                "request_id": request_id,
                "engine_replica_id": self._replica_identifier(),
                "checkpoint": active_checkpoint,
            }

            def usage_fields(request_output: Any) -> dict[str, Any]:
                return _stream_usage_fields(
                    request_output, len(completion_token_ids), **usage_context
                )

            # ``thinking`` rides the ready event because it must be known before the first delta.
            # usage rides it too because vllm has already produced its first output, and a client may
            # disconnect before the first text delta reaches the front door.
            yield {
                "type": "ready",
                "thinking": thinking_default,
                **_stream_usage_fields(first_output, first_completion_tokens, **usage_context),
            }
            final_output = None
            previous_text = ""
            out = first_output
            try:
                while True:
                    final_output = out
                    output = out.outputs[0]
                    text = output.text or ""
                    token_ids = list(getattr(output, "token_ids", []) or [])
                    if token_ids:
                        # _stream_generate pins delta output, so each chunk is new tokens appended
                        # without inspecting contents; test_serve_structured_outputs.py pins the
                        # sampling-param output kind that enforces this contract.
                        completion_token_ids.extend(token_ids)
                    delta = ""
                    if text:
                        delta, previous_text = _stream_text_delta(
                            text, previous_text, cumulative_output=False
                        )
                    yield {
                        "type": "delta",
                        "text": delta,
                        **usage_fields(out),
                    }
                    try:
                        out = await anext(output_stream)
                    except StopAsyncIteration:
                        break
            except Exception:
                self._self_heal_if_dead("stream_generate")
                raise
            if final_output is None:
                raise RuntimeError("vLLM returned no output")
            output = final_output.outputs[0]
            yield {
                "type": "final",
                "ok": True,
                "adapter_id": payload.adapter_id,
                "finish_reason": getattr(output, "finish_reason", None),
                **usage_fields(final_output),
                # see generate(): the rendered thinking mode, which the streamed text alone
                # cannot reveal.
                "thinking": thinking_default,
            }
        finally:
            try:
                if output_stream is not None:
                    close = getattr(output_stream, "aclose", None)
                    if close is not None:
                        active_exception = sys.exc_info()[0] is not None
                        try:
                            result = close()
                            if inspect.isawaitable(result):
                                await result
                        # only swallow ordinary close errors while already unwinding;
                        # control-flow exceptions (CancelledError, KeyboardInterrupt,
                        # SystemExit) must always propagate rather than be masked here.
                        except Exception:
                            if not active_exception:
                                raise
            finally:
                self._close_prompt_images(prompt_input)

    async def _unregister(
        self,
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> dict[str, Any]:
        # undeploy under the per-adapter lock so register and stale background cleanup have one order.
        lock = await self._adapter_lock(adapter_id)
        async with lock:
            current = self.registry.get(adapter_id)
            current_generation = current.deployment_generation if current is not None else None
            stale = current is not None and (
                current_generation is not None
                if expected_generation is None
                else current_generation != expected_generation
            )
            if stale:
                return {
                    "ok": True,
                    "removed": None,
                    "skipped_stale_generation": True,
                    "base_model": self.base_model,
                }
            self.registry.remove(adapter_id)
            await self._evict_loaded_lora(adapter_id)
        return {"ok": True, "removed": adapter_id, "base_model": self.base_model}

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

        from flash.serving.src import settings as cfg

        _ov = engine_overrides_for(self.base_model)
        _served_model = _ov.get("serve_model_id") or self.base_model
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
            # Effective weight quantization of the served checkpoint. Every base now serves a
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
