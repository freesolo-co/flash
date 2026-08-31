"""The vLLM multi-LoRA engine implementation for the Modal serving app, plus its pure helpers.

``_LoraEngineImpl`` is the non-Modal base that ``modal_app._build_engine`` wraps per GPU tier.
This stays import-light: importing it triggers no Modal registration, and vllm/transformers remain
lazy inside engine methods; only pure serving support modules are imported at module scope.
"""

import asyncio
import hashlib
import inspect
import json
import math
import os
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Adapter-cache paths and token accounting live in engine_support.py, alongside
# _RESERVED_CHAT_TEMPLATE_KWARGS (the apply_chat_template args a caller must never re-supply) and
# the vllm build probes engine_boot uses.
from flash.content.thinking import messages_for_chat_template
from flash.serve.app.progress import boot_elapsed_seconds
from flash.serve.contract.provenance import engine_adapter_name, record_key
from flash.serve.request.tool_calls import (
    detached_template_messages,
    normalize_tools,
    tools_active,
    tools_wire,
)
from flash.serving.src.engine.lora_entries import _LoraEntry, cached_lora_request, entries_for
from flash.serving.src.engine.model_config import (
    engine_overrides_for,
    gpu_for,
    image_limit_for,
    immutable_serving_revisions,
    supports_image_input,
    tokenizer_model_for,
    tool_parser_for,
)
from flash.serving.src.engine.support import (
    _adapter_cache_path,
    _adapter_cache_ready,
    _adapter_source_cache_dir,
    _adapter_source_ident,
    _assert_source_cache_containment,
    _engine_is_dead,
    _load_adapters_for_base,
    _safe_chat_template_kwargs,
    enforce_expected_checkpoint,
)


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

        from flash.serving.src.engine.boot import load_engine_config, pin_loras_default
        from flash.serving.src.store import settings as cfg
        from flash.serving.src.store.registry import AdapterRegistry
        from flash.serving.src.store.settings import ADAPTER_CACHE_DIR, Settings

        self._replica_id = uuid.uuid4().hex
        self._replica_in_flight_requests = 0
        self._replica_first_request_pending = True
        self._replica_boot_duration_seconds = None
        self.settings = Settings()
        self.registry = AdapterRegistry()
        self._adapter_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._adapter_locks_guard = asyncio.Lock()
        self._source_locks: dict[tuple[str, str, str, str, str | None], asyncio.Lock] = {}
        self._source_locks_guard = asyncio.Lock()
        self._source_paths: dict[tuple[str, str, str, str, str | None], Path] = {}
        self._lora_entries: dict[tuple[str, str], _LoraEntry] = {}
        self._prompt_token_cache: OrderedDict[tuple[str, str], tuple[int, ...]] = OrderedDict()
        self._prompt_cache_size = cfg.PROMPT_TOKEN_CACHE_SIZE
        base_model_adapters = _load_adapters_for_base(self.settings, self.base_model)
        self.registry.hydrate(base_model_adapters)
        ADAPTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self.processor, self.tokenizer, overrides, kwargs = load_engine_config(
            self.base_model,
            self.settings,
            cfg,
        )
        self._pin_loras = pin_loras_default(overrides, cfg)
        self.reasoning_parser = kwargs.get("reasoning_parser")
        self.tool_parser = tool_parser_for(self.base_model)
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
        try:
            self._replica_boot_duration_seconds = boot_elapsed_seconds()
        except Exception:
            # capacity telemetry is observational and must never make a healthy replica fail.
            self._replica_boot_duration_seconds = None

    def _admit_generation(self) -> dict[str, Any]:
        """record honest process-local demand at request admission."""

        try:
            if not hasattr(self, "_replica_in_flight_requests"):
                return {}
            requests = max(0, int(self._replica_in_flight_requests)) + 1
            self._replica_in_flight_requests = requests
            freshly_booted = bool(getattr(self, "_replica_first_request_pending", False))
            self._replica_first_request_pending = False
            snapshot: dict[str, Any] = {
                "replica_in_flight_requests_at_admission": requests,
                "replica_freshly_booted": freshly_booted,
            }
            boot_duration = getattr(self, "_replica_boot_duration_seconds", None)
            if isinstance(boot_duration, (int, float)) and not isinstance(boot_duration, bool):
                normalized_boot_duration = float(boot_duration)
                if math.isfinite(normalized_boot_duration) and normalized_boot_duration >= 0:
                    snapshot["replica_boot_duration_seconds"] = normalized_boot_duration
            return snapshot
        except Exception:
            # capacity telemetry is observational and must never reject inference.
            return {}

    def _release_generation(self) -> None:
        """release process-local demand without raising into inference cleanup."""

        try:
            if not hasattr(self, "_replica_in_flight_requests"):
                return
            requests = int(self._replica_in_flight_requests) - 1
            self._replica_in_flight_requests = max(0, requests)
        except Exception:
            # a broken counter must not replace a successful generation with an error.
            self._replica_in_flight_requests = 0

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

    async def _adapter_lock(self, adapter_key: tuple[str, str]) -> asyncio.Lock:
        async with self._adapter_locks_guard:
            lock = self._adapter_locks.get(adapter_key)
            if lock is None:
                lock = asyncio.Lock()
                self._adapter_locks[adapter_key] = lock
            return lock

    async def _source_lock(self, record: Any) -> asyncio.Lock:
        ident = _adapter_source_ident(record)
        async with self._source_locks_guard:
            lock = self._source_locks.get(ident)
            if lock is None:
                lock = asyncio.Lock()
                self._source_locks[ident] = lock
            return lock

    def _entries(self) -> dict[tuple[str, str], _LoraEntry]:
        return entries_for(self)

    async def _evict_loaded_lora(self, adapter_key: tuple[str, str]) -> None:
        entries = self._entries()
        if (entry := entries.get(adapter_key)) is None:
            return
        if entry.state == "reserved":
            entries.pop(adapter_key)
            return
        entries[adapter_key] = _LoraEntry(entry.source_ident, entry.lora_request, "unconfirmed")
        remove = getattr(self.engine, "remove_lora", None)
        if remove is None:
            raise RuntimeError("vLLM cannot confirm LoRA removal")
        result = remove(entry.lora_request.lora_int_id)
        if inspect.isawaitable(result):
            result = await result
        if result is False:
            raise RuntimeError("vLLM rejected LoRA removal")
        entries.pop(adapter_key)

    async def _pin_lora(self, lora_request: Any) -> None:
        pin = getattr(self.engine, "pin_lora", None)
        if pin is None:
            return
        result = pin(lora_request.lora_int_id)
        if inspect.isawaitable(result):
            await result

    async def _add_lora_locked(self, record: Any, path: Path) -> None:
        adapter_key = record_key(record)
        entries = self._entries()
        lora_request = self._cached_lora_request_locked(record, path)
        was_reserved = entries[adapter_key].state == "reserved"
        try:
            added = await self.engine.add_lora(lora_request)
        except Exception:
            if was_reserved:
                entries.pop(adapter_key)
            raise
        if added is False and was_reserved:
            entries.pop(adapter_key)
            raise RuntimeError("vLLM rejected a new LoRA registration")
        entry = entries[adapter_key]
        entries[adapter_key] = _LoraEntry(entry.source_ident, lora_request, "loaded")
        if self._pin_loras:  # capped pools stay unpinned so surplus adapters can lru-swap
            await self._pin_lora(lora_request)

    async def _preload_cached_loras(self) -> None:
        from flash.serving.src.store.settings import ADAPTER_CACHE_DIR

        for record in self.registry.list_ready():
            source_ident = _adapter_source_ident(record)
            local_dir = _adapter_source_cache_dir(ADAPTER_CACHE_DIR, record)
            subfolder = getattr(record, "subfolder", None)
            path = _adapter_cache_path(local_dir, subfolder)
            if not _adapter_cache_ready(path):
                continue
            lock = await self._adapter_lock(record_key(record))
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
        # Download body; caller must already hold self._adapter_lock(record_key(record)).
        import anyio
        from huggingface_hub import snapshot_download

        from flash.serving.src.store.settings import ADAPTER_CACHE_DIR

        local_dir = _adapter_source_cache_dir(ADAPTER_CACHE_DIR, record)
        subfolder = getattr(record, "subfolder", None)
        cached_path = _adapter_cache_path(local_dir, subfolder)
        # Stale cached path (source changed) -> evict the old LoRA before re-downloading; check
        # before local_path(), which clears the stale entry.
        if self.registry.local_path_is_stale(record):
            # eviction releases this adapter's single lifecycle entry after confirmed removal.
            await self._evict_loaded_lora(record_key(record))
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
                            revision=record.artifact_revision,
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
        from flash.serving.src.io.schemas import AdapterRecord

        if record_dict is None:
            raise ValueError("tenant-scoped adapter resolution requires the forwarded record")
        forwarded = AdapterRecord.model_validate(record_dict)
        if forwarded.adapter_id != adapter_id:
            raise ValueError("forwarded adapter record does not match the requested checkpoint")
        if forwarded.serve_base_model:
            self.registry.upsert(forwarded)
            return None, forwarded
        adapter_key = record_key(forwarded)
        # lock across read + download so a concurrent unregister can't slip in between.
        lock = await self._adapter_lock(adapter_key)
        async with lock:
            self.registry.upsert(forwarded)
            record = self.registry.get(*adapter_key)
            if record is None or record.status != "ready":
                raise ValueError(f"Unknown adapter id on {self.base_model}: {adapter_id}")
            if not record.serve_base_model and not record.is_checkpoint:
                raise ValueError(f"Unknown adapter id on {self.base_model}: {adapter_id}")
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
                    messages_for_chat_template(detached_template_messages(template_messages)),
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
        from flash.serving.src.io.schemas import AdapterRecord

        record = AdapterRecord.model_validate(record_dict).model_copy(
            update={"deployment_generation": deployment_generation}
        )
        if record.serve_base_model:
            self.registry.upsert(record, revive=True)
            return {"ok": True, "adapter_id": record.adapter_id, "base_model": self.base_model}
        if not record.is_checkpoint:
            raise ValueError("only immutable checkpoints can be registered")
        lock = await self._adapter_lock(record_key(record))
        async with lock:  # _locked variant: we hold the lock (the public one would deadlock)
            path = await self._ensure_adapter_local_locked(record)
            await self._add_lora_locked(record, path)
            self.registry.upsert(record, revive=True)
        return {"ok": True, "adapter_id": record.adapter_id, "base_model": self.base_model}

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
    ) -> dict[str, Any]:
        from flash.serving.src.engine.generation import generate

        capacity = self._admit_generation()
        try:
            return await generate(
                self,
                payload_dict,
                record_dict,
                expected_checkpoint,
                generation_id,
                capacity,
            )
        finally:
            self._release_generation()

    async def _stream_generate(
        self,
        payload_dict: dict[str, Any],
        record_dict: dict[str, Any] | None = None,
        expected_checkpoint: str | None = None,
        generation_id: str | None = None,
    ):
        from flash.serving.src.engine.generation import stream_generate

        capacity = self._admit_generation()
        stream = stream_generate(
            self,
            payload_dict,
            record_dict,
            expected_checkpoint,
            generation_id,
            capacity,
        )
        try:
            async for event in stream:
                yield event
        finally:
            try:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
            finally:
                self._release_generation()

    async def _unregister(
        self,
        org_id: str,
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> dict[str, Any]:
        # undeploy under the tenant-scoped lock so sibling organizations cannot interfere.
        adapter_key = (org_id, adapter_id)
        lock = await self._adapter_lock(adapter_key)
        async with lock:
            current = self.registry.get(org_id, adapter_id)
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
            self.registry.remove(org_id, adapter_id)
            await self._evict_loaded_lora(adapter_key)
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
