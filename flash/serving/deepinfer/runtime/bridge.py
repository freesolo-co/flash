"""Bounded async bridge to the cooperative DeepInfer GPU processes."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import multiprocessing
import queue
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace

from anyio import CancelScope

from flash.serve.runtime.prompt_cache import cached_prompt_tokens
from flash.serving.deepinfer.contract.config import (
    MAX_INPUTS,
    MODEL_CACHE,
    MODEL_ID,
    MODEL_REVISION,
    RANK_COUNT,
    REQUEST_SECONDS,
    SEQUENCES_PER_RANK,
    STARTUP_SECONDS,
    WORKER_STALL_SECONDS,
)
from flash.serving.deepinfer.runtime.routing import CacheRouter
from flash.serving.deepinfer.runtime.worker import run_rank


@dataclass
class Pending:
    rank: int
    events: asyncio.Queue
    cancelled: asyncio.Event
    work: int = 0
    prefill: int = 0


class Runtime:
    def __init__(self):
        self.processes = []
        self.prompt_cache = OrderedDict()
        self.token_prefix_cache = None
        self.pending: dict[str, Pending] = {}
        self.loads = [0] * RANK_COUNT
        self.ready: set[int] = set()
        self.provenance: dict[int, dict] = {}
        self.failure: str | None = None
        self.pump = None
        self.cache_router = CacheRouter(RANK_COUNT)

    def _load_assets(self):
        from huggingface_hub import snapshot_download

        from flash.serving.deepinfer.contract.tokenizer import load_tokenizer, make_prefix_cache

        self.model_path = snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=MODEL_CACHE
        )
        self.tokenizer, self.encode_messages = load_tokenizer(self.model_path)
        self.token_prefix_cache = make_prefix_cache(self.tokenizer)

    async def start(self):
        await asyncio.to_thread(self._load_assets)
        context = multiprocessing.get_context("spawn")
        self.stopped = context.Event()
        self.outgoing = context.Queue(maxsize=4096)
        self.incoming = [context.Queue(maxsize=MAX_INPUTS * 2) for _ in range(RANK_COUNT)]
        self.last_seen = [time.monotonic()] * RANK_COUNT
        try:
            for rank in range(RANK_COUNT):
                process = context.Process(
                    target=run_rank,
                    args=(
                        rank,
                        self.model_path,
                        self.incoming[rank],
                        self.outgoing,
                        self.stopped,
                    ),
                )
                process.start()
                self.processes.append(process)
            self.pump = asyncio.create_task(self._dispatch())
            async with asyncio.timeout(STARTUP_SECONDS):
                while len(self.ready) != RANK_COUNT:
                    self.assert_healthy()
                    await asyncio.sleep(0.1)
                from flash.serving.deepinfer.runtime.warmup import warmup

                await warmup(self)
        except BaseException:
            await self.close()
            raise

    def assert_healthy(self):
        if self.failure:
            raise RuntimeError(self.failure)
        if any(not process.is_alive() for process in self.processes):
            raise RuntimeError("DeepInfer rank exited")

    def _read_event(self):
        try:
            return self.outgoing.get(timeout=0.2)
        except queue.Empty:
            return None

    async def _dispatch(self):
        try:
            while True:
                self.assert_healthy()
                event = await asyncio.to_thread(self._read_event)
                if event is not None:
                    self._deliver(event)
                if (
                    len(self.ready) == RANK_COUNT
                    and time.monotonic() - min(self.last_seen) > WORKER_STALL_SECONDS
                ):
                    raise RuntimeError("DeepInfer scheduling stalled")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failure = "DeepInfer replica is unavailable"
            for pending in self.pending.values():
                # Make room for a terminal error even if a client is slow.
                while not pending.events.empty():
                    pending.events.get_nowait()
                pending.events.put_nowait({"type": "fatal"})

    def _deliver(self, event):
        if event["type"] == "fatal":
            raise RuntimeError("DeepInfer rank failed")
        if event["type"] == "cache_directory":
            self.cache_router.update(event["rank"], event["prefixes"])
            return
        if event["type"] in {"ready", "heartbeat"}:
            self.last_seen[event["rank"]] = time.monotonic()
            if event["type"] == "ready":
                self.ready.add(event["rank"])
                self.provenance[event["rank"]] = event.get("provenance", {})
                if len(self.ready) == RANK_COUNT:
                    self.last_seen = [time.monotonic()] * RANK_COUNT
            return
        pending = self.pending.get(event["id"])
        if pending is not None:
            if event["type"] == "tokens":
                pending.work = max(0, pending.work - pending.prefill - 16 * len(event["tokens"]))
                pending.prefill = 0
            if event["type"] == "cancelled":
                pending.cancelled.set()
                return
            if pending.events.full():
                # Never let one slow reader block progress on other ranks.
                self.incoming[pending.rank].put_nowait({"type": "cancel", "id": event["id"]})
                while not pending.events.empty():
                    pending.events.get_nowait()
                pending.events.put_nowait({"type": "fatal"})
            else:
                pending.events.put_nowait(event)

    async def prepare_tokens(self, request):
        key = hashlib.sha256(
            json.dumps(
                [MODEL_REVISION, request.cache_namespace, request.messages, request.thinking],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        tokens = await cached_prompt_tokens(
            self.prompt_cache,
            key,
            lambda: self.tokenize(request),
            max_entries=128,
        )
        # max_tokens varies independently of the cached prompt.
        self._validate_length(tokens, request.max_tokens)
        return tokens

    async def prefix_length(self, request, tokens):
        if not request.cache_namespace or len(tokens) < 130:
            return 0
        # Reuse only a token-exact stable conversation prefix. Single-message prompts
        # fall back to full-prompt reuse; no target hit ratio is manufactured.
        messages = [*request.messages[:-1], {**request.messages[-1], "content": ""}]
        empty_tail = await self.prepare_tokens(replace(request, messages=messages, max_tokens=1))
        common = 0
        for left, right in zip(tokens, empty_tail, strict=False):
            if left != right:
                break
            common += 1
        return min(common if common >= 128 else len(tokens) - 2, len(tokens) - 2)

    @staticmethod
    def _validate_length(tokens, max_tokens):
        from flash.serving.deepinfer.contract.tokenizer import validate_length

        validate_length(tokens, max_tokens)

    def tokenize(self, request):
        from flash.serving.deepinfer.contract.tokenizer import encode_request

        return encode_request(
            self.tokenizer, self.encode_messages, request, prefix_cache=self.token_prefix_cache
        )

    async def generate(self, request, tokens):
        self.assert_healthy()
        rank = self.cache_router.choose(request.cache_namespace, tokens, self.loads, self.pending)
        if self.loads[rank] >= SEQUENCES_PER_RANK:
            raise OverflowError("DeepInfer replica is at capacity")
        request_id = uuid.uuid4().hex
        pending = Pending(
            rank,
            asyncio.Queue(maxsize=256),
            asyncio.Event(),
            len(tokens) + 16 * request.max_tokens,
            len(tokens),
        )
        self.pending[request_id] = pending
        self.loads[rank] += 1
        try:
            self.incoming[rank].put_nowait(
                {
                    "type": "generate",
                    "id": request_id,
                    "tokens": tokens,
                    "cache_namespace": request.cache_namespace,
                    "prefix_tokens": request.prefix_tokens,
                    "sampling": {
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "max_tokens": request.max_tokens,
                        "seq_seed": request.seed,
                    },
                }
            )
            loop = asyncio.get_running_loop()
            deadline = loop.time() + REQUEST_SECONDS
            while True:
                # Each anext() can run in a different task. Never keep a task-bound
                # timeout open across yield, or reset the budget for the next event.
                if loop.time() >= deadline:
                    raise TimeoutError("DeepInfer generation deadline exceeded")
                async with asyncio.timeout_at(deadline):
                    event = await pending.events.get()
                if event["type"] in {"fatal", "error"}:
                    raise RuntimeError("DeepInfer generation failed")
                yield event
                if event.get("finish_reason"):
                    return
        finally:
            # Starlette cancels its stream task on disconnect. Shield the bounded abort
            # handshake so its slot cannot be reused while native work remains alive.
            with CancelScope(shield=True):
                try:
                    self.incoming[rank].put_nowait({"type": "cancel", "id": request_id})
                    await asyncio.wait_for(pending.cancelled.wait(), timeout=5)
                except (queue.Full, TimeoutError):
                    self.failure = "DeepInfer cancellation was not confirmed"
                finally:
                    self.pending.pop(request_id, None)
                    self.loads[rank] -= 1

    async def close(self):
        if self.pump is not None:
            self.pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.pump
        if not hasattr(self, "stopped"):
            return
        self.stopped.set()
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            await asyncio.to_thread(process.join, 2)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 2)
        for channel in [*self.incoming, self.outgoing]:
            channel.cancel_join_thread()
            channel.close()
