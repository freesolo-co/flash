"""Bounded async bridge to the four cooperative DeepInfer processes."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import multiprocessing
import queue
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from anyio import CancelScope

from flash.serving.deepinfer.config import (
    GPU_COUNT,
    MAX_INPUTS,
    MAX_MODEL_LEN,
    MODEL_CACHE,
    MODEL_ID,
    MODEL_REVISION,
    REQUEST_SECONDS,
    SEQUENCES_PER_RANK,
    STARTUP_SECONDS,
)
from flash.serving.deepinfer.worker import run_rank


@dataclass
class Pending:
    rank: int
    events: asyncio.Queue
    cancelled: asyncio.Event


class Runtime:
    def __init__(self):
        self.processes = []
        self.pending: dict[str, Pending] = {}
        self.loads = [0] * GPU_COUNT
        self.ready: set[int] = set()
        self.failure: str | None = None
        self.pump = None

    def _load_assets(self):
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        self.model_path = snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=MODEL_CACHE
        )
        self.tokenizer = Tokenizer.from_file(str(Path(self.model_path) / "tokenizer.json"))
        spec = importlib.util.spec_from_file_location(
            "deepseek_1017_encoding", Path(self.model_path) / "encoding" / "encoding.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Pinned DeepSeek encoding module is missing")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.encode_messages = module.encode_messages

    async def start(self):
        await asyncio.to_thread(self._load_assets)
        context = multiprocessing.get_context("spawn")
        self.stopped = context.Event()
        self.outgoing = context.Queue(maxsize=4096)
        self.incoming = [context.Queue(maxsize=MAX_INPUTS * 2) for _ in range(GPU_COUNT)]
        self.last_seen = [time.monotonic()] * GPU_COUNT
        try:
            for rank in range(GPU_COUNT):
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
                while len(self.ready) != GPU_COUNT:
                    self.assert_healthy()
                    await asyncio.sleep(0.1)
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
                    len(self.ready) == GPU_COUNT
                    and time.monotonic() - min(self.last_seen) > REQUEST_SECONDS
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
        if event["type"] in {"ready", "heartbeat"}:
            self.last_seen[event["rank"]] = time.monotonic()
            if event["type"] == "ready":
                self.ready.add(event["rank"])
                if len(self.ready) == GPU_COUNT:
                    self.last_seen = [time.monotonic()] * GPU_COUNT
            return
        pending = self.pending.get(event["id"])
        if pending is not None:
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

    def tokenize(self, request):
        prompt = self.encode_messages(
            request.messages, thinking_mode="thinking" if request.thinking else "chat"
        )
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if not tokens or len(tokens) + request.max_tokens + 64 > MAX_MODEL_LEN:
            raise ValueError(
                f"prompt plus output exceeds the {MAX_MODEL_LEN}-token serving context"
            )
        return tokens

    async def generate(self, request, tokens):
        self.assert_healthy()
        rank = min(range(GPU_COUNT), key=self.loads.__getitem__)
        if self.loads[rank] >= SEQUENCES_PER_RANK:
            raise OverflowError("DeepInfer replica is at capacity")
        request_id = uuid.uuid4().hex
        pending = Pending(rank, asyncio.Queue(maxsize=256), asyncio.Event())
        self.pending[request_id] = pending
        self.loads[rank] += 1
        try:
            self.incoming[rank].put_nowait(
                {
                    "type": "generate",
                    "id": request_id,
                    "tokens": tokens,
                    "sampling": {
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "max_tokens": request.max_tokens,
                        "seq_seed": request.seed,
                    },
                }
            )
            async with asyncio.timeout(REQUEST_SECONDS):
                while True:
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
