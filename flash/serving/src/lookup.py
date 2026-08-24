"""Adapter lookup with a bounded background refresh.

Split out of router.py's app builder. Owns the reload bookkeeping the app factory used to hold as
a bare dict: when the router last rehydrated, and the in-flight refresh task.

A container that missed a (un)registration performed on another container still has to resolve the
adapter, so a miss reloads once before 404-ing, and a hit reloads at most once per interval in the
background.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, status

from flash.serving.src.routing import AdapterRouter
from flash.serving.src.schemas import AdapterRecord
from flash.serving.src.serving_io import _assert_supported_base_model


class AdapterLookup:
    def __init__(
        self,
        router: AdapterRouter,
        reload_records: Any = None,
        *,
        lookup_record: Callable[[str], AdapterRecord | None] | None = None,
        reload_interval_seconds: float = 30.0,
    ) -> None:
        self._router = router
        self._reload_records = reload_records
        self._lookup_record = lookup_record
        self._reload_interval_seconds = reload_interval_seconds
        # `at` is when the last fetch COMPLETED and drives the ttl. `fetched_at` is when that
        # fetch STARTED, which is what decides whether a waiting caller may reuse its result.
        self._last_reload: dict[str, Any] = {
            "at": float("-inf"),
            "fetched_at": float("-inf"),
            "task": None,
        }
        self._reload_lock = asyncio.Lock()

    async def reload(self) -> None:
        # to_thread: reload_records is a sync Supabase call; don't block the event loop.
        assert self._reload_records is not None
        started = time.monotonic()
        # serialized: the fetch suspends, so two concurrent misses could otherwise hydrate out of
        # order -- the slower fetch's older records landing last and overwriting fresher state,
        # then stamping a newer timestamp over it. hydrate and timestamp move together under here.
        async with self._reload_lock:
            # coalesce onto the in-flight reload only if its fetch STARTED after we did. comparing
            # against completion time instead let a reload that snapshotted storage before this
            # caller existed, and merely finished late, satisfy it -- so an adapter committed
            # before the request arrived stayed invisible and the caller got a 404.
            if self._last_reload["fetched_at"] >= started:
                return
            fetch_started = time.monotonic()
            records = await asyncio.to_thread(self._reload_records)
            self._router.hydrate(records)
            self._last_reload["fetched_at"] = fetch_started
            self._last_reload["at"] = time.monotonic()

    async def _reload_safe(self) -> bool:
        try:
            await self.reload()
        except Exception as exc:  # a refresh cannot fail a cached hit
            print(f"adapter refresh skipped: {exc!r}", flush=True)
            return False
        return True

    async def refresh_periodically(self) -> None:
        if self._reload_records is None:
            return
        while True:
            await asyncio.sleep(self._reload_interval_seconds)
            await self._reload_safe()

    def _schedule_reload(self) -> None:
        task = self._last_reload.get("task")
        if task is not None and not task.done():
            return
        self._last_reload["task"] = asyncio.create_task(self._reload_safe())

    def _is_stale(self) -> bool:
        return time.monotonic() - self._last_reload["at"] >= self._reload_interval_seconds

    async def resolve(
        self, adapter_id: str, *, require_supported_base_model: bool = True
    ) -> tuple[AdapterRecord, AdapterRecord]:
        resolved = self._router.resolve(adapter_id)
        stale = resolved is not None and self._is_stale()
        if resolved is not None and self._reload_records is not None:
            if resolved[0].is_alias:
                if await self._reload_safe():
                    resolved = self._router.resolve(adapter_id)
            elif stale:
                self._schedule_reload()
        elif resolved is None and self._reload_records is not None:
            if self._router.is_unqualified_adapter(adapter_id):
                await self._reload_safe()
            else:
                await self.reload()
            resolved = self._router.resolve(adapter_id)
        if resolved is None:
            if self._router.is_unqualified_adapter(adapter_id):
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "adapter base model is not qualified for this deployment",
                )
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")
        if require_supported_base_model:
            _assert_supported_base_model(resolved[1].base_model)
        return resolved

    async def get_exact(self, adapter_id: str) -> AdapterRecord:
        record = self._router.get(adapter_id)
        cached_ready = record is not None and record.status == "ready"
        if cached_ready and self._is_stale() and self._reload_records is not None:
            self._schedule_reload()
        if not cached_ready and self._lookup_record is not None:
            # lifecycle reads need disabled rows, but routing hydration must stay ready-only. fetch
            # this id without mutating the registry so visibility never makes it routable.
            record = await asyncio.to_thread(self._lookup_record, adapter_id)
        elif not cached_ready and self._reload_records is not None:
            await self.reload()
            record = self._router.get(adapter_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")
        if not record.serve_base_model and not (record.is_alias or record.is_revision):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")
        return record
