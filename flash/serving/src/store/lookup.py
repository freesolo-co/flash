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

from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.requests import _assert_supported_base_model
from flash.serving.src.io.schemas import AdapterRecord


class AdapterLookup:
    def __init__(
        self,
        router: AdapterRouter,
        reload_records: Any = None,
        *,
        lookup_record: Callable[[str, str], AdapterRecord | None] | None = None,
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

    def _schedule_reload(self) -> None:
        task = self._last_reload.get("task")
        if task is not None and not task.done():
            return
        self._last_reload["task"] = asyncio.create_task(self._reload_safe())

    def _is_stale(self) -> bool:
        return time.monotonic() - self._last_reload["at"] >= self._reload_interval_seconds

    async def resolve(
        self,
        adapter_id: str,
        *,
        org_id: str | None,
        require_supported_base_model: bool = True,
    ) -> tuple[AdapterRecord, AdapterRecord]:
        resolved = self._router.resolve(adapter_id, org_id=org_id)
        stale = resolved is not None and self._is_stale()
        if resolved is not None and self._reload_records is not None and stale:
            self._schedule_reload()
        elif resolved is None and self._reload_records is not None:
            await self.reload()
            resolved = self._router.resolve(adapter_id, org_id=org_id)
        if resolved is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")
        if require_supported_base_model:
            _assert_supported_base_model(resolved[1].base_model)
        return resolved

    async def get_exact(self, org_id: str, adapter_id: str) -> AdapterRecord:
        record = self._router.get(adapter_id, org_id=org_id)
        cached_ready = record is not None and record.status == "ready"
        if cached_ready and self._is_stale() and self._reload_records is not None:
            self._schedule_reload()
        if not cached_ready and self._lookup_record is not None:
            # lifecycle reads need disabled rows, but routing hydration must stay ready-only. fetch
            # this id without mutating the registry so visibility never makes it routable.
            record = await asyncio.to_thread(self._lookup_record, org_id, adapter_id)
        elif not cached_ready and self._reload_records is not None:
            await self.reload()
            record = self._router.get(adapter_id, org_id=org_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")
        if not record.serve_base_model and not record.is_checkpoint:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")
        return record
