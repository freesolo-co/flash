"""Batched buffered generation: many requests' engine calls dispatched as one.

``POST /v1/chat/completions/batch`` runs every item through the ordinary ``/v1/chat/completions``
handler concurrently. Each handler builds, admits, meters and shapes its own request exactly as
it would alone; ``GenerationBatcher`` only replaces the engine dispatch: once every item has
either reached ``pool.generate`` or finished without it, the reached ones go to the pool as one
``generate_batch`` call, which one engine replica runs as one batched prefill.

Per-item failures cross the Modal boundary as envelopes (``error_envelope``) and are rebuilt
into the exception the single-request path would have raised (``raise_envelope``), so the
router's error mapping is unchanged.
"""

from __future__ import annotations

import asyncio
import contextvars
from typing import Any

from flash.serving.lambda_cloud.transport import WorkerError

MAX_BATCH_ITEMS = 256

_MEMBER: contextvars.ContextVar[tuple[GenerationBatcher, _Member] | None] = contextvars.ContextVar(
    "flash_generation_batch", default=None
)


class _Member:
    __slots__ = ("dispatched",)

    def __init__(self) -> None:
        self.dispatched = False


class _Call:
    __slots__ = ("base_model", "expected_checkpoint", "future", "payload", "pool", "record")

    def __init__(self, pool, base_model, payload, record, expected_checkpoint, future) -> None:
        self.pool, self.base_model, self.payload, self.record = pool, base_model, payload, record
        self.expected_checkpoint, self.future = expected_checkpoint, future


def error_envelope(exc: BaseException) -> dict[str, Any]:
    """A picklable, type-preserving stand-in for one item's failure."""
    if isinstance(exc, WorkerError):
        kind = "worker_error"
    elif isinstance(exc, ValueError):
        kind = "value_error"
    elif isinstance(exc, TimeoutError):
        kind = "timeout_error"
    else:
        kind = "runtime_error"
    return {
        "ok": False,
        "error": kind,
        "message": str(exc),
        "status_code": getattr(exc, "status_code", None),
    }


def envelope(outcome: Any) -> dict[str, Any]:
    if isinstance(outcome, BaseException):
        return error_envelope(outcome)
    return {"ok": True, "result": outcome}


def outcome_of(item: Any) -> Any:
    """The result, or the exception instance, an envelope stands for."""
    if not isinstance(item, dict) or not isinstance(item.get("ok"), bool):
        return RuntimeError("serving engine returned a malformed batch item")
    if item["ok"]:
        return item.get("result")
    message = str(item.get("message") or "")
    kind = item.get("error")
    if kind == "worker_error":
        status = item.get("status_code")
        return WorkerError(message, status if type(status) is int else 502)
    if kind == "value_error":
        return ValueError(message)
    if kind == "timeout_error":
        return TimeoutError(message)
    return RuntimeError(message or "serving engine failed")


async def gather_outcomes(calls) -> list[Any]:
    """Run awaitables concurrently; each result or exception, in order.

    Cancellation is not an outcome: it cancels the rest and propagates.
    """
    results = await asyncio.gather(*calls, return_exceptions=True)
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            raise result
    return list(results)


class GenerationBatcher:
    """Collects the engine dispatches of ``size`` concurrent request handlers."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._arrived = 0
        self._calls: list[_Call] = []
        self._flush: asyncio.Task | None = None

    async def run(self, handler) -> Any:
        """Await ``handler`` (a coroutine) as one member of the batch."""
        member = _Member()
        token = _MEMBER.set((self, member))
        try:
            return await handler
        finally:
            _MEMBER.reset(token)
            if not member.dispatched:
                self._arrive(None)

    def _arrive(self, call: _Call | None) -> None:
        self._arrived += 1
        if call is not None:
            self._calls.append(call)
        if self._arrived == self._size and self._calls:
            self._flush = asyncio.ensure_future(self._dispatch(self._calls))

    async def _dispatch(self, calls: list[_Call]) -> None:
        groups: dict[tuple, list[_Call]] = {}
        for call in calls:
            key = (
                id(call.pool),
                call.base_model,
                call.record.model_id,
                getattr(call.record, "org_id", None),
                getattr(call.record, "deployment_generation", None),
                call.expected_checkpoint,
            )
            groups.setdefault(key, []).append(call)
        await asyncio.gather(*(self._dispatch_group(group) for group in groups.values()))

    async def _dispatch_group(self, calls: list[_Call]) -> None:
        first = calls[0]
        pool = first.pool
        try:
            batch = getattr(pool, "generate_batch", None)
            if batch is not None and len(calls) > 1:
                outcomes = await batch(
                    first.base_model,
                    [call.payload for call in calls],
                    first.record,
                    expected_checkpoint=first.expected_checkpoint,
                )
                if len(outcomes) != len(calls):
                    raise RuntimeError("serving engine returned a batch of the wrong size")
            else:
                outcomes = await gather_outcomes(
                    call.pool.generate(
                        call.base_model,
                        call.payload,
                        call.record,
                        expected_checkpoint=call.expected_checkpoint,
                    )
                    for call in calls
                )
        except Exception as exc:
            outcomes = [exc] * len(calls)
        except BaseException as exc:
            for call in calls:
                if not call.future.done():
                    call.future.set_exception(exc)
            raise
        for call, outcome in zip(calls, outcomes, strict=True):
            if call.future.done():
                continue
            if isinstance(outcome, BaseException):
                call.future.set_exception(outcome)
            else:
                call.future.set_result(outcome)


async def dispatch_generate(
    pool: Any,
    base_model: str,
    payload: Any,
    record: Any,
    *,
    expected_checkpoint: str | None,
) -> dict[str, Any]:
    """``pool.generate``, shared with the current batch when this handler is a batch member."""
    current = _MEMBER.get()
    if current is None or current[1].dispatched:
        return await pool.generate(
            base_model, payload, record, expected_checkpoint=expected_checkpoint
        )
    batcher, member = current
    member.dispatched = True
    future = asyncio.get_running_loop().create_future()
    batcher._arrive(_Call(pool, base_model, payload, record, expected_checkpoint, future))
    return await future
