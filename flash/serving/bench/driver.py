"""The load driver: issues concurrent requests against a live engine and records evidence.

Runs INSIDE the GPU container, driving ``_LoraEngineImpl._stream_generate`` directly. Two reasons it
is not an external HTTP client:

* The production router mandates a durable usage outbox that settles billing against the real
  backend. Driving a benchmark through it would write synthetic usage into production billing.
* An external client measures the network path to Modal as much as the engine. The question here is
  the engine's capacity on one B200, so the driver sits next to it.

The tradeoff is explicit: these numbers EXCLUDE router overhead, ingress, and network latency, and
the report says so. They are an engine capacity envelope, not an end-to-end client SLA.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from flash.serving.bench.metrics import (
    ERROR_CACHE_CONTAMINATED,
    ERROR_CACHE_UNVERIFIED,
    ERROR_ENGINE,
    ERROR_MALFORMED_STREAM,
    ERROR_MISSING_USAGE,
    ERROR_NO_FINISH_REASON,
    ERROR_TIMEOUT,
    ERROR_TOKEN_MISMATCH,
    CellResult,
    RequestRecord,
    reduce_cell,
)
from flash.serving.bench.workload import (
    ENABLE_THINKING,
    TEMPERATURE,
    TOP_P,
    Bucket,
    fit_prompt_to_tokens,
    request_uid,
)

# A request that has produced nothing for this long is counted as a timeout rather than waited on.
# Sized well above the near-32k prefill so a slow-but-working request is never miscounted as a
# failure; the point is to bound a hung cell, not to trim the tail.
REQUEST_TIMEOUT_SECONDS = 900.0


def base_model_record(base_model: str) -> dict[str, Any]:
    """The open, no-LoRA record that makes ``base_model`` addressable by name.

    ``serve_base_model=True`` is what routes generation to the base weights with ``lora_request=None``
    and what lets the caller's ``enable_thinking`` win (see ``lora_engine._thinking_default``). A
    trained adapter would override thinking with its own persisted value, which is why this campaign
    is base-only.
    """
    return {
        "adapter_id": base_model,
        "repo_id": base_model,
        "base_model": base_model,
        "serve_base_model": True,
        "thinking": False,
        "org_id": None,
        "status": "ready",
        "private": True,
    }


def _payload_for(
    base_model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    uid: str,
) -> dict[str, Any]:
    return {
        "adapter_id": base_model,
        "generation_id": f"bench-{uid}",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }


class _StreamOutcome:
    """Mutable accumulator for one streamed request."""

    __slots__ = (
        "cached_tokens",
        "cached_tokens_reported",
        "completion_tokens",
        "delta_tokens",
        "finish_reason",
        "first_token_at",
        "prompt_tokens",
        "reasoning_tokens",
        "replica_id",
        "saw_final",
    )

    def __init__(self) -> None:
        self.first_token_at: float | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.cached_tokens: int | None = None
        self.cached_tokens_reported: bool | None = None
        self.reasoning_tokens: int | None = None
        self.finish_reason: str | None = None
        self.replica_id: str | None = None
        self.saw_final = False
        self.delta_tokens = 0


def _absorb_event(event: dict[str, Any], outcome: _StreamOutcome, now: float) -> None:
    """Fold one engine event into the outcome.

    The engine emits usage fields on EVERY event, so the latest values win; the terminal ``final``
    event carries the authoritative totals.
    """
    kind = event.get("type")
    if event.get("engine_replica_id"):
        outcome.replica_id = str(event["engine_replica_id"])
    for field in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"):
        value = event.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            setattr(outcome, field, value)
    # Whether the engine MEASURED the cached-token count, which is not the same as it being zero.
    # `type(...) is bool` so a truthy non-bool cannot pose as a measurement.
    reported = event.get("cached_tokens_reported")
    if type(reported) is bool:
        outcome.cached_tokens_reported = reported

    if kind == "delta":
        text = event.get("text") or ""
        if text:
            outcome.delta_tokens += 1
            if outcome.first_token_at is None:
                outcome.first_token_at = now
    elif kind == "choice_finished":
        reason = event.get("finish_reason")
        if isinstance(reason, str) and reason:
            outcome.finish_reason = reason
    elif kind == "final":
        outcome.saw_final = True


def _validate(outcome: _StreamOutcome, record: RequestRecord) -> None:
    """Apply the success contract; mark the record ok or with a normalized error.

    Order matters: the most specific, most diagnostic failure wins, so an invalid cell says WHY it
    is invalid rather than collapsing everything into a generic engine error.
    """
    if not outcome.saw_final:
        record.error = ERROR_MALFORMED_STREAM
        record.error_detail = "stream ended without a final event"
        return
    if outcome.finish_reason is None:
        record.error = ERROR_NO_FINISH_REASON
        record.error_detail = "no terminal finish reason"
        return
    if outcome.prompt_tokens is None or outcome.completion_tokens is None:
        record.error = ERROR_MISSING_USAGE
        record.error_detail = "engine did not report authoritative usage"
        return
    if outcome.completion_tokens <= 0:
        record.error = ERROR_TOKEN_MISMATCH
        record.error_detail = "engine reported zero completion tokens"
        return
    # An UNREPORTED cached-token count is not a cache miss. `_num_cached_tokens` (support.py) returns
    # 0 when the attribute is absent or None, so a build that stopped reporting would make every
    # request look perfectly uncached and the whole campaign would read as clean while being
    # unverified. Dev emits `cached_tokens_reported` precisely to separate those cases.
    if not outcome.cached_tokens_reported:
        record.error = ERROR_CACHE_UNVERIFIED
        record.error_detail = "engine did not report a cached-token measurement"
        return
    # A nonzero cached_tokens means this request reused another's prefix, so its TTFT and throughput
    # describe a cache hit rather than capacity. Invalid, not fast.
    if outcome.cached_tokens:
        record.error = ERROR_CACHE_CONTAMINATED
        record.error_detail = f"prefix cache served {outcome.cached_tokens} tokens"
        return
    if outcome.first_token_at is None:
        record.error = ERROR_MALFORMED_STREAM
        record.error_detail = "no content delta observed"
        return
    record.ok = True


async def run_request(
    engine: Any,
    base_model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    uid: str,
    *,
    bucket: str,
    concurrency: int,
    block: int,
    origin: float,
) -> RequestRecord:
    """Issue one streamed request and return its evidence record. Never raises."""
    record = RequestRecord(
        uid=uid,
        base_model=base_model,
        bucket=bucket,
        concurrency=concurrency,
        block=block,
        started_at=time.monotonic() - origin,
    )
    outcome = _StreamOutcome()
    payload = _payload_for(base_model, messages, max_tokens, uid)
    forwarded = base_model_record(base_model)

    async def _consume() -> None:
        async for event in engine._stream_generate(
            payload, forwarded, None, payload["generation_id"]
        ):
            _absorb_event(event, outcome, time.monotonic() - origin)

    try:
        await asyncio.wait_for(_consume(), timeout=REQUEST_TIMEOUT_SECONDS)
    except TimeoutError:
        record.error = ERROR_TIMEOUT
        record.error_detail = f"exceeded {REQUEST_TIMEOUT_SECONDS}s"
    except Exception as exc:
        record.error = ERROR_ENGINE
        # Type and message only. A full traceback can carry paths and, in principle, credentials.
        record.error_detail = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        record.finished_at = time.monotonic() - origin
        record.first_token_at = outcome.first_token_at
        record.prompt_tokens = outcome.prompt_tokens
        record.completion_tokens = outcome.completion_tokens
        record.cached_tokens = outcome.cached_tokens
        record.cached_tokens_reported = outcome.cached_tokens_reported
        record.reasoning_tokens = outcome.reasoning_tokens
        record.finish_reason = outcome.finish_reason
        record.engine_replica_id = outcome.replica_id

    if record.error is None:
        _validate(outcome, record)
    return record


def _build_prompt_pool(
    tokenizer: Any,
    bucket: Bucket,
    *,
    concurrency: int,
    block: int,
    min_requests: int,
) -> list[tuple[str, list[dict[str, Any]], int]]:
    """Fit every prompt the cell can need BEFORE the measured window opens.

    Each entry is request-unique from its first token, so no two requests share a prefix and the
    engine's prefix cache cannot turn a measurement into a lookup. The pool is sized past the
    request floor because a cell may overrun it while waiting on `min_seconds`; if it does wrap, it
    reuses a prompt, and that reuse is caught by the cache-contamination check rather than passing
    silently as a fast success.
    """
    size = max(concurrency, min_requests + concurrency)
    pool: list[tuple[str, list[dict[str, Any]], int]] = []
    for index in range(size):
        uid = request_uid(bucket.name, concurrency, block, index)
        messages, exact = fit_prompt_to_tokens(tokenizer, uid, bucket.target_input_tokens)
        pool.append((uid, messages, exact))
    return pool


async def run_cell(
    engine: Any,
    tokenizer: Any,
    base_model: str,
    bucket: Bucket,
    concurrency: int,
    block: int,
    *,
    min_seconds: float | None = None,
    min_requests: int | None = None,
    max_seconds: float | None = None,
) -> tuple[CellResult, list[RequestRecord]]:
    """Hold ``concurrency`` requests in flight until the cell's floors are met.

    Closed-loop: a finished request is immediately replaced, so exactly ``concurrency`` requests are
    in flight throughout. This measures the engine's capacity AT that concurrency, which is the
    quantity the envelope is about. It deliberately does not model an open-loop arrival process.

    Prompts are built up front so tokenization never competes with the measured window for CPU.

    The depth floors default to the bucket's own, since how many requests are affordable is a
    property of the request shape rather than of the call site. An explicit argument overrides them.
    """
    min_seconds = bucket.min_seconds if min_seconds is None else min_seconds
    min_requests = bucket.min_requests if min_requests is None else min_requests
    max_seconds = bucket.max_seconds if max_seconds is None else max_seconds

    records: list[RequestRecord] = []

    # Fitting is repeated synchronous tokenization. Done lazily it would run ON the event loop: the
    # first `concurrency` fits land inside the measured window as idle wall time, and every
    # replacement fit blocks consumption of the OTHER in-flight streams, inflating their TTFT and
    # latency. At 8k and 31k input that distortion is larger than the effect being measured. So the
    # whole pool is built BEFORE the clock starts, and the measured loop only pops from it.
    pool = _build_prompt_pool(
        tokenizer,
        bucket,
        concurrency=concurrency,
        block=block,
        min_requests=min_requests,
    )
    issued = 0

    def _next_prompt() -> tuple[str, list[dict[str, Any]], int]:
        nonlocal issued
        uid, messages, exact = pool[issued % len(pool)]
        issued += 1
        return uid, messages, exact

    origin = time.monotonic()
    spawned_at: dict[asyncio.Task[RequestRecord], float] = {}
    spawned_uid: dict[asyncio.Task[RequestRecord], str] = {}

    def _spawn() -> asyncio.Task[RequestRecord]:
        uid, messages, _ = _next_prompt()
        started_at = time.monotonic() - origin
        task = asyncio.create_task(
            run_request(
                engine,
                base_model,
                messages,
                bucket.max_output_tokens,
                uid,
                bucket=bucket.name,
                concurrency=concurrency,
                block=block,
                origin=origin,
            )
        )
        spawned_at[task] = started_at
        spawned_uid[task] = uid
        return task

    in_flight = {_spawn() for _ in range(concurrency)}
    try:
        while True:
            done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
            in_flight = set(pending)
            records.extend(task.result() for task in done)
            elapsed = time.monotonic() - origin
            enough = len(records) >= min_requests and elapsed >= min_seconds
            if enough or elapsed >= max_seconds:
                break
            for _ in done:
                in_flight.add(_spawn())
    finally:
        # Drain rather than abandon. Two reasons, and the ORDER matters:
        #
        # 1. An orphaned request keeps occupying the engine during the NEXT cell and contaminates it.
        # 2. Cancelling first and awaiting second silently DELETES those requests from the attempt
        #    denominator: `task.cancel()` raises CancelledError at the await point, the suppress
        #    swallows it, and the append never runs. An overloaded cell then reports a cleaner error
        #    rate than it earned, which is the exact direction a capacity claim must not err.
        #
        # So let issued work finish on its own (each request already carries its own
        # REQUEST_TIMEOUT_SECONDS, so this terminates) and count every record. Only a task still
        # pending after that bound is cancelled, and it is recorded as a timeout rather than dropped.
        if in_flight:
            done, still_pending = await asyncio.wait(in_flight, timeout=REQUEST_TIMEOUT_SECONDS)
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    records.append(task.result())
            for task in still_pending:
                task.cancel()
            for task in still_pending:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                # `started_at` is REQUIRED and has no default. Omitting it raised TypeError here,
                # which aborted the whole bucket and discarded every record accumulated during the
                # expensive sweep -- the exact loss this drain exists to prevent. The task's own
                # start offset is carried on the task so a drained record keeps its real origin.
                records.append(
                    RequestRecord(
                        uid=spawned_uid.get(task, f"drain-{id(task):x}"),
                        base_model=base_model,
                        bucket=bucket.name,
                        concurrency=concurrency,
                        block=block,
                        started_at=spawned_at.get(task, 0.0),
                        error=ERROR_TIMEOUT,
                        error_detail="request did not finish within the drain bound",
                    )
                )

    wall_seconds = time.monotonic() - origin
    result = reduce_cell(
        records,
        base_model=base_model,
        bucket=bucket.name,
        concurrency=concurrency,
        block=block,
        wall_seconds=wall_seconds,
    )
    return result, records


__all__ = [
    "REQUEST_TIMEOUT_SECONDS",
    "base_model_record",
    "run_cell",
    "run_request",
]
