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
import faulthandler
import os
import sys
import time
from collections.abc import Callable
from typing import Any

from flash.serving.bench.metrics import (
    ERROR_CACHE_CONTAMINATED,
    ERROR_CACHE_UNVERIFIED,
    ERROR_ENGINE,
    ERROR_MALFORMED_STREAM,
    ERROR_MISSING_USAGE,
    ERROR_NO_FINISH_REASON,
    ERROR_PROMPT_LENGTH,
    ERROR_TIMEOUT,
    ERROR_TOKEN_MISMATCH,
    CellResult,
    RequestRecord,
    reduce_cell,
)
from flash.serving.bench.workload import (
    ENABLE_THINKING,
    PROMPT_TOKEN_TOLERANCE,
    TEMPERATURE,
    TOP_P,
    Bucket,
    corpus_seed,
    fit_prompt_to_tokens,
    request_uid,
    reseed_prompt,
)

# A request that has produced nothing for this long is counted as a timeout rather than waited on.
# Sized well above the near-32k prefill so a slow-but-working request is never miscounted as a
# failure; the point is to bound a hung cell, not to trim the tail.
REQUEST_TIMEOUT_SECONDS = 900.0

# How long a cancelled-but-unfinished task is given to actually die. Small relative to the drain
# allowance: by this point the request has already exceeded REQUEST_TIMEOUT_SECONDS and been
# cancelled, so this covers cleanup, not work.
_DRAIN_REAP_SECONDS = 30.0


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

    if kind == "ready":
        # The authoritative first-output timestamp. Production yields `ready` immediately after
        # `anext(output_stream)` returns vLLM's first output (see serving/src/engine/generation.py),
        # so it marks the end of prefill regardless of what that output DECODES to.
        #
        # Timestamping from the first non-empty delta instead would measure time-to-first-visible-
        # TEXT: a first token that decodes to "" -- a control token, or a partial UTF-8 sequence the
        # detokenizer is still buffering -- pushes TTFT out to a later token and overstates prefill.
        # That error is silent and always in the same direction.
        if outcome.first_token_at is None:
            outcome.first_token_at = now
    elif kind == "delta":
        text = event.get("text") or ""
        if text:
            outcome.delta_tokens += 1
            # Fallback only. A stream that somehow emitted no `ready` still gets a TTFT rather
            # than None, but `ready` always wins when present because it fires strictly earlier.
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
    # The fitter measured this prompt's assembled length offline; the engine reports what it
    # actually received. A bucket claims a specific input size, so a request the engine sized
    # differently belongs to a DIFFERENT bucket and must not be averaged into this one. The pooled
    # prompts are reseeded per request, which moves the header but not the body, so the tolerance
    # this compares against is the same one the fit was accepted under.
    expected = record.expected_prompt_tokens
    if expected is not None and abs(outcome.prompt_tokens - expected) > PROMPT_TOKEN_TOLERANCE:
        record.error = ERROR_PROMPT_LENGTH
        record.error_detail = (
            f"engine reported {outcome.prompt_tokens} prompt tokens; fitted {expected} "
            f"(tolerance {PROMPT_TOKEN_TOLERANCE})"
        )
        return
    # Against the BUCKET TARGET as well, not only the fitted count. A reseeded prompt carries the
    # pooled prompt's `exact`, but rewriting the header moves its real length, so the check above
    # measures drift from a stale value. The fit itself may already sit a full tolerance from the
    # target, so the two allowances compose: a wrapped request could land 2x tolerance out of bucket
    # and still be counted in it. The bucket label is the claim being published, so it is what the
    # engine's own count has to satisfy.
    target = record.bucket_target_tokens
    if target is not None and abs(outcome.prompt_tokens - target) > PROMPT_TOKEN_TOLERANCE:
        record.error = ERROR_PROMPT_LENGTH
        record.error_detail = (
            f"engine reported {outcome.prompt_tokens} prompt tokens; bucket target {target} "
            f"(tolerance {PROMPT_TOKEN_TOLERANCE})"
        )
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
    expected_prompt_tokens: int | None = None,
    bucket_target_tokens: int | None = None,
) -> RequestRecord:
    """Issue one streamed request and return its evidence record. Never raises."""
    record = RequestRecord(
        uid=uid,
        base_model=base_model,
        bucket=bucket,
        concurrency=concurrency,
        block=block,
        started_at=time.monotonic() - origin,
        expected_prompt_tokens=expected_prompt_tokens,
        bucket_target_tokens=bucket_target_tokens,
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


# Prompts fitted beyond `min_requests` so a cell that runs ahead of `min_seconds` does not wrap.
# Fixed, NOT a function of concurrency: a concurrency-dependent period makes the wrap point differ
# between points on the same curve, so two points would send different corpora and the difference
# would be read as a load effect.
_POOL_PERIOD_SLACK = 64

# Conservative per-call bound on one `fit_prompt_to_tokens` tokenization. Deliberately far slower
# than a real fast tokenizer: this funds a reservation, so it must err toward refusing a run.
_PROMPT_FIT_FIXED_SECONDS = 0.005
_PROMPT_FIT_SECONDS_PER_TOKEN = 2e-5  # 50k tokens/s
_PROMPT_FIT_MAX_ITERATIONS = 12


def prompt_fit_seconds_bound(bucket: Bucket, *, min_requests: int | None = None) -> float:
    """Upper bound on the wall time one cell spends fitting prompts.

    This time is NOT inside `bucket.max_seconds` and must not be: fitting runs before the clock
    starts, precisely so tokenization never competes with the measured window (see
    `_build_prompt_pool`). But it still runs INSIDE the GPU container, so it is billed. Excluding it
    from the window is a measurement decision; excluding it from the reservation was a funding gap,
    and at 31k input it is the largest unreserved term in a cell.

    Bounded at the binary search's iteration cap rather than its typical depth, because a
    reservation is an authorization to spend.
    """
    depth = bucket.min_requests if min_requests is None else min_requests
    prompts = depth + _POOL_PERIOD_SLACK
    per_call = _PROMPT_FIT_FIXED_SECONDS + _PROMPT_FIT_SECONDS_PER_TOKEN * float(
        bucket.target_input_tokens
    )
    return prompts * _PROMPT_FIT_MAX_ITERATIONS * per_call


_FITTING_WATCHDOG_GRACE_SECONDS = 30.0
_fitting_watchdog_armed = False


def _arm_fitting_watchdog(bound: float, *, label: str) -> None:
    """Arm a watchdog that can end the container from OUTSIDE a blocked tokenizer call.

    The per-prompt deadline check in the pool loop cannot fire while a fit is executing:
    `fit_prompt_to_tokens` calls into the tokenizer, which for the fast tokenizers is a Rust
    extension holding the GIL for the duration. A call that blocks -- or merely starts just under
    the deadline and runs long, which is likeliest on the final `near_32k` entry -- never returns
    control to the loop, so the deadline is checked again only after the overrun it exists to
    prevent. Meanwhile the container keeps billing against the class-wide `TIMEOUT_SECONDS`, which
    is sized for the largest bucket and is therefore far more GPU time than a short lane reserved.

    `faulthandler` rather than a `threading.Timer`, and the difference is the whole point. A timer
    fires a PYTHON callback, so it must acquire the GIL to run -- the same GIL the stalled Rust call
    is holding. In the exact scenario this watchdog exists for, a timer therefore never executes and
    the bound is unenforced; measured directly, a timer armed at 1s did not fire during a 60s C call,
    while `dump_traceback_later` terminated the process on schedule. `faulthandler`'s watchdog thread
    runs in C and calls `_exit` without touching the GIL, so it fires regardless of what the
    tokenizer is doing. It also prints the stalled thread's traceback, which names the blocked call.

    `exit=True` is what makes this an enforcement rather than a diagnostic: without it the handler
    dumps a traceback and lets the container keep billing.

    The grace period covers the case where the LAST call starts legitimately just inside the bound:
    without it the watchdog would kill a lane whose fitting was about to complete on time.
    """
    global _fitting_watchdog_armed
    print(
        f"[bench] fitting watchdog armed for {label}: the container ends if a single tokenizer "
        f"call blocks past {bound:.0f}s + {_FITTING_WATCHDOG_GRACE_SECONDS:.0f}s grace",
        flush=True,
    )
    # Cancels any watchdog already armed, so nesting cannot leave an earlier deadline running.
    faulthandler.dump_traceback_later(bound + _FITTING_WATCHDOG_GRACE_SECONDS, exit=True)
    _fitting_watchdog_armed = True


def _disarm_fitting_watchdog() -> None:
    global _fitting_watchdog_armed
    if _fitting_watchdog_armed:
        faulthandler.cancel_dump_traceback_later()
        _fitting_watchdog_armed = False


@contextlib.contextmanager
def fitting_watchdog(bound: float, *, label: str) -> Any:
    """Scope a fitting watchdog around any synchronous fit, not just the cell pool.

    The warmup fits its prompts the same way a cell does, but OUTSIDE `run_request`, so the request
    timeout that bounds every other generation never applies to it. Both estimators price a warmup
    at exactly `REQUEST_TIMEOUT_SECONDS`; a tokenizer stalling there kept `certify`/`run_bucket`
    alive to the class timeout instead, billing GPU-seconds no reservation covered.
    """
    _arm_fitting_watchdog(bound, label=label)
    try:
        yield
    finally:
        _disarm_fitting_watchdog()


def fitting_watchdog_grace_seconds() -> float:
    """Grace the fitting watchdog allows past a bound before ending the container.

    Exposed so the estimators reserve what the watchdog actually terminates at rather than the
    nominal bound. A lane that reserves the bound but permits the grace can bill past its own
    authorization, which is the failure `BudgetLedger` exists to make impossible.
    """
    return _FITTING_WATCHDOG_GRACE_SECONDS


def drain_reap_seconds() -> float:
    """How long a drain waits for a task that ignored cancellation, past the drain's own timeout.

    Exposed for the same reason as the watchdog grace: it is bounded, billed, and happens at every
    concurrency point, so it belongs in the reservation rather than in unfunded slack.
    """
    return _DRAIN_REAP_SECONDS


def _build_prompt_pool(
    tokenizer: Any,
    bucket: Bucket,
    *,
    concurrency: int,
    block: int,
    min_requests: int,
    invocation: str = "",
) -> list[tuple[str, list[dict[str, Any]], int]]:
    """Fit every prompt the cell can need BEFORE the measured window opens.

    Each entry is request-unique from its first token, so no two requests share a prefix and the
    engine's prefix cache cannot turn a measurement into a lookup.

    The filler body is seeded WITHOUT the concurrency point (`corpus_seed`), so every point on a
    curve sends the same corpus and differs only in offered load. The per-request header still
    carries the uid digest, so requests remain mutually unique.

    A cell can still outrun this pool while waiting on `min_seconds`, so the caller reseeds a pooled
    prompt rather than re-sending it; see `reseed_prompt`. The pool is therefore sized for the depth
    the cell is expected to need, not for a bound it is forbidden to exceed.

    The pool PERIOD is deliberately independent of `concurrency`. Sizing it as
    `min_requests + concurrency` made the wrap point differ per point on the curve, so request 301
    reused corpus 0 at c=1 but corpus 301 at c=16 -- two points on one curve then sent different
    semantic prompt sequences, which moves completion lengths and the derived knee for a reason that
    has nothing to do with offered load. `_POOL_PERIOD_SLACK` widens the pool enough that a cell
    running ahead of `min_seconds` still has unwrapped prompts, without making the period a function
    of the point being measured. Requests stay mutually unique past the wrap because the reseeded
    header carries the uid digest.
    """
    size = min_requests + _POOL_PERIOD_SLACK
    # No `max(concurrency, ...)` floor: reinstating one would put concurrency back into the
    # period. A pool shorter than one in-flight set is harmless anyway -- wrapped requests are
    # RESEEDED, so they diverge at character zero and cannot share a cache block.
    pool: list[tuple[str, list[dict[str, Any]], int]] = []
    # The bound this loop was RESERVED under, enforced rather than assumed. `prompt_fit_seconds_bound`
    # funds a tokenization rate the sweep estimate is built on; nothing made the fit obey it, so a
    # tokenizer running slower than assumed would keep billing against the class's shared
    # `TIMEOUT_SECONDS` -- which for a short-only sweep is sized for `near_32k`, i.e. far more GPU
    # time than this lane reserved. Checked per prompt because that is the only point the loop
    # yields: a single blocked tokenizer call is a C call Python cannot interrupt.
    bound = prompt_fit_seconds_bound(bucket, min_requests=min_requests)
    deadline = time.monotonic() + bound
    _arm_fitting_watchdog(bound, label=f"{bucket.name} pool")
    for index in range(size):
        if time.monotonic() > deadline:
            # Ends the process rather than raising, matching `_probe_in_container_within_bound`. A
            # raise would unwind into the same container that is still billing, and the reservation
            # this loop just exceeded is the one that authorized the spend. Flushed first because
            # `os._exit` skips atexit handlers, and this line is the only record of why the lane died.
            print(
                f"[bench] prompt fitting for {bucket.name} exceeded its reserved bound after "
                f"{index}/{size} prompts; ending the container rather than billing past the "
                f"reservation",
                flush=True,
            )
            sys.stderr.flush()
            os._exit(75)
        uid = request_uid(bucket.name, concurrency, block, index, invocation)
        messages, exact = fit_prompt_to_tokens(
            tokenizer,
            uid,
            bucket.target_input_tokens,
            # No invocation nonce here, on purpose: the BODY stays keyed to the grid coordinates so
            # a rerun sends the same corpus, while the uid-derived header makes each request unique.
            corpus=corpus_seed(bucket.name, block, index),
        )
        pool.append((uid, messages, exact))
    _disarm_fitting_watchdog()
    return pool


async def _drain(
    in_flight: set[asyncio.Task[RequestRecord]],
    *,
    base_model: str,
    bucket: str,
    concurrency: int,
    block: int,
    spawned_at: dict[asyncio.Task[RequestRecord], float],
    spawned_uid: dict[asyncio.Task[RequestRecord], str],
) -> list[RequestRecord]:
    """Let the still-in-flight requests finish, and return a record for every one of them.

    Drain rather than abandon. Two reasons, and the ORDER matters:

    1. An orphaned request keeps occupying the engine during the NEXT cell and contaminates it.
    2. Cancelling first and awaiting second silently DELETES those requests from the attempt
       denominator: `task.cancel()` raises CancelledError at the await point, the suppress swallows
       it, and the append never runs. An overloaded cell then reports a cleaner error rate than it
       earned, which is the exact direction a capacity claim must not err.

    So issued work finishes on its own (each request carries its own REQUEST_TIMEOUT_SECONDS, so
    this terminates) and every record is counted. Only a task still pending after that bound is
    cancelled, and it is recorded as a timeout rather than dropped.
    """
    drained: list[RequestRecord] = []
    done, still_pending = await asyncio.wait(in_flight, timeout=REQUEST_TIMEOUT_SECONDS)
    for task in done:
        with contextlib.suppress(asyncio.CancelledError):
            drained.append(task.result())
    for task in still_pending:
        task.cancel()
    for task in still_pending:
        # Bounded. `Task.cancel()` only REQUESTS cancellation: if the engine's stream does not
        # cooperate -- an async-generator close blocked in a backend call, say -- an unbounded
        # `await task` sits here until the class-wide method timeout kills the container, losing the
        # whole bucket artifact this drain exists to preserve. The reap gets a slice of the same
        # allowance the sweep already reserved for the drain.
        with contextlib.suppress(asyncio.CancelledError, TimeoutError, Exception):
            await asyncio.wait_for(asyncio.shield(task), timeout=_DRAIN_REAP_SECONDS)
        if not task.done():
            # The record below is still appended, so the request keeps its place in the attempt
            # denominator; what is unrecoverable is the container, which now holds a task pinned
            # inside the engine and would contaminate every later cell on it.
            print(
                f"[bench] {bucket} c={concurrency} left an uncancellable request after "
                f"{_DRAIN_REAP_SECONDS:.0f}s; ending the container rather than measuring around it",
                flush=True,
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(75)
        # `started_at` is REQUIRED and has no default. Omitting it raised TypeError here, which
        # aborted the whole bucket and discarded every record accumulated during the expensive
        # sweep -- the exact loss this drain exists to prevent. The task's own start offset is
        # carried on the task so a drained record keeps its real origin.
        drained.append(
            RequestRecord(
                uid=spawned_uid.get(task, f"drain-{id(task):x}"),
                base_model=base_model,
                bucket=bucket,
                concurrency=concurrency,
                block=block,
                started_at=spawned_at.get(task, 0.0),
                error=ERROR_TIMEOUT,
                error_detail="request did not finish within the drain bound",
            )
        )
    return drained


def _prompt_issuer(
    tokenizer: Any,
    bucket: Bucket,
    *,
    concurrency: int,
    block: int,
    min_requests: int,
    invocation: str = "",
) -> Callable[[], tuple[str, list[dict[str, Any]], int]]:
    """Build the cell's whole prompt pool, and return a callable handing out one prompt per call.

    Each call returns (uid, messages, fitted_length).

    Past the end of the pool the prompt is RESEEDED, never reused. Re-sending a pooled prompt would
    hit the engine's prefix cache, which `_validate` correctly rejects as ERROR_CACHE_CONTAMINATED
    -- so a cell that ran FASTER, and therefore wrapped, would earn an error rate for being fast.
    Reseeding rewrites only the per-request header, so the new prompt diverges at character zero
    without paying for tokenization inside the measured window.
    """
    pool = _build_prompt_pool(
        tokenizer,
        bucket,
        concurrency=concurrency,
        block=block,
        min_requests=min_requests,
        invocation=invocation,
    )
    issued = 0

    def _next_prompt() -> tuple[str, list[dict[str, Any]], int]:
        nonlocal issued
        index = issued
        issued += 1
        uid, messages, exact = pool[index % len(pool)]
        if index < len(pool):
            return uid, messages, exact
        wrapped_uid = request_uid(bucket.name, concurrency, block, index, invocation)
        return wrapped_uid, reseed_prompt(messages, wrapped_uid), exact

    return _next_prompt


_CONCLUSIVE_FAILURE_ATTEMPTS = 64


def _cell_is_conclusively_failed(records: list[RequestRecord]) -> bool:
    """True once enough attempts have failed outright that more would add nothing.

    Deliberately requires a floor of attempts rather than tripping on the first errors: a cell that
    is merely overloaded produces a MIX, and its error rate is the measurement. What this catches is
    the degenerate case -- many attempts, zero successes -- where the cell's verdict is already
    fixed and every further respawn only inflates the artifact.

    Not a substitute for the error rate. The records collected so far are still reduced and
    reported, so the failed cell appears in the curve as a failed cell.
    """
    if len(records) < _CONCLUSIVE_FAILURE_ATTEMPTS:
        return False
    return not any(record.error is None for record in records)


def _make_spawner(
    engine: Any,
    tokenizer: Any,
    base_model: str,
    bucket: Bucket,
    *,
    concurrency: int,
    block: int,
    min_requests: int,
    invocation: str,
) -> tuple[float, Callable[[], asyncio.Task[RequestRecord]], dict[Any, float], dict[Any, str]]:
    """A launcher for one cell's requests, plus the books the drain needs to account for them.

    Fitting is repeated synchronous tokenization. Done lazily it would run ON the event loop: the
    first ``concurrency`` fits land inside the measured window as idle wall time, and every
    replacement fit blocks consumption of the OTHER in-flight streams, inflating their TTFT and
    latency. At 8k and 31k input that distortion is larger than the effect being measured. So the
    whole pool is built HERE, and the cell's clock starts only AFTER it -- hence the returned
    ``origin``. Starting the clock first would charge the cell's ``max_seconds`` for tokenization
    that deliberately happens outside the measured window, shortening a near-32k window by minutes.

    ``spawned_at`` and ``spawned_uid`` are returned rather than kept private because a request that
    never finishes has no record of its own: the drain reconstructs one from these, and a task
    missing from them would silently vanish from the attempt denominator.
    """
    _next_prompt = _prompt_issuer(
        tokenizer,
        bucket,
        concurrency=concurrency,
        block=block,
        min_requests=min_requests,
        invocation=invocation,
    )
    # The measured window opens HERE, after every prompt is fitted.
    origin = time.monotonic()
    spawned_at: dict[Any, float] = {}
    spawned_uid: dict[Any, str] = {}

    def _spawn() -> asyncio.Task[RequestRecord]:
        uid, messages, exact = _next_prompt()
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
                expected_prompt_tokens=exact,
                bucket_target_tokens=bucket.target_input_tokens,
            )
        )
        spawned_at[task] = started_at
        spawned_uid[task] = uid
        return task

    return origin, _spawn, spawned_at, spawned_uid


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
    invocation: str = "",
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

    # `origin` comes back from the spawner because the clock starts AFTER the pool is fitted.
    origin, _spawn, spawned_at, spawned_uid = _make_spawner(
        engine,
        tokenizer,
        base_model,
        bucket,
        concurrency=concurrency,
        block=block,
        min_requests=min_requests,
        invocation=invocation,
    )

    in_flight = {_spawn() for _ in range(concurrency)}
    # Set when the steady-state window closes, BEFORE draining. See the wall-clock note below.
    measured_seconds: float | None = None
    try:
        while True:
            # Bounded by the cell's own remaining time. An unbounded wait returns only when a request
            # completes, so `max_seconds` could not bind: a cell whose requests all stalled -- the
            # exact shape of an overloaded engine -- sat until every request hit its individual
            # 900s timeout instead of stopping at the bucket's bound. `max_seconds` is what keeps a
            # degenerate cell from spending the sweep's whole budget.
            remaining = max_seconds - (time.monotonic() - origin)
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(
                in_flight, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            in_flight = set(pending)
            records.extend(task.result() for task in done)
            elapsed = time.monotonic() - origin
            enough = len(records) >= min_requests and elapsed >= min_seconds
            if enough or elapsed >= max_seconds:
                break
            if _cell_is_conclusively_failed(records):
                # A dead engine fails `_stream_generate` immediately, so every replacement completes
                # as an error at once and the loop becomes a tight respawn against `min_seconds`.
                # Self-healing is deliberately disabled, so nothing interrupts it: the cell
                # accumulates hundreds of thousands of records and an artifact large enough to
                # exhaust memory, destroying the evidence of the failure it already established.
                # The result is conclusive long before the floor -- keep it and stop.
                print(
                    f"[bench] {bucket.name} c={concurrency} failed "
                    f"{len(records)}/{len(records)} requests with no successes; ending the cell "
                    f"rather than respawning against its time floor",
                    flush=True,
                )
                break
            for _ in done:
                in_flight.add(_spawn())
    finally:
        # The measured window ends HERE, with `concurrency` requests still in flight. Everything
        # after this point is teardown at falling load.
        measured_seconds = time.monotonic() - origin
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
            records.extend(
                await _drain(
                    in_flight,
                    base_model=base_model,
                    bucket=bucket.name,
                    concurrency=concurrency,
                    block=block,
                    spawned_at=spawned_at,
                    spawned_uid=spawned_uid,
                )
            )

    # Rates are divided by the STEADY-STATE window, not by wall time including the drain.
    #
    # The drain runs at falling concurrency: no request is replaced, so the last one finishes alone
    # on an otherwise idle engine. Counting that tail in the denominator divides steady-state work by
    # steady-state-plus-idle time and understates every rate. The bias is worst exactly where the
    # numbers matter most -- a near-32k cell can drain for minutes after a 60s window -- and it grows
    # with concurrency, so it would bend the curve downward at the high end and manufacture a knee
    # that the engine does not have.
    #
    # Drained records are still COUNTED in the attempt denominator and the error breakdown. What
    # they are excluded from is the RATE and LATENCY numerators: a request that completed during the
    # tail did its work at falling concurrency, so crediting it to the window would divide
    # window-plus-tail output by window-only time. `window_seconds` is what lets `reduce_cell` tell
    # the two apart -- records carry `finished_at` on the same monotonic origin.
    total_seconds = time.monotonic() - origin
    window_seconds = measured_seconds if measured_seconds else total_seconds
    result = reduce_cell(
        records,
        base_model=base_model,
        bucket=bucket.name,
        concurrency=concurrency,
        block=block,
        wall_seconds=window_seconds,
        drain_seconds=total_seconds - window_seconds,
        window_seconds=window_seconds,
    )
    return result, records


__all__ = [
    "REQUEST_TIMEOUT_SECONDS",
    "base_model_record",
    "drain_reap_seconds",
    "fitting_watchdog",
    "fitting_watchdog_grace_seconds",
    "prompt_fit_seconds_bound",
    "run_cell",
    "run_request",
]
