"""The warmup a cold container runs before any measured cell opens.

A sibling of `driver` rather than part of it, and NOT part of the entrypoint script. The split from
`driver` is only the file-size gate; what matters is that this stays inside `flash.serving.bench`,
because `_execution_digest` reads that package. A warmup contract defined in the script sat outside
the digest entirely, so the count, the prompt shape or the sequencing could all move while
`workload_checksum` stayed byte-identical -- and two campaigns that excluded different startup costs
compared as if they had measured the same thing.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from flash.serving.bench.driver import (
    fitting_watchdog,
    prompt_fit_seconds_bound,
    run_request_within_bound,
)
from flash.serving.bench.workload import BUCKETS_BY_NAME, fit_prompt_to_tokens

# How many warmup requests a cold container issues before any cell opens. Digested by value: the
# warmup source names this constant without revealing it, and the number itself decides how much
# compilation and lazy-initialization cost is moved out of the measured window. Five warmups and one
# warmup exclude materially different startup cost from the published curve.
CANARY_WARMUP_REQUESTS = 5


def warmup_fit_seconds_bound() -> float:
    """Wall time one warmup prompt fit is allowed.

    A warmup fits ONE `short_interactive` prompt, so it is priced from that bucket's own bound at a
    pool of one rather than from a cell's whole pool.
    """
    return prompt_fit_seconds_bound(BUCKETS_BY_NAME["short_interactive"], min_requests=1)


async def run_warmup(engine: Any, requests: int) -> dict[str, Any]:
    """Sequential warmups on ``engine``, reported separately from the envelope.

    Lives here rather than in the entrypoint script for the same reason `grid_should_halt` does:
    `_execution_digest` reads this module, so a warmup contract defined in the script sat outside
    its reach entirely. Every canary and every cold replacement runs this before a cell opens, and
    the warmup is what moves compilation and lazy-initialization cost OUT of the measured window --
    so changing the count, the prompt shape or the sequencing changes which costs the published
    curve excludes. With the contract in the script, all of that moved while `workload_checksum`
    stayed byte-identical, and two campaigns that excluded different startup costs compared as if
    they had measured the same thing.

    Sequential on purpose. Warmups issued concurrently would compile under contention and leave a
    different residue than the one-at-a-time path every measured cell is preceded by.
    """
    bucket = BUCKETS_BY_NAME["short_interactive"]
    origin = time.monotonic()
    out = []
    exact = 0
    # Prompts are derived from the UID, so a fixed `warmup-{i}` reissued the SAME five prompts on
    # every invocation. Within the 120s scaledown window the container survives, and the second
    # canary hits a retained prefix cache -- which the driver correctly scores as
    # ERROR_CACHE_CONTAMINATED, refusing the sweep even though generation was healthy. A nonce per
    # invocation makes each warmup prompt request-unique from its first token, like every other
    # prompt the harness issues.
    nonce = uuid.uuid4().hex[:12]
    for index in range(requests):
        uid = f"warmup-{nonce}-{index}"
        # Bounded by the same enforcement the cell pool uses. This fit runs BEFORE `run_request`, so
        # the 900s request timeout that both estimators price a warmup at does not cover it; without
        # a watchdog a stalled tokenizer here billed to the class-wide timeout instead.
        with fitting_watchdog(warmup_fit_seconds_bound(), label=f"warmup-{index}"):
            messages, exact = fit_prompt_to_tokens(
                engine.tokenizer, uid, bucket.target_input_tokens
            )
        # Bounded, unlike a bare `run_request`. Its `wait_for` waits for cancellation CLEANUP to
        # finish, so a stream whose close blocks keeps the call open past the request timeout with
        # no exception raised. A measured cell survives that because `_drain` awaits a shielded
        # task; this warmup awaits directly, so it needs the same enforcement or one hung close
        # bills to the class-wide method timeout against a `REQUEST_TIMEOUT_SECONDS` reservation.
        record = await run_request_within_bound(
            engine,
            engine.base_model,
            messages,
            bucket.max_output_tokens,
            uid,
            bucket="warmup",
            concurrency=1,
            block=0,
            origin=origin,
            # The warmup is the gate that runs before any sweep, so it is the cheapest place to
            # discover that the engine sizes prompts differently than the fitter does.
            expected_prompt_tokens=exact,
        )
        out.append(record.to_json())
    # Marked HERE, after the requests ran, and only when they all succeeded. Set before the loop it
    # recorded intent rather than outcome: a warmup that returned failed records -- or raised part
    # way, since `fitting_watchdog` and the request bound both raise -- still left the container
    # flagged warm. `_ensure_warm` rejects that attempt, but the flag outlives the rejection, so a
    # retry inside the scaledown window short-circuits at the flag check and measures on a container
    # whose generation path was never proven. Every caller reads the same flag, so the one place
    # that knows whether the warmup actually worked is the one place that sets it.
    if out and all(record.get("ok") for record in out):
        engine._bench_warmed = True
    return {"warmups": out, "assembled_prompt_tokens": exact}
