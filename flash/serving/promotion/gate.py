"""Sequence the three proofs a promotion needs, and fail the deploy without them.

Run as `python -m flash.serving.promotion.gate` from `.github/workflows/deploy-modal.yml`, after the
readiness poll. Order is load-bearing and short-circuits:

  health  -> is the live router THIS release, with engines to route to?
  stream  -> did a real authenticated request generate real tokens over a real stream?
  usage   -> is this release's durable delivery loop running rather than wedged?

Each stage is skipped when an earlier one failed. Streaming against a router that is not this
release would prove nothing about it, and inspecting an accounting backlog for a generation that
never happened would just burn the deadline before failing anyway.

The exact-head binding comes from stages 1 and 2 together: the router proved its identity, and then
that same router served a real generation. Stage 3 is deliberately weaker than "the canary's own row
settled" -- `serving_usage_backlog_snapshot` aggregates every generation in flight and offers no
per-correlation read, so a zero-backlog assertion would be fail-open under concurrent traffic and
flaky besides. See `verify_accounting`.

Credentials are read from the environment only, never from argv: `run:` lines are echoed into the
public build log, so an interpolated key leaks on every run including the green ones.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

from flash.serving.promotion.canary import (
    CANARY_TRANSPORT_FAILURE,
    CanaryError,
    CanaryRequest,
    correlation_id_for,
    run_stream_canary,
)
from flash.serving.promotion.evidence import (
    ACCOUNTING_MALFORMED,
    PromotionVerdict,
    StreamEvidence,
    parse_accounting,
    parse_health,
    verify_accounting,
    verify_health,
    verify_stream,
)
from flash.serving.src.store.settings import Settings
from flash.serving.src.store.supabase_rest import supabase_headers

# `Any` return types are the honest shape here: these load UNVALIDATED json off the wire, and
# narrowing them is precisely what `parse_health`/`parse_accounting` exist to do. The httpx client,
# by contrast, IS statically known -- CI installs the `serving` extra -- so it is typed under
# TYPE_CHECKING, which keeps the runtime import deferred into `_run` without costing the annotation.
HealthLoader = Callable[[], Awaitable[Any]]
StreamRunner = Callable[[], Awaitable[StreamEvidence]]
AccountingLoader = Callable[[], Awaitable[Any]]
BacklogReader = Callable[["httpx.AsyncClient"], Awaitable[Any]]

GATE_CONFIG_INCOMPLETE = "gate_config_incomplete"
HEALTH_UNREACHABLE = "health_unreachable"

# the canary is the FIRST request to a freshly deployed release, and engines are scale-to-zero
# (`MIN_CONTAINERS = 0`), so this budget must cover a full GPU cold start before the first token.
# `/healthz` cannot have pre-warmed anything: `health_body` reports CONFIGURED per-model tiers and
# says so explicitly ("remains stable when demand-driven containers scale to zero"), so a passing
# identity check proves a router booted, not that any engine is up. Sized for a cold engine boot
# with headroom rather than the warm-path latency, because a promotion that fails on `canary_timeout`
# does not merely fail -- it rolls a healthy release back to its predecessor.
_DEFAULT_CANARY_TIMEOUT_SECONDS = 420.0
# long enough for a lease that expired to be RECLAIMED, not merely observed. `modal deploy` replaces
# the router containers, and a container holding a claimed row when it goes away leaves that lease to
# expire rather than releasing it -- so a deploy itself produces the exact signal `stalled` reads.
# `DurableUsageOutbox` leases for 60s and the replacement worker reclaims on its next sweep, so a
# deploy-induced expiry clears well inside this window while a genuinely wedged loop never does. A
# deadline at or below the lease would turn every deploy into a rollback of a healthy release.
#
# this is deliberately SHORTER than `_STALL_AGE_SECONDS`, and the two are not in tension: this
# deadline bounds how long the gate waits for a reclaim, while the threshold decides what counts
# as wedged in the first place. a row still inside the outbox retry budget already reads healthy
# on the first poll, so there is nothing for a longer deadline here to wait out.
_DEFAULT_ACCOUNTING_DEADLINE_SECONDS = 180.0
_ACCOUNTING_POLL_SECONDS = 5.0
_BACKLOG_SNAPSHOT_RPC = "serving_usage_backlog_snapshot"
_MAX_TOKENS = 32
_REQUIRED_ENV = (
    "FREESOLO_INTERNAL_KEY",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "GITHUB_SHA",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
)


async def evaluate_promotion(
    *,
    health_loader: HealthLoader,
    stream_runner: StreamRunner,
    accounting_loader: AccountingLoader,
    expected_sha: str,
    expected_deployment_id: str,
    accounting_deadline_seconds: float = _DEFAULT_ACCOUNTING_DEADLINE_SECONDS,
    poll_seconds: float = _ACCOUNTING_POLL_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> PromotionVerdict:
    try:
        payload = await health_loader()
    # an unreachable router is a failed promotion, not a crashed step: a crash skips the rollback.
    except Exception:
        return PromotionVerdict(ok=False, reason=HEALTH_UNREACHABLE)
    verdict = verify_health(
        parse_health(payload),
        expected_sha=expected_sha,
        expected_deployment_id=expected_deployment_id,
    )
    if not verdict.ok:
        return verdict

    try:
        evidence = await stream_runner()
    except CanaryError as exc:
        return PromotionVerdict(ok=False, reason=str(exc))
    # `run_stream_canary` already collapses transport faults into `CanaryError`, but that guarantee
    # lives in another module and `stream_runner` is injected. The consequence of being wrong lands
    # here, not there: an escaped exception crashes the step, and `failure()` cannot tell a crash
    # from a failure, so it would roll production back to its predecessor. Same reasoning as the
    # health load above.
    except Exception:
        return PromotionVerdict(ok=False, reason=CANARY_TRANSPORT_FAILURE)
    verdict = verify_stream(evidence)
    if not verdict.ok:
        return verdict

    return await _await_accounting(
        accounting_loader,
        deadline_seconds=accounting_deadline_seconds,
        poll_seconds=poll_seconds,
        sleep=sleep,
    )


async def _await_accounting(
    loader: AccountingLoader,
    *,
    deadline_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], Awaitable[None]],
) -> PromotionVerdict:
    """Poll until delivery looks healthy, bounded.

    Delivery is asynchronous, so a single stalled-looking read right after a generation is not yet
    evidence of anything: a lease can expire and be recovered, and an undelivered row ages until the
    worker claims it. Only a stall that persists across the deadline indicates a wedged loop, which
    is why this retries rather than failing on the first bad read.
    """
    waited = 0.0
    # the standing verdict before any read succeeds: an accounting stage that never got a readable
    # snapshot has proven nothing, so it must not fall through as a pass.
    verdict = PromotionVerdict(ok=False, reason=ACCOUNTING_MALFORMED)
    while True:
        try:
            snapshot = await loader()
        # an unreadable snapshot must never pass as settled.
        except Exception:
            verdict = PromotionVerdict(ok=False, reason=ACCOUNTING_MALFORMED)
            snapshot = None
        if snapshot is not None:
            verdict = verify_accounting(parse_accounting(snapshot))
            if verdict.ok:
                return verdict
        if waited >= deadline_seconds:
            return verdict
        await sleep(poll_seconds)
        waited += poll_seconds


class GateConfigError(Exception):
    """The gate cannot be built from this environment, so it never gets to run.

    Carries no detail on purpose: every input here is either a secret or derived from one, and the
    caller turns this into a log line.
    """


@dataclass(frozen=True)
class GatePlan:
    """A gate that is already known to be constructible; the type is the proof.

    Holding the resolved request and reader together is what lets `_resolve` own every rejection and
    leaves `_run` with nothing left to validate.
    """

    request: CanaryRequest
    read_backlog: BacklogReader
    expected_sha: str
    expected_deployment_id: str


def _resolve(base_url: str, env: dict[str, str]) -> GatePlan:
    """Build the plan from the step's environment, or refuse before any request is made.

    Every rejection lives HERE, above `asyncio.run`, because a gate that raises is strictly worse
    than one that fails: the rollback step keys off `failure()`, which fires on a crashed step
    exactly as it does on a failed one, so a misconfigured secret would roll production back to its
    predecessor instead of simply declining to promote it. All three rejections -- an incomplete
    environment, `supabase_headers` refusing a key that is not `sb_secret_`-shaped, and an empty
    base-model catalog -- are knowable before the first request, so each is answered with a reason
    code rather than a traceback.

    Taking credentials as an argument rather than through `get_settings()` also keeps this
    constructible in a test: `get_settings` is `lru_cache`d over real process env, so it cannot be
    exercised offline without leaking global state between tests.
    """
    # naming the missing variable would be friendlier, but a secret-shaped value must never
    # influence log output, and "which one" is one bisection away from the value itself.
    if not base_url or not all(env.get(name) for name in _REQUIRED_ENV):
        raise GateConfigError

    from flash.serving.src.engine.model_config import base_models

    models = base_models()
    if not models:
        raise GateConfigError

    key = env["SUPABASE_SERVICE_ROLE_KEY"]
    url = f"{env['SUPABASE_URL'].rstrip('/')}/rest/v1/rpc/{_BACKLOG_SNAPSHOT_RPC}"
    # one RPC read, deliberately NOT through `DurableUsageOutbox`: that is the delivery WORKER, and
    # constructing one demands a `worker_id`, a `backend_url`, and a `deployment_id` that a single
    # read never uses. reuse the canonical header helper so the service-role format check and the
    # postgrest schema routing stay identical to every other supabase caller in the repo.
    # a REAL `Settings`, not a look-alike: `model_construct` skips validation and env resolution, so
    # it neither reads the runner's ambient environment nor loads a `.env`, and the two fields the
    # helper touches are the two supplied here. a `SimpleNamespace` would satisfy the call at runtime
    # while lying to the type checker about what `supabase_headers` accepts.
    try:
        headers = supabase_headers(
            Settings.model_construct(
                supabase_url=env["SUPABASE_URL"], supabase_service_role_key=key
            ),
            "public",
        )
    except RuntimeError as exc:
        raise GateConfigError from exc
    headers["Authorization"] = f"Bearer {key}"

    async def read_backlog(client: httpx.AsyncClient) -> Any:
        response = await client.post(url, headers=headers, json={}, timeout=15)
        response.raise_for_status()
        return response.json()

    run_id, attempt = env["GITHUB_RUN_ID"], env["GITHUB_RUN_ATTEMPT"]
    return GatePlan(
        request=CanaryRequest(
            base_url=base_url,
            # ONE model, not all of `base_models()`. This is a deliberate limit and it is narrower
            # than the health check beside it: `/healthz` reports `gpus` as the count of CONFIGURED
            # tiers (3 as of the 27B activation), so a passing gate must not be read as three
            # proven engines. A release that breaks only the 27B or 35B path promotes cleanly.
            #
            # Looping all three would put three scale-to-zero cold starts inside one 420s budget,
            # and a canary that times out fails the gate -- which redeploys the predecessor over a
            # healthy release. Widening coverage by making spurious rollbacks likelier is a bad
            # trade; per-model gating needs its own budget, not a longer single wait.
            model=models[0],
            api_key=env["FREESOLO_INTERNAL_KEY"],
            correlation_id=correlation_id_for(run_id, attempt),
            timeout_seconds=_DEFAULT_CANARY_TIMEOUT_SECONDS,
            max_tokens=_MAX_TOKENS,
        ),
        read_backlog=read_backlog,
        expected_sha=env["GITHUB_SHA"],
        expected_deployment_id=f"{run_id}-{attempt}",
    )


async def _run(plan: GatePlan) -> PromotionVerdict:
    import httpx

    base_url = plan.request.base_url.rstrip("/")
    async with httpx.AsyncClient() as client:

        async def health_loader() -> Any:
            response = await client.get(f"{base_url}/healthz", timeout=15)
            response.raise_for_status()
            return response.json()

        async def stream_runner() -> StreamEvidence:
            return await run_stream_canary(plan.request, client=client)

        async def accounting_loader() -> Any:
            return await plan.read_backlog(client)

        return await evaluate_promotion(
            health_loader=health_loader,
            stream_runner=stream_runner,
            accounting_loader=accounting_loader,
            expected_sha=plan.expected_sha,
            expected_deployment_id=plan.expected_deployment_id,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prove a hosted serving promotion")
    # only non-secret configuration is accepted as an argument.
    parser.add_argument("--base-url", default=os.environ.get("SERVING_BASE_URL", ""))
    args = parser.parse_args(argv)
    env = {name: os.environ.get(name, "") for name in _REQUIRED_ENV}
    try:
        plan = _resolve(args.base_url, env)
    except GateConfigError:
        print(f"promotion gate failed: {GATE_CONFIG_INCOMPLETE}", file=sys.stderr)
        return 1
    verdict = asyncio.run(_run(plan))
    if verdict.ok:
        print("promotion gate passed")
        return 0
    print(f"promotion gate failed: {verdict.reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
