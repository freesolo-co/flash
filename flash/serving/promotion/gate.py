"""Sequence the three proofs a promotion needs, and fail the deploy without them.

Run as `python -m flash.serving.promotion.gate` from `.github/workflows/deploy-modal.yml`, after the
readiness poll. Order is load-bearing and short-circuits:

  health  -> is the live router THIS release, with engines to route to?
  stream  -> did a real authenticated request generate real tokens over a real stream?
  usage   -> is this release's durable delivery loop running rather than wedged?

Each stage is skipped when an earlier one failed. Streaming against a router that is not this
release would prove nothing about it, and inspecting an accounting backlog for a generation that
never happened would just burn the deadline before failing anyway.

The accounting RPC scopes durable settlement to the canary's correlation id, organization, model,
deployment id, and release SHA. Fresh pending usage is not evidence of delivery.

Credentials are read from the environment only, never from argv: `run:` lines are echoed into the
public build log, so an interpolated key leaks on every run including the green ones.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

from flash.serving.app.timeouts import CANARY_TIMEOUT_SECONDS
from flash.serving.promotion.canary import (
    CANARY_TRANSPORT_FAILURE,
    CanaryError,
    CanaryRequest,
    correlation_id_for,
    run_stream_canary,
)
from flash.serving.promotion.evidence import (
    ACCOUNTING_MALFORMED,
    HEALTH_DEPLOYMENT_ID_MISMATCH,
    HEALTH_SHA_MISMATCH,
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

# long enough for a lease that expired to be RECLAIMED, not merely observed. `modal deploy` replaces
# the router containers, and a container holding a claimed row when it goes away leaves that lease to
# expire rather than releasing it -- so a deploy itself produces the exact signal `stalled` reads.
# `DurableUsageOutbox` leases for 60s and the replacement worker reclaims on its next sweep, so a
# deploy-induced expiry clears well inside this window while a genuinely wedged loop never does. A
# deadline at or below the lease would turn every deploy into a rollback of a healthy release.
#
_DEFAULT_ACCOUNTING_DEADLINE_SECONDS = 180.0
_ACCOUNTING_POLL_SECONDS = 5.0
# Modal can briefly route health requests to both releases during a rollout, even
# after the workflow's first successful readiness probe.
_HEALTH_ROLLOUT_DEADLINE_SECONDS = 60.0
_BACKLOG_SNAPSHOT_RPC = "serving_promotion_accounting_snapshot"
_MAX_TOKENS = 32
_REQUIRED_ENV = (
    "FREESOLO_INTERNAL_KEY",
    "SERVING_CANARY_ORG_ID",
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
    health_deadline_seconds: float = _HEALTH_ROLLOUT_DEADLINE_SECONDS,
    accounting_deadline_seconds: float = _DEFAULT_ACCOUNTING_DEADLINE_SECONDS,
    poll_seconds: float = _ACCOUNTING_POLL_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> PromotionVerdict:
    verdict = await _await_health(
        health_loader,
        expected_sha=expected_sha,
        expected_deployment_id=expected_deployment_id,
        deadline_seconds=health_deadline_seconds,
        poll_seconds=poll_seconds,
        sleep=sleep,
        clock=clock,
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
        clock=clock,
    )


async def _await_health(
    loader: HealthLoader,
    *,
    expected_sha: str,
    expected_deployment_id: str,
    deadline_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], Awaitable[None]],
    clock: Callable[[], float],
) -> PromotionVerdict:
    """Wait out rollout identity drift without sending a canary to a stale release."""
    deadline = clock() + max(0.0, deadline_seconds)
    verdict = PromotionVerdict(ok=False, reason=HEALTH_UNREACHABLE)
    while (remaining := deadline - clock()) > 0:
        try:
            payload = await asyncio.wait_for(loader(), timeout=remaining)
        except TimeoutError:
            return verdict
        except Exception:
            return PromotionVerdict(ok=False, reason=HEALTH_UNREACHABLE)
        if clock() >= deadline:
            return verdict
        verdict = verify_health(
            parse_health(payload),
            expected_sha=expected_sha,
            expected_deployment_id=expected_deployment_id,
        )
        if verdict.reason not in (HEALTH_SHA_MISMATCH, HEALTH_DEPLOYMENT_ID_MISMATCH):
            return verdict
        remaining = deadline - clock()
        if remaining <= 0:
            break
        await sleep(min(max(0.0, poll_seconds), remaining))
    return verdict


async def _await_accounting(
    loader: AccountingLoader,
    *,
    deadline_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], Awaitable[None]],
    clock: Callable[[], float] = time.monotonic,
) -> PromotionVerdict:
    """wait for durable canary delivery within one wall-clock budget, including RPC reads."""
    deadline = clock() + max(0.0, deadline_seconds)
    verdict = PromotionVerdict(ok=False, reason=ACCOUNTING_MALFORMED)
    while (remaining := deadline - clock()) > 0:
        try:
            snapshot = await asyncio.wait_for(loader(), timeout=remaining)
        except TimeoutError:
            return verdict
        except Exception:
            verdict = PromotionVerdict(ok=False, reason=ACCOUNTING_MALFORMED)
        else:
            # even a successful response arriving at the deadline cannot promote the release.
            if clock() >= deadline:
                return verdict
            verdict = verify_accounting(parse_accounting(snapshot))
            if verdict.ok:
                return verdict
        remaining = deadline - clock()
        if remaining <= 0:
            break
        await sleep(min(max(0.0, poll_seconds), remaining))
    return verdict


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

    try:
        org_id = str(uuid.UUID(env["SERVING_CANARY_ORG_ID"].strip()))
    except ValueError as exc:
        raise GateConfigError from exc

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

    run_id, attempt = env["GITHUB_RUN_ID"], env["GITHUB_RUN_ATTEMPT"]
    correlation_id = correlation_id_for(run_id, attempt)
    deployment_id = f"{run_id}-{attempt}"
    accounting_scope = {
        "p_org_id": org_id,
        "p_correlation_id": correlation_id,
        "p_serving_deployment_id": deployment_id,
        "p_serving_release": env["GITHUB_SHA"],
        "p_public_model_id": models[0],
    }

    async def read_backlog(client: httpx.AsyncClient) -> Any:
        response = await client.post(url, headers=headers, json=accounting_scope, timeout=15)
        response.raise_for_status()
        return response.json()

    return GatePlan(
        request=CanaryRequest(
            base_url=base_url,
            # ONE model, not all of `base_models()`. This is a deliberate limit and it is narrower
            # than the health check beside it: `/healthz` reports `gpus` as the count of CONFIGURED
            # tiers (3 as of the 27B activation), so a passing gate must not be read as three
            # proven engines. A release that breaks only the 27B or 35B path promotes cleanly.
            #
            # Looping all three would put three scale-to-zero cold starts inside one cold-start budget,
            # and a canary that times out fails the gate -- which redeploys the predecessor over a
            # healthy release. Widening coverage by making spurious rollbacks likelier is a bad
            # trade; per-model gating needs its own budget, not a longer single wait.
            model=models[0],
            api_key=env["FREESOLO_INTERNAL_KEY"],
            org_id=org_id,
            correlation_id=correlation_id,
            timeout_seconds=CANARY_TIMEOUT_SECONDS,
            max_tokens=_MAX_TOKENS,
        ),
        read_backlog=read_backlog,
        expected_sha=env["GITHUB_SHA"],
        expected_deployment_id=deployment_id,
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
