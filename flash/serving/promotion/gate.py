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
from typing import Any

from flash.serving.promotion.canary import (
    CanaryError,
    CanaryRequest,
    correlation_id_for,
    run_stream_canary,
)
from flash.serving.promotion.evidence import (
    ACCOUNTING_MALFORMED,
    PromotionVerdict,
    StreamEvidence,
    accounting_from_snapshot,
    parse_health,
    verify_accounting,
    verify_health,
    verify_stream,
)

HealthLoader = Callable[[], Awaitable[Any]]
StreamRunner = Callable[[], Awaitable[StreamEvidence]]
AccountingLoader = Callable[[], Awaitable[Any]]

GATE_CONFIG_INCOMPLETE = "gate_config_incomplete"
HEALTH_UNREACHABLE = "health_unreachable"

_DEFAULT_CANARY_TIMEOUT_SECONDS = 180.0
_DEFAULT_ACCOUNTING_DEADLINE_SECONDS = 180.0
_ACCOUNTING_POLL_SECONDS = 5.0
_MAX_COMPLETION_TOKENS = 32
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
    verdict = verify_accounting(None)
    while True:
        try:
            snapshot = await loader()
        # an unreadable snapshot must never pass as settled.
        except Exception:
            verdict = PromotionVerdict(ok=False, reason=ACCOUNTING_MALFORMED)
            snapshot = None
        if snapshot is not None:
            verdict = verify_accounting(accounting_from_snapshot(snapshot))
            if verdict.ok:
                return verdict
        if waited >= deadline_seconds:
            return verdict
        await sleep(poll_seconds)
        waited += poll_seconds


def _canary_model() -> str:
    from flash.serving.src.engine.model_config import base_models

    models = base_models()
    if not models:
        raise ValueError("no hosted base models are configured")
    return models[0]


async def _run(base_url: str, env: dict[str, str]) -> PromotionVerdict:
    import httpx

    from flash.serving.src.accounting.usage_outbox import DurableUsageOutbox
    from flash.serving.src.store.settings import get_settings

    expected_deployment_id = f"{env['GITHUB_RUN_ID']}-{env['GITHUB_RUN_ATTEMPT']}"
    correlation_id = correlation_id_for(env["GITHUB_RUN_ID"], env["GITHUB_RUN_ATTEMPT"])
    request = CanaryRequest(
        base_url=base_url,
        model=_canary_model(),
        api_key=env["FREESOLO_INTERNAL_KEY"],
        correlation_id=correlation_id,
        timeout_seconds=_DEFAULT_CANARY_TIMEOUT_SECONDS,
        max_completion_tokens=_MAX_COMPLETION_TOKENS,
    )
    outbox = DurableUsageOutbox(get_settings())
    async with httpx.AsyncClient() as client:

        async def health_loader() -> Any:
            response = await client.get(f"{base_url.rstrip('/')}/healthz", timeout=15)
            response.raise_for_status()
            return response.json()

        async def stream_runner() -> StreamEvidence:
            return await run_stream_canary(request, client=client)

        return await evaluate_promotion(
            health_loader=health_loader,
            stream_runner=stream_runner,
            accounting_loader=outbox.snapshot,
            expected_sha=env["GITHUB_SHA"],
            expected_deployment_id=expected_deployment_id,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="prove a hosted serving promotion")
    # only non-secret configuration is accepted as an argument.
    parser.add_argument("--base-url", default=os.environ.get("SERVING_BASE_URL", ""))
    args = parser.parse_args(argv)
    if not args.base_url:
        print(f"promotion gate failed: {GATE_CONFIG_INCOMPLETE}", file=sys.stderr)
        return 1
    env = {name: os.environ.get(name, "") for name in _REQUIRED_ENV}
    if not all(env.values()):
        # naming which variable is missing would be friendlier, but the value of a secret-shaped
        # variable must never influence log output.
        print(f"promotion gate failed: {GATE_CONFIG_INCOMPLETE}", file=sys.stderr)
        return 1
    verdict = asyncio.run(_run(args.base_url, env))
    if verdict.ok:
        print("promotion gate passed")
        return 0
    print(f"promotion gate failed: {verdict.reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
