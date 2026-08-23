"""Fire-and-forget usage metering for the serving front door.

Split out of router.py's app builder. Owns the detached-task set that keeps in-flight billing
reports alive, so the app factory holds a reporter rather than bare bookkeeping state.
"""

import asyncio
import uuid
from typing import Any

from flash.serving.src.io.schemas import AdapterRecord


def usage_payload(
    record: AdapterRecord,
    result: dict[str, Any],
    caller_org: str | None,
    deployment_id: str,
) -> dict[str, Any] | None:
    """Backend usage-report body, or None when there's nothing billable to report.

    A LoRA serve reports by ``adapterId`` — the backend resolves the OWNING org and bills it. A
    base-model serve has no owner, so it carries the CALLER's ``orgId`` and OMITS ``adapterId``
    (which is authoritative when present and would fail to resolve for a base model); the backend
    bills that org. When a base serve has no known caller org (an internal-key server-to-server
    caller), it is dropped rather than misbilled.
    """
    # The engine result is snake_case (the serving API's internal contract); this OUTBOUND
    # billing payload keeps the platform backend's camelCase keys (a separate service's
    # contract), so we read snake here and write camel below.
    prompt_tokens = result.get("prompt_tokens")
    completion_tokens = result.get("completion_tokens")
    # The engine reports token counts; without them there is nothing to meter.
    if prompt_tokens is None or completion_tokens is None:
        return None
    payload: dict[str, Any] = {
        "baseModel": record.base_model,
        "promptTokens": int(prompt_tokens),
        "completionTokens": int(completion_tokens),
        # Prefix-cached subset of the prompt (engine num_cached_tokens); billed at a
        # discount by the backend. Defaults to 0 when the engine doesn't report it.
        "cachedTokens": int(result.get("cached_tokens") or 0),
        "cachedTokensReported": result.get("cached_tokens_reported") is True,
        "gpuSeconds": result.get("inference_time_seconds"),
        # stable per-generation id from the engine is the backend idempotency key for retries.
        # fall back to a fresh id only when an offline test pool did not supply one.
        "requestId": result.get("request_id") or str(uuid.uuid4()),
        "engineReplicaId": result.get("engine_replica_id"),
        "servingDeploymentId": deployment_id or None,
    }
    if record.serve_base_model:
        if not caller_org:
            return None
        payload["orgId"] = caller_org
    else:
        payload["adapterId"] = record.adapter_id
    return payload


class UsageReporter:
    """Schedules usage reports as detached tasks and drains them on shutdown.

    Holds strong refs to the fire-and-forget tasks. asyncio only keeps a WEAK reference to a bare
    create_task(), so without this the GC can collect a still-pending billing report mid event
    loop (silently dropping it and emitting "Task was destroyed but it is pending"). The
    done-callback discards each task once it settles, so the set stays bounded by in-flight count.
    """

    def __init__(
        self,
        report: Any,
        *,
        deployment_id: str = "",
        drain_timeout_seconds: float = 45.0,
    ) -> None:
        self._report = report
        self._deployment_id = deployment_id
        self._drain_timeout_seconds = drain_timeout_seconds
        self._pending: set[Any] = set()

    async def _report_safe(self, usage: dict[str, Any]) -> None:
        # fire-and-forget: the reporter performs its bounded idempotent retries, and any final
        # failure remains isolated from the response that has already been sent.
        assert self._report is not None
        try:
            await self._report(usage)
        except Exception as exc:  # never fail the (already-sent) response
            print(f"serving usage report dropped for {usage.get('requestId')}: {exc!r}", flush=True)

    def schedule(
        self,
        record: AdapterRecord,
        result: dict[str, Any],
        caller_org: str | None,
    ) -> None:
        if self._report is None:
            return
        payload = usage_payload(record, result, caller_org, self._deployment_id)
        if payload is None:
            return
        task = asyncio.create_task(self._report_safe(payload))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        """Settle in-flight reports before their shared client closes.

        The timeout bounds graceful shutdown; any remainder is cancelled and settled so no task
        can wake against a closed client after a rolling deployment.
        """
        if not self._pending:
            return
        _, pending = await asyncio.wait(tuple(self._pending), timeout=self._drain_timeout_seconds)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
