"""offline-safe orchestration for hosted inference load-test scenarios."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from flash.serving.loadtest.artifacts import ResultDirectory
from flash.serving.loadtest.metrics import summarize_events
from flash.serving.loadtest.protocol import (
    RequestObservation,
    get_health,
    resolve_discovered_models,
    stream_chat,
)
from flash.serving.loadtest.schedule import (
    Clock,
    ScheduledRequest,
    SystemClock,
    build_schedule,
    phase_duration_seconds,
)
from flash.serving.loadtest.schema import (
    CLAIM_LIMITATIONS,
    NO_CAPACITY_CONTRACT_LIMITATION,
    BaseTarget,
    ColdBurstPhase,
    OverloadPhase,
    ResolvedScenario,
    Scenario,
    WarmPhase,
)

ClientFactory = Callable[..., httpx.AsyncClient]


async def discover_scenario(
    scenario: Scenario,
    credential: str,
    *,
    client_factory: ClientFactory = httpx.AsyncClient,
) -> ResolvedScenario:
    async with _client(scenario, credential, client_factory) as client:
        health = await get_health(
            client,
            str(scenario.endpoint).rstrip("/"),
            scenario.expected_deployment,
            scenario.required_capabilities,
        )
    targets = list(scenario.targets)
    if scenario.discovery.enabled:
        models = resolve_discovered_models(
            health.base_models,
            scenario.discovery.include,
            scenario.discovery.exclude,
            scenario.discovery.require,
        )
        explicit_models = {target.model for target in targets}
        targets.extend(
            BaseTarget(name=model, model=model) for model in models if model not in explicit_models
        )
    if not targets:
        raise ValueError("resolved scenario has no targets")
    target_names = [target.name for target in targets]
    if len(set(target_names)) != len(target_names):
        raise ValueError("resolved target names must be unique")
    target_base_models = {
        target.base_model if hasattr(target, "base_model") else target.model for target in targets
    }
    missing_models = sorted(target_base_models - set(health.base_models))
    if missing_models:
        raise ValueError(
            f"resolved targets require unavailable models: {', '.join(missing_models)}"
        )
    cold = {
        phase.name: phase.cold_attestation
        for phase in scenario.phases
        if isinstance(phase, ColdBurstPhase)
    }
    capacity = {
        phase.name: phase.expects_capacity_contract
        for phase in scenario.phases
        if isinstance(phase, OverloadPhase)
    }
    limitations = list(CLAIM_LIMITATIONS)
    if capacity and not all(capacity.values()):
        limitations.append(NO_CAPACITY_CONTRACT_LIMITATION)
    return ResolvedScenario(
        authored=scenario,
        health=health,
        targets=targets,
        phase_cold_attestations=cold,
        phase_capacity_expectations=capacity,
        claim_limitations=limitations,
    )


async def run_scenario(
    scenario: Scenario,
    credential: str,
    result_path: Path,
    *,
    clock: Clock | None = None,
    client_factory: ClientFactory = httpx.AsyncClient,
) -> dict[str, Any]:
    run_clock = clock or SystemClock()
    result = ResultDirectory.create(result_path, scenario, credential)
    try:
        resolved = await discover_scenario(
            scenario,
            credential,
            client_factory=client_factory,
        )
        result.write_resolved(resolved)
        schedules = build_schedule(
            scenario.phases,
            resolved.targets,
            scenario.profiles,
            seed=scenario.seed,
        )
        async with _client(scenario, credential, client_factory) as client:
            await _health_event(result, client, resolved, "run_start", run_clock)
            for phase, scheduled in zip(scenario.phases, schedules, strict=True):
                await _run_phase(
                    result,
                    client,
                    resolved,
                    phase,
                    scheduled,
                    credential,
                    run_clock,
                )
            await _health_event(result, client, resolved, "run_end", run_clock)
        events = _events_for_summary(result.path)
        summary = summarize_events(
            events,
            fake=scenario.fake,
            capacity_expectations=resolved.phase_capacity_expectations,
        )
        result.complete(summary)
        return summary
    except BaseException:
        result.abort()
        raise


async def _run_phase(
    result: ResultDirectory,
    client: httpx.AsyncClient,
    resolved: ResolvedScenario,
    phase: Any,
    scheduled: list[ScheduledRequest],
    credential: str,
    clock: Clock,
) -> None:
    phase_start = clock.monotonic_ns()
    result.events.write(
        {
            "type": "phase_start",
            "phase_name": phase.name,
            "phase_kind": phase.kind,
            "monotonic_ns": phase_start,
            "scheduled_requests": len(scheduled),
            "cold_attestation": resolved.phase_cold_attestations.get(phase.name),
        }
    )
    await _health_event(result, client, resolved, f"phase:{phase.name}:start", clock)
    try:
        if isinstance(phase, WarmPhase):
            terminals = await _run_warm_phase(
                client, resolved, phase, scheduled, credential, phase_start, clock
            )
        else:
            terminals = await _run_open_loop_phase(
                result,
                client,
                resolved,
                phase,
                scheduled,
                credential,
                phase_start,
                clock,
            )
    except BaseException:
        now_ns = clock.monotonic_ns()
        for item in scheduled:
            result.events.write(_interrupted_terminal(item, phase_start, now_ns, phase))
        result.events.write(
            {
                "type": "phase_interrupted",
                "phase_name": phase.name,
                "phase_kind": phase.kind,
                "monotonic_ns": now_ns,
            }
        )
        raise
    for terminal in terminals:
        result.events.write(terminal)
    await _health_event(result, client, resolved, f"phase:{phase.name}:end", clock)
    result.events.write(
        {
            "type": "phase_end",
            "phase_name": phase.name,
            "phase_kind": phase.kind,
            "monotonic_ns": clock.monotonic_ns(),
            "terminal_requests": len(terminals),
        }
    )


async def _run_warm_phase(
    client: httpx.AsyncClient,
    resolved: ResolvedScenario,
    phase: WarmPhase,
    scheduled: list[ScheduledRequest],
    credential: str,
    phase_start_ns: int,
    clock: Clock,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(phase.concurrency)
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def execute(item: ScheduledRequest) -> dict[str, Any]:
        nonlocal in_flight, peak
        async with semaphore:
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
                current = in_flight
            observation = await stream_chat(
                client,
                str(resolved.authored.endpoint).rstrip("/"),
                credential,
                item.target,
                item.profile,
                item.request_id,
                clock,
            )
            async with lock:
                in_flight -= 1
            return _terminal_event(
                item,
                observation,
                phase_start_ns,
                phase_duration_seconds(phase),
                current,
            )

    terminals = await asyncio.gather(*(execute(item) for item in scheduled))
    for terminal in terminals:
        terminal["phase_peak_client_in_flight"] = peak
    return terminals


async def _run_open_loop_phase(
    result: ResultDirectory,
    client: httpx.AsyncClient,
    resolved: ResolvedScenario,
    phase: Any,
    scheduled: list[ScheduledRequest],
    credential: str,
    phase_start_ns: int,
    clock: Clock,
) -> list[dict[str, Any]]:
    slots = asyncio.Semaphore(resolved.authored.client.max_in_flight)
    max_lag_ns = round(resolved.authored.client.max_scheduling_lag_ms * 1_000_000)
    active = 0
    peak = 0
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    terminals: list[dict[str, Any]] = []
    duration = phase_duration_seconds(phase)
    midpoint_ns = phase_start_ns + round(duration * 500_000_000) if duration is not None else None
    during_health_done = False

    async def execute(item: ScheduledRequest, current: int) -> dict[str, Any]:
        nonlocal active
        try:
            observation = await stream_chat(
                client,
                str(resolved.authored.endpoint).rstrip("/"),
                credential,
                item.target,
                item.profile,
                item.request_id,
                clock,
            )
            return _terminal_event(
                item,
                observation,
                phase_start_ns,
                phase_duration_seconds(phase),
                current,
            )
        finally:
            active -= 1
            slots.release()

    try:
        for item in scheduled:
            deadline_ns = phase_start_ns + item.scheduled_offset_ns
            if midpoint_ns is not None and not during_health_done and deadline_ns >= midpoint_ns:
                await clock.sleep_until_ns(midpoint_ns)
                await _health_event(result, client, resolved, f"phase:{phase.name}:during", clock)
                during_health_done = True
            await clock.sleep_until_ns(deadline_ns)
            now_ns = clock.monotonic_ns()
            if now_ns - deadline_ns > max_lag_ns or slots.locked():
                terminals.append(_admission_missed(item, phase_start_ns, now_ns, phase))
                continue
            await slots.acquire()
            active += 1
            peak = max(peak, active)
            tasks.append(asyncio.create_task(execute(item, active)))
        if midpoint_ns is not None and not during_health_done:
            await clock.sleep_until_ns(midpoint_ns)
            await _health_event(result, client, resolved, f"phase:{phase.name}:during", clock)
        if tasks:
            terminals.extend(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    terminals.sort(key=lambda event: event["phase_request_index"])
    for terminal in terminals:
        terminal["phase_peak_client_in_flight"] = peak
    return terminals


def _interrupted_terminal(
    item: ScheduledRequest, phase_start_ns: int, now_ns: int, phase: Any
) -> dict[str, Any]:
    observation = RequestObservation(
        request_id=item.request_id,
        outcome="interrupted",
        error_class="interrupted",
        error_detail="run interrupted before the phase produced a complete terminal result",
        completed_ns=now_ns,
    )
    return _terminal_event(
        item,
        observation,
        phase_start_ns,
        phase_duration_seconds(phase),
        0,
    )


def _admission_missed(
    item: ScheduledRequest, phase_start_ns: int, now_ns: int, phase: Any
) -> dict[str, Any]:
    observation = RequestObservation(
        request_id=item.request_id,
        outcome="client_admission_missed",
        error_class="client_admission_missed",
        error_detail="client max in-flight or scheduling-lag budget was unavailable",
        completed_ns=now_ns,
    )
    return _terminal_event(
        item,
        observation,
        phase_start_ns,
        phase_duration_seconds(phase),
        0,
    )


def _terminal_event(
    item: ScheduledRequest,
    observation: RequestObservation,
    phase_start_ns: int,
    window_seconds: float | None,
    in_flight: int,
) -> dict[str, Any]:
    return {
        "type": "request_terminal",
        **asdict(observation),
        "phase_name": item.phase_name,
        "phase_kind": item.phase_kind,
        "phase_index": item.phase_index,
        "phase_request_index": item.phase_request_index,
        "stage_index": item.stage_index,
        "target_name": item.target.name,
        "target_kind": item.target.kind,
        "profile_name": item.profile.name,
        "scheduled_ns": phase_start_ns + item.scheduled_offset_ns,
        "authored_window_seconds": window_seconds,
        "client_in_flight_at_dispatch": in_flight,
    }


async def _health_event(
    result: ResultDirectory,
    client: httpx.AsyncClient,
    resolved: ResolvedScenario,
    label: str,
    clock: Clock,
) -> None:
    health = await get_health(
        client,
        str(resolved.authored.endpoint).rstrip("/"),
        resolved.authored.expected_deployment,
        resolved.authored.required_capabilities,
    )
    missing_models = sorted(
        {
            target.base_model if hasattr(target, "base_model") else target.model
            for target in resolved.targets
        }
        - set(health.base_models)
    )
    if missing_models:
        raise RuntimeError(f"healthz lost required base models: {', '.join(missing_models)}")
    result.events.write(
        {
            "type": "health",
            "label": label,
            "monotonic_ns": clock.monotonic_ns(),
            "deployment_sha": health.deployment_sha,
            "deployment_id": health.deployment_id,
            "base_models": health.base_models,
            "capabilities": health.capabilities,
        }
    )


def _client(
    scenario: Scenario,
    credential: str,
    client_factory: ClientFactory,
) -> httpx.AsyncClient:
    timeout = httpx.Timeout(
        connect=float(scenario.client.connect_timeout_seconds),
        read=float(scenario.client.read_timeout_seconds),
        write=float(scenario.client.write_timeout_seconds),
        pool=float(scenario.client.pool_timeout_seconds),
    )
    limits = httpx.Limits(
        max_connections=scenario.client.max_in_flight,
        max_keepalive_connections=scenario.client.max_in_flight,
    )
    return client_factory(
        timeout=timeout,
        limits=limits,
        headers={"Authorization": f"Bearer {credential}"},
    )


def _events_for_summary(path: Path) -> list[dict[str, Any]]:
    from flash.serving.loadtest.artifacts import load_events

    return load_events(path / "events.jsonl")
