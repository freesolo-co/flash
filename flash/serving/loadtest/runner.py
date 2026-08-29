"""offline-safe orchestration for hosted inference load-test scenarios."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from flash.serving.loadtest.artifacts import ResultDirectory, load_events
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
    AdapterTarget,
    BaseTarget,
    ColdBurstPhase,
    Phase,
    ResolvedScenario,
    Scenario,
    Target,
    WarmPhase,
    capacity_expectations,
    claim_limitations,
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
            scenario.origin,
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
    target_base_models = _required_base_models(targets)
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
    capacity = capacity_expectations(scenario)
    return ResolvedScenario(
        authored=scenario,
        health=health,
        targets=targets,
        phase_cold_attestations=cold,
        phase_capacity_expectations=capacity,
        claim_limitations=claim_limitations(capacity, fake=scenario.fake),
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
        events = load_events(result.path / "events.jsonl")
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


class PhaseRecorder:
    """settle each scheduled request exactly once, in authored order.

    the runners hand results here as they settle rather than returning a batch at the end. that
    is what makes an interruption truthful: a request whose outcome was already observed keeps
    that outcome, and only requests that never settled become ``interrupted``. batching would
    force the interrupt path to reconstruct rows it cannot know, overwriting real successes.
    """

    def __init__(
        self,
        result: ResultDirectory,
        scheduled: list[ScheduledRequest],
        phase: Phase,
        start_ns: int,
    ) -> None:
        self._result = result
        self._scheduled = scheduled
        self._window = phase_duration_seconds(phase)
        self._settled: dict[str, dict[str, Any]] = {}
        self.start_ns = start_ns
        self.peak_in_flight = 0

    def settle(
        self, item: ScheduledRequest, observation: RequestObservation, in_flight: int
    ) -> None:
        if item.request_id in self._settled:
            return
        self.peak_in_flight = max(self.peak_in_flight, in_flight)
        self._settled[item.request_id] = _terminal_event(
            item, observation, self.start_ns, self._window, in_flight
        )

    def miss(self, item: ScheduledRequest, now_ns: int) -> None:
        self.settle(item, _admission_missed_observation(now_ns), 0)

    def flush(self, now_ns: int) -> int:
        """write exactly one row per scheduled request, marking only unsettled ones interrupted.

        called once per phase: either the interrupt path flushes and re-raises, or the phase
        completes and the tail flushes. the two are mutually exclusive.
        """
        for item in self._scheduled:
            event = self._settled.get(item.request_id)
            if event is None:
                event = _terminal_event(
                    item, _interrupted_observation(now_ns), self.start_ns, self._window, 0
                )
            event["phase_peak_client_in_flight"] = self.peak_in_flight
            self._result.events.write(event)
        return len(self._scheduled)


async def _run_phase(
    result: ResultDirectory,
    client: httpx.AsyncClient,
    resolved: ResolvedScenario,
    phase: Phase,
    scheduled: list[ScheduledRequest],
    credential: str,
    clock: Clock,
) -> None:
    # the opening probe runs first so the phase origin is the moment traffic actually begins.
    # stamping it before a blocking round trip would charge the probe's latency to the earliest
    # arrivals' lag budget and manufacture client_admission_missed rows out of instrumentation.
    await _health_event(result, client, resolved, f"phase:{phase.name}:start", clock)
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
    recorder = PhaseRecorder(result, scheduled, phase, phase_start)
    try:
        if isinstance(phase, WarmPhase):
            await _run_warm_phase(client, resolved, phase, scheduled, credential, recorder, clock)
        else:
            await _run_open_loop_phase(
                result, client, resolved, phase, scheduled, credential, recorder, clock
            )
    except BaseException:
        now_ns = clock.monotonic_ns()
        recorder.flush(now_ns)
        result.events.write(
            {
                "type": "phase_interrupted",
                "phase_name": phase.name,
                "phase_kind": phase.kind,
                "monotonic_ns": now_ns,
            }
        )
        raise
    terminal_count = recorder.flush(clock.monotonic_ns())
    await _health_event(result, client, resolved, f"phase:{phase.name}:end", clock)
    result.events.write(
        {
            "type": "phase_end",
            "phase_name": phase.name,
            "phase_kind": phase.kind,
            "monotonic_ns": clock.monotonic_ns(),
            "terminal_requests": terminal_count,
        }
    )


async def _run_warm_phase(
    client: httpx.AsyncClient,
    resolved: ResolvedScenario,
    phase: WarmPhase,
    scheduled: list[ScheduledRequest],
    credential: str,
    recorder: PhaseRecorder,
    clock: Clock,
) -> None:
    """the only closed-loop phase: a bounded-concurrency sweep of the authored request count."""
    semaphore = asyncio.Semaphore(phase.concurrency)
    in_flight = 0

    async def execute(item: ScheduledRequest) -> None:
        nonlocal in_flight
        async with semaphore:
            in_flight += 1
            current = in_flight
            try:
                observation = await _dispatch(client, resolved, credential, item, clock)
                recorder.settle(item, observation, current)
            finally:
                in_flight -= 1

    tasks = [asyncio.create_task(execute(item)) for item in scheduled]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        await _drain(tasks)
        raise


async def _run_open_loop_phase(
    result: ResultDirectory,
    client: httpx.AsyncClient,
    resolved: ResolvedScenario,
    phase: Phase,
    scheduled: list[ScheduledRequest],
    credential: str,
    recorder: PhaseRecorder,
    clock: Clock,
) -> None:
    """dispatch precomputed arrivals without ever waiting on a previous completion.

    an arrival that cannot be admitted within the client's in-flight and lag budget is settled as
    a miss rather than deferred, so client saturation never masquerades as server latency.
    """
    limits = resolved.authored.client
    slots = asyncio.Semaphore(limits.max_in_flight)
    max_lag_ns = round(limits.max_scheduling_lag_ms * 1_000_000)
    active = 0
    tasks: list[asyncio.Task[None]] = []
    health = _MidpointHealth(result, client, resolved, phase, recorder.start_ns, clock)

    async def execute(item: ScheduledRequest, current: int) -> None:
        nonlocal active
        try:
            observation = await _dispatch(client, resolved, credential, item, clock)
            recorder.settle(item, observation, current)
        finally:
            active -= 1
            slots.release()

    try:
        for item in scheduled:
            deadline_ns = recorder.start_ns + item.scheduled_offset_ns
            await health.before(deadline_ns)
            await clock.sleep_until_ns(deadline_ns)
            now_ns = clock.monotonic_ns()
            # lag means the client could not keep up; time spent inside a health probe is the
            # harness's own cost, so it is excluded from the admission decision. the recorded
            # timestamps still show the real delay, it just is not blamed on client saturation.
            lag_ns = now_ns - deadline_ns - health.credit_ns(deadline_ns)
            if lag_ns > max_lag_ns or slots.locked():
                recorder.miss(item, now_ns)
                continue
            await slots.acquire()
            active += 1
            tasks.append(asyncio.create_task(execute(item, active)))
        await health.finish()
        if tasks:
            await asyncio.gather(*tasks)
    except BaseException:
        await _drain(tasks)
        raise


class _MidpointHealth:
    """emit one health observation at the phase midpoint of a duration-bounded phase.

    duration phases are the only ones with a meaningful midpoint, so a phase without a duration
    simply never fires. keeping this here rather than as a parallel task means it cannot race the
    dispatch loop or outlive an interrupted phase.

    ``credit_ns`` reports how much of the probe an arrival actually waited through, so the caller
    can keep instrumentation cost out of the client scheduling-lag budget without handing later
    arrivals a discount they did not earn.
    """

    def __init__(
        self,
        result: ResultDirectory,
        client: httpx.AsyncClient,
        resolved: ResolvedScenario,
        phase: Phase,
        start_ns: int,
        clock: Clock,
    ) -> None:
        duration = phase_duration_seconds(phase)
        self._at_ns = start_ns + round(duration * 500_000_000) if duration is not None else None
        self._done = False
        self._probe = partial(
            _health_event, result, client, resolved, f"phase:{phase.name}:during", clock
        )
        self._clock = clock
        self._started_ns: int | None = None
        self._finished_ns: int | None = None

    async def before(self, deadline_ns: int) -> None:
        if self._at_ns is not None and not self._done and deadline_ns >= self._at_ns:
            await self._fire()

    async def finish(self) -> None:
        if self._at_ns is not None and not self._done:
            await self._fire()

    def credit_ns(self, deadline_ns: int) -> int:
        """the part of the probe that delayed an arrival due at ``deadline_ns``.

        an arrival due before the probe returned was held by it and is credited only for the
        overlap; one due after it returned waited on nothing and is credited zero. a standing
        credit would be subtracted from every later arrival in the phase and would mask genuine
        client_admission_missed events for the rest of it, which is the same metric the credit
        exists to protect, failing in the opposite direction.
        """
        if self._started_ns is None or self._finished_ns is None:
            return 0
        return max(0, self._finished_ns - max(self._started_ns, deadline_ns))

    async def _fire(self) -> None:
        await self._clock.sleep_until_ns(self._at_ns)
        self._started_ns = self._clock.monotonic_ns()
        await self._probe()
        self._finished_ns = self._clock.monotonic_ns()
        self._done = True


def _required_base_models(targets: list[Target]) -> set[str]:
    """the base model each target needs the deployment to serve.

    an adapter rides on its base model's engine, so its requirement is ``base_model``; a base
    target is its own requirement. this reads the discriminated union directly rather than
    sniffing for an attribute, so a new target kind fails typing instead of silently resolving
    to the wrong field.
    """
    return {
        target.base_model if isinstance(target, AdapterTarget) else target.model
        for target in targets
    }


async def _dispatch(
    client: httpx.AsyncClient,
    resolved: ResolvedScenario,
    credential: str,
    item: ScheduledRequest,
    clock: Clock,
) -> RequestObservation:
    return await stream_chat(
        client,
        resolved.authored.origin,
        credential,
        item.target,
        item.profile,
        clock,
    )


async def _drain(tasks: list[asyncio.Task[None]]) -> None:
    """cancel in-flight work and await it, so no request is abandoned mid-settle."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _interrupted_observation(now_ns: int) -> RequestObservation:
    return RequestObservation(
        outcome="interrupted",
        error_class="interrupted",
        error_detail="run interrupted before the phase produced a complete terminal result",
        completed_ns=now_ns,
    )


def _admission_missed_observation(now_ns: int) -> RequestObservation:
    return RequestObservation(
        outcome="client_admission_missed",
        error_class="client_admission_missed",
        error_detail="client max in-flight or scheduling-lag budget was unavailable",
        completed_ns=now_ns,
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
        "request_id": item.request_id,
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
        resolved.authored.origin,
        resolved.authored.expected_deployment,
        resolved.authored.required_capabilities,
    )
    missing_models = sorted(_required_base_models(resolved.targets) - set(health.base_models))
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
    # one connection above max_in_flight is reserved for health probes. sizing the pool to exactly
    # max_in_flight means a midpoint probe at saturation waits on the pool and can time out, which
    # aborts the phase over instrumentation rather than over anything the deployment did.
    limits = httpx.Limits(
        max_connections=scenario.client.max_in_flight + 1,
        max_keepalive_connections=scenario.client.max_in_flight + 1,
    )
    return client_factory(
        timeout=timeout,
        limits=limits,
        headers={"Authorization": f"Bearer {credential}"},
    )
