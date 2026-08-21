"""bounded waiting for a runpod pod to prove itself, and reclaiming its one-shot token.

`runpod.py` reached the 1000-line file limit a second time. The group that came out cleanest here
is the readiness wait: polling the provider until the exact pod is running and its endpoint answers
a probe, then deleting the hydration artifact the pod no longer needs. It is the one part of the
lifecycle every entry point shares -- create, adopt, and read-only reconcile all end in this same
wait -- and it touches provider state only through the observe and delete callables handed to it.

The split follows `_runpod_lifecycle`'s rule: this module imports nothing from `runpod.py`. The
pieces it needs from that file arrive as parameters (`observe`, `delete_secret`) rather than
imports, which is also what lets the tests drive it without a transport.
"""

from __future__ import annotations

from collections.abc import Callable

from flash.serve.control import DeploymentResult, RunPodProviderHandle

from ._common import Clock, DeploymentBundle, LifecycleFailure, Sleeper, failed_deployment_result
from ._runpod_plan import RunPodCreatePlan
from ._runpod_protocol import RunPodObservation, RunPodSecretObservation
from ._runpod_resources import (
    RunPodResourceConflict,
    build_handle,
    ensure_unique_resources,
    exact_core_resources,
    readiness_state,
)
from ._runpod_transport import RunPodTransportFailure

READINESS_POLL_SECONDS = 2.0
MAX_PROBE_TIMEOUT_SECONDS = 30.0

Observe = Callable[[RunPodCreatePlan], RunPodObservation]
DeleteSecret = Callable[[str], None]


class EndpointProbe:
    """structural type for the readiness probe (see `RunPodEndpointProbe`)."""

    def __call__(
        self,
        public_url: str,
        inference_token: str,
        bundle: DeploymentBundle,
        timeout_seconds: float,
    ) -> bool: ...


def failure_result(
    plan: RunPodCreatePlan,
    failure: LifecycleFailure,
    *,
    handle: RunPodProviderHandle | None = None,
) -> DeploymentResult:
    return failed_deployment_result(
        plan.bundle.spec,
        failure.code,
        outcome_unknown=failure.outcome_unknown,
        handle=handle,
    )


def unknown_result(
    plan: RunPodCreatePlan,
    *,
    handle: RunPodProviderHandle | None = None,
) -> DeploymentResult:
    return failure_result(
        plan,
        LifecycleFailure("resource_ambiguous", outcome_unknown=True),
        handle=handle,
    )


def sleep_until_poll(deadline_at: float, clock: Clock, sleep: Sleeper) -> bool:
    remaining = deadline_at - clock()
    if remaining <= 0:
        return False
    sleep(min(READINESS_POLL_SECONDS, remaining))
    return clock() < deadline_at


def probe_with_deadline(
    probe: EndpointProbe,
    handle: RunPodProviderHandle,
    inference_token: str,
    plan: RunPodCreatePlan,
    *,
    deadline_at: float,
    clock: Clock,
) -> bool:
    remaining = deadline_at - clock()
    if remaining <= 0:
        return False
    try:
        accepted = probe(
            handle.public_url,
            inference_token,
            plan.bundle,
            min(MAX_PROBE_TIMEOUT_SECONDS, remaining),
        )
    except Exception:
        return False
    return accepted is True and clock() < deadline_at


def read_only_reconcile(
    plan: RunPodCreatePlan,
    observe: Observe,
    inference_token: str,
    *,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
    allow_transient_artifact: bool = False,
    unproven_is_failure: bool = True,
) -> DeploymentResult:
    last_handle: RunPodProviderHandle | None = None
    while True:
        observation = observe(plan)
        if observation.resource_count == 0:
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        try:
            secret, template, volume, pod = exact_core_resources(plan, observation)
            last_handle = build_handle(plan, secret, template, volume, pod)
        except RunPodResourceConflict:
            return failure_result(plan, LifecycleFailure("conflict"))
        state = readiness_state(pod.desired_status)
        if state == "invalid":
            return failure_result(plan, LifecycleFailure("conflict"), handle=last_handle)
        if state == "failed":
            return failure_result(plan, LifecycleFailure("readiness_failed"), handle=last_handle)
        if (
            state == "running"
            and (not observation.artifact_secrets or allow_transient_artifact)
            and probe_with_deadline(
                probe,
                last_handle,
                inference_token,
                plan,
                deadline_at=deadline_at,
                clock=clock,
            )
        ):
            return DeploymentResult.from_spec(
                plan.bundle.spec,
                status="ready",
                handle=last_handle,
            )
        if observation.artifact_secrets and not allow_transient_artifact:
            if not sleep_until_poll(deadline_at, clock, sleep):
                return unknown_result(plan, handle=last_handle)
            continue
        if not sleep_until_poll(deadline_at, clock, sleep):
            if state == "pending":
                return DeploymentResult.from_spec(
                    plan.bundle.spec,
                    status="provisioning",
                    handle=last_handle,
                )
            # a running pod that never proved itself is only *definitely* failed to a caller that
            # can undo what it made. the create path can: it aborts its own ledger immediately
            # after. adoption cannot -- it owns resources it did not create, and answering
            # "failed" there invites the supervisor to drop the deployment record while a customer
            # pod keeps billing. it reports unproven instead, which is the truth in both cases and
            # the retry-safe one in the second.
            if not unproven_is_failure:
                return unknown_result(plan, handle=last_handle)
            return failure_result(
                plan,
                LifecycleFailure("readiness_failed"),
                handle=last_handle,
            )


def confirm_artifact_absence(
    plan: RunPodCreatePlan,
    observe: Observe,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> bool:
    """confirm the artifact is gone *and* the deployment it belonged to is still there.

    Checking only the artifact made a vanished deployment indistinguishable from a cleaned-up one:
    an empty observation has no artifact secret either, so a pod deleted between the readiness
    probe and this confirmation still returned `ready` -- with a handle whose url resolves to
    nothing. Re-proving the exact core resources is what separates "the token is gone" from
    "everything is gone", and it re-runs the same identity match the probe relied on, so drift as
    well as disappearance is caught.
    """

    while True:
        observation = observe(plan)
        try:
            ensure_unique_resources(observation)
            exact_core_resources(plan, observation)
        except RunPodResourceConflict:
            return False
        if not observation.artifact_secrets:
            return True
        if not sleep_until_poll(deadline_at, clock, sleep):
            return False


def delete_artifact_and_confirm(
    plan: RunPodCreatePlan,
    observe: Observe,
    delete_secret: DeleteSecret,
    artifact: RunPodSecretObservation,
    handle: RunPodProviderHandle,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    try:
        delete_secret(artifact.id)
    except RunPodTransportFailure:
        return unknown_result(plan, handle=handle)
    try:
        absent = confirm_artifact_absence(
            plan,
            observe,
            deadline_at=deadline_at,
            clock=clock,
            sleep=sleep,
        )
    except RunPodTransportFailure:
        return unknown_result(plan, handle=handle)
    if not absent:
        return unknown_result(plan, handle=handle)
    return DeploymentResult.from_spec(plan.bundle.spec, status="ready", handle=handle)


def await_ready_and_reclaim(
    plan: RunPodCreatePlan,
    observe: Observe,
    delete_secret: DeleteSecret,
    inference_token: str,
    artifact: RunPodSecretObservation | None,
    *,
    unproven_is_failure: bool,
    on_ready: Callable[[], None] = lambda: None,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    """wait for the pod to prove itself, then reclaim the one-shot artifact token.

    Both the create path and a rerun that adopts its own pod need exactly this, so they call it
    rather than each keeping a copy. Adoption used to keep a shortened copy that checked readiness
    once and returned `outcome_unknown` with the whole timeout unspent -- which made rerunning
    `serve deploy`, the documented way to continue a deployment that returned `provisioning`,
    structurally unable to ever finish one.

    `artifact` is the id observed *before* the wait, never a re-read one. The wait can be minutes
    and these names are deterministic, so a re-read can return a successor generation's artifact
    and delete a token that generation still needs. Deleting an already-deleted id is the safe
    direction: the delete tolerates `not_found` and `confirm_artifact_absence` refuses to report
    clean while any artifact remains.

    `on_ready` fires the instant readiness is proven, before the artifact is reclaimed. The create
    path's interrupt cleanup keys off it: everything before that point is a half-built deployment
    worth tearing down on Ctrl-C, and everything after is a live pod the user paid to warm up.
    """

    ready = read_only_reconcile(
        plan,
        observe,
        inference_token,
        deadline_at=deadline_at,
        probe=probe,
        clock=clock,
        sleep=sleep,
        allow_transient_artifact=True,
        unproven_is_failure=unproven_is_failure,
    )
    if ready.status != "ready":
        return ready
    on_ready()
    if artifact is None:
        return ready
    return delete_artifact_and_confirm(
        plan,
        observe,
        delete_secret,
        artifact,
        ready.handle,
        deadline_at=deadline_at,
        clock=clock,
        sleep=sleep,
    )
