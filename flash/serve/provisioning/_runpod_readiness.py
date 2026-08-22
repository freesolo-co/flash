"""bounded waiting for a runpod pod to prove itself, and reclaiming its one-shot token.

`runpod.py` reached the 1000-line file limit a second time. The group that came out cleanest here
is the readiness wait: polling the provider until the exact pod is running and its endpoint answers
a probe, then deleting the hydration artifact the pod no longer needs. It is the one part of the
lifecycle every entry point shares -- create, adopt, and read-only reconcile all end in this same
wait -- and it touches provider state only through the observe, patch, and delete callables handed
to it.

The split follows `_runpod_lifecycle`'s rule: this module imports nothing from `runpod.py`. The
pieces it needs from that file arrive as callables rather than imports, which is also what lets the
tests drive it without a transport.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import Protocol, cast

from flash.serve.control import DeploymentErrorReason, DeploymentResult, RunPodProviderHandle

from ._common import Clock, DeploymentBundle, LifecycleFailure, Sleeper, failed_deployment_result
from ._runpod_plan import RunPodCreatePlan
from ._runpod_protocol import (
    RunPodObservation,
    RunPodPodObservation,
    RunPodSecretObservation,
)
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
PatchTemplate = Callable[[str, dict[str, object]], None]
PatchPod = Callable[[str, dict[str, object]], None]
DeleteSecret = Callable[[str], None]


class _Confirmation(Enum):
    CONFIRMED = auto()
    TIMED_OUT = auto()
    OBSERVATION_FAILED = auto()
    CONFLICT = auto()
    IDENTITY_DRIFT = auto()


class EndpointProbe(Protocol):
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
        error_reason=failure.reason,
    )


def unknown_result(
    plan: RunPodCreatePlan,
    *,
    reason: DeploymentErrorReason,
    handle: RunPodProviderHandle | None = None,
) -> DeploymentResult:
    return failure_result(
        plan,
        LifecycleFailure("resource_ambiguous", outcome_unknown=True, reason=reason),
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
        try:
            observation = observe(plan)
        except RunPodTransportFailure as exc:
            if not sleep_until_poll(deadline_at, clock, sleep):
                return failure_result(
                    plan,
                    LifecycleFailure(
                        exc.code,
                        outcome_unknown=exc.outcome_unknown,
                        reason="readiness_observation_failed",
                    ),
                    handle=last_handle,
                )
            continue
        if observation.resource_count == 0:
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        try:
            secret, template, volume, pod = exact_core_resources(plan, observation)
            last_handle = build_handle(plan, secret, template, volume, pod)
        except RunPodResourceConflict:
            return failure_result(
                plan,
                LifecycleFailure("conflict", reason="readiness_resource_conflict"),
            )
        state = readiness_state(pod.desired_status)
        if state == "invalid":
            return failure_result(
                plan,
                LifecycleFailure("conflict", reason="readiness_status_invalid"),
                handle=last_handle,
            )
        if state == "failed":
            # same reasoning as the unproven case below: a terminal pod is only *definitely*
            # failed to a caller that can undo what it made. adoption and the read-only
            # reconciliation path cannot -- the pod, template, volume, and secrets are still in
            # the customer account, so answering "failed" lets the supervisor drop the record and
            # strand them, and suppresses the cli's outcome-unknown reconciliation warning.
            if not unproven_is_failure:
                return unknown_result(
                    plan,
                    reason="readiness_terminal",
                    handle=last_handle,
                )
            return failure_result(
                plan,
                LifecycleFailure("readiness_failed", reason="readiness_terminal"),
                handle=last_handle,
            )
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
                return unknown_result(
                    plan,
                    reason="readiness_artifact_present",
                    handle=last_handle,
                )
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
                return unknown_result(
                    plan,
                    reason="readiness_deadline_unproven",
                    handle=last_handle,
                )
            return failure_result(
                plan,
                LifecycleFailure("readiness_failed", reason="readiness_deadline_unproven"),
                handle=last_handle,
            )


def confirm_artifact_absence(
    plan: RunPodCreatePlan,
    observe: Observe,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> _Confirmation:
    """confirm the artifact is gone *and* the deployment it belonged to is still there.

    Checking only the artifact made a vanished deployment indistinguishable from a cleaned-up one:
    an empty observation has no artifact secret either, so a pod deleted between the readiness
    probe and this confirmation still returned `ready` -- with a handle whose url resolves to
    nothing. Re-proving the exact core resources is what separates "the token is gone" from
    "everything is gone", and it re-runs the same identity match the probe relied on, so drift as
    well as disappearance is caught.
    """

    while True:
        try:
            observation = observe(plan)
            ensure_unique_resources(observation)
            exact_core_resources(plan, observation)
        except RunPodTransportFailure:
            if not sleep_until_poll(deadline_at, clock, sleep):
                return _Confirmation.OBSERVATION_FAILED
            continue
        except RunPodResourceConflict:
            return _Confirmation.CONFLICT
        if not observation.artifact_secrets:
            return _Confirmation.CONFIRMED
        if not sleep_until_poll(deadline_at, clock, sleep):
            return _Confirmation.TIMED_OUT


def _pod_environment_is_stripped(plan: RunPodCreatePlan, pod: RunPodPodObservation) -> bool:
    environment = dict(pod.environment)
    return "FLASH_ARTIFACT_TOKEN" not in environment and all(
        environment.get(key) == value for key, value in plan.environment_without_artifact
    )


def _await_stripped_resources(
    plan: RunPodCreatePlan,
    observe: Observe,
    template_id: str,
    pod_id: str,
    handle: RunPodProviderHandle,
    inference_token: str,
    *,
    require_endpoint_probe: bool,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> _Confirmation:
    """wait until both artifact references are absent and a restarted pod serves again."""

    while True:
        try:
            _secret, template, _volume, pod = exact_core_resources(plan, observe(plan))
        except RunPodTransportFailure:
            if not sleep_until_poll(deadline_at, clock, sleep):
                return _Confirmation.OBSERVATION_FAILED
            continue
        except RunPodResourceConflict:
            return _Confirmation.CONFLICT
        # a successful patch response is not proof that the next read still names the resources we
        # patched. deterministic names can be reused after replacement, so ids bind cleanup to the
        # exact template and pod whose artifact reference was present before the transition.
        if template.id != template_id or pod.id != pod_id:
            return _Confirmation.IDENTITY_DRIFT
        resources_are_stripped = (
            template.environment == plan.environment_without_artifact
            and _pod_environment_is_stripped(plan, pod)
        )
        if resources_are_stripped and (
            not require_endpoint_probe
            or (
                readiness_state(pod.desired_status) == "running"
                and probe_with_deadline(
                    probe,
                    handle,
                    inference_token,
                    plan,
                    deadline_at=deadline_at,
                    clock=clock,
                )
            )
        ):
            return _Confirmation.CONFIRMED
        if not sleep_until_poll(deadline_at, clock, sleep):
            return _Confirmation.TIMED_OUT


def _unconfirmed_cleanup_result(
    plan: RunPodCreatePlan,
    confirmation: _Confirmation,
    handle: RunPodProviderHandle,
    *,
    timeout_reason: DeploymentErrorReason,
) -> DeploymentResult:
    if confirmation is _Confirmation.TIMED_OUT:
        return failure_result(
            plan,
            LifecycleFailure(
                "artifact_cleanup_timeout",
                reason=timeout_reason,
            ),
            handle=handle,
        )
    reason_by_confirmation: dict[_Confirmation, DeploymentErrorReason] = {
        _Confirmation.OBSERVATION_FAILED: "artifact_cleanup_observation_failed",
        _Confirmation.CONFLICT: "artifact_cleanup_conflict",
        _Confirmation.IDENTITY_DRIFT: "artifact_cleanup_identity_drift",
    }
    return unknown_result(
        plan,
        reason=reason_by_confirmation[confirmation],
        handle=handle,
    )


def delete_artifact_and_confirm(
    plan: RunPodCreatePlan,
    observe: Observe,
    patch_template: PatchTemplate,
    patch_pod: PatchPod,
    delete_secret: DeleteSecret,
    artifact: RunPodSecretObservation,
    handle: RunPodProviderHandle,
    inference_token: str,
    *,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    # the template and live pod must stop referring to the one-shot secret before it is deleted.
    # the stored value is only a named secret reference, not the credential itself, but observing
    # both stripped environments proves no dangling artifact entry remains on a later restart.
    while True:
        try:
            _secret, template, _volume, pod = exact_core_resources(plan, observe(plan))
        except RunPodTransportFailure:
            if not sleep_until_poll(deadline_at, clock, sleep):
                return unknown_result(
                    plan,
                    reason="artifact_cleanup_observation_failed",
                    handle=handle,
                )
            continue
        except RunPodResourceConflict:
            return unknown_result(
                plan,
                reason="artifact_cleanup_conflict",
                handle=handle,
            )
        needs_template_patch = template.environment != plan.environment_without_artifact
        needs_pod_patch = not _pod_environment_is_stripped(plan, pod)
        try:
            if needs_template_patch:
                patch_template(template.id, plan.template_payload(False))
            if needs_pod_patch:
                # the payload carries only flash-authored entries. if runpod replaces rather than merges
                # the map, provider-added values such as PUBLIC_KEY are dropped; the serving workload does
                # not consume them. if runpod retains them, the subset check below deliberately allows it.
                patch_pod(pod.id, plan.pod_environment_payload())
        except RunPodTransportFailure:
            if sleep_until_poll(deadline_at, clock, sleep):
                continue
            return unknown_result(
                plan,
                reason="artifact_cleanup_patch_unknown",
                handle=handle,
            )
        break
    if needs_template_patch or needs_pod_patch:
        # the endpoint probe authenticates with the inference token, not the artifact token. prove
        # the restarted workload before deleting the artifact secret so an unproven transition
        # remains recoverable, while the stripped references prevent the restarted pod using it.
        confirmation = _await_stripped_resources(
            plan,
            observe,
            template.id,
            pod.id,
            handle,
            inference_token,
            require_endpoint_probe=needs_pod_patch,
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
        if confirmation is not _Confirmation.CONFIRMED:
            return _unconfirmed_cleanup_result(
                plan,
                confirmation,
                handle,
                timeout_reason="artifact_cleanup_unproven",
            )
    try:
        delete_secret(artifact.id)
    except RunPodTransportFailure:
        return unknown_result(
            plan,
            reason="artifact_cleanup_delete_unknown",
            handle=handle,
        )
    confirmation = confirm_artifact_absence(
        plan,
        observe,
        deadline_at=deadline_at,
        clock=clock,
        sleep=sleep,
    )
    if confirmation is not _Confirmation.CONFIRMED:
        return _unconfirmed_cleanup_result(
            plan,
            confirmation,
            handle,
            timeout_reason="artifact_cleanup_delete_unknown",
        )
    return DeploymentResult.from_spec(plan.bundle.spec, status="ready", handle=handle)


def await_ready_and_reclaim(
    plan: RunPodCreatePlan,
    observe: Observe,
    patch_template: PatchTemplate,
    patch_pod: PatchPod,
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
        patch_template,
        patch_pod,
        delete_secret,
        artifact,
        cast("RunPodProviderHandle", ready.handle),
        inference_token,
        deadline_at=deadline_at,
        probe=probe,
        clock=clock,
        sleep=sleep,
    )
