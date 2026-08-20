"""request-scoped modal serving deployment lifecycle."""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from flash.serve.control import (
    DeploymentErrorCode,
    DeploymentResult,
    ModalCredentials,
    ModalProviderHandle,
)
from flash.serve.control.types import validate_modal_handle

from ._common import DeploymentBundle, ServingRuntimeSecrets, failed_deployment_result
from ._modal_plan import ModalCreatePlan, build_modal_create_plan
from ._modal_probe import ModalEndpointProbe
from ._modal_resources import (
    ModalResourceConflict,
    build_handle,
    ensure_unique_resources,
    exact_core_resources,
    exact_teardown_resources,
    resources_are_absent,
)
from ._modal_sdk import (
    ModalNamedResource,
    ModalObservation,
    ModalSdk,
    ModalSdkFactory,
    ModalSdkFailure,
    create_modal_sdk,
)

_READINESS_POLL_SECONDS = 2.0
_MAX_PROBE_TIMEOUT_SECONDS = 30.0

Clock = Callable[[], float]
Sleeper = Callable[[float], None]


class EndpointProbe(Protocol):
    def __call__(
        self,
        public_url: str,
        inference_token: str,
        bundle: DeploymentBundle,
        timeout_seconds: float,
    ) -> bool: ...


_DEFAULT_ENDPOINT_PROBE = ModalEndpointProbe()


@dataclass(frozen=True, slots=True)
class _LifecycleFailure:
    code: DeploymentErrorCode
    outcome_unknown: bool = False


@dataclass(frozen=True, slots=True)
class _ExpectedResources:
    app_id: str | None
    volume_id: str
    inference_secret_id: str
    artifact_secret_id: str | None


@dataclass(slots=True)
class _CreatedResources:
    """what this invocation actually created, recorded as each create returns.

    Mutable and written in place so an interrupt handler can read it: the create sequence may be
    abandoned between any two steps, and only the resources already made need tearing down.

    `app_deployed` is the expensive one. The secrets and the volume are cheap storage; the app is
    the live GPU deployment, and it starts billing the moment `deploy_app` returns, which is well
    before the readiness probe the user is waiting on. Tracking it separately from the named
    resources is what lets abort stop compute first, the same order canonical teardown uses.
    """

    inference: ModalNamedResource | None = None
    artifact: ModalNamedResource | None = None
    volume: ModalNamedResource | None = None
    app_deployed: bool = False

    @property
    def any_created(self) -> bool:
        return any((self.inference, self.artifact, self.volume)) or self.app_deployed


@dataclass(frozen=True, slots=True)
class _PhaseProof:
    handle: ModalProviderHandle
    artifact: ModalNamedResource | None


@dataclass(frozen=True, slots=True)
class _TransientPhase:
    plan: ModalCreatePlan
    artifact_present: bool


def _validate_deadline(deadline_at: float, clock: Clock) -> None:
    if type(deadline_at) not in {int, float} or not math.isfinite(float(deadline_at)):
        raise ValueError("deadline_at must be finite")
    if float(deadline_at) <= clock():
        raise ValueError("deadline_at must be in the future")


def _validate_runtime_inputs(
    credentials: ModalCredentials,
    runtime_secrets: ServingRuntimeSecrets,
    deadline_at: float,
    clock: Clock,
) -> None:
    if type(credentials) is not ModalCredentials:
        raise ValueError("modal credentials must use the exact credential type")
    if type(runtime_secrets) is not ServingRuntimeSecrets:
        raise ValueError("runtime secrets must use the exact secret boundary")
    _validate_deadline(deadline_at, clock)


def _validate_control_inputs(
    credentials: ModalCredentials,
    deadline_at: float,
    clock: Clock,
) -> None:
    if type(credentials) is not ModalCredentials:
        raise ValueError("modal credentials must use the exact credential type")
    _validate_deadline(deadline_at, clock)


def _failure_result(
    plan: ModalCreatePlan,
    failure: _LifecycleFailure,
    *,
    handle: ModalProviderHandle | None = None,
) -> DeploymentResult:
    return failed_deployment_result(
        plan.bundle.spec,
        failure.code,
        outcome_unknown=failure.outcome_unknown,
        handle=handle,
    )


def _unknown_result(
    plan: ModalCreatePlan,
    *,
    handle: ModalProviderHandle | None = None,
) -> DeploymentResult:
    return _failure_result(
        plan,
        _LifecycleFailure("resource_ambiguous", outcome_unknown=True),
        handle=handle,
    )


def _from_sdk_failure(exc: ModalSdkFailure) -> _LifecycleFailure:
    return _LifecycleFailure(exc.code, exc.outcome_unknown)


def _open_sdk(
    factory: ModalSdkFactory,
    credentials: ModalCredentials,
    plan: ModalCreatePlan,
) -> ModalSdk:
    try:
        sdk = factory(credentials, plan)
    except ModalSdkFailure:
        raise
    except Exception:
        raise ModalSdkFailure("transport_failed") from None
    if sdk.workspace_name != plan.placement.workspace_name:
        with contextlib.suppress(Exception):
            sdk.close()
        raise ModalSdkFailure("authentication_failed")
    if sdk.environment_name != plan.placement.environment:
        with contextlib.suppress(Exception):
            sdk.close()
        raise ModalSdkFailure("conflict")
    return sdk


def _observe(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    *,
    app_id_hint: str | None = None,
) -> ModalObservation:
    try:
        observation = sdk.observe(plan, app_id_hint=app_id_hint)
    except ModalSdkFailure:
        raise
    except Exception:
        raise ModalSdkFailure("transport_failed") from None
    if type(observation) is not ModalObservation:
        raise ModalSdkFailure("transport_failed")
    if (
        observation.workspace_name != plan.placement.workspace_name
        or observation.environment_name != plan.placement.environment
    ):
        raise ModalSdkFailure("authentication_failed")
    return observation


def _mutation(operation: Callable[[], object]) -> object:
    try:
        return operation()
    except ModalSdkFailure:
        raise
    except Exception:
        raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True) from None


def _sleep_until_poll(deadline_at: float, clock: Clock, sleep: Sleeper) -> bool:
    remaining = deadline_at - clock()
    if remaining <= 0:
        return False
    sleep(min(_READINESS_POLL_SECONDS, remaining))
    return clock() < deadline_at


def _probe_with_deadline(
    probe: EndpointProbe,
    handle: ModalProviderHandle,
    inference_token: str,
    plan: ModalCreatePlan,
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
            min(_MAX_PROBE_TIMEOUT_SECONDS, remaining),
        )
    except Exception:
        return False
    return accepted is True and clock() < deadline_at


def _phase_proof(
    plan: ModalCreatePlan,
    observation: ModalObservation,
    *,
    artifact_present: bool,
    expected: _ExpectedResources | None,
) -> _PhaseProof:
    app, volume, inference = exact_core_resources(plan, observation)
    artifacts = observation.artifact_secrets
    if len(artifacts) != int(artifact_present):
        raise ModalResourceConflict("modal artifact phase does not match the exact deployment")
    artifact = artifacts[0] if artifacts else None
    handle = build_handle(plan, app, volume, inference)
    if expected is not None and (
        (expected.app_id is not None and handle.app_id != expected.app_id)
        or handle.volume_id != expected.volume_id
        or handle.inference_secret_id != expected.inference_secret_id
        or (artifact_present and (artifact is None or artifact.id != expected.artifact_secret_id))
    ):
        raise ModalResourceConflict("modal provider ids drifted across deployment phases")
    return _PhaseProof(handle, artifact)


def _matches_transient(
    observation: ModalObservation,
    transient: _TransientPhase,
    expected: _ExpectedResources | None,
) -> bool:
    try:
        _phase_proof(
            transient.plan,
            observation,
            artifact_present=transient.artifact_present,
            expected=expected,
        )
    except ModalResourceConflict:
        return False
    return True


def _wait_for_phase(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    inference_token: str,
    *,
    artifact_present: bool,
    expected: _ExpectedResources | None,
    transient_phases: tuple[_TransientPhase, ...],
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> _PhaseProof | None:
    app_id_hint = None if expected is None else expected.app_id
    while True:
        observation = _observe(plan, sdk, app_id_hint=app_id_hint)
        proof: _PhaseProof | None = None
        if observation.resource_count:
            try:
                proof = _phase_proof(
                    plan,
                    observation,
                    artifact_present=artifact_present,
                    expected=expected,
                )
            except ModalResourceConflict:
                if not any(
                    _matches_transient(observation, transient, expected)
                    for transient in transient_phases
                ):
                    raise
        if proof is not None and _probe_with_deadline(
            probe,
            proof.handle,
            inference_token,
            plan,
            deadline_at=deadline_at,
            clock=clock,
        ):
            return proof
        if not _sleep_until_poll(deadline_at, clock, sleep):
            return None


def _create_resources(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    inference_token: str,
    artifact_token: str | None,
    created: _CreatedResources | None = None,
) -> _ExpectedResources:
    # each resource is recorded the instant it exists, before the next create runs. an interrupt
    # between two creates has to tear down what already landed, so "what did we build" cannot be
    # assembled only at the end of a function that may not reach its end.
    record = created if created is not None else _CreatedResources()
    inference = _mutation(lambda: sdk.create_inference_secret(plan, inference_token))
    record.inference = inference
    artifact = None
    if artifact_token is not None:
        artifact = _mutation(lambda: sdk.create_artifact_secret(plan, artifact_token))
        record.artifact = artifact
    volume = _mutation(lambda: sdk.create_volume(plan))
    record.volume = volume
    assert type(inference) is ModalNamedResource
    assert artifact is None or type(artifact) is ModalNamedResource
    assert type(volume) is ModalNamedResource
    return _ExpectedResources(
        app_id=None,
        volume_id=volume.id,
        inference_secret_id=inference.id,
        artifact_secret_id=None if artifact is None else artifact.id,
    )


def _deploy_once_then_wait(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    inference_token: str,
    expected: _ExpectedResources,
    *,
    artifact_present: bool,
    transient_phases: tuple[_TransientPhase, ...],
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
    created: _CreatedResources | None = None,
) -> _PhaseProof | None:
    deployed_app_id: str | None = None
    try:
        # marked before the call, not after: an interrupt or an ambiguous failure can land with the
        # app already deployed and billing, and cleanup that only knows about confirmed returns
        # would walk past it. `stop_app` on an app that never deployed is a no-op the abort path
        # already suppresses, so over-recording is safe and under-recording leaks a live gpu.
        if created is not None:
            created.app_deployed = True
        deployed = _mutation(lambda: sdk.deploy_app(plan))
        if type(deployed) is not str:
            raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True)
        deployed_app_id = deployed
    except ModalSdkFailure as exc:
        if not exc.outcome_unknown:
            raise
    expected_after_deploy = _ExpectedResources(
        app_id=deployed_app_id or expected.app_id,
        volume_id=expected.volume_id,
        inference_secret_id=expected.inference_secret_id,
        artifact_secret_id=expected.artifact_secret_id,
    )
    return _wait_for_phase(
        plan,
        sdk,
        inference_token,
        artifact_present=artifact_present,
        expected=expected_after_deploy,
        transient_phases=transient_phases,
        deadline_at=deadline_at,
        probe=probe,
        clock=clock,
        sleep=sleep,
    )


def _delete_artifact_and_confirm(
    finalized_plan: ModalCreatePlan,
    sdk: ModalSdk,
    proof: _PhaseProof,
    inference_token: str,
    *,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    artifact = proof.artifact
    if artifact is None:
        return DeploymentResult.from_spec(
            finalized_plan.bundle.spec,
            status="ready",
            handle=proof.handle,
        )
    expected = _ExpectedResources(
        app_id=proof.handle.app_id,
        volume_id=proof.handle.volume_id,
        inference_secret_id=proof.handle.inference_secret_id,
        artifact_secret_id=artifact.id,
    )
    try:
        _mutation(lambda: sdk.delete_secret(finalized_plan, artifact.name))
    except ModalSdkFailure as exc:
        if not exc.outcome_unknown:
            return _failure_result(finalized_plan, _from_sdk_failure(exc), handle=proof.handle)
    try:
        cleaned = _wait_for_phase(
            finalized_plan,
            sdk,
            inference_token,
            artifact_present=False,
            expected=expected,
            transient_phases=(_TransientPhase(finalized_plan, True),),
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
    except (ModalResourceConflict, ModalSdkFailure):
        return _unknown_result(finalized_plan, handle=proof.handle)
    if cleaned is None:
        return _unknown_result(finalized_plan, handle=proof.handle)
    return DeploymentResult.from_spec(
        finalized_plan.bundle.spec,
        status="ready",
        handle=cleaned.handle,
    )


def _finalize_bootstrap(
    bootstrap_plan: ModalCreatePlan,
    finalized_plan: ModalCreatePlan,
    sdk: ModalSdk,
    bootstrap: _PhaseProof,
    inference_token: str,
    *,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    artifact = bootstrap.artifact
    if artifact is None:
        return _unknown_result(finalized_plan, handle=bootstrap.handle)
    expected = _ExpectedResources(
        app_id=bootstrap.handle.app_id,
        volume_id=bootstrap.handle.volume_id,
        inference_secret_id=bootstrap.handle.inference_secret_id,
        artifact_secret_id=artifact.id,
    )
    try:
        finalized = _deploy_once_then_wait(
            finalized_plan,
            sdk,
            inference_token,
            expected,
            artifact_present=True,
            transient_phases=(_TransientPhase(bootstrap_plan, True),),
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
    except (ModalResourceConflict, ModalSdkFailure):
        return _unknown_result(finalized_plan, handle=bootstrap.handle)
    if finalized is None:
        return _unknown_result(finalized_plan, handle=bootstrap.handle)
    return _delete_artifact_and_confirm(
        finalized_plan,
        sdk,
        finalized,
        inference_token,
        deadline_at=deadline_at,
        probe=probe,
        clock=clock,
        sleep=sleep,
    )


def _adopt_existing(
    bootstrap_plan: ModalCreatePlan,
    finalized_plan: ModalCreatePlan,
    sdk: ModalSdk,
    observation: ModalObservation,
    inference_token: str,
    *,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    finalized: _PhaseProof | None = None
    with contextlib.suppress(ModalResourceConflict):
        finalized = _phase_proof(
            finalized_plan,
            observation,
            artifact_present=False,
            expected=None,
        )
    if finalized is not None:
        if _probe_with_deadline(
            probe,
            finalized.handle,
            inference_token,
            finalized_plan,
            deadline_at=deadline_at,
            clock=clock,
        ):
            return DeploymentResult.from_spec(
                finalized_plan.bundle.spec,
                status="ready",
                handle=finalized.handle,
            )
        return _unknown_result(finalized_plan, handle=finalized.handle)
    with contextlib.suppress(ModalResourceConflict):
        finalized = _phase_proof(
            finalized_plan,
            observation,
            artifact_present=True,
            expected=None,
        )
    if finalized is not None:
        if not _probe_with_deadline(
            probe,
            finalized.handle,
            inference_token,
            finalized_plan,
            deadline_at=deadline_at,
            clock=clock,
        ):
            return _unknown_result(finalized_plan, handle=finalized.handle)
        return _delete_artifact_and_confirm(
            finalized_plan,
            sdk,
            finalized,
            inference_token,
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
    bootstrap = _phase_proof(
        bootstrap_plan,
        observation,
        artifact_present=True,
        expected=None,
    )
    if not _probe_with_deadline(
        probe,
        bootstrap.handle,
        inference_token,
        bootstrap_plan,
        deadline_at=deadline_at,
        clock=clock,
    ):
        return _unknown_result(finalized_plan, handle=bootstrap.handle)
    artifact = bootstrap.artifact
    assert artifact is not None
    expected = _ExpectedResources(
        app_id=bootstrap.handle.app_id,
        volume_id=bootstrap.handle.volume_id,
        inference_secret_id=bootstrap.handle.inference_secret_id,
        artifact_secret_id=artifact.id,
    )
    finalized = _wait_for_phase(
        finalized_plan,
        sdk,
        inference_token,
        artifact_present=True,
        expected=expected,
        transient_phases=(_TransientPhase(bootstrap_plan, True),),
        deadline_at=deadline_at,
        probe=probe,
        clock=clock,
        sleep=sleep,
    )
    if finalized is None:
        return _unknown_result(finalized_plan, handle=bootstrap.handle)
    return _delete_artifact_and_confirm(
        finalized_plan,
        sdk,
        finalized,
        inference_token,
        deadline_at=deadline_at,
        probe=probe,
        clock=clock,
        sleep=sleep,
    )


def provision_modal_deployment(
    bundle: DeploymentBundle,
    credentials: ModalCredentials,
    runtime_secrets: ServingRuntimeSecrets,
    *,
    deadline_at: float,
    sdk_factory: ModalSdkFactory = create_modal_sdk,
    probe: EndpointProbe = _DEFAULT_ENDPOINT_PROBE,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> DeploymentResult:
    """create or prove one exact modal deployment without blind mutation retries."""

    _validate_runtime_inputs(credentials, runtime_secrets, deadline_at, clock)
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    finalized_plan = build_modal_create_plan(bundle, phase="finalized")
    sdk: ModalSdk | None = None
    created = _CreatedResources()
    reached_ready = False
    try:
        sdk = _open_sdk(sdk_factory, credentials, finalized_plan)
        observation = _observe(finalized_plan, sdk)
        ensure_unique_resources(observation)
        inference_token, artifact_token = runtime_secrets._reveal_for_launch()
        if observation.resource_count:
            return _adopt_existing(
                bootstrap_plan,
                finalized_plan,
                sdk,
                observation,
                inference_token,
                deadline_at=deadline_at,
                probe=probe,
                clock=clock,
                sleep=sleep,
            )
        create_plan = bootstrap_plan if artifact_token is not None else finalized_plan
        expected = _create_resources(
            create_plan,
            sdk,
            inference_token,
            artifact_token,
            created,
        )
        try:
            phase = _deploy_once_then_wait(
                create_plan,
                sdk,
                inference_token,
                expected,
                artifact_present=artifact_token is not None,
                transient_phases=(),
                deadline_at=deadline_at,
                probe=probe,
                clock=clock,
                sleep=sleep,
                created=created,
            )
        except (ModalResourceConflict, ModalSdkFailure):
            return _unknown_result(finalized_plan)
        if phase is None:
            return _unknown_result(finalized_plan)
        # the app is deployed and has answered the readiness probe. from here the only remaining
        # work is swapping the bootstrap phase out for the finalized one, so an interrupt must
        # leave the deployment standing rather than delete what the user just waited for.
        reached_ready = True
        if artifact_token is None:
            return DeploymentResult.from_spec(
                bundle.spec,
                status="ready",
                handle=phase.handle,
            )
        return _finalize_bootstrap(
            bootstrap_plan,
            finalized_plan,
            sdk,
            phase,
            inference_token,
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
    except ModalResourceConflict:
        return _failure_result(finalized_plan, _LifecycleFailure("conflict"))
    except ModalSdkFailure as exc:
        return _failure_result(finalized_plan, _from_sdk_failure(exc))
    except BaseException:
        # Ctrl-C derives from BaseException, so neither handler above sees it. Without this the
        # app, its volume, and its secrets stay live in the customer's Modal account and keep
        # billing, with nothing but a traceback that reads like nothing happened.
        # Bounded by `not reached_ready` for the same reason as the RunPod path: once the probe has
        # answered, tearing down would destroy a working deployment, and a half-finalized app is
        # recoverable by re-running the command.
        if sdk is not None and created.any_created and not reached_ready:
            _abort_created_resources(finalized_plan, sdk, created)
        raise
    finally:
        if sdk is not None:
            sdk.close()


def reconcile_modal_deployment(
    bundle: DeploymentBundle,
    credentials: ModalCredentials,
    runtime_secrets: ServingRuntimeSecrets,
    *,
    deadline_at: float,
    sdk_factory: ModalSdkFactory = create_modal_sdk,
    probe: EndpointProbe = _DEFAULT_ENDPOINT_PROBE,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> DeploymentResult:
    """read and prove one finalized modal deployment without provider mutation."""

    _validate_runtime_inputs(credentials, runtime_secrets, deadline_at, clock)
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    finalized_plan = build_modal_create_plan(bundle, phase="finalized")
    sdk: ModalSdk | None = None
    try:
        sdk = _open_sdk(sdk_factory, credentials, finalized_plan)
        initial = _observe(finalized_plan, sdk)
        ensure_unique_resources(initial)
        if initial.resource_count == 0:
            return DeploymentResult.from_spec(bundle.spec, status="absent")
        inference_token, _artifact_token = runtime_secrets._reveal_for_launch()
        proof = _wait_for_phase(
            finalized_plan,
            sdk,
            inference_token,
            artifact_present=False,
            expected=None,
            transient_phases=(
                _TransientPhase(bootstrap_plan, True),
                _TransientPhase(finalized_plan, True),
            ),
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
        if proof is None:
            return _unknown_result(finalized_plan)
        return DeploymentResult.from_spec(bundle.spec, status="ready", handle=proof.handle)
    except ModalResourceConflict:
        return _failure_result(finalized_plan, _LifecycleFailure("conflict"))
    except ModalSdkFailure as exc:
        return _failure_result(finalized_plan, _from_sdk_failure(exc))
    finally:
        if sdk is not None:
            sdk.close()


def _validate_handle(plan: ModalCreatePlan, handle: ModalProviderHandle) -> None:
    if type(handle) is not ModalProviderHandle:
        raise ValueError("handle must be an exact ModalProviderHandle")
    validate_modal_handle(handle)
    if (
        handle.deployment_id != plan.bundle.spec.deployment_id
        or handle.generation != plan.bundle.spec.generation
        or handle.engine_id != plan.bundle.spec.engine.engine_id
        or handle.workspace_name != plan.placement.workspace_name
        or handle.environment != plan.placement.environment
        or handle.region != plan.placement.region
        or handle.image_digest != plan.bundle.image.digest
        or handle.app_name != plan.names.app_or_pod
        or handle.volume_name != plan.names.volume
        or handle.inference_secret_name != plan.names.inference_secret
        or handle.public_url != plan.expected_public_url
    ):
        raise ValueError("modal handle does not match the exact deployment generation")


def resize_modal_volume(
    bundle: DeploymentBundle,
    handle: ModalProviderHandle,
    credentials: ModalCredentials,
    target_size_gb: int,
    *,
    sdk_factory: ModalSdkFactory = create_modal_sdk,
) -> DeploymentResult:
    """reject modal volume resizing locally because modal exposes no size api."""

    if type(credentials) is not ModalCredentials:
        raise ValueError("modal credentials must use the exact credential type")
    if type(target_size_gb) is not int or target_size_gb <= 0:
        raise ValueError("target_size_gb must be a positive integer")
    plan = build_modal_create_plan(bundle)
    _validate_handle(plan, handle)
    if sdk_factory is None:
        raise ValueError("sdk_factory must be callable")
    return failed_deployment_result(plan.bundle.spec, "invalid_request", handle=handle)


def _wait_for_terminal_app(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    handle: ModalProviderHandle,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> ModalObservation | None:
    while True:
        observation = _observe(plan, sdk, app_id_hint=handle.app_id)
        app, _volume, _inference, _artifact = exact_teardown_resources(
            plan,
            handle,
            observation,
        )
        if app is not None and app.state in {"stopped", "failed"}:
            return observation if app.running_containers == 0 else None
        if not _sleep_until_poll(deadline_at, clock, sleep):
            return None


def _abort_created_resources(
    plan: ModalCreatePlan, sdk: ModalSdk, created: _CreatedResources
) -> None:
    """best-effort teardown of a half-built deployment, stopping compute first.

    Every step is suppressed individually. This runs from an interrupt handler, so a failure here
    must neither replace the exception that brought us in -- the user pressed Ctrl-C, and a
    `ModalSdkFailure` surfacing instead would read as an unrelated provider bug -- nor stop the
    remaining deletes from being attempted. `Exception` rather than `BaseException`, so a second
    Ctrl-C during cleanup still gets out.
    """

    if created.app_deployed:
        # the app is the billable gpu deployment and it starts charging when `deploy_app` returns,
        # long before the readiness probe the user is waiting on. it also holds the volume mount,
        # and modal refuses to delete a volume an app still has attached, so stopping it first is
        # what makes the deletes below able to succeed at all. canonical teardown uses this order
        # for the same reason.
        with contextlib.suppress(Exception):
            _mutation(lambda: sdk.stop_app(plan))
    for secret in (created.artifact, created.inference):
        if secret is not None:
            with contextlib.suppress(Exception):
                _mutation(lambda name=secret.name: sdk.delete_secret(plan, name))
    if created.volume is not None:
        with contextlib.suppress(Exception):
            _mutation(lambda: sdk.delete_volume(plan))


def _delete_teardown_resources(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    volume: ModalNamedResource | None,
    inference: ModalNamedResource | None,
    artifact: ModalNamedResource | None,
) -> None:
    if artifact is not None:
        _mutation(lambda: sdk.delete_secret(plan, artifact.name))
    if inference is not None:
        _mutation(lambda: sdk.delete_secret(plan, inference.name))
    if volume is not None:
        _mutation(lambda: sdk.delete_volume(plan))


def teardown_modal_deployment(
    bundle: DeploymentBundle,
    handle: ModalProviderHandle,
    credentials: ModalCredentials,
    *,
    deadline_at: float,
    sdk_factory: ModalSdkFactory = create_modal_sdk,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> DeploymentResult:
    """stop one exact app, delete exact resources once, and prove terminal absence."""

    _validate_control_inputs(credentials, deadline_at, clock)
    plan = build_modal_create_plan(bundle, phase="finalized")
    _validate_handle(plan, handle)
    sdk: ModalSdk | None = None
    mutation_attempted = False
    try:
        sdk = _open_sdk(sdk_factory, credentials, plan)
        observation = _observe(plan, sdk, app_id_hint=handle.app_id)
        app, _volume, _inference, _artifact = exact_teardown_resources(plan, handle, observation)
        if app is None:
            return _unknown_result(plan, handle=handle)
        if app.state == "deployed":
            mutation_attempted = True
            try:
                _mutation(lambda: sdk.stop_app(plan))
            except ModalSdkFailure as exc:
                if not exc.outcome_unknown:
                    return _failure_result(plan, _from_sdk_failure(exc), handle=handle)
        terminal = _wait_for_terminal_app(
            plan,
            sdk,
            handle,
            deadline_at=deadline_at,
            clock=clock,
            sleep=sleep,
        )
        if terminal is None:
            return _unknown_result(plan, handle=handle)
        _app, volume, inference, artifact = exact_teardown_resources(plan, handle, terminal)
        if any(resource is not None for resource in (volume, inference, artifact)):
            mutation_attempted = True
        _delete_teardown_resources(plan, sdk, volume, inference, artifact)
        final = _observe(plan, sdk, app_id_hint=handle.app_id)
        exact_teardown_resources(plan, handle, final)
        if resources_are_absent(final, allow_terminal_app=True):
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        return _unknown_result(plan, handle=handle)
    except ModalResourceConflict:
        if mutation_attempted:
            return _unknown_result(plan, handle=handle)
        return _failure_result(plan, _LifecycleFailure("conflict"), handle=handle)
    except ModalSdkFailure as exc:
        if mutation_attempted:
            return _unknown_result(plan, handle=handle)
        return _failure_result(plan, _from_sdk_failure(exc), handle=handle)
    finally:
        if sdk is not None:
            sdk.close()


def confirm_modal_absence(
    bundle: DeploymentBundle,
    credentials: ModalCredentials,
    *,
    deadline_at: float,
    sdk_factory: ModalSdkFactory = create_modal_sdk,
    clock: Clock = time.monotonic,
) -> DeploymentResult:
    """report absent only after authoritative workspace resource confirmation."""

    _validate_control_inputs(credentials, deadline_at, clock)
    plan = build_modal_create_plan(bundle, phase="finalized")
    sdk: ModalSdk | None = None
    try:
        sdk = _open_sdk(sdk_factory, credentials, plan)
        observation = _observe(plan, sdk)
        if resources_are_absent(observation, allow_terminal_app=False):
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        return _failure_result(plan, _LifecycleFailure("conflict"))
    except (ModalResourceConflict, ModalSdkFailure) as exc:
        if isinstance(exc, ModalSdkFailure):
            return _failure_result(plan, _from_sdk_failure(exc))
        return _failure_result(plan, _LifecycleFailure("conflict"))
    finally:
        if sdk is not None:
            sdk.close()
