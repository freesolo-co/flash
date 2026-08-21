"""request-scoped modal serving deployment lifecycle."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass

from flash.serve.control import (
    DeploymentResult,
    ModalCredentials,
    ModalProviderHandle,
)
from flash.serve.control.types import validate_modal_handle

from ._common import (
    Clock,
    DeploymentBundle,
    InterruptedProvisioning,
    LifecycleFailure,
    ServingRuntimeSecrets,
    Sleeper,
    failed_deployment_result,
)
from ._modal_lifecycle import (
    mutation,
    observe,
    open_sdk,
    validate_control_inputs,
    validate_runtime_inputs,
)
from ._modal_plan import ModalCreatePlan, build_modal_create_plan
from ._modal_probe import ModalEndpointProbe
from ._modal_readiness import (
    EndpointProbe,
    ExpectedResources,
    PhaseProof,
    TransientPhase,
    failure_result,
    from_sdk_failure,
    matches_transient,
    phase_proof,
    probe_with_deadline,
    sleep_until_poll,
    unknown_result,
    wait_for_phase,
)
from ._modal_resources import (
    ModalResourceConflict,
    ensure_unique_resources,
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

_DEFAULT_ENDPOINT_PROBE = ModalEndpointProbe()


@dataclass(slots=True)
class _CreatedResources:
    """what this invocation may have created, each flag set *before* its create is issued.

    Mutable and written in place so an interrupt handler can read it: the create sequence may be
    abandoned between any two steps, and only the resources already attempted need tearing down.

    Attempted, not confirmed. A create that Modal accepted but whose return value never reached us
    -- Ctrl-C landing between the accept and the assignment -- leaves the resource live and
    billing. Recording after the call meant cleanup walked past exactly that resource and still
    reported success, so the CLI printed a plain abort over a volume the customer keeps paying for.
    Marking first inverts the error: the worst case is deleting a resource that was never made,
    and every delete here is name-addressed with `allow_missing=True`, so that case is a no-op.

    Flags rather than handles because nothing needs the ids. Deletes address secrets and the volume
    by their plan-derived names, which are known before any call is issued -- which is also what
    makes marking-before-the-call able to clean up a resource whose handle was never seen.

    `app_deployed` is the expensive one. The secrets and the volume are cheap storage; the app is
    the live GPU deployment, and it starts billing the moment `deploy_app` returns, which is well
    before the readiness probe the user is waiting on. Tracking it separately from the named
    resources is what lets abort stop compute first, the same order canonical teardown uses.
    """

    inference: bool = False
    artifact: bool = False
    volume: bool = False
    app_deployed: bool = False

    @property
    def any_created(self) -> bool:
        return self.inference or self.artifact or self.volume or self.app_deployed


def _create_resources(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    inference_token: str,
    artifact_token: str | None,
    created: _CreatedResources | None = None,
) -> ExpectedResources:
    # each resource is marked before its create is issued, never after. an interrupt between two
    # creates has to tear down what may already have landed, and a resource Modal accepted whose
    # return value never arrived is exactly the one that leaks. see `_CreatedResources`.
    record = created if created is not None else _CreatedResources()
    record.inference = True
    inference = mutation(lambda: sdk.create_inference_secret(plan, inference_token))
    artifact = None
    if artifact_token is not None:
        record.artifact = True
        artifact = mutation(lambda: sdk.create_artifact_secret(plan, artifact_token))
    record.volume = True
    volume = mutation(lambda: sdk.create_volume(plan))
    assert type(inference) is ModalNamedResource
    assert artifact is None or type(artifact) is ModalNamedResource
    assert type(volume) is ModalNamedResource
    return ExpectedResources(
        app_id=None,
        volume_id=volume.id,
        inference_secret_id=inference.id,
        artifact_secret_id=None if artifact is None else artifact.id,
    )


def _deploy_once_then_wait(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    inference_token: str,
    expected: ExpectedResources,
    *,
    artifact_present: bool,
    transient_phases: tuple[TransientPhase, ...],
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
    created: _CreatedResources | None = None,
) -> PhaseProof | None:
    deployed_app_id: str | None = None
    try:
        # marked before the call, not after: an interrupt or an ambiguous failure can land with the
        # app already deployed and billing, and cleanup that only knows about confirmed returns
        # would walk past it. `stop_app` on an app that never deployed is a no-op the abort path
        # already suppresses, so over-recording is safe and under-recording leaks a live gpu.
        if created is not None:
            created.app_deployed = True
        deployed = mutation(lambda: sdk.deploy_app(plan))
        if type(deployed) is not str:
            raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True)
        deployed_app_id = deployed
    except ModalSdkFailure as exc:
        if not exc.outcome_unknown:
            raise
    expected_after_deploy = ExpectedResources(
        app_id=deployed_app_id or expected.app_id,
        volume_id=expected.volume_id,
        inference_secret_id=expected.inference_secret_id,
        artifact_secret_id=expected.artifact_secret_id,
    )
    return wait_for_phase(
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
    proof: PhaseProof,
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
    expected = ExpectedResources(
        app_id=proof.handle.app_id,
        volume_id=proof.handle.volume_id,
        inference_secret_id=proof.handle.inference_secret_id,
        artifact_secret_id=artifact.id,
    )
    try:
        mutation(lambda: sdk.delete_secret(finalized_plan, artifact.name))
    except ModalSdkFailure as exc:
        if not exc.outcome_unknown:
            return failure_result(finalized_plan, from_sdk_failure(exc), handle=proof.handle)
    try:
        cleaned = wait_for_phase(
            finalized_plan,
            sdk,
            inference_token,
            artifact_present=False,
            expected=expected,
            transient_phases=(TransientPhase(finalized_plan, True),),
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
    except (ModalResourceConflict, ModalSdkFailure):
        return unknown_result(finalized_plan, handle=proof.handle)
    if cleaned is None:
        return unknown_result(finalized_plan, handle=proof.handle)
    return DeploymentResult.from_spec(
        finalized_plan.bundle.spec,
        status="ready",
        handle=cleaned.handle,
    )


def _finalize_bootstrap(
    bootstrap_plan: ModalCreatePlan,
    finalized_plan: ModalCreatePlan,
    sdk: ModalSdk,
    bootstrap: PhaseProof,
    inference_token: str,
    *,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    artifact = bootstrap.artifact
    if artifact is None:
        return unknown_result(finalized_plan, handle=bootstrap.handle)
    expected = ExpectedResources(
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
            transient_phases=(TransientPhase(bootstrap_plan, True),),
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
    except (ModalResourceConflict, ModalSdkFailure):
        return unknown_result(finalized_plan, handle=bootstrap.handle)
    if finalized is None:
        return unknown_result(finalized_plan, handle=bootstrap.handle)
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


def _adopt_uncleaned(
    finalized_plan: ModalCreatePlan,
    sdk: ModalSdk,
    finalized: PhaseProof,
    inference_token: str,
    *,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    """adopt a finalized app whose artifact secret was never reclaimed, then reclaim it.

    The bounded wait targets the phase this observation already showed -- artifact present.
    Targeting the *cleaned* phase could never build a proof while the secret is still there, so
    nothing would ever be probed: the solo case, where no one else is reclaiming, would burn the
    entire deadline and then hand an exhausted one to the reclaim.
    """

    try:
        proved = wait_for_phase(
            finalized_plan,
            sdk,
            inference_token,
            artifact_present=True,
            expected=None,
            transient_phases=(),
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
    except ModalResourceConflict:
        # the artifact can vanish underneath the wait: a concurrent `serve deploy`, or the run that
        # created this app finishing its own reclaim. that leaves the deployment
        # finalized-and-cleaned -- the success state -- yet a phase mismatch here, which would
        # otherwise surface as a definite `conflict` for a healthy, billing app.
        #
        # only a vanished artifact is tolerable. re-read and prove that is what happened: identity
        # drift -- a *replacement* app under the same names -- must stay a definite conflict rather
        # than be mistaken for someone else's successful reclaim.
        if not matches_transient(
            observe(finalized_plan, sdk),
            TransientPhase(finalized_plan, False),
            None,
        ):
            raise
        # hand it to the reclaim: the delete is name-addressed and `allow_missing`, so it is a
        # no-op against a secret already gone, and its own wait -- which targets the cleaned phase
        # this observation just showed -- is what decides ready versus unknown.
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
    if proved is None:
        # the deadline ran out with the endpoint never answering. the artifact is this app's
        # bootstrap credential and it is still here, so reclaiming it now would strip a container
        # that has not yet proven it finished hydrating -- and leave nothing for the rerun this
        # `outcome_unknown` invites. reclaim follows proof, never precedes it.
        return unknown_result(finalized_plan, handle=finalized.handle)
    return _delete_artifact_and_confirm(
        finalized_plan,
        sdk,
        proved,
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
    finalized: PhaseProof | None = None
    with contextlib.suppress(ModalResourceConflict):
        finalized = phase_proof(
            finalized_plan,
            observation,
            artifact_present=False,
            expected=None,
        )
    if finalized is not None:
        # poll rather than probe once. a cold gpu container can need longer than the probe's
        # 30-second cap to load, and a single probe answered `outcome_unknown` with most of the
        # caller's deadline still unspent -- so a rerun meant to follow an existing deployment
        # could never reach it, while fresh creates and explicit reconciliation both wait here.
        proved = wait_for_phase(
            finalized_plan,
            sdk,
            inference_token,
            artifact_present=False,
            expected=None,
            transient_phases=(),
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
        if proved is not None:
            return DeploymentResult.from_spec(
                finalized_plan.bundle.spec,
                status="ready",
                handle=proved.handle,
            )
        return unknown_result(finalized_plan, handle=finalized.handle)
    with contextlib.suppress(ModalResourceConflict):
        finalized = phase_proof(
            finalized_plan,
            observation,
            artifact_present=True,
            expected=None,
        )
    if finalized is not None:
        return _adopt_uncleaned(
            finalized_plan,
            sdk,
            finalized,
            inference_token,
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
    bootstrap = phase_proof(
        bootstrap_plan,
        observation,
        artifact_present=True,
        expected=None,
    )
    if not probe_with_deadline(
        probe,
        bootstrap.handle,
        inference_token,
        bootstrap_plan,
        deadline_at=deadline_at,
        clock=clock,
    ):
        return unknown_result(finalized_plan, handle=bootstrap.handle)
    artifact = bootstrap.artifact
    assert artifact is not None
    expected = ExpectedResources(
        app_id=bootstrap.handle.app_id,
        volume_id=bootstrap.handle.volume_id,
        inference_secret_id=bootstrap.handle.inference_secret_id,
        artifact_secret_id=artifact.id,
    )
    finalized = wait_for_phase(
        finalized_plan,
        sdk,
        inference_token,
        artifact_present=True,
        expected=expected,
        transient_phases=(TransientPhase(bootstrap_plan, True),),
        deadline_at=deadline_at,
        probe=probe,
        clock=clock,
        sleep=sleep,
    )
    if finalized is None:
        return unknown_result(finalized_plan, handle=bootstrap.handle)
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

    validate_runtime_inputs(credentials, runtime_secrets, deadline_at, clock)
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    finalized_plan = build_modal_create_plan(bundle, phase="finalized")
    sdk: ModalSdk | None = None
    created = _CreatedResources()
    reached_ready = False
    try:
        sdk = open_sdk(sdk_factory, credentials, finalized_plan)
        observation = observe(finalized_plan, sdk)
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
            return unknown_result(finalized_plan)
        if phase is None:
            return unknown_result(finalized_plan)
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
        return failure_result(finalized_plan, LifecycleFailure("conflict"))
    except ModalSdkFailure as exc:
        return failure_result(finalized_plan, from_sdk_failure(exc))
    except BaseException:
        # Ctrl-C derives from BaseException, so neither handler above sees it. Without this the
        # app, its volume, and its secrets stay live in the customer's Modal account and keep
        # billing, with nothing but a traceback that reads like nothing happened.
        # Bounded by `not reached_ready` for the same reason as the RunPod path: once the probe has
        # answered, tearing down would destroy a working deployment, and a half-finalized app is
        # recoverable by re-running the command.
        if (
            sdk is not None
            and created.any_created
            and not reached_ready
            and not _abort_created_resources(finalized_plan, sdk, created)
        ):
            # a step failed and was suppressed, so resources may still be live and billing.
            # replacing the interrupt with this carrier is the only way the cli can say so:
            # the generic handler prints "aborted", which reads as "nothing was created".
            raise InterruptedProvisioning("modal") from None
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

    validate_runtime_inputs(credentials, runtime_secrets, deadline_at, clock)
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    finalized_plan = build_modal_create_plan(bundle, phase="finalized")
    sdk: ModalSdk | None = None
    try:
        sdk = open_sdk(sdk_factory, credentials, finalized_plan)
        initial = observe(finalized_plan, sdk)
        ensure_unique_resources(initial)
        if initial.resource_count == 0:
            return DeploymentResult.from_spec(bundle.spec, status="absent")
        inference_token, _artifact_token = runtime_secrets._reveal_for_launch()
        proof = wait_for_phase(
            finalized_plan,
            sdk,
            inference_token,
            artifact_present=False,
            expected=None,
            transient_phases=(
                TransientPhase(bootstrap_plan, True),
                TransientPhase(finalized_plan, True),
            ),
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
        if proof is None:
            return unknown_result(finalized_plan)
        return DeploymentResult.from_spec(bundle.spec, status="ready", handle=proof.handle)
    except ModalResourceConflict:
        return failure_result(finalized_plan, LifecycleFailure("conflict"))
    except ModalSdkFailure as exc:
        return failure_result(finalized_plan, from_sdk_failure(exc))
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
        observation = observe(plan, sdk, app_id_hint=handle.app_id)
        app, _volume, _inference, _artifact = exact_teardown_resources(
            plan,
            handle,
            observation,
        )
        if app is not None and app.state in {"stopped", "failed"}:
            return observation if app.running_containers == 0 else None
        if not sleep_until_poll(deadline_at, clock, sleep):
            return None


def _suppressed(step: Callable[[], object]) -> bool:
    """run one teardown step, reporting whether it succeeded instead of hiding that.

    `Exception` rather than `BaseException`, so a second Ctrl-C during cleanup still gets out.
    """

    try:
        step()
    except Exception:
        return False
    return True


def _abort_created_resources(
    plan: ModalCreatePlan, sdk: ModalSdk, created: _CreatedResources
) -> bool:
    """best-effort teardown of a half-built deployment, stopping compute first.

    Every step is suppressed individually. This runs from an interrupt handler, so a failure here
    must neither replace the exception that brought us in -- the user pressed Ctrl-C, and a
    `ModalSdkFailure` surfacing instead would read as an unrelated provider bug -- nor stop the
    remaining deletes from being attempted. `Exception` rather than `BaseException`, so a second
    Ctrl-C during cleanup still gets out.
    """

    confirmed = True
    if created.app_deployed:
        # the app is the billable gpu deployment and it starts charging when `deploy_app` returns,
        # long before the readiness probe the user is waiting on. it also holds the volume mount,
        # and modal refuses to delete a volume an app still has attached, so stopping it first is
        # what makes the deletes below able to succeed at all. canonical teardown uses this order
        # for the same reason.
        confirmed &= _suppressed(lambda: mutation(lambda: sdk.stop_app(plan)))
    # the plan's names, not names read back off a create response: an attempted create whose
    # handle never arrived still has to be deletable, and `allow_missing=True` makes deleting one
    # that was never made a no-op.
    for attempted, name in (
        (created.artifact, plan.names.artifact_secret),
        (created.inference, plan.names.inference_secret),
    ):
        if attempted:
            confirmed &= _suppressed(
                lambda name=name: mutation(lambda: sdk.delete_secret(plan, name))
            )
    if created.volume:
        confirmed &= _suppressed(lambda: mutation(lambda: sdk.delete_volume(plan)))
    return confirmed


def _delete_teardown_resources(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    volume: ModalNamedResource | None,
    inference: ModalNamedResource | None,
    artifact: ModalNamedResource | None,
) -> None:
    if artifact is not None:
        mutation(lambda: sdk.delete_secret(plan, artifact.name))
    if inference is not None:
        mutation(lambda: sdk.delete_secret(plan, inference.name))
    if volume is not None:
        mutation(lambda: sdk.delete_volume(plan))


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

    validate_control_inputs(credentials, deadline_at, clock)
    plan = build_modal_create_plan(bundle, phase="finalized")
    _validate_handle(plan, handle)
    sdk: ModalSdk | None = None
    mutation_attempted = False
    try:
        sdk = open_sdk(sdk_factory, credentials, plan)
        observation = observe(plan, sdk, app_id_hint=handle.app_id)
        app, _volume, _inference, _artifact = exact_teardown_resources(plan, handle, observation)
        if app is None:
            return unknown_result(plan, handle=handle)
        if app.state == "deployed":
            mutation_attempted = True
            try:
                mutation(lambda: sdk.stop_app(plan))
            except ModalSdkFailure as exc:
                if not exc.outcome_unknown:
                    return failure_result(plan, from_sdk_failure(exc), handle=handle)
        terminal = _wait_for_terminal_app(
            plan,
            sdk,
            handle,
            deadline_at=deadline_at,
            clock=clock,
            sleep=sleep,
        )
        if terminal is None:
            return unknown_result(plan, handle=handle)
        _app, volume, inference, artifact = exact_teardown_resources(plan, handle, terminal)
        if any(resource is not None for resource in (volume, inference, artifact)):
            mutation_attempted = True
        _delete_teardown_resources(plan, sdk, volume, inference, artifact)
        final = observe(plan, sdk, app_id_hint=handle.app_id)
        exact_teardown_resources(plan, handle, final)
        if resources_are_absent(final, allow_terminal_app=True):
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        return unknown_result(plan, handle=handle)
    except ModalResourceConflict:
        if mutation_attempted:
            return unknown_result(plan, handle=handle)
        return failure_result(plan, LifecycleFailure("conflict"), handle=handle)
    except ModalSdkFailure as exc:
        if mutation_attempted:
            return unknown_result(plan, handle=handle)
        return failure_result(plan, from_sdk_failure(exc), handle=handle)
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

    validate_control_inputs(credentials, deadline_at, clock)
    plan = build_modal_create_plan(bundle, phase="finalized")
    sdk: ModalSdk | None = None
    try:
        sdk = open_sdk(sdk_factory, credentials, plan)
        observation = observe(plan, sdk)
        if resources_are_absent(observation, allow_terminal_app=False):
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        return failure_result(plan, LifecycleFailure("conflict"))
    except (ModalResourceConflict, ModalSdkFailure) as exc:
        if isinstance(exc, ModalSdkFailure):
            return failure_result(plan, from_sdk_failure(exc))
        return failure_result(plan, LifecycleFailure("conflict"))
    finally:
        if sdk is not None:
            sdk.close()
