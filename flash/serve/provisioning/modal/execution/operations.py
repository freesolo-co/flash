"""request-scoped modal serving deployment lifecycle."""

from __future__ import annotations

import time

from flash.serve.control import (
    DeploymentResult,
    ModalCredentials,
    ModalProviderHandle,
)
from flash.serve.control.types import validate_modal_handle
from flash.serve.provisioning.common.records import (
    Clock,
    DeploymentBundle,
    FreshDeploymentArtifactTokenRequired,
    InterruptedProvisioning,
    LifecycleFailure,
    ServingRuntimeSecrets,
    Sleeper,
)
from flash.serve.provisioning.modal.execution.deployment import (
    _adopt_existing,
    _CreatedResources,
    _finalize_bootstrap,
    _start_fresh_deployment,
)
from flash.serve.provisioning.modal.execution.lifecycle import (
    mutation,
    observe,
    open_sdk,
    validate_control_inputs,
    validate_runtime_inputs,
)
from flash.serve.provisioning.modal.execution.sdk import (
    ModalObservation,
    ModalSdk,
    ModalSdkFactory,
    ModalSdkFailure,
    create_modal_sdk,
)
from flash.serve.provisioning.modal.planning.plan import ModalCreatePlan, build_modal_create_plan
from flash.serve.provisioning.modal.planning.resources import (
    ModalResourceConflict,
    ensure_unique_resources,
    exact_teardown_resources,
)
from flash.serve.provisioning.modal.readiness_checks.probe import ModalEndpointProbe
from flash.serve.provisioning.modal.readiness_checks.readiness import (
    EndpointProbe,
    ExpectedResources,
    TransientPhase,
    failure_result,
    from_sdk_failure,
    phase_proof,
    unknown_result,
    wait_for_phase,
)
from flash.serve.provisioning.modal.readiness_checks.teardown import (
    confirm_teardown_absence,
    confirmed_abort_handle,
    delete_confirmed_abort_resources,
    delete_teardown_resources,
    suppressed,
    wait_for_terminal_app,
)

_DEFAULT_ENDPOINT_PROBE = ModalEndpointProbe()


def _readiness_timeout_result(
    create_plan: ModalCreatePlan,
    finalized_plan: ModalCreatePlan,
    sdk: ModalSdk,
    created: _CreatedResources,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    cleaned = _abort_created_resources(
        create_plan,
        sdk,
        created,
        expected=created.confirmed,
        deadline_at=deadline_at,
        clock=clock,
        sleep=sleep,
    )
    handle = confirmed_abort_handle(finalized_plan, created.confirmed)
    if not cleaned:
        return unknown_result(finalized_plan, handle=handle)
    return failure_result(finalized_plan, LifecycleFailure("readiness_failed"))


def _resource_conflict_result(
    create_plan: ModalCreatePlan | None,
    finalized_plan: ModalCreatePlan,
    sdk: ModalSdk | None,
    created: _CreatedResources,
    reached_ready: bool,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    if create_plan is None or sdk is None or not created.any_created or reached_ready:
        return failure_result(finalized_plan, LifecycleFailure("conflict"))
    cleaned = _abort_created_resources(
        create_plan,
        sdk,
        created,
        expected=created.confirmed,
        deadline_at=deadline_at,
        clock=clock,
        sleep=sleep,
    )
    if cleaned:
        return failure_result(finalized_plan, LifecycleFailure("conflict"))
    handle = confirmed_abort_handle(finalized_plan, created.confirmed)
    return unknown_result(finalized_plan, handle=handle)


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
    create_plan: ModalCreatePlan | None = None
    created = _CreatedResources()
    reached_ready = False
    try:
        sdk = open_sdk(sdk_factory, credentials, finalized_plan, deadline_at, clock)
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
        if artifact_token is None:
            raise FreshDeploymentArtifactTokenRequired
        create_plan = bootstrap_plan
        phase = _start_fresh_deployment(
            create_plan,
            finalized_plan,
            sdk,
            inference_token,
            artifact_token,
            created,
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
        if isinstance(phase, DeploymentResult):
            return phase
        if phase is None:
            return _readiness_timeout_result(
                create_plan, finalized_plan, sdk, created, deadline_at, clock, sleep
            )
        # the app is deployed and has answered the readiness probe. from here the only remaining
        # work is swapping the bootstrap phase out for the finalized one, so an interrupt must
        # leave the deployment standing rather than delete what the user just waited for.
        reached_ready = True
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
        # pre-create conflicts remain non-destructive. after this invocation has confirmed provider
        # ids, cleanup can target only that generation and must not strand a live billing app.
        return _resource_conflict_result(
            create_plan,
            finalized_plan,
            sdk,
            created,
            reached_ready,
            deadline_at=deadline_at,
            clock=clock,
            sleep=sleep,
        )
    except ModalSdkFailure as exc:
        # an ambiguous mutation may have landed under the deterministic name after our initial empty
        # observation, so deleting it could destroy a concurrent deployment. a definite failure makes
        # this create terminal, so its attempted resources are aborted. once readiness was proved, the
        # deployment is working and the interrupt-path bound applies: leave it recoverable.
        if (
            not exc.outcome_unknown
            and sdk is not None
            and created.any_created
            and not reached_ready
            and not _abort_created_resources(
                create_plan,
                sdk,
                created,
                expected=created.confirmed,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            )
        ):
            return unknown_result(finalized_plan)
        return failure_result(finalized_plan, from_sdk_failure(exc))
    except BaseException:
        # Ctrl-C derives from BaseException, so neither handler above sees it. Without this the
        # app, its volume, and its secrets stay live in the customer's Modal account and keep
        # billing, with nothing but a traceback that reads like nothing happened.
        # bounded by `not reached_ready`: once the probe has answered, tearing down would destroy a
        # working deployment, and a half-finalized app is
        # recoverable by re-running the command.
        if (
            sdk is not None
            and created.any_created
            and not reached_ready
            and not _abort_created_resources(
                create_plan,
                sdk,
                created,
                expected=created.confirmed,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            )
        ):
            # a step failed and was suppressed, so resources may still be live and billing.
            # replacing the interrupt with this carrier is the only way the cli can say so:
            # the generic handler prints "aborted", which reads as "nothing was created".
            raise InterruptedProvisioning("modal") from None
        raise
    finally:
        if sdk is not None:
            sdk.close()


def _observed_modal_handle(
    bootstrap_plan: ModalCreatePlan,
    finalized_plan: ModalCreatePlan,
    observation: ModalObservation,
) -> ModalProviderHandle:
    for plan in (finalized_plan, bootstrap_plan):
        try:
            return phase_proof(
                plan,
                observation,
                artifact_present=bool(observation.artifact_secrets),
                expected=None,
            ).handle
        except ModalResourceConflict:
            continue
    raise ModalResourceConflict("modal resources do not match a deployment phase")


def reconcile_modal_deployment(
    bundle: DeploymentBundle,
    credentials: ModalCredentials,
    runtime_secrets: ServingRuntimeSecrets | None,
    *,
    deadline_at: float,
    sdk_factory: ModalSdkFactory = create_modal_sdk,
    probe: EndpointProbe = _DEFAULT_ENDPOINT_PROBE,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> DeploymentResult:
    """read and prove one finalized modal deployment without provider mutation."""

    validate_control_inputs(credentials, deadline_at, clock)
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    finalized_plan = build_modal_create_plan(bundle, phase="finalized")
    sdk: ModalSdk | None = None
    try:
        sdk = open_sdk(sdk_factory, credentials, finalized_plan, deadline_at, clock)
        initial = observe(finalized_plan, sdk)
        ensure_unique_resources(initial)
        if initial.resource_count == 0:
            return DeploymentResult.from_spec(bundle.spec, status="absent")
        if runtime_secrets is None:
            handle = _observed_modal_handle(bootstrap_plan, finalized_plan, initial)
            return unknown_result(
                finalized_plan,
                reason="readiness_deadline_unproven",
                handle=handle,
            )
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
            # the probe never converged, but the resources observed above are live and billing.
            # report them so a teardown has ids to target. a phase mismatch here is not fatal on
            # its own: an unnamed ambiguous result is still better than losing the outcome.
            try:
                handle = _observed_modal_handle(bootstrap_plan, finalized_plan, initial)
            except ModalResourceConflict:
                handle = None
            return unknown_result(
                finalized_plan,
                reason="readiness_deadline_unproven",
                handle=handle,
            )
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


def _abort_created_resources(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    created: _CreatedResources,
    *,
    expected: ExpectedResources | None,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> bool:
    """best-effort teardown of a half-built deployment, with observed absence as proof."""

    try:
        initial = observe(plan, sdk)
        ensure_unique_resources(initial)
    except (ModalResourceConflict, ModalSdkFailure):
        return False

    if initial.apps:
        if expected is None or expected.app_id is None:
            # plan identity cannot distinguish this invocation from a concurrent deploy of the same
            # generation: both callers produce byte-identical names, tags, and app identity. an
            # attempted deploy therefore proves nothing about the observed app. refusing teardown
            # may leave our own ambiguous create for later proof-based reclaim, while stopping a
            # race winner destroys a live deployment and cannot be recovered. ambiguity stays live.
            return False
        try:
            proof = phase_proof(
                plan,
                initial,
                artifact_present=plan.phase == "bootstrap",
                expected=expected,
            )
        except ModalResourceConflict:
            return False
        # stop acknowledgement is not terminal proof. modal may retain the mount until lifecycle
        # observation reaches a zero-container terminal state, so use the same wait as teardown.
        suppressed(lambda: mutation(lambda: sdk.stop_app(plan, proof.handle.app_id)))
        try:
            terminal = wait_for_terminal_app(
                plan,
                sdk,
                proof.handle,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            )
        except (ModalResourceConflict, ModalSdkFailure):
            return False
        if terminal is None:
            return False
        try:
            _app, volume, inference, artifact = exact_teardown_resources(
                plan,
                proof.handle,
                terminal,
            )
        except ModalResourceConflict:
            return False
        # cleanup runs inside an interrupt handler. attempt every independent delete even if one
        # raises, then let the authoritative observation below decide whether anything remains.
        delete_teardown_resources(
            plan,
            sdk,
            volume,
            inference,
            artifact,
            suppress_failures=True,
        )
        try:
            return confirm_teardown_absence(plan, sdk, proof.handle)
        except (ModalResourceConflict, ModalSdkFailure):
            return False

    # plan identity is byte-identical across same-generation racers. only provider ids returned to
    # this invocation can authorize deletion of an observed secret or volume.
    return delete_confirmed_abort_resources(
        plan,
        sdk,
        initial,
        expected,
        artifact_attempted=created.artifact,
        inference_attempted=created.inference,
        volume_attempted=created.volume,
        app_deployed=created.app_deployed,
    )


def _teardown_plan(
    finalized_plan: ModalCreatePlan,
    bootstrap_plan: ModalCreatePlan,
    handle: ModalProviderHandle,
    observation: ModalObservation,
) -> ModalCreatePlan:
    try:
        exact_teardown_resources(finalized_plan, handle, observation)
    except ModalResourceConflict:
        exact_teardown_resources(bootstrap_plan, handle, observation)
        return bootstrap_plan
    return finalized_plan


def _reclaim_plan(
    finalized_plan: ModalCreatePlan,
    bootstrap_plan: ModalCreatePlan,
    observation: ModalObservation,
) -> ModalCreatePlan:
    try:
        exact_teardown_resources(finalized_plan, None, observation)
    except ModalResourceConflict:
        exact_teardown_resources(bootstrap_plan, None, observation)
        return bootstrap_plan
    return finalized_plan


def _reclaim_modal_deployment(
    finalized_plan: ModalCreatePlan,
    bootstrap_plan: ModalCreatePlan,
    sdk: ModalSdk,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    """reclaim deterministic resources after an ambiguous create returned no handle."""

    plan = finalized_plan
    mutation_attempted = False
    try:
        observation = observe(plan, sdk)
        if observation.apps:
            plan = _reclaim_plan(finalized_plan, bootstrap_plan, observation)
        app, volume, inference, artifact = exact_teardown_resources(plan, None, observation)
        if app is None and observation.resource_count == 0:
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        app_id = None if app is None else app.app_id
        if app is not None and app.state == "deployed":
            mutation_attempted = True
            try:
                mutation(lambda: sdk.stop_app(plan, app.app_id))
            except ModalSdkFailure as exc:
                if not exc.outcome_unknown:
                    return failure_result(plan, from_sdk_failure(exc))
            terminal = wait_for_terminal_app(
                plan,
                sdk,
                None,
                app_id=app.app_id,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            )
            if terminal is None:
                return unknown_result(plan)
            _app, volume, inference, artifact = exact_teardown_resources(
                plan,
                None,
                terminal,
                app_id_hint=app.app_id,
            )
        if any(resource is not None for resource in (volume, inference, artifact)):
            mutation_attempted = True
        delete_teardown_resources(plan, sdk, volume, inference, artifact)
        if confirm_teardown_absence(plan, sdk, None, app_id=app_id):
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        return unknown_result(plan)
    except ModalResourceConflict:
        if mutation_attempted:
            return unknown_result(plan)
        return failure_result(plan, LifecycleFailure("conflict"))
    except ModalSdkFailure as exc:
        if mutation_attempted:
            return unknown_result(plan)
        return failure_result(plan, from_sdk_failure(exc))


def teardown_modal_deployment(
    bundle: DeploymentBundle,
    handle: ModalProviderHandle | None,
    credentials: ModalCredentials,
    *,
    deadline_at: float,
    sdk_factory: ModalSdkFactory = create_modal_sdk,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> DeploymentResult:
    """stop one exact app, delete exact resources once, and prove terminal absence.

    A provider handle keeps the ordinary path bound to provider-returned ids. ``None`` is reserved
    for explicit undeploy after an ambiguous create returned no ids: the immutable deployment bundle
    authorizes reclaim by exact deterministic identity.
    """

    validate_control_inputs(credentials, deadline_at, clock)
    finalized_plan = build_modal_create_plan(bundle, phase="finalized")
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    if handle is not None:
        _validate_handle(finalized_plan, handle)
    plan = finalized_plan
    sdk: ModalSdk | None = None
    mutation_attempted = False
    try:
        sdk = open_sdk(sdk_factory, credentials, plan, deadline_at, clock)
        if handle is None:
            return _reclaim_modal_deployment(
                finalized_plan,
                bootstrap_plan,
                sdk,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            )
        observation = observe(plan, sdk, app_id_hint=handle.app_id)
        if observation.apps and observation.apps[0].state == "deployed":
            plan = _teardown_plan(finalized_plan, bootstrap_plan, handle, observation)
        app, _volume, _inference, _artifact = exact_teardown_resources(plan, handle, observation)
        if app is None:
            if observation.resource_count == 0:
                return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
            return unknown_result(plan, handle=handle)
        if app.state == "deployed":
            mutation_attempted = True
            try:
                mutation(lambda: sdk.stop_app(plan, handle.app_id))
            except ModalSdkFailure as exc:
                if not exc.outcome_unknown:
                    return failure_result(plan, from_sdk_failure(exc), handle=handle)
        terminal = wait_for_terminal_app(
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
        delete_teardown_resources(plan, sdk, volume, inference, artifact)
        if confirm_teardown_absence(plan, sdk, handle):
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
