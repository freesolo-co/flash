"""fresh-create and adoption phases for modal serving deployments."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from flash.serve.control import DeploymentResult
from flash.serve.provisioning.common.records import Clock, LifecycleFailure, Sleeper
from flash.serve.provisioning.modal.execution.lifecycle import mutation, observe
from flash.serve.provisioning.modal.execution.sdk import (
    ModalNamedResource,
    ModalObservation,
    ModalSdk,
    ModalSdkFailure,
)
from flash.serve.provisioning.modal.planning.plan import ModalCreatePlan
from flash.serve.provisioning.modal.planning.resources import ModalResourceConflict
from flash.serve.provisioning.modal.readiness_checks.readiness import (
    EndpointProbe,
    ExpectedResources,
    PhaseProof,
    TransientPhase,
    failure_result,
    from_sdk_failure,
    matches_transient,
    phase_proof,
    unknown_result,
    wait_for_phase,
)

_CLEANUP_RESERVE_SECONDS = 30.0


def _work_deadline(deadline_at: float, clock: Clock) -> float:
    remaining_seconds = max(0.0, deadline_at - clock())
    return deadline_at - min(_CLEANUP_RESERVE_SECONDS, remaining_seconds / 2.0)


@dataclass(slots=True)
class _CreatedResources:
    """what this invocation may have created, each flag set *before* its create is issued.

    Mutable and written in place so an interrupt handler can read it: the create sequence may be
    abandoned between any two steps, and only the resources already attempted need tearing down.

    Attempted, not confirmed. A create that Modal accepted but whose return value never reached us
    -- Ctrl-C landing between the accept and the assignment -- leaves the resource live and
    billing. Recording after the call meant cleanup walked past exactly that resource and still
    reported success, so the CLI printed a plain abort over a volume the customer keeps paying for.
    marking first keeps an unresolved create visible even when no provider id returns. cleanup then
    declines destructive cleanup and reports ambiguity instead of falsely claiming clean absence.

    The flags remain separate from `confirmed`: an attempt keeps ambiguous cleanup from reporting
    success, while only provider ids returned to this invocation authorize destructive cleanup.
    provider ids enter `confirmed` only after the mutation return reaches this invocation.

    `app_deployed` is the expensive one. The secrets and the volume are cheap storage; the app is
    the live GPU deployment, and it starts billing the moment `deploy_app` returns, which is well
    before the readiness probe the user is waiting on. Tracking it separately from the named
    resources is what lets abort stop compute first, the same order canonical teardown uses.
    """

    inference: bool = False
    artifact: bool = False
    volume: bool = False
    app_deployed: bool = False
    confirmed: ExpectedResources | None = None

    @property
    def any_created(self) -> bool:
        return self.inference or self.artifact or self.volume or self.app_deployed

    def confirm(
        self,
        *,
        volume_id: str | None = None,
        inference_secret_id: str | None = None,
        artifact_secret_id: str | None = None,
    ) -> None:
        """record one provider id as its create returns to this invocation."""

        current = self.confirmed or ExpectedResources(
            app_id=None,
            volume_id="",
            inference_secret_id="",
            artifact_secret_id=None,
        )
        self.confirmed = ExpectedResources(
            app_id=current.app_id,
            volume_id=volume_id if volume_id is not None else current.volume_id,
            inference_secret_id=(
                inference_secret_id
                if inference_secret_id is not None
                else current.inference_secret_id
            ),
            artifact_secret_id=(
                artifact_secret_id if artifact_secret_id is not None else current.artifact_secret_id
            ),
        )


def _create_resources(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    inference_token: str,
    artifact_token: str | None,
    *,
    deadline_at: float,
    created: _CreatedResources | None = None,
) -> ExpectedResources:
    # each resource is marked before its create is issued, never after. an interrupt between two
    # creates has to tear down what may already have landed, and a resource Modal accepted whose
    # return value never arrived is exactly the one that leaks. see `_CreatedResources`.
    record = created if created is not None else _CreatedResources()
    record.inference = True
    inference = mutation(
        lambda: sdk.create_inference_secret(
            plan,
            inference_token,
            deadline_at=deadline_at,
        )
    )
    assert type(inference) is ModalNamedResource
    record.confirm(inference_secret_id=inference.id)
    artifact = None
    if artifact_token is not None:
        record.artifact = True
        artifact = mutation(
            lambda: sdk.create_artifact_secret(
                plan,
                artifact_token,
                deadline_at=deadline_at,
            )
        )
        assert type(artifact) is ModalNamedResource
        record.confirm(artifact_secret_id=artifact.id)
    record.volume = True
    volume = mutation(lambda: sdk.create_volume(plan, deadline_at=deadline_at))
    assert type(volume) is ModalNamedResource
    record.confirm(volume_id=volume.id)
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
        deployed = mutation(lambda: sdk.deploy_app(plan, deadline_at=deadline_at))
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
    if created is not None:
        created.confirmed = expected_after_deploy
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


def _start_fresh_deployment(
    create_plan: ModalCreatePlan,
    finalized_plan: ModalCreatePlan,
    sdk: ModalSdk,
    inference_token: str,
    artifact_token: str | None,
    created: _CreatedResources,
    *,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> PhaseProof | DeploymentResult | None:
    work_deadline = _work_deadline(deadline_at, clock)
    expected = _create_resources(
        create_plan,
        sdk,
        inference_token,
        artifact_token,
        deadline_at=work_deadline,
        created=created,
    )
    try:
        return _deploy_once_then_wait(
            create_plan,
            sdk,
            inference_token,
            expected,
            artifact_present=artifact_token is not None,
            transient_phases=(),
            deadline_at=work_deadline,
            probe=probe,
            clock=clock,
            sleep=sleep,
            created=created,
        )
    except ModalSdkFailure as exc:
        # a definite failure here is terminal, and the secrets, volume and possibly the app are
        # already created and billing. returning from this catch would report the failure with
        # those resources left standing, because the outer handler that calls
        # `_abort_created_resources` never runs. only an ambiguous outcome stops here: the
        # mutation may have landed, so deleting by the deterministic names could destroy a
        # concurrent deployment.
        if exc.outcome_unknown:
            return unknown_result(finalized_plan)
        raise


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
        mutation(
            lambda: sdk.delete_secret(
                finalized_plan,
                artifact.id,
                deadline_at=deadline_at,
            )
        )
    except ModalSdkFailure as exc:
        if not exc.outcome_unknown:
            failure = from_sdk_failure(exc)
            return failure_result(
                finalized_plan,
                LifecycleFailure(failure.code, reason="artifact_cleanup_delete_rejected"),
                handle=proof.handle,
            )
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
    except ModalResourceConflict:
        return unknown_result(
            finalized_plan,
            reason="artifact_cleanup_conflict",
            handle=proof.handle,
        )
    except ModalSdkFailure:
        return unknown_result(
            finalized_plan,
            reason="artifact_cleanup_observation_failed",
            handle=proof.handle,
        )
    if cleaned is None:
        return unknown_result(
            finalized_plan,
            reason="artifact_cleanup_delete_unknown",
            handle=proof.handle,
        )
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
    expected: ExpectedResources | None = None,
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
            expected=expected,
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
            observe(
                finalized_plan,
                sdk,
                app_id_hint=None if expected is None else expected.app_id,
                deadline_at=deadline_at,
            ),
            TransientPhase(finalized_plan, False),
            expected,
        ):
            raise
        # hand it to the reclaim: the id-bound delete may find the secret already gone, and its own
        # wait -- which targets the cleaned phase
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


def _adopt_bootstrap(
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
    assert artifact is not None
    expected = ExpectedResources(
        app_id=bootstrap.handle.app_id,
        volume_id=bootstrap.handle.volume_id,
        inference_secret_id=bootstrap.handle.inference_secret_id,
        artifact_secret_id=artifact.id,
    )
    try:
        proved = wait_for_phase(
            bootstrap_plan,
            sdk,
            inference_token,
            artifact_present=True,
            expected=expected,
            transient_phases=(),
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
    except ModalResourceConflict:
        successor = observe(
            finalized_plan,
            sdk,
            app_id_hint=expected.app_id,
            deadline_at=deadline_at,
        )
        with_artifact = TransientPhase(finalized_plan, True)
        if matches_transient(successor, with_artifact, expected):
            finalized = phase_proof(
                finalized_plan,
                successor,
                artifact_present=True,
                expected=expected,
            )
            return _adopt_uncleaned(
                finalized_plan,
                sdk,
                finalized,
                inference_token,
                expected=expected,
                deadline_at=deadline_at,
                probe=probe,
                clock=clock,
                sleep=sleep,
            )
        cleaned = TransientPhase(finalized_plan, False)
        if not matches_transient(successor, cleaned, expected):
            raise
        finalized = phase_proof(
            finalized_plan,
            successor,
            artifact_present=False,
            expected=expected,
        )
        proved = wait_for_phase(
            finalized_plan,
            sdk,
            inference_token,
            artifact_present=False,
            expected=expected,
            # the artifact can still flicker back into view between polls while the concurrent
            # finalize settles. tolerate it exactly as the post-delete wait does, so a healthy
            # billing app is never reported as a definite conflict over a transient reading.
            transient_phases=(with_artifact,),
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
    if proved is None:
        return unknown_result(finalized_plan, handle=bootstrap.handle)
    return _finalize_bootstrap(
        bootstrap_plan,
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
    return _adopt_bootstrap(
        bootstrap_plan,
        finalized_plan,
        sdk,
        bootstrap,
        inference_token,
        deadline_at=deadline_at,
        probe=probe,
        clock=clock,
        sleep=sleep,
    )
