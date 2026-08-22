"""request-scoped persistent runpod pod provisioning lifecycle."""

from __future__ import annotations

import time
from collections.abc import Callable

from flash.serve.control import (
    DeploymentErrorReason,
    DeploymentResult,
    RunPodCredentials,
    RunPodProviderHandle,
)
from flash.serve.control.types import validate_runpod_handle

from ._common import (
    Clock,
    DeploymentBundle,
    InterruptedProvisioning,
    LifecycleFailure,
    ServingRuntimeSecrets,
    Sleeper,
)
from ._runpod_lifecycle import (
    TransportFactory,
    identity,
    mutation_call,
    open_transport,
    read_call,
    validate_control_inputs,
    validate_runtime_inputs,
)
from ._runpod_mutations import MutationKind, MutationLedger
from ._runpod_plan import RunPodCreatePlan, build_runpod_create_plan
from ._runpod_probe import RunPodEndpointProbe
from ._runpod_protocol import (
    CREATE_SECRET,
    DELETE_SECRET,
    LIST_ACCOUNT_SECRETS,
    RunPodObservation,
    RunPodPodObservation,
    RunPodSecretObservation,
    RunPodTemplateObservation,
    RunPodVolumeObservation,
    parse_account_secrets,
    parse_created_pod,
    parse_created_secret,
    parse_created_template,
    parse_created_volume,
    parse_deleted_secret,
    parse_pods,
    parse_templates,
    parse_volumes,
)
from ._runpod_readiness import (
    DeleteSecret,
    EndpointProbe,
    Observe,
    PatchPod,
    PatchTemplate,
    adopt_existing_generation,
    await_ready_and_reclaim,
    failure_result,
    read_only_reconcile,
    sleep_until_poll,
    unknown_result,
)
from ._runpod_resources import (
    RunPodResourceConflict,
    build_handle,
    complete_resource_set,
    ensure_unique_resources,
    exact_teardown_resources,
    merge_resource_observations,
    pod_identity_matches,
    template_identity_matches,
    volume_identity_matches,
)
from ._runpod_transport import (
    RunPodTransport,
    RunPodTransportFailure,
    StdlibRunPodTransport,
)

_DEFAULT_ENDPOINT_PROBE = RunPodEndpointProbe()

# how much of the caller's deadline is held back for teardown. one observe, one pod delete, a
# short absence wait, then four deletes and a confirming observe. the same bounded interval lets an
# explicit reclaim discover a create that became visible only after its ambiguous response.
_CLEANUP_RESERVE_SECONDS = 30.0


def _work_deadline(deadline_at: float, clock: Clock) -> float:
    """the deadline for every phase of a create that can leave a resource behind.

    Held back from the caller's deadline so the teardown that follows still has a live one. Every
    transport call takes a deadline and rejects outright once it has passed, so cleanup handed an
    exhausted deadline issues no deletes at all: the pod, its template, volume, and both secrets
    stay live and billing while the result claims the creation ledger was aborted. Readiness is
    what usually reaches the deadline -- a pod that runs but never answers the probe polls until
    there is nothing left -- but a create slow enough to expire mid-sequence lands in the same
    place, so the reserve covers the whole ledger-bearing stretch rather than the wait alone.

    Half the remaining budget when that is smaller than the reserve, so a caller whose deadline is
    shorter than the reserve gets both phases instead of a readiness wait that starts expired. At or
    past the caller's deadline, no reserve is subtracted and the work deadline remains unchanged.

    Adoption does not use this: it has no ledger, so it never reaches teardown and spending its
    whole deadline proving an existing pod healthy is exactly what it should do.
    """

    remaining_seconds = max(0.0, deadline_at - clock())
    return deadline_at - min(_CLEANUP_RESERVE_SECONDS, remaining_seconds / 2.0)


def _observe(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    *,
    deadline_at: float,
) -> RunPodObservation:
    account_id, secrets = read_call(
        lambda: transport.graphql(
            LIST_ACCOUNT_SECRETS,
            {},
            mutation=False,
            deadline_at=deadline_at,
        ),
        parse_account_secrets,
    )
    templates = read_call(
        lambda: transport.rest("GET", "/templates", None, mutation=False, deadline_at=deadline_at),
        # only flash's own template is parsed strictly; foreign rows in the customer's account may
        # omit fields this parser requires and must not fail the whole observation.
        lambda raw: parse_templates(raw, keep_name=plan.names.template),
    )
    volumes = read_call(
        lambda: transport.rest(
            "GET", "/networkvolumes", None, mutation=False, deadline_at=deadline_at
        ),
        parse_volumes,
    )
    pods = read_call(
        # runpod omits the machine and network-volume objects unless asked, and gpu type and data
        # center live only inside them. without these flags a pod's gpuTypeId is always absent, so
        # adoption could never confirm a pod matches the plan's placement.
        lambda: transport.rest(
            "GET",
            "/pods",
            None,
            mutation=False,
            deadline_at=deadline_at,
            query={"includeMachine": "true", "includeNetworkVolume": "true"},
        ),
        # only flash's own pod is parsed strictly; foreign account pods may be cpu-only or omit
        # fields this deployment requires and must not fail the whole observation.
        lambda raw: parse_pods(raw, keep_name=plan.names.app_or_pod),
    )
    assert type(account_id) is str
    assert type(secrets) is tuple
    assert type(templates) is tuple
    assert type(volumes) is tuple
    assert type(pods) is tuple
    if account_id != plan.placement.account_id:
        raise RunPodTransportFailure("authentication_failed")
    return RunPodObservation(
        account_id=account_id,
        inference_secrets=tuple(
            item for item in secrets if item.name == plan.names.inference_secret
        ),
        artifact_secrets=tuple(item for item in secrets if item.name == plan.names.artifact_secret),
        templates=templates,
        volumes=tuple(item for item in volumes if item.name == plan.names.volume),
        pods=tuple(item for item in pods if item.name == plan.names.app_or_pod),
    )


def _from_transport_failure(exc: RunPodTransportFailure) -> LifecycleFailure:
    reason = "mutation_outcome_unknown" if exc.outcome_unknown else None
    return LifecycleFailure(exc.code, exc.outcome_unknown, reason)


def _bind_observe(transport: RunPodTransport, *, deadline_at: float) -> Observe:
    """give the readiness module a way to re-read state without handing it the transport."""

    return lambda plan: _observe(plan, transport, deadline_at=deadline_at)


def _bind_patch_template(transport: RunPodTransport, *, deadline_at: float) -> PatchTemplate:
    def patch(template_id: str, payload: dict[str, object]) -> None:
        mutation_call(
            lambda: transport.rest(
                "PATCH",
                f"/templates/{template_id}",
                payload,
                mutation=True,
                deadline_at=deadline_at,
            ),
            identity,
        )

    return patch


def _bind_patch_pod(transport: RunPodTransport, *, deadline_at: float) -> PatchPod:
    def patch(pod_id: str, payload: dict[str, object]) -> None:
        mutation_call(
            lambda: transport.rest(
                "PATCH",
                f"/pods/{pod_id}",
                payload,
                mutation=True,
                deadline_at=deadline_at,
            ),
            identity,
        )

    return patch


def _bind_delete_secret(transport: RunPodTransport, *, deadline_at: float) -> DeleteSecret:
    return lambda secret_id: _delete_secret_once(transport, secret_id, deadline_at=deadline_at)


def reconcile_runpod_deployment(
    bundle: DeploymentBundle,
    credentials: RunPodCredentials,
    runtime_secrets: ServingRuntimeSecrets,
    *,
    deadline_at: float,
    transport_factory: TransportFactory = StdlibRunPodTransport,
    probe: EndpointProbe = _DEFAULT_ENDPOINT_PROBE,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> DeploymentResult:
    """read and prove one deterministic deployment without mutating provider state."""

    validate_runtime_inputs(credentials, runtime_secrets, deadline_at, clock)
    plan = build_runpod_create_plan(bundle)
    inference_token, _artifact_token = runtime_secrets._reveal_for_launch()
    try:
        transport = open_transport(transport_factory, credentials)
        return read_only_reconcile(
            plan,
            _bind_observe(transport, deadline_at=deadline_at),
            inference_token,
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
            # this function mutates nothing by contract, so it can never be the caller that undoes
            # a pod it could not prove. `failed` is reserved for the create path, which aborts its
            # own ledger in the same breath; reporting it from here would tell the supervisor to
            # discard the deployment record while the pod stays live and billing.
            unproven_is_failure=False,
        )
    except RunPodTransportFailure as exc:
        return failure_result(plan, _from_transport_failure(exc))


def _delete_secret_once(
    transport: RunPodTransport,
    secret_id: str,
    *,
    deadline_at: float,
) -> None:
    try:
        mutation_call(
            lambda: transport.graphql(
                DELETE_SECRET,
                {"id": secret_id},
                mutation=True,
                deadline_at=deadline_at,
            ),
            parse_deleted_secret,
        )
    except RunPodTransportFailure as exc:
        if exc.code != "not_found":
            raise


def _delete_rest_once(
    transport: RunPodTransport,
    path: str,
    *,
    deadline_at: float,
) -> None:
    try:
        mutation_call(
            lambda: transport.rest("DELETE", path, None, mutation=True, deadline_at=deadline_at),
            identity,
        )
    except RunPodTransportFailure as exc:
        if exc.code != "not_found":
            raise


def _delete_tolerating_ambiguity(delete: Callable[[], None]) -> None:
    try:
        delete()
    except RunPodTransportFailure as exc:
        if not exc.outcome_unknown:
            raise


def _create_secret(
    transport: RunPodTransport,
    ledger: MutationLedger,
    kind: MutationKind,
    name: str,
    value: str,
    *,
    deadline_at: float,
) -> RunPodSecretObservation:
    ledger.begin(kind)
    created = mutation_call(
        lambda: transport.graphql(
            CREATE_SECRET,
            {"name": name, "value": value},
            mutation=True,
            deadline_at=deadline_at,
        ),
        parse_created_secret,
    )
    assert type(created) is RunPodSecretObservation
    if created.name != name:
        raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)
    ledger.confirm(kind, created.id)
    return created


def _create_resources(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    ledger: MutationLedger,
    inference_token: str,
    artifact_token: str | None,
    *,
    deadline_at: float,
) -> tuple[
    RunPodSecretObservation,
    RunPodSecretObservation | None,
    RunPodTemplateObservation,
    RunPodVolumeObservation,
    RunPodPodObservation,
]:
    inference = _create_secret(
        transport,
        ledger,
        "inference_secret",
        plan.names.inference_secret,
        inference_token,
        deadline_at=deadline_at,
    )
    artifact = None
    if artifact_token is not None:
        artifact = _create_secret(
            transport,
            ledger,
            "artifact_secret",
            plan.names.artifact_secret,
            artifact_token,
            deadline_at=deadline_at,
        )
    ledger.begin("volume")
    volume = mutation_call(
        lambda: transport.rest(
            "POST",
            "/networkvolumes",
            plan.volume_payload(),
            mutation=True,
            deadline_at=deadline_at,
        ),
        parse_created_volume,
    )
    assert type(volume) is RunPodVolumeObservation
    if not volume_identity_matches(plan, volume):
        raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)
    ledger.confirm("volume", volume.id)

    ledger.begin("template")
    template = mutation_call(
        lambda: transport.rest(
            "POST",
            "/templates",
            plan.template_payload(artifact_token is not None),
            mutation=True,
            deadline_at=deadline_at,
        ),
        parse_created_template,
    )
    assert type(template) is RunPodTemplateObservation
    if not template_identity_matches(plan, template):
        raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)
    ledger.confirm("template", template.id)

    ledger.begin("pod")
    pod = mutation_call(
        lambda: transport.rest(
            "POST",
            "/pods",
            plan.pod_payload(template_id=template.id, volume_id=volume.id),
            mutation=True,
            deadline_at=deadline_at,
        ),
        parse_created_pod,
    )
    assert type(pod) is RunPodPodObservation
    if not pod_identity_matches(plan, pod, template_id=template.id, volume_id=volume.id):
        raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)
    ledger.confirm("pod", pod.id)
    return inference, artifact, template, volume, pod


def _abort_observation_matches(
    plan: RunPodCreatePlan,
    ledger: MutationLedger,
    observation: RunPodObservation,
) -> bool:
    try:
        ensure_unique_resources(observation)
    except RunPodResourceConflict:
        return False
    resources = (
        ("inference_secret", observation.inference_secrets),
        ("artifact_secret", observation.artifact_secrets),
        ("volume", observation.volumes),
        ("template", observation.templates),
        ("pod", observation.pods),
    )
    for kind, values in resources:
        if not values:
            continue
        confirmed_id = ledger.confirmed_id(kind)
        if confirmed_id is None:
            # unconfirmed, so plan identity is the only evidence -- and it proves nothing. two
            # `serve deploy` runs for one deployment generation build byte-identical names and
            # identities, so the loser's plan matches the winner's live resources exactly.
            # deleting on that basis tears down a running deployment the user is paying for.
            #
            # this also refuses to delete our own create if it landed but its id never reached us.
            # automatic cleanup cannot distinguish those cases. explicit identity-authorized
            # undeploy is the recovery path; ambiguous provider errors stay unknown rather than
            # guessing ownership inside the losing create process.
            return False
        # confirmed: ours only if the id matches the one the provider returned to us.
        if values[0].id != confirmed_id:
            return False
    template = observation.templates[0] if observation.templates else None
    volume = observation.volumes[0] if observation.volumes else None
    pod = observation.pods[0] if observation.pods else None
    if template is not None and not template_identity_matches(plan, template):
        return False
    if volume is not None and not volume_identity_matches(plan, volume):
        return False
    if pod is not None:
        template_id = template.id if template is not None else ledger.confirmed_id("template")
        volume_id = volume.id if volume is not None else ledger.confirmed_id("volume")
        if template_id is None or volume_id is None:
            return False
        if not pod_identity_matches(plan, pod, template_id=template_id, volume_id=volume_id):
            return False
    return True


def _wait_for_pod_absence(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> bool:
    while True:
        observation = _observe(plan, transport, deadline_at=deadline_at)
        if not observation.pods:
            return True
        if len(observation.pods) > 1:
            return False
        if not sleep_until_poll(deadline_at, clock, sleep):
            return False


def _abort_created_resources(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    ledger: MutationLedger,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> bool:
    try:
        observation = _observe(plan, transport, deadline_at=deadline_at)
        if not _abort_observation_matches(plan, ledger, observation):
            return False
        pod = observation.pods[0] if observation.pods else None
        template = observation.templates[0] if observation.templates else None
        volume = observation.volumes[0] if observation.volumes else None
        artifact = observation.artifact_secrets[0] if observation.artifact_secrets else None
        inference = observation.inference_secrets[0] if observation.inference_secrets else None
        if pod is not None:
            _delete_tolerating_ambiguity(
                lambda: _delete_rest_once(transport, f"/pods/{pod.id}", deadline_at=deadline_at)
            )
            if not _wait_for_pod_absence(
                plan,
                transport,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            ):
                return False
        if template is not None:
            _delete_tolerating_ambiguity(
                lambda: _delete_rest_once(
                    transport, f"/templates/{template.id}", deadline_at=deadline_at
                )
            )
        if volume is not None:
            _delete_tolerating_ambiguity(
                lambda: _delete_rest_once(
                    transport,
                    f"/networkvolumes/{volume.id}",
                    deadline_at=deadline_at,
                )
            )
        if artifact is not None:
            _delete_tolerating_ambiguity(
                lambda: _delete_secret_once(transport, artifact.id, deadline_at=deadline_at)
            )
        if inference is not None:
            _delete_tolerating_ambiguity(
                lambda: _delete_secret_once(transport, inference.id, deadline_at=deadline_at)
            )
        while True:
            if _observe(plan, transport, deadline_at=deadline_at).resource_count == 0:
                return True
            if not sleep_until_poll(deadline_at, clock, sleep):
                return False
    except (RunPodResourceConflict, RunPodTransportFailure):
        return False


def _failure_after_create_attempt(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    ledger: MutationLedger,
    failure: LifecycleFailure,
    *,
    handle: RunPodProviderHandle | None,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    if not ledger.has_attempted_creations:
        return failure_result(plan, failure, handle=handle)
    absent = _abort_created_resources(
        plan,
        transport,
        ledger,
        deadline_at=deadline_at,
        clock=clock,
        sleep=sleep,
    )
    if absent and not failure.outcome_unknown:
        return failure_result(plan, failure)
    cleanup_reason_by_failure: dict[DeploymentErrorReason, DeploymentErrorReason] = {
        "readiness_deadline_unproven": "readiness_deadline_cleanup_unconfirmed",
        "readiness_observation_failed": "readiness_observation_cleanup_unconfirmed",
        "readiness_resource_conflict": "readiness_resource_conflict_cleanup_unconfirmed",
        "readiness_status_invalid": "readiness_status_invalid_cleanup_unconfirmed",
        "readiness_terminal": "readiness_terminal_cleanup_unconfirmed",
    }
    if failure.reason in cleanup_reason_by_failure:
        reason = cleanup_reason_by_failure[failure.reason]
    elif failure.outcome_unknown:
        reason = "mutation_outcome_unknown"
    else:
        reason = "create_cleanup_unconfirmed"
    return unknown_result(plan, reason=reason, handle=handle)


def provision_runpod_deployment(
    bundle: DeploymentBundle,
    credentials: RunPodCredentials,
    runtime_secrets: ServingRuntimeSecrets,
    *,
    deadline_at: float,
    transport_factory: TransportFactory = StdlibRunPodTransport,
    probe: EndpointProbe = _DEFAULT_ENDPOINT_PROBE,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> DeploymentResult:
    """create one exact persistent pod generation with bounded abort cleanup."""

    validate_runtime_inputs(credentials, runtime_secrets, deadline_at, clock)
    plan = build_runpod_create_plan(bundle)
    inference_token, artifact_token = runtime_secrets._reveal_for_launch()
    ledger = MutationLedger()
    handle: RunPodProviderHandle | None = None
    transport: RunPodTransport | None = None
    reached_ready = False

    def _mark_ready() -> None:
        nonlocal reached_ready
        reached_ready = True

    try:
        transport = open_transport(transport_factory, credentials)
        discovery_deadline_at = min(deadline_at, clock() + _CLEANUP_RESERVE_SECONDS)
        adopted = adopt_existing_generation(
            plan,
            _bind_observe(transport, deadline_at=deadline_at),
            _bind_patch_template(transport, deadline_at=deadline_at),
            _bind_patch_pod(transport, deadline_at=deadline_at),
            _bind_delete_secret(transport, deadline_at=deadline_at),
            inference_token,
            discovery_deadline_at=discovery_deadline_at,
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
        if adopted is not None:
            return adopted
        # from here on this call owns resources, so every phase stops short of the caller's
        # deadline to leave the teardown below a live one.
        work_deadline_at = _work_deadline(deadline_at, clock)
        secret, artifact, template, volume, pod = _create_resources(
            plan,
            transport,
            ledger,
            inference_token,
            artifact_token,
            deadline_at=work_deadline_at,
        )
        handle = build_handle(plan, secret, template, volume, pod)
        ready = await_ready_and_reclaim(
            plan,
            _bind_observe(transport, deadline_at=work_deadline_at),
            _bind_patch_template(transport, deadline_at=work_deadline_at),
            _bind_patch_pod(transport, deadline_at=work_deadline_at),
            _bind_delete_secret(transport, deadline_at=work_deadline_at),
            inference_token,
            artifact,
            unproven_is_failure=True,
            absent_after_deadline=False,
            on_ready=_mark_ready,
            deadline_at=work_deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
        if ready.status == "ready":
            return ready
        if ready.error_code in {"readiness_timeout", "artifact_cleanup_timeout"}:
            return ready
        if ready.status == "failed":
            return _failure_after_create_attempt(
                plan,
                transport,
                ledger,
                LifecycleFailure(
                    ready.error_code or "readiness_failed",
                    reason=ready.error_reason,
                ),
                handle=handle,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            )
        return ready
    except RunPodTransportFailure as exc:
        failure = _from_transport_failure(exc)
        if transport is None:
            return failure_result(plan, failure)
        return _failure_after_create_attempt(
            plan,
            transport,
            ledger,
            failure,
            handle=handle,
            deadline_at=deadline_at,
            clock=clock,
            sleep=sleep,
        )
    except BaseException:
        # Ctrl-C derives from BaseException, so the handler above never sees it: interrupting a
        # slow readiness poll leaves the pod, volume, and secrets live and billing, silently. The
        # ledger already holds what bounded cleanup needs, so tear down, then re-raise.
        # `readiness_failed` rather than a new code: failure codes are a public serialization
        # surface and nothing serializes this one -- only the cleanup it drives, and whether that
        # cleanup could be confirmed, matter here.
        # `not reached_ready` bounds this to the half-built window. After the probe reports healthy
        # the only work left is deleting the hydration secret, and tearing down there would destroy
        # a live pod the user just paid to warm up. A leftover secret is recoverable by re-running;
        # a deleted pod is not.
        if transport is not None and ledger.has_attempted_creations and not reached_ready:
            cleanup = _failure_after_create_attempt(
                plan,
                transport,
                ledger,
                LifecycleFailure("readiness_failed"),
                handle=handle,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            )
            if cleanup.status == "outcome_unknown":
                # cleanup could not prove the pod and volume are gone -- often because the
                # original deadline already expired -- so they may still be live and billing.
                # the generic cli handler prints "aborted", which reads as "nothing was created";
                # this carrier is what lets the cli say the outcome is actually unknown.
                raise InterruptedProvisioning("runpod") from None
        raise


def _validate_handle(plan: RunPodCreatePlan, handle: RunPodProviderHandle) -> None:
    if type(handle) is not RunPodProviderHandle:
        raise ValueError("handle must be an exact RunPodProviderHandle")
    validate_runpod_handle(handle)
    if (
        handle.deployment_id != plan.bundle.spec.deployment_id
        or handle.generation != plan.bundle.spec.generation
        or handle.engine_id != plan.bundle.spec.engine.engine_id
        or handle.account_id != plan.placement.account_id
        or handle.data_center_id != plan.placement.data_center_id
        or handle.image_digest != plan.bundle.image.digest
        or handle.pod_name != plan.names.app_or_pod
        or handle.network_volume_name != plan.names.volume
        or handle.template_name != plan.names.template
        or handle.inference_secret_name != plan.names.inference_secret
    ):
        raise ValueError("runpod handle does not match the exact deployment generation")


def _reclaim_runpod_deployment(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    """reclaim resources after an ambiguous create returned no provider handle."""

    mutation_attempted = False
    known: RunPodObservation | None = None
    attempted_deletes: set[tuple[str, str]] = set()
    discovery_deadline_at = min(deadline_at, clock() + _CLEANUP_RESERVE_SECONDS)
    discovery_settled = False
    try:
        while True:
            observation = _observe(plan, transport, deadline_at=deadline_at)
            if not discovery_settled:
                complete = complete_resource_set(plan, observation, allow_duplicates=True)
                if not complete:
                    if sleep_until_poll(discovery_deadline_at, clock, sleep):
                        continue
                    return unknown_result(plan, reason="teardown_cleanup_unconfirmed")
                known = observation
                discovery_settled = True
            else:
                complete_resource_set(plan, observation, allow_duplicates=True)
                assert known is not None
                known = merge_resource_observations(known, observation)
                complete_resource_set(plan, known, allow_duplicates=True)
            if observation.resource_count == 0 and all(
                (kind, item.id) in attempted_deletes
                for kind, values in (
                    ("pod", known.pods),
                    ("template", known.templates),
                    ("volume", known.volumes),
                    ("artifact_secret", known.artifact_secrets),
                    ("inference_secret", known.inference_secrets),
                )
                for item in values
            ):
                return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
            pending_pods = tuple(
                item for item in known.pods if ("pod", item.id) not in attempted_deletes
            )
            resources = (
                (("pod", item) for item in pending_pods)
                if pending_pods or observation.pods
                else (
                    *(("template", item) for item in known.templates),
                    *(("volume", item) for item in known.volumes),
                    *(("artifact_secret", item) for item in known.artifact_secrets),
                    *(("inference_secret", item) for item in known.inference_secrets),
                )
            )
            issued_delete = False
            for kind, resource in resources:
                key = (kind, resource.id)
                if key in attempted_deletes:
                    continue
                attempted_deletes.add(key)
                issued_delete = True
                mutation_attempted = True
                if kind == "pod":
                    _delete_tolerating_ambiguity(
                        lambda resource_id=resource.id: _delete_rest_once(
                            transport, f"/pods/{resource_id}", deadline_at=deadline_at
                        )
                    )
                elif kind == "template":
                    _delete_tolerating_ambiguity(
                        lambda resource_id=resource.id: _delete_rest_once(
                            transport, f"/templates/{resource_id}", deadline_at=deadline_at
                        )
                    )
                elif kind == "volume":
                    _delete_tolerating_ambiguity(
                        lambda resource_id=resource.id: _delete_rest_once(
                            transport,
                            f"/networkvolumes/{resource_id}",
                            deadline_at=deadline_at,
                        )
                    )
                else:
                    _delete_tolerating_ambiguity(
                        lambda resource_id=resource.id: _delete_secret_once(
                            transport, resource_id, deadline_at=deadline_at
                        )
                    )
            if not issued_delete and not sleep_until_poll(deadline_at, clock, sleep):
                return unknown_result(plan, reason="teardown_cleanup_unconfirmed")
    except RunPodResourceConflict:
        if mutation_attempted:
            return unknown_result(plan, reason="teardown_cleanup_unconfirmed")
        return failure_result(plan, LifecycleFailure("conflict"))
    except RunPodTransportFailure as exc:
        if mutation_attempted:
            return unknown_result(plan, reason="teardown_cleanup_unconfirmed")
        return failure_result(plan, _from_transport_failure(exc))


def teardown_runpod_deployment(
    bundle: DeploymentBundle,
    handle: RunPodProviderHandle | None,
    credentials: RunPodCredentials,
    *,
    deadline_at: float,
    transport_factory: TransportFactory = StdlibRunPodTransport,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> DeploymentResult:
    """delete one exact generation and prove authoritative absence.

    A provider handle keeps the ordinary path bound to provider-returned ids. ``None`` is reserved
    for explicit undeploy after an ambiguous create returned no ids: the immutable deployment bundle
    authorizes reclaim by exact deterministic identity, including duplicate resources from retries.
    """

    validate_control_inputs(credentials, deadline_at, clock)
    plan = build_runpod_create_plan(bundle)
    if handle is None:
        try:
            transport = open_transport(transport_factory, credentials)
        except RunPodTransportFailure as exc:
            return failure_result(plan, _from_transport_failure(exc))
        return _reclaim_runpod_deployment(
            plan,
            transport,
            deadline_at=deadline_at,
            clock=clock,
            sleep=sleep,
        )
    _validate_handle(plan, handle)
    # once a delete has been issued, nothing after it can report a plain `failed`: the resource may
    # already be gone, or may still be live and billing. `failed` reads as "nothing changed" and
    # suppresses the cli's reconcile warning, inviting a retry that double-provisions. this mirrors
    # `mutation_attempted` in the modal teardown, which returns unknown for exactly these cases.
    mutation_attempted = False
    try:
        transport = open_transport(transport_factory, credentials)
        observation = _observe(plan, transport, deadline_at=deadline_at)
        inference, artifact, template, volume, pod = exact_teardown_resources(
            plan, handle, observation
        )
        if pod is not None:
            mutation_attempted = True
            _delete_rest_once(transport, f"/pods/{pod.id}", deadline_at=deadline_at)
            if not _wait_for_pod_absence(
                plan,
                transport,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            ):
                # the delete was accepted but the pod was still listed at the deadline. absence was
                # never proved, so this is ambiguous rather than a confirmed failure.
                return unknown_result(
                    plan,
                    reason="teardown_cleanup_unconfirmed",
                    handle=handle,
                )
        if any(resource is not None for resource in (template, volume, inference, artifact)):
            mutation_attempted = True
        if template is not None:
            _delete_rest_once(transport, f"/templates/{template.id}", deadline_at=deadline_at)
        if volume is not None:
            _delete_rest_once(
                transport,
                f"/networkvolumes/{volume.id}",
                deadline_at=deadline_at,
            )
        if inference is not None:
            _delete_secret_once(transport, inference.id, deadline_at=deadline_at)
        if artifact is not None:
            _delete_secret_once(transport, artifact.id, deadline_at=deadline_at)
        final = _observe(plan, transport, deadline_at=deadline_at)
        if final.resource_count == 0:
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        return unknown_result(
            plan,
            reason="teardown_cleanup_unconfirmed",
            handle=handle,
        )
    except RunPodResourceConflict:
        if mutation_attempted:
            return unknown_result(
                plan,
                reason="teardown_cleanup_unconfirmed",
                handle=handle,
            )
        return failure_result(plan, LifecycleFailure("conflict"), handle=handle)
    except RunPodTransportFailure as exc:
        if mutation_attempted:
            return unknown_result(
                plan,
                reason="teardown_cleanup_unconfirmed",
                handle=handle,
            )
        return failure_result(plan, _from_transport_failure(exc), handle=handle)
