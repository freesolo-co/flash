"""request-scoped persistent runpod pod provisioning lifecycle."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from flash.serve.control import (
    DeploymentErrorCode,
    DeploymentResult,
    RunPodCredentials,
    RunPodProviderHandle,
)
from flash.serve.control.types import validate_runpod_handle

from ._common import DeploymentBundle, ServingRuntimeSecrets, failed_deployment_result
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
from ._runpod_resources import (
    RunPodResourceConflict,
    build_handle,
    ensure_unique_resources,
    exact_core_resources,
    exact_teardown_resources,
    pod_identity_matches,
    readiness_state,
    template_identity_matches,
    volume_identity_matches,
)
from ._runpod_transport import (
    RunPodTransport,
    RunPodTransportFailure,
    StdlibRunPodTransport,
)

_READINESS_POLL_SECONDS = 2.0
_MAX_PROBE_TIMEOUT_SECONDS = 30.0

TransportFactory = Callable[[str], RunPodTransport]
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


_DEFAULT_ENDPOINT_PROBE = RunPodEndpointProbe()


@dataclass(frozen=True, slots=True)
class _LifecycleFailure:
    code: DeploymentErrorCode
    outcome_unknown: bool = False


def _validate_deadline(deadline_at: float, clock: Clock) -> None:
    if type(deadline_at) not in {int, float} or not math.isfinite(float(deadline_at)):
        raise ValueError("deadline_at must be finite")
    if float(deadline_at) <= clock():
        raise ValueError("deadline_at must be in the future")


def _validate_runtime_inputs(
    credentials: RunPodCredentials,
    runtime_secrets: ServingRuntimeSecrets,
    deadline_at: float,
    clock: Clock,
) -> None:
    if type(credentials) is not RunPodCredentials:
        raise ValueError("runpod credentials must use the exact credential type")
    if type(runtime_secrets) is not ServingRuntimeSecrets:
        raise ValueError("runtime secrets must use the exact secret boundary")
    _validate_deadline(deadline_at, clock)


def _validate_control_inputs(
    credentials: RunPodCredentials,
    deadline_at: float,
    clock: Clock,
) -> None:
    if type(credentials) is not RunPodCredentials:
        raise ValueError("runpod credentials must use the exact credential type")
    _validate_deadline(deadline_at, clock)


def _transport(factory: TransportFactory, credentials: RunPodCredentials) -> RunPodTransport:
    try:
        return factory(credentials.reveal())
    except Exception:
        raise RunPodTransportFailure("transport_failed") from None


def _read_call(operation: Callable[[], object], parser: Callable[[object], object]):
    try:
        return parser(operation())
    except RunPodTransportFailure:
        raise
    except Exception:
        raise RunPodTransportFailure("transport_failed") from None


def _mutation_call(operation: Callable[[], object], parser: Callable[[object], object]):
    try:
        return parser(operation())
    except RunPodTransportFailure:
        raise
    except Exception:
        raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True) from None


def _identity(value: object) -> object:
    return value


def _observe(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    *,
    deadline_at: float,
) -> RunPodObservation:
    account_id, secrets = _read_call(
        lambda: transport.graphql(
            LIST_ACCOUNT_SECRETS,
            {},
            mutation=False,
            deadline_at=deadline_at,
        ),
        parse_account_secrets,
    )
    templates = _read_call(
        lambda: transport.rest("GET", "/templates", None, mutation=False, deadline_at=deadline_at),
        parse_templates,
    )
    volumes = _read_call(
        lambda: transport.rest(
            "GET", "/networkvolumes", None, mutation=False, deadline_at=deadline_at
        ),
        parse_volumes,
    )
    pods = _read_call(
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
        parse_pods,
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
        templates=tuple(item for item in templates if item.name == plan.names.template),
        volumes=tuple(item for item in volumes if item.name == plan.names.volume),
        pods=tuple(item for item in pods if item.name == plan.names.app_or_pod),
    )


def _failure_result(
    plan: RunPodCreatePlan,
    failure: _LifecycleFailure,
    *,
    handle: RunPodProviderHandle | None = None,
) -> DeploymentResult:
    return failed_deployment_result(
        plan.bundle.spec,
        failure.code,
        outcome_unknown=failure.outcome_unknown,
        handle=handle,
    )


def _unknown_result(
    plan: RunPodCreatePlan,
    *,
    handle: RunPodProviderHandle | None = None,
) -> DeploymentResult:
    return _failure_result(
        plan,
        _LifecycleFailure("resource_ambiguous", outcome_unknown=True),
        handle=handle,
    )


def _from_transport_failure(exc: RunPodTransportFailure) -> _LifecycleFailure:
    return _LifecycleFailure(exc.code, exc.outcome_unknown)


def _sleep_until_poll(deadline_at: float, clock: Clock, sleep: Sleeper) -> bool:
    remaining = deadline_at - clock()
    if remaining <= 0:
        return False
    sleep(min(_READINESS_POLL_SECONDS, remaining))
    return clock() < deadline_at


def _probe_with_deadline(
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
            min(_MAX_PROBE_TIMEOUT_SECONDS, remaining),
        )
    except Exception:
        return False
    return accepted is True and clock() < deadline_at


def _read_only_reconcile(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    inference_token: str,
    *,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
    allow_transient_artifact: bool = False,
) -> DeploymentResult:
    last_handle: RunPodProviderHandle | None = None
    while True:
        observation = _observe(plan, transport, deadline_at=deadline_at)
        if observation.resource_count == 0:
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        try:
            secret, template, volume, pod = exact_core_resources(plan, observation)
            last_handle = build_handle(plan, secret, template, volume, pod)
        except RunPodResourceConflict:
            return _failure_result(plan, _LifecycleFailure("conflict"))
        state = readiness_state(pod.desired_status)
        if state == "invalid":
            return _failure_result(plan, _LifecycleFailure("conflict"), handle=last_handle)
        if state == "failed":
            return _failure_result(plan, _LifecycleFailure("readiness_failed"), handle=last_handle)
        if (
            state == "running"
            and (not observation.artifact_secrets or allow_transient_artifact)
            and _probe_with_deadline(
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
            if not _sleep_until_poll(deadline_at, clock, sleep):
                return _unknown_result(plan, handle=last_handle)
            continue
        if not _sleep_until_poll(deadline_at, clock, sleep):
            if state == "pending":
                return DeploymentResult.from_spec(
                    plan.bundle.spec,
                    status="provisioning",
                    handle=last_handle,
                )
            return _failure_result(
                plan,
                _LifecycleFailure("readiness_failed"),
                handle=last_handle,
            )


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

    _validate_runtime_inputs(credentials, runtime_secrets, deadline_at, clock)
    plan = build_runpod_create_plan(bundle)
    inference_token, _artifact_token = runtime_secrets._reveal_for_launch()
    try:
        transport = _transport(transport_factory, credentials)
        return _read_only_reconcile(
            plan,
            transport,
            inference_token,
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
        )
    except RunPodTransportFailure as exc:
        return _failure_result(plan, _from_transport_failure(exc))


def _delete_secret_once(
    transport: RunPodTransport,
    secret_id: str,
    *,
    deadline_at: float,
) -> None:
    try:
        _mutation_call(
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
        _mutation_call(
            lambda: transport.rest("DELETE", path, None, mutation=True, deadline_at=deadline_at),
            _identity,
        )
    except RunPodTransportFailure as exc:
        if exc.code != "not_found":
            raise


def _confirm_artifact_absence(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> bool:
    while True:
        observation = _observe(plan, transport, deadline_at=deadline_at)
        try:
            ensure_unique_resources(observation)
        except RunPodResourceConflict:
            return False
        if not observation.artifact_secrets:
            return True
        if not _sleep_until_poll(deadline_at, clock, sleep):
            return False


def _delete_artifact_and_confirm(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    artifact: RunPodSecretObservation,
    handle: RunPodProviderHandle,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    try:
        _delete_secret_once(transport, artifact.id, deadline_at=deadline_at)
    except RunPodTransportFailure:
        return _unknown_result(plan, handle=handle)
    try:
        absent = _confirm_artifact_absence(
            plan,
            transport,
            deadline_at=deadline_at,
            clock=clock,
            sleep=sleep,
        )
    except RunPodTransportFailure:
        return _unknown_result(plan, handle=handle)
    if not absent:
        return _unknown_result(plan, handle=handle)
    return DeploymentResult.from_spec(plan.bundle.spec, status="ready", handle=handle)


def _adopt_existing(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    observation: RunPodObservation,
    inference_token: str,
    *,
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    try:
        secret, template, volume, pod = exact_core_resources(plan, observation)
        handle = build_handle(plan, secret, template, volume, pod)
    except RunPodResourceConflict:
        return _failure_result(plan, _LifecycleFailure("conflict"))
    state = readiness_state(pod.desired_status)
    if state == "invalid":
        return _failure_result(plan, _LifecycleFailure("conflict"), handle=handle)
    if state == "failed":
        return _failure_result(plan, _LifecycleFailure("readiness_failed"), handle=handle)
    if state != "running" or not _probe_with_deadline(
        probe,
        handle,
        inference_token,
        plan,
        deadline_at=deadline_at,
        clock=clock,
    ):
        return _unknown_result(plan, handle=handle)
    if not observation.artifact_secrets:
        return DeploymentResult.from_spec(plan.bundle.spec, status="ready", handle=handle)
    artifact = observation.artifact_secrets[0]
    return _delete_artifact_and_confirm(
        plan,
        transport,
        artifact,
        handle,
        deadline_at=deadline_at,
        clock=clock,
        sleep=sleep,
    )


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
    created = _mutation_call(
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
    volume = _mutation_call(
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
    template = _mutation_call(
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
    pod = _mutation_call(
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
        confirmed_id = ledger.confirmed_id(kind)
        if values and confirmed_id is not None and values[0].id != confirmed_id:
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
        if not _sleep_until_poll(deadline_at, clock, sleep):
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
            _delete_rest_once(transport, f"/pods/{pod.id}", deadline_at=deadline_at)
            if not _wait_for_pod_absence(
                plan,
                transport,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            ):
                return False
        if template is not None:
            _delete_rest_once(transport, f"/templates/{template.id}", deadline_at=deadline_at)
        if volume is not None:
            _delete_rest_once(
                transport,
                f"/networkvolumes/{volume.id}",
                deadline_at=deadline_at,
            )
        if artifact is not None:
            _delete_secret_once(transport, artifact.id, deadline_at=deadline_at)
        if inference is not None:
            _delete_secret_once(transport, inference.id, deadline_at=deadline_at)
        return _observe(plan, transport, deadline_at=deadline_at).resource_count == 0
    except (RunPodResourceConflict, RunPodTransportFailure):
        return False


def _failure_after_create_attempt(
    plan: RunPodCreatePlan,
    transport: RunPodTransport,
    ledger: MutationLedger,
    failure: _LifecycleFailure,
    *,
    handle: RunPodProviderHandle | None,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> DeploymentResult:
    if not ledger.has_attempted_creations:
        return _failure_result(plan, failure, handle=handle)
    absent = _abort_created_resources(
        plan,
        transport,
        ledger,
        deadline_at=deadline_at,
        clock=clock,
        sleep=sleep,
    )
    if absent and not failure.outcome_unknown:
        return _failure_result(plan, failure)
    return _unknown_result(plan, handle=handle)


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

    _validate_runtime_inputs(credentials, runtime_secrets, deadline_at, clock)
    plan = build_runpod_create_plan(bundle)
    inference_token, artifact_token = runtime_secrets._reveal_for_launch()
    ledger = MutationLedger()
    handle: RunPodProviderHandle | None = None
    transport: RunPodTransport | None = None
    reached_ready = False
    try:
        transport = _transport(transport_factory, credentials)
        observation = _observe(plan, transport, deadline_at=deadline_at)
        try:
            ensure_unique_resources(observation)
        except RunPodResourceConflict:
            return _failure_result(plan, _LifecycleFailure("conflict"))
        if observation.resource_count:
            return _adopt_existing(
                plan,
                transport,
                observation,
                inference_token,
                deadline_at=deadline_at,
                probe=probe,
                clock=clock,
                sleep=sleep,
            )
        secret, artifact, template, volume, pod = _create_resources(
            plan,
            transport,
            ledger,
            inference_token,
            artifact_token,
            deadline_at=deadline_at,
        )
        handle = build_handle(plan, secret, template, volume, pod)
        ready = _read_only_reconcile(
            plan,
            transport,
            inference_token,
            deadline_at=deadline_at,
            probe=probe,
            clock=clock,
            sleep=sleep,
            allow_transient_artifact=True,
        )
        reached_ready = ready.status == "ready"
        if reached_ready and artifact is not None:
            return _delete_artifact_and_confirm(
                plan,
                transport,
                artifact,
                ready.handle,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            )
        if ready.status == "failed":
            return _failure_after_create_attempt(
                plan,
                transport,
                ledger,
                _LifecycleFailure(ready.error_code or "readiness_failed"),
                handle=handle,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            )
        return ready
    except RunPodTransportFailure as exc:
        failure = _from_transport_failure(exc)
        if transport is None:
            return _failure_result(plan, failure)
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
        # surface and this result is discarded anyway -- only the cleanup it drives matters.
        # `not reached_ready` bounds this to the half-built window. After the probe reports healthy
        # the only work left is deleting the hydration secret, and tearing down there would destroy
        # a live pod the user just paid to warm up. A leftover secret is recoverable by re-running;
        # a deleted pod is not.
        if transport is not None and ledger.has_attempted_creations and not reached_ready:
            _failure_after_create_attempt(
                plan,
                transport,
                ledger,
                _LifecycleFailure("readiness_failed"),
                handle=handle,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            )
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


def grow_runpod_volume(
    bundle: DeploymentBundle,
    handle: RunPodProviderHandle,
    credentials: RunPodCredentials,
    target_size_gb: int,
    *,
    deadline_at: float,
    transport_factory: TransportFactory = StdlibRunPodTransport,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> DeploymentResult:
    """grow the exact generation volume once and confirm its authoritative size."""

    _validate_control_inputs(credentials, deadline_at, clock)
    plan = build_runpod_create_plan(bundle)
    _validate_handle(plan, handle)
    if type(target_size_gb) is not int or target_size_gb < plan.placement.volume_size_gb:
        raise ValueError("target volume size cannot shrink the planned volume")
    try:
        transport = _transport(transport_factory, credentials)
        observation = _observe(plan, transport, deadline_at=deadline_at)
        _secret, _template, volume, _pod = exact_core_resources(plan, observation)
        if volume.id != handle.network_volume_id:
            return _failure_result(plan, _LifecycleFailure("conflict"), handle=handle)
        if target_size_gb < volume.size_gb:
            return _failure_result(plan, _LifecycleFailure("invalid_request"), handle=handle)
        if target_size_gb == volume.size_gb:
            return DeploymentResult.from_spec(plan.bundle.spec, status="ready", handle=handle)
        resized = _mutation_call(
            lambda: transport.rest(
                "PATCH",
                f"/networkvolumes/{volume.id}",
                {"size": target_size_gb},
                mutation=True,
                deadline_at=deadline_at,
            ),
            parse_created_volume,
        )
        assert type(resized) is RunPodVolumeObservation
        if (
            resized.id != volume.id
            or resized.name != volume.name
            or resized.data_center_id != volume.data_center_id
            or resized.size_gb != target_size_gb
        ):
            raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)
        while True:
            current = _observe(plan, transport, deadline_at=deadline_at)
            _current_secret, _current_template, current_volume, _current_pod = exact_core_resources(
                plan, current
            )
            if current_volume.size_gb == target_size_gb:
                return DeploymentResult.from_spec(plan.bundle.spec, status="ready", handle=handle)
            if current_volume.size_gb > target_size_gb:
                return _failure_result(plan, _LifecycleFailure("conflict"), handle=handle)
            if not _sleep_until_poll(deadline_at, clock, sleep):
                return _unknown_result(plan, handle=handle)
    except RunPodResourceConflict:
        return _failure_result(plan, _LifecycleFailure("conflict"), handle=handle)
    except RunPodTransportFailure as exc:
        return _failure_result(plan, _from_transport_failure(exc), handle=handle)


def teardown_runpod_deployment(
    bundle: DeploymentBundle,
    handle: RunPodProviderHandle,
    credentials: RunPodCredentials,
    *,
    deadline_at: float,
    transport_factory: TransportFactory = StdlibRunPodTransport,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> DeploymentResult:
    """delete one exact generation once per resource and prove authoritative absence."""

    _validate_control_inputs(credentials, deadline_at, clock)
    plan = build_runpod_create_plan(bundle)
    _validate_handle(plan, handle)
    try:
        transport = _transport(transport_factory, credentials)
        observation = _observe(plan, transport, deadline_at=deadline_at)
        inference, artifact, template, volume, pod = exact_teardown_resources(
            plan, handle, observation
        )
        if pod is not None:
            _delete_rest_once(transport, f"/pods/{pod.id}", deadline_at=deadline_at)
            if not _wait_for_pod_absence(
                plan,
                transport,
                deadline_at=deadline_at,
                clock=clock,
                sleep=sleep,
            ):
                return _failure_result(
                    plan,
                    _LifecycleFailure("readiness_failed"),
                    handle=handle,
                )
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
        return _failure_result(plan, _LifecycleFailure("readiness_failed"), handle=handle)
    except RunPodResourceConflict:
        return _failure_result(plan, _LifecycleFailure("conflict"), handle=handle)
    except RunPodTransportFailure as exc:
        return _failure_result(plan, _from_transport_failure(exc), handle=handle)


def confirm_runpod_absence(
    bundle: DeploymentBundle,
    credentials: RunPodCredentials,
    *,
    deadline_at: float,
    transport_factory: TransportFactory = StdlibRunPodTransport,
    clock: Clock = time.monotonic,
) -> DeploymentResult:
    """report absent only after authoritative account-scoped list confirmation."""

    _validate_control_inputs(credentials, deadline_at, clock)
    plan = build_runpod_create_plan(bundle)
    try:
        transport = _transport(transport_factory, credentials)
        observation = _observe(plan, transport, deadline_at=deadline_at)
        if observation.resource_count == 0:
            return DeploymentResult.from_spec(plan.bundle.spec, status="absent")
        return _failure_result(plan, _LifecycleFailure("conflict"))
    except RunPodTransportFailure as exc:
        return _failure_result(plan, _from_transport_failure(exc))
