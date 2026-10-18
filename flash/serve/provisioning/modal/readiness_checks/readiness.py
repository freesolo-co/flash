"""readiness proofs for a modal deployment: what phase the provider is in, and whether it serves.

Split out of `modal.py` when that file reached the 1000-line limit. Everything here answers "is
this deployment in the exact phase I expect, and does its endpoint answer", and nothing here
mutates provider state. The lifecycle in `modal.py`
owns the mutations and decides what to do with these answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from flash.serve.control import DeploymentErrorReason, DeploymentResult, ModalProviderHandle
from flash.serve.provisioning.common.records import (
    Clock,
    DeploymentBundle,
    LifecycleFailure,
    ServingRuntimeSecrets,
    Sleeper,
    failed_deployment_result,
)
from flash.serve.provisioning.modal.execution.lifecycle import observe
from flash.serve.provisioning.modal.execution.sdk import (
    ModalNamedResource,
    ModalObservation,
    ModalSdk,
    ModalSdkFailure,
)
from flash.serve.provisioning.modal.planning.plan import ModalCreatePlan
from flash.serve.provisioning.modal.planning.resources import (
    ModalResourceConflict,
    build_handle,
    exact_core_resources,
)

__all__ = [
    "MAX_PROBE_TIMEOUT_SECONDS",
    "READINESS_POLL_SECONDS",
    "EndpointProbe",
    "ExpectedResources",
    "PhaseProof",
    "ServingRuntimeSecrets",
    "TransientPhase",
    "failure_result",
    "from_sdk_failure",
    "matches_transient",
    "phase_proof",
    "probe_with_deadline",
    "sleep_until_poll",
    "unknown_result",
    "wait_for_phase",
]

READINESS_POLL_SECONDS = 2.0
MAX_PROBE_TIMEOUT_SECONDS = 30.0


class EndpointProbe(Protocol):
    def __call__(
        self,
        public_url: str,
        inference_token: str,
        bundle: DeploymentBundle,
        timeout_seconds: float,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class ExpectedResources:
    app_id: str | None
    volume_id: str
    inference_secret_id: str
    artifact_secret_id: str | None


@dataclass(frozen=True, slots=True)
class PhaseProof:
    handle: ModalProviderHandle
    artifact: ModalNamedResource | None


@dataclass(frozen=True, slots=True)
class TransientPhase:
    plan: ModalCreatePlan
    artifact_present: bool


def failure_result(
    plan: ModalCreatePlan,
    failure: LifecycleFailure,
    *,
    handle: ModalProviderHandle | None = None,
) -> DeploymentResult:
    return failed_deployment_result(
        plan.bundle.spec,
        failure.code,
        outcome_unknown=failure.outcome_unknown,
        handle=handle,
        error_reason=failure.reason,
    )


def unknown_result(
    plan: ModalCreatePlan,
    *,
    reason: DeploymentErrorReason | None = None,
    handle: ModalProviderHandle | None = None,
) -> DeploymentResult:
    return failure_result(
        plan,
        LifecycleFailure("resource_ambiguous", outcome_unknown=True, reason=reason),
        handle=handle,
    )


def from_sdk_failure(exc: ModalSdkFailure) -> LifecycleFailure:
    return LifecycleFailure(exc.code, exc.outcome_unknown)


def sleep_until_poll(deadline_at: float, clock: Clock, sleep: Sleeper) -> bool:
    remaining = deadline_at - clock()
    if remaining <= 0:
        return False
    sleep(min(READINESS_POLL_SECONDS, remaining))
    return clock() < deadline_at


def probe_with_deadline(
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
            min(MAX_PROBE_TIMEOUT_SECONDS, remaining),
        )
    except Exception:
        return False
    return accepted is True and clock() < deadline_at


def phase_proof(
    plan: ModalCreatePlan,
    observation: ModalObservation,
    *,
    artifact_present: bool,
    expected: ExpectedResources | None,
) -> PhaseProof:
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
    return PhaseProof(handle, artifact)


def matches_transient(
    observation: ModalObservation,
    transient: TransientPhase,
    expected: ExpectedResources | None,
) -> bool:
    try:
        phase_proof(
            transient.plan,
            observation,
            artifact_present=transient.artifact_present,
            expected=expected,
        )
    except ModalResourceConflict:
        return False
    return True


def wait_for_phase(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    inference_token: str,
    *,
    artifact_present: bool,
    expected: ExpectedResources | None,
    transient_phases: tuple[TransientPhase, ...],
    deadline_at: float,
    probe: EndpointProbe,
    clock: Clock,
    sleep: Sleeper,
) -> PhaseProof | None:
    """poll until the deployment is in `artifact_present` phase and its endpoint answers.

    Returns `None` when the deadline runs out. That is a timeout, not a proof of anything: the
    caller must not treat it as licence to mutate what it could not verify.
    """

    app_id_hint = None if expected is None else expected.app_id
    while True:
        observation = observe(
            plan,
            sdk,
            app_id_hint=app_id_hint,
            deadline_at=deadline_at,
        )
        proof: PhaseProof | None = None
        if observation.resource_count:
            try:
                proof = phase_proof(
                    plan,
                    observation,
                    artifact_present=artifact_present,
                    expected=expected,
                )
            except ModalResourceConflict:
                if not any(
                    matches_transient(observation, transient, expected)
                    for transient in transient_phases
                ):
                    raise
        if proof is not None and probe_with_deadline(
            probe,
            proof.handle,
            inference_token,
            plan,
            deadline_at=deadline_at,
            clock=clock,
        ):
            return proof
        if not sleep_until_poll(deadline_at, clock, sleep):
            return None
