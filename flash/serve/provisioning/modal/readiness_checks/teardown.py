"""shared terminal-state and absence proofs for modal resource teardown."""

from __future__ import annotations

from collections.abc import Callable

from flash.serve.control import ModalProviderHandle
from flash.serve.provisioning.common.records import Clock, Sleeper
from flash.serve.provisioning.modal.execution.lifecycle import mutation, observe
from flash.serve.provisioning.modal.execution.sdk import (
    ModalNamedResource,
    ModalObservation,
    ModalSdk,
    ModalSdkFailure,
)
from flash.serve.provisioning.modal.planning.plan import ModalCreatePlan
from flash.serve.provisioning.modal.planning.resources import (
    ModalResourceConflict,
    exact_teardown_resources,
    resources_are_absent,
)
from flash.serve.provisioning.modal.readiness_checks.readiness import (
    ExpectedResources,
    sleep_until_poll,
)


def wait_for_terminal_app(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    handle: ModalProviderHandle | None,
    *,
    app_id: str | None = None,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> ModalObservation | None:
    target_app_id = handle.app_id if handle is not None else app_id
    if target_app_id is None:
        raise ValueError("modal teardown requires an exact app id")
    while True:
        observation = observe(plan, sdk, app_id_hint=target_app_id)
        app, _volume, _inference, _artifact = exact_teardown_resources(
            plan,
            handle,
            observation,
            app_id_hint=target_app_id if handle is None else None,
        )
        if app is not None and app.state in {"stopped", "failed"}:
            return observation if app.running_containers == 0 else None
        if not sleep_until_poll(deadline_at, clock, sleep):
            return None


def suppressed(step: Callable[[], object]) -> bool:
    """run one teardown step without replacing the exception that initiated abort cleanup."""

    try:
        step()
    except Exception:
        return False
    return True


def delete_teardown_resources(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    volume: ModalNamedResource | None,
    inference: ModalNamedResource | None,
    artifact: ModalNamedResource | None,
    *,
    suppress_failures: bool = False,
) -> None:
    def run(step: Callable[[], object]) -> None:
        if suppress_failures:
            suppressed(step)
        else:
            step()

    if artifact is not None:
        run(lambda: mutation(lambda: sdk.delete_secret(plan, artifact.id)))
    if inference is not None:
        run(lambda: mutation(lambda: sdk.delete_secret(plan, inference.id)))
    if volume is not None:
        run(lambda: mutation(lambda: sdk.delete_volume(plan, volume.id)))


def delete_confirmed_abort_resources(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    initial: ModalObservation,
    expected: ExpectedResources | None,
    *,
    artifact_attempted: bool,
    inference_attempted: bool,
    volume_attempted: bool,
    app_deployed: bool,
) -> bool:
    """delete no-app abort resources only after proving this invocation owns them."""

    if app_deployed:
        # no observed app means its name cannot be bound to the id this invocation received.
        return False
    resources = (
        (
            artifact_attempted,
            initial.artifact_secrets[0] if initial.artifact_secrets else None,
            None if expected is None else expected.artifact_secret_id,
            plan.names.artifact_secret,
        ),
        (
            inference_attempted,
            initial.inference_secrets[0] if initial.inference_secrets else None,
            None if expected is None else expected.inference_secret_id,
            plan.names.inference_secret,
        ),
        (
            volume_attempted,
            initial.volumes[0] if initial.volumes else None,
            None if expected is None else expected.volume_id,
            plan.names.volume,
        ),
    )
    if any(
        resource is not None
        and (not attempted or resource.id != confirmed_id or resource.name != expected_name)
        for attempted, resource, confirmed_id, expected_name in resources
    ):
        return False
    delete_teardown_resources(
        plan,
        sdk,
        initial.volumes[0] if initial.volumes and volume_attempted else None,
        initial.inference_secrets[0] if initial.inference_secrets and inference_attempted else None,
        initial.artifact_secrets[0] if initial.artifact_secrets and artifact_attempted else None,
        suppress_failures=True,
    )
    try:
        final = observe(plan, sdk)
        return resources_are_absent(final, allow_terminal_app=False)
    except (ModalResourceConflict, ModalSdkFailure):
        return False


def confirmed_abort_handle(
    plan: ModalCreatePlan,
    expected: ExpectedResources | None,
) -> ModalProviderHandle | None:
    """build the recovery handle only from provider ids confirmed by this invocation."""

    if expected is None or expected.app_id is None:
        return None
    return ModalProviderHandle(
        deployment_id=plan.bundle.spec.deployment_id,
        generation=plan.bundle.spec.generation,
        engine_id=plan.bundle.spec.engine.engine_id,
        workspace_name=plan.placement.workspace_name,
        app_id=expected.app_id,
        app_name=plan.names.app_or_pod,
        volume_id=expected.volume_id,
        volume_name=plan.names.volume,
        inference_secret_id=expected.inference_secret_id,
        inference_secret_name=plan.names.inference_secret,
        environment=plan.placement.environment,
        region=plan.placement.region,
        image_digest=plan.bundle.image.digest,
        public_url=plan.expected_public_url,
    )


def confirm_teardown_absence(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    handle: ModalProviderHandle | None,
    *,
    app_id: str | None = None,
) -> bool:
    target_app_id = handle.app_id if handle is not None else app_id
    final = observe(plan, sdk, app_id_hint=target_app_id)
    exact_teardown_resources(
        plan,
        handle,
        final,
        app_id_hint=target_app_id if handle is None else None,
    )
    return resources_are_absent(final, allow_terminal_app=True)
