"""pure modal identity matching and operation-specific state policy."""

from __future__ import annotations

from flash.serve.control import ModalProviderHandle
from flash.serve.control.types import validate_modal_provider_id
from flash.serve.provisioning.modal.execution.sdk import (
    ModalAppObservation,
    ModalNamedResource,
    ModalObservation,
)
from flash.serve.provisioning.modal.planning.plan import ModalCreatePlan


class ModalResourceConflict(RuntimeError):
    pass


def ensure_unique_resources(observation: ModalObservation) -> None:
    groups = (
        observation.apps,
        observation.volumes,
        observation.inference_secrets,
        observation.artifact_secrets,
    )
    if any(len(group) > 1 for group in groups):
        raise ModalResourceConflict("deterministic modal resource name is duplicated")


def _one(values: tuple[object, ...], name: str):
    if len(values) != 1:
        raise ModalResourceConflict(f"{name} is not unique")
    return values[0]


def _valid_provider_id(value: object, role: str) -> bool:
    try:
        validate_modal_provider_id(value, role)
    except ValueError:
        return False
    return True


def deployed_app_matches(plan: ModalCreatePlan, app: ModalAppObservation) -> bool:
    return (
        _valid_provider_id(app.app_id, "app")
        and app.app_name == plan.names.app_or_pod
        and app.state == "deployed"
        and type(app.running_containers) is int
        and 0 <= app.running_containers <= plan.max_containers
        and app.tags == plan.tags
        and _valid_provider_id(app.function_id, "function")
        and app.function_name == plan.function_name
        and app.public_url == plan.expected_public_url
    )


def teardown_deployed_app_matches(
    plan: ModalCreatePlan,
    handle: ModalProviderHandle,
    app: ModalAppObservation,
) -> bool:
    return (
        _valid_provider_id(app.app_id, "app")
        and app.app_id == handle.app_id
        and app.app_name == plan.names.app_or_pod
        and app.state == "deployed"
        and type(app.running_containers) is int
        and 0 <= app.running_containers <= plan.max_containers
        and app.tags == plan.tags
        and _valid_provider_id(app.function_id, "function")
        and app.function_name == plan.function_name
    )


def terminal_app_matches(
    plan: ModalCreatePlan,
    handle: ModalProviderHandle,
    app: ModalAppObservation,
) -> bool:
    return (
        _valid_provider_id(app.app_id, "app")
        and app.app_id == handle.app_id
        and app.app_name == plan.names.app_or_pod
        and app.state in {"stopped", "failed"}
        and app.running_containers == 0
        and app.function_id is None
        and app.function_name is None
        and app.public_url is None
    )


def lifecycle_pending_app_matches(
    plan: ModalCreatePlan,
    handle: ModalProviderHandle,
    app: ModalAppObservation,
) -> bool:
    return (
        _valid_provider_id(app.app_id, "app")
        and app.app_id == handle.app_id
        and app.app_name == plan.names.app_or_pod
        and app.state == "lifecycle_pending"
        and app.running_containers is None
        and app.function_id is None
        and app.function_name is None
        and app.public_url is None
    )


def _named_matches(value: ModalNamedResource, name: str, role: str) -> bool:
    return _valid_provider_id(value.id, role) and value.name == name


def exact_core_resources(
    plan: ModalCreatePlan,
    observation: ModalObservation,
) -> tuple[ModalAppObservation, ModalNamedResource, ModalNamedResource]:
    """return exact deployed app, inference secret, and volume identities."""

    ensure_unique_resources(observation)
    app = _one(observation.apps, "modal app")
    volume = _one(observation.volumes, "modal volume")
    inference = _one(observation.inference_secrets, "modal inference secret")
    assert type(app) is ModalAppObservation
    assert type(volume) is ModalNamedResource
    assert type(inference) is ModalNamedResource
    if not deployed_app_matches(plan, app):
        raise ModalResourceConflict("modal app does not match the exact deployment")
    if not _named_matches(volume, plan.names.volume, "volume"):
        raise ModalResourceConflict("modal volume does not match the exact deployment")
    if not _named_matches(inference, plan.names.inference_secret, "secret"):
        raise ModalResourceConflict("modal inference secret does not match the exact deployment")
    if observation.artifact_secrets and not _named_matches(
        observation.artifact_secrets[0],
        plan.names.artifact_secret,
        "secret",
    ):
        raise ModalResourceConflict("modal artifact secret does not match the exact deployment")
    return app, volume, inference


def build_handle(
    plan: ModalCreatePlan,
    app: ModalAppObservation,
    volume: ModalNamedResource,
    inference: ModalNamedResource,
) -> ModalProviderHandle:
    assert app.public_url is not None
    return ModalProviderHandle(
        deployment_id=plan.bundle.spec.deployment_id,
        generation=plan.bundle.spec.generation,
        engine_id=plan.bundle.spec.engine.engine_id,
        workspace_name=plan.placement.workspace_name,
        app_id=app.app_id,
        app_name=app.app_name,
        volume_id=volume.id,
        volume_name=volume.name,
        inference_secret_id=inference.id,
        inference_secret_name=inference.name,
        environment=plan.placement.environment,
        region=plan.placement.region,
        image_digest=plan.bundle.image.digest,
        public_url=app.public_url,
    )


def _reclaim_app_matches(
    plan: ModalCreatePlan,
    app: ModalAppObservation,
    app_id_hint: str | None,
) -> bool:
    if app.state == "deployed":
        return deployed_app_matches(plan, app) and (
            app_id_hint is None or app.app_id == app_id_hint
        )
    if (
        app_id_hint is None
        or not _valid_provider_id(app.app_id, "app")
        or app.app_id != app_id_hint
        or app.app_name != plan.names.app_or_pod
    ):
        return False
    if app.state == "lifecycle_pending":
        return (
            app.running_containers is None
            and app.function_id is None
            and app.function_name is None
            and app.public_url is None
        )
    return (
        app.state in {"stopped", "failed"}
        and app.running_containers == 0
        and app.function_id is None
        and app.function_name is None
        and app.public_url is None
    )


def exact_teardown_resources(
    plan: ModalCreatePlan,
    handle: ModalProviderHandle | None,
    observation: ModalObservation,
    *,
    app_id_hint: str | None = None,
) -> tuple[
    ModalAppObservation | None,
    ModalNamedResource | None,
    ModalNamedResource | None,
    ModalNamedResource | None,
]:
    """validate exact handle ids or deterministic identity for handleless reclaim."""

    ensure_unique_resources(observation)
    app = observation.apps[0] if observation.apps else None
    volume = observation.volumes[0] if observation.volumes else None
    inference = observation.inference_secrets[0] if observation.inference_secrets else None
    artifact = observation.artifact_secrets[0] if observation.artifact_secrets else None
    if handle is None:
        if app is not None and not _reclaim_app_matches(plan, app, app_id_hint):
            raise ModalResourceConflict("modal app does not match the exact deployment")
        resources = (
            (volume, plan.names.volume, "volume"),
            (inference, plan.names.inference_secret, "secret"),
            (artifact, plan.names.artifact_secret, "secret"),
        )
        if any(
            value is not None and not _named_matches(value, expected_name, role)
            for value, expected_name, role in resources
        ):
            raise ModalResourceConflict("modal resource does not match the exact deployment")
        return app, volume, inference, artifact
    if app is not None:
        if app.state == "deployed":
            if not teardown_deployed_app_matches(plan, handle, app):
                raise ModalResourceConflict("modal app does not match the exact handle")
        elif app.state == "lifecycle_pending":
            if not lifecycle_pending_app_matches(plan, handle, app):
                raise ModalResourceConflict("pending modal app does not match the exact handle")
        elif not terminal_app_matches(plan, handle, app):
            raise ModalResourceConflict("terminal modal app does not match the exact handle")
    identities = (
        (volume, handle.volume_id, plan.names.volume),
        (inference, handle.inference_secret_id, plan.names.inference_secret),
    )
    if any(
        value is not None and (value.id != expected_id or value.name != expected_name)
        for value, expected_id, expected_name in identities
    ):
        raise ModalResourceConflict("modal resource does not match the exact handle")
    if artifact is not None and not _named_matches(
        artifact,
        plan.names.artifact_secret,
        "secret",
    ):
        raise ModalResourceConflict("modal artifact secret does not match the exact deployment")
    return app, volume, inference, artifact


def resources_are_absent(observation: ModalObservation, *, allow_terminal_app: bool) -> bool:
    ensure_unique_resources(observation)
    if observation.volumes or observation.inference_secrets or observation.artifact_secrets:
        return False
    if not observation.apps:
        return True
    app = observation.apps[0]
    return allow_terminal_app and app.state in {"stopped", "failed"} and app.running_containers == 0
