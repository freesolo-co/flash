from __future__ import annotations

from dataclasses import replace

import pytest

from flash.serve.app.manifest import build_serving_manifest
from flash.serve.control import ModalPlacement, ModalProviderHandle
from flash.serve.provisioning import DeploymentBundle, ServingImage
from flash.serve.provisioning.modal.execution.sdk import ModalAppObservation, ModalObservation
from flash.serve.provisioning.modal.planning.plan import build_modal_create_plan
from flash.serve.provisioning.modal.planning.resources import (
    ModalResourceConflict,
    deployed_app_matches,
    exact_teardown_resources,
)
from tests.test_serve_app_manifest import _spec_and_inputs

APP_ID = "ap-" + "A" * 22
OTHER_APP_ID = "ap-" + "R" * 22
FUNCTION_ID = "fu-" + "F" * 22
INFERENCE_SECRET_ID = "st-" + "I" * 22
VOLUME_ID = "vo-" + "V" * 22


def _plan_and_handle():
    original, inputs = _spec_and_inputs()
    spec = replace(
        original,
        provider="modal",
        placement=ModalPlacement(
            workspace_name="workspace",
            environment="main",
            gpu="B200",
            region="us-east-1",
            gpu_count=1,
        ),
    )
    bundle = DeploymentBundle(
        spec=spec,
        manifest=build_serving_manifest(spec, inputs),
        image=ServingImage(
            reference=f"registry.example/flash/serve@{spec.engine.image_digest}",
            digest=spec.engine.image_digest,
        ),
    )
    plan = build_modal_create_plan(bundle)
    handle = ModalProviderHandle(
        deployment_id=spec.deployment_id,
        generation=spec.generation,
        engine_id=spec.engine.engine_id,
        workspace_name=plan.placement.workspace_name,
        app_id=APP_ID,
        app_name=plan.names.app_or_pod,
        volume_id=VOLUME_ID,
        volume_name=plan.names.volume,
        inference_secret_id=INFERENCE_SECRET_ID,
        inference_secret_name=plan.names.inference_secret,
        environment=plan.placement.environment,
        region=plan.placement.region,
        image_digest=bundle.image.digest,
        public_url=plan.expected_public_url,
    )
    return plan, handle


def _observation(plan, *, app_id: str = APP_ID) -> ModalObservation:
    app = ModalAppObservation(
        app_id=app_id,
        app_name=plan.names.app_or_pod,
        state="deployed",
        running_containers=1,
        tags=plan.tags,
        function_id=FUNCTION_ID,
        function_name=plan.function_name,
        public_url="https://workspace-changed--replacement.modal.run",
    )
    return ModalObservation(
        workspace_name=plan.placement.workspace_name,
        environment_name=plan.placement.environment,
        apps=(app,),
        volumes=(),
        inference_secrets=(),
        artifact_secrets=(),
    )


def test_teardown_ownership_does_not_depend_on_mutable_public_url() -> None:
    plan, handle = _plan_and_handle()
    observation = _observation(plan)

    assert not deployed_app_matches(plan, observation.apps[0])
    app, volume, inference, artifact = exact_teardown_resources(plan, handle, observation)

    assert app is observation.apps[0]
    assert volume is None
    assert inference is None
    assert artifact is None


def test_teardown_ownership_still_requires_exact_app_id() -> None:
    plan, handle = _plan_and_handle()

    with pytest.raises(ModalResourceConflict, match="exact handle"):
        exact_teardown_resources(plan, handle, _observation(plan, app_id=OTHER_APP_ID))
