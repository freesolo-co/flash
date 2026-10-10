"""offline modal serving lifecycle, identity, and secret-boundary coverage."""

from __future__ import annotations

import asyncio
import base64
import email.message
import io
import json
import re
import subprocess
import sys
import urllib.response
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.serve.app.manifest import build_serving_manifest
from flash.serve.control import (
    DeploymentSpec,
    ModalCredentials,
    ModalPlacement,
    ModalProviderHandle,
)
from flash.serve.deployment.profiles import get_profile, placement_for
from flash.serve.provisioning import (
    DeploymentBundle,
    FreshDeploymentArtifactTokenRequired,
    InterruptedProvisioning,
    ServingImage,
    ServingRuntimeSecrets,
    serving_resource_names,
)
from flash.serve.provisioning.modal.execution.deployment import _work_deadline
from flash.serve.provisioning.modal.execution.operations import (
    _abort_created_resources,
    _CreatedResources,
    provision_modal_deployment,
    reconcile_modal_deployment,
    teardown_modal_deployment,
)
from flash.serve.provisioning.modal.execution.sdk import (
    ModalAppObservation,
    ModalNamedResource,
    ModalObservation,
    ModalSdkFailure,
    PinnedModalSdk,
)
from flash.serve.provisioning.modal.planning.plan import (
    MODAL_APP_TAG_LIMIT,
    MODAL_DEPLOYMENT_TAG_LIMIT,
    MODAL_STARTUP_TIMEOUT_SECONDS,
    build_modal_create_plan,
)
from flash.serve.provisioning.modal.readiness_checks.probe import (
    ModalEndpointProbe,
    _provenance_matches,
)
from flash.serve.provisioning.modal.readiness_checks.readiness import ExpectedResources
from tests.test_serve_app_manifest import _profile_spec_and_inputs, _spec_and_inputs

PROVIDER_ID = "provider-id-sentinel"
PROVIDER_SECRET = "provider-secret-sentinel"
INFERENCE_SECRET = "inference-secret-sentinel"
ARTIFACT_SECRET = "artifact-secret-sentinel"
APP_ID = "ap-" + "A" * 22
OTHER_APP_ID = "ap-" + "R" * 22
FUNCTION_ID = "fu-" + "F" * 22
INFERENCE_SECRET_ID = "st-" + "I" * 22
ARTIFACT_SECRET_ID = "st-" + "A" * 22
OPAQUE_SECRET_ID = "st-" + "O" * 22
VOLUME_ID = "vo-" + "V" * 22
ROOT = Path(__file__).resolve().parents[1]


def _bundle(
    *,
    region: str | None = "us-east-1",
    workspace_name: str = "workspace",
    web_suffix: str | None = None,
    generation: int | None = None,
) -> DeploymentBundle:
    original, inputs = _spec_and_inputs()
    spec = DeploymentSpec(
        deployment_id=original.deployment_id,
        generation=original.generation if generation is None else generation,
        provider="modal",
        placement=ModalPlacement(
            workspace_name=workspace_name,
            environment="main",
            gpu="B200",
            region=region,
            gpu_count=1,
            web_suffix=web_suffix,
        ),
        engine=original.engine,
        adapters=original.adapters,
    )
    manifest = build_serving_manifest(spec, inputs)
    return DeploymentBundle(
        spec=spec,
        manifest=manifest,
        image=ServingImage(
            reference=f"registry.example/flash/serve@{spec.engine.image_digest}",
            digest=spec.engine.image_digest,
        ),
    )


@pytest.mark.parametrize(
    ("model_id", "expected_gpu_request"),
    [
        ("Qwen/Qwen3.5-9B", "L40S:1"),
        ("Qwen/Qwen3.8-27B", "H100!:1"),
        ("Qwen/Qwen3.6-35B-A3B", "H200:1"),
    ],
)
def test_profile_modal_plan_preserves_exact_engine_and_placement(
    model_id: str, expected_gpu_request: str
) -> None:
    spec, inputs = _profile_spec_and_inputs(model_id)
    profile = get_profile(model_id)
    placement = placement_for(
        profile,
        "modal",
        workspace_name="workspace",
        environment="main",
        region="us-east",
    )
    modal_spec = replace(spec, provider="modal", placement=placement)
    manifest = build_serving_manifest(modal_spec, inputs)
    bundle = DeploymentBundle(
        spec=modal_spec,
        manifest=manifest,
        image=ServingImage(
            reference=f"registry.example/flash/serve@{modal_spec.engine.image_digest}",
            digest=modal_spec.engine.image_digest,
        ),
    )

    plan = build_modal_create_plan(bundle, phase="finalized")

    assert plan.placement.gpu == profile.modal_gpu_request
    assert plan.placement.gpu_count == profile.tensor_parallel_size
    assert plan.gpu_request == expected_gpu_request
    assert plan.bundle.manifest == manifest
    assert plan.bundle.image.reference.endswith(f"@{bundle.image.digest}")
    assert plan.encoded_manifest


def _models_payload(bundle: DeploymentBundle) -> dict[str, object]:
    manifest = bundle.manifest
    data = []
    for adapter in manifest.adapters:
        checkpoint_id = adapter.checkpoint_id
        data.append(
            {
                "id": checkpoint_id,
                "flash_provenance": {
                    "deployment_id": bundle.spec.deployment_id,
                    "spec_id": bundle.spec.spec_id,
                    "manifest_id": manifest.manifest_id,
                    "engine_id": bundle.spec.engine.engine_id,
                    "image_digest": bundle.image.digest,
                    "logical_base_model": manifest.logical_base_model,
                    "logical_base_revision": manifest.logical_base_revision,
                    "served_checkpoint": manifest.engine.served_model,
                    "served_checkpoint_revision": manifest.engine.model_revision,
                    "tokenizer_model": manifest.engine.tokenizer_model,
                    "tokenizer_revision": manifest.engine.tokenizer_revision,
                    "requested_model": checkpoint_id,
                    "checkpoint_id": checkpoint_id,
                },
            }
        )
    return {"data": data}


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.parametrize(
    ("deadline_at", "now", "expected"),
    [
        (160.0, 100.0, 130.0),
        (110.0, 100.0, 105.0),
        (100.0, 100.0, 100.0),
        (100.0, 160.0, 100.0),
        (100.0, 300.0, 100.0),
    ],
)
def test_work_deadline_never_exceeds_the_caller_deadline(
    deadline_at: float,
    now: float,
    expected: float,
) -> None:
    work_deadline = _work_deadline(deadline_at, lambda: now)

    assert work_deadline <= deadline_at
    assert work_deadline == expected


class _Probe:
    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.calls: list[tuple[str, bool, DeploymentBundle, float]] = []

    def __call__(
        self,
        url: str,
        token: str,
        bundle: DeploymentBundle,
        timeout_seconds: float,
    ) -> bool:
        self.calls.append((url, token == INFERENCE_SECRET, bundle, timeout_seconds))
        return self.accepted


class _FakeSdk:
    def __init__(self, plan) -> None:
        self.plan = plan
        self.workspace_name = plan.placement.workspace_name
        self.environment_name = plan.placement.environment
        self.apps: list[ModalAppObservation] = []
        self.volumes: list[ModalNamedResource] = []
        self.inference: list[ModalNamedResource] = []
        self.artifact: list[ModalNamedResource] = []
        self.calls: list[tuple[str, object]] = []
        self.fail_operation: str | None = None
        self.fail_with_sdk = False
        self.closed = False

    def _fail(self, operation: str) -> None:
        if self.fail_operation != operation:
            return
        if self.fail_with_sdk:
            raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True)
        raise RuntimeError(PROVIDER_SECRET)

    def observe(self, plan, *, app_id_hint=None, deadline_at=None) -> ModalObservation:
        self.calls.append(("observe", app_id_hint))
        return ModalObservation(
            workspace_name=self.workspace_name,
            environment_name=self.environment_name,
            apps=tuple(self.apps),
            volumes=tuple(self.volumes),
            inference_secrets=tuple(self.inference),
            artifact_secrets=tuple(self.artifact),
        )

    def create_inference_secret(self, plan, value: str, *, deadline_at=None) -> ModalNamedResource:
        self.calls.append(("create_inference", value == INFERENCE_SECRET))
        self._fail("create_inference")
        resource = ModalNamedResource(INFERENCE_SECRET_ID, plan.names.inference_secret)
        self.inference.append(resource)
        return resource

    def create_artifact_secret(self, plan, value: str, *, deadline_at=None) -> ModalNamedResource:
        self.calls.append(("create_artifact", value == ARTIFACT_SECRET))
        self._fail("create_artifact")
        resource = ModalNamedResource(ARTIFACT_SECRET_ID, plan.names.artifact_secret)
        self.artifact.append(resource)
        return resource

    def create_volume(self, plan, *, deadline_at=None) -> ModalNamedResource:
        self.calls.append(("create_volume", None))
        self._fail("create_volume")
        resource = ModalNamedResource(VOLUME_ID, plan.names.volume)
        self.volumes.append(resource)
        return resource

    def deploy_app(self, plan, *, deadline_at=None) -> str:
        self.calls.append(("deploy_app", plan.phase))
        self._fail(f"deploy_{plan.phase}")
        self.apps = [
            ModalAppObservation(
                app_id=APP_ID,
                app_name=plan.names.app_or_pod,
                state="deployed",
                running_containers=0,
                tags=plan.tags,
                function_id=FUNCTION_ID,
                function_name=plan.function_name,
                public_url=plan.expected_public_url,
            )
        ]
        return APP_ID

    def stop_app(self, plan, app_id: str, *, deadline_at=None) -> None:
        self.calls.append(("stop_app", None))
        self._fail("stop_app")
        app = self.apps[0]
        assert app.app_id == app_id
        self.apps = [
            replace(
                app,
                state="stopped",
                running_containers=0,
                tags=(),
                function_id=None,
                function_name=None,
                public_url=None,
            )
        ]

    def delete_secret(self, plan, secret_id: str, *, deadline_at=None) -> None:
        role = "artifact" if secret_id == ARTIFACT_SECRET_ID else "inference"
        assert secret_id in {ARTIFACT_SECRET_ID, INFERENCE_SECRET_ID}
        self.calls.append((f"delete_{role}", None))
        self._fail(f"delete_{role}")
        if role == "artifact":
            self.artifact.clear()
        else:
            self.inference.clear()

    def delete_volume(self, plan, volume_id: str, *, deadline_at=None) -> None:
        self.calls.append(("delete_volume", None))
        assert volume_id == VOLUME_ID
        self._fail("delete_volume")
        self.volumes.clear()

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []
        self.sdk: _FakeSdk | None = None
        self.workspace_override: str | None = None
        self.environment_override: str | None = None
        # lets a test swap in a _FakeSdk subclass that fails one exact operation, without
        # reimplementing the factory's credential assertions.
        self.sdk_class: type[_FakeSdk] = _FakeSdk

    def __call__(
        self,
        credentials: ModalCredentials,
        plan,
        _deadline_at: float,
        _clock,
    ) -> _FakeSdk:
        token_id, token_secret = credentials.reveal()
        self.calls.append((token_id == PROVIDER_ID, token_secret == PROVIDER_SECRET))
        sdk = self.sdk_class(plan)
        if self.workspace_override is not None:
            sdk.workspace_name = self.workspace_override
        if self.environment_override is not None:
            sdk.environment_name = self.environment_override
        self.sdk = sdk
        return sdk


def _seed_exact(sdk: _FakeSdk, *, artifact: bool = False) -> ModalProviderHandle:
    plan = sdk.plan
    sdk.inference = [ModalNamedResource(INFERENCE_SECRET_ID, plan.names.inference_secret)]
    if artifact:
        sdk.artifact = [ModalNamedResource(ARTIFACT_SECRET_ID, plan.names.artifact_secret)]
    sdk.volumes = [ModalNamedResource(VOLUME_ID, plan.names.volume)]
    sdk.deploy_app(plan)
    sdk.calls.clear()
    app = sdk.apps[0]
    return ModalProviderHandle(
        deployment_id=plan.bundle.spec.deployment_id,
        generation=plan.bundle.spec.generation,
        engine_id=plan.bundle.spec.engine.engine_id,
        workspace_name=plan.placement.workspace_name,
        app_id=app.app_id,
        app_name=app.app_name,
        volume_id=sdk.volumes[0].id,
        volume_name=sdk.volumes[0].name,
        inference_secret_id=sdk.inference[0].id,
        inference_secret_name=sdk.inference[0].name,
        environment=plan.placement.environment,
        region=plan.placement.region,
        image_digest=plan.bundle.image.digest,
        public_url=plan.expected_public_url,
    )


def _provision(
    bundle: DeploymentBundle,
    factory: _Factory,
    *,
    artifact_token: str | None = ARTIFACT_SECRET,
    probe: _Probe | None = None,
):
    clock = _Clock()
    selected_probe = probe or _Probe()
    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, artifact_token),
        deadline_at=100.0,
        sdk_factory=factory,
        probe=selected_probe,
        clock=clock,
        sleep=clock.sleep,
    )
    return result, selected_probe


def test_invalid_spec_credentials_and_deadline_fail_before_client_construction() -> None:
    factory = _Factory()
    clock = _Clock()
    with pytest.raises(ValueError, match="exact lowercase region"):
        provision_modal_deployment(
            _bundle(region=None),
            ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
            ServingRuntimeSecrets(INFERENCE_SECRET),
            deadline_at=100.0,
            sdk_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        )
    with pytest.raises(ValueError, match="too long"):
        provision_modal_deployment(
            _bundle(workspace_name="w" * 42),
            ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
            ServingRuntimeSecrets(INFERENCE_SECRET),
            deadline_at=100.0,
            sdk_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        )
    with pytest.raises(ValueError, match="credential type"):
        provision_modal_deployment(
            _bundle(),
            object(),
            ServingRuntimeSecrets(INFERENCE_SECRET),
            deadline_at=100.0,
            sdk_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        )
    with pytest.raises(ValueError, match="future"):
        provision_modal_deployment(
            _bundle(),
            ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
            ServingRuntimeSecrets(INFERENCE_SECRET),
            deadline_at=0.0,
            sdk_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        )
    assert factory.calls == []


def test_deployment_tag_fits_modals_limit_in_every_phase() -> None:
    # modal rejects a tag over 50 characters at app.deploy(), which happens AFTER the inference
    # secret and volume are created -- so an over-long tag does not fail cleanly, it fails as
    # outcome_unknown with orphaned billable resources. this asserts the shape modal accepts:
    # within the length limit, in modal's charset, and still distinct per phase.
    bundle = _bundle()
    tags = {
        phase: build_modal_create_plan(bundle, phase=phase).deployment_tag
        for phase in ("bootstrap", "finalized")
    }

    for phase, tag in tags.items():
        assert len(tag) <= MODAL_DEPLOYMENT_TAG_LIMIT, f"{phase} tag is {len(tag)} chars"
        assert re.fullmatch(r"[A-Za-z0-9._-]+", tag), f"{phase} tag has invalid characters"
        assert tag.startswith(f"fsm1-{phase}-")
    assert tags["bootstrap"] != tags["finalized"]


def test_app_tag_set_fits_modals_tag_budget_in_every_phase() -> None:
    # modal caps an app at 8 tags server-side and its client does not check, so a ninth key fails
    # inside app.deploy() -- after the inference secret and volume exist -- and surfaces as
    # outcome_unknown with orphaned billable resources. the plan carried 9 and every deploy died
    # here. asserting the budget rather than the exact key list keeps this from turning red for a
    # deliberate rename, while still failing the moment a tag is added without removing one.
    for phase in ("bootstrap", "finalized"):
        tags = build_modal_create_plan(_bundle(), phase=phase).tags
        assert len(tags) <= MODAL_APP_TAG_LIMIT, f"{phase} carries {len(tags)} tags"
        assert len({key for key, _ in tags}) == len(tags), "tag keys must be unique"


def _default_serve_timeout_seconds() -> float:
    """The real `--timeout` default the deploy command ships with.

    Read from the parser rather than restated, so the budget assertion below cannot drift away
    from the value users actually get.
    """
    import argparse

    from flash.cli.parsing.serve_parser import _add_deployment_arguments

    command = argparse.ArgumentParser()
    _add_deployment_arguments(command)
    timeout = next(a for a in command._actions if a.dest == "timeout")
    return float(timeout.default)


def test_plan_is_complete_secret_free_and_binds_pinned_image() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    bootstrap = build_modal_create_plan(bundle, phase="bootstrap")
    rendered = repr(plan)
    assert plan.phase == "finalized"
    assert bootstrap.phase == "bootstrap"
    assert dict(plan.tags)["flash-phase"] == "finalized"
    assert dict(bootstrap.tags)["flash-phase"] == "bootstrap"
    assert plan.names == bootstrap.names
    assert plan.names == serving_resource_names(
        bundle.spec.deployment_id,
        bundle.spec.generation,
        bundle.spec.engine.engine_id,
        workload_role="app",
    )
    assert plan.gpu_request == "B200:1"
    assert plan.expected_public_url == (f"https://workspace--{plan.endpoint_label}.modal.run")
    assert plan.include_source is False
    assert plan.min_containers == 0
    assert plan.max_containers == 1
    assert plan.buffer_containers == 0
    assert plan.startup_timeout_seconds == MODAL_STARTUP_TIMEOUT_SECONDS == 1800
    # modal's container-boot limit must leave room under the cli deadline for finalize and
    # cleanup, otherwise a slow boot strands a billable half-finalized app.
    assert _default_serve_timeout_seconds() > MODAL_STARTUP_TIMEOUT_SECONDS
    assert plan.scaledown_window_seconds > 0
    assert plan.environment
    assert "FLASH_SERVING_MANIFEST" in dict(plan.environment)
    assert "FLASH_SERVING_CACHE_ROOT" in dict(plan.environment)
    assert all(
        secret not in rendered
        for secret in (PROVIDER_ID, PROVIDER_SECRET, INFERENCE_SECRET, ARTIFACT_SECRET)
    )
    expected_image_tag = (
        base64.urlsafe_b64encode(bytes.fromhex(bundle.image.digest.removeprefix("sha256:")))
        .decode("ascii")
        .rstrip("=")
    )
    assert dict(plan.tags)["flash-image"] == expected_image_tag
    assert plan.bundle.spec.engine.image_digest == bundle.image.digest


def test_tokenless_fresh_create_is_rejected_before_every_mutation() -> None:
    factory = _Factory()
    failure: FreshDeploymentArtifactTokenRequired | None = None

    try:
        _provision(_bundle(), factory, artifact_token=None)
    except FreshDeploymentArtifactTokenRequired as exc:
        failure = exc

    sdk = factory.sdk
    assert sdk is not None
    assert sdk.calls == [("observe", None)]
    assert all(not name.startswith(("create_", "deploy_")) for name, _value in sdk.calls)
    assert failure is not None
    assert failure.code == "invalid_request"
    assert failure.outcome_unknown is False
    assert str(failure) == (
        "a new deployment hydrates its serving cache from the hub before the engine starts, and "
        "that hydration requires a token even when the repositories are public"
    )


def test_tokenless_existing_generation_is_adopted_without_create_mutations() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    expected_handle = _seed_exact(sdk)
    clock = _Clock()

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "ready"
    assert result.handle == expected_handle
    assert all(not name.startswith(("create_", "deploy_")) for name, _value in sdk.calls)


def test_happy_create_uses_exact_resources_endpoint_and_cleanup_order() -> None:
    bundle = _bundle()
    factory = _Factory()
    result, probe = _provision(bundle, factory)
    sdk = factory.sdk
    assert sdk is not None

    assert result.status == "ready"
    assert result.handle is not None
    assert result.handle.public_url == sdk.plan.expected_public_url
    assert result.handle.workspace_name == "workspace"
    assert result.handle.environment == "main"
    assert result.handle.region == "us-east-1"
    assert result.handle.image_digest == bundle.image.digest
    assert probe.calls == [
        (result.handle.public_url, True, bundle, 30.0),
        (result.handle.public_url, True, bundle, 30.0),
        (result.handle.public_url, True, bundle, 30.0),
    ]
    assert sdk.artifact == []
    assert sdk.closed is True
    assert [name for name, _value in sdk.calls if name != "observe"] == [
        "create_inference",
        "create_artifact",
        "create_volume",
        "deploy_app",
        "deploy_app",
        "delete_artifact",
    ]
    assert [value for name, value in sdk.calls if name == "deploy_app"] == [
        "bootstrap",
        "finalized",
    ]
    assert not hasattr(result.handle, "artifact_secret_id")


def test_request_secrets_are_confined_to_one_shot_sinks_and_sanitized_results() -> None:
    bundle = _bundle()
    factory = _Factory()
    result, probe = _provision(bundle, factory)
    sdk = factory.sdk
    assert sdk is not None

    rendered = repr((result.spec, result.status, result.handle, result.error_code)) + repr(
        sdk.calls
    )
    assert factory.calls == [(True, True)]
    assert probe.calls[0][1] is True
    assert ("create_inference", True) in sdk.calls
    assert ("create_artifact", True) in sdk.calls
    for secret in (PROVIDER_ID, PROVIDER_SECRET, INFERENCE_SECRET, ARTIFACT_SECRET):
        assert secret not in rendered
    assert "FREESOLO_INTERNAL_KEY" not in rendered


def test_adoption_waits_out_a_cold_container_instead_of_probing_once() -> None:
    """a rerun must follow an existing deployment to readiness, like a fresh create does.

    A cold gpu container can need longer than the probe's 30-second cap to load. Probing once
    answered `outcome_unknown` with nearly the whole deadline unspent, so rerunning `serve deploy`
    against an app that was still warming could never reach it -- while `wait_for_phase` already
    knew how to wait for exactly this on the create and reconcile paths.
    """

    class _ColdProbe:
        def __init__(self, accept_on: int) -> None:
            self.accept_on = accept_on
            self.calls = 0

        def __call__(
            self,
            _url: str,
            _token: str,
            _bundle: DeploymentBundle,
            _timeout_seconds: float,
        ) -> bool:
            self.calls += 1
            return self.calls >= self.accept_on

    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk)
    probe = _ColdProbe(accept_on=3)
    clock = _Clock()

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=600.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "ready", "adoption gave up on a container that warmed up in time"
    assert result.handle == handle
    assert probe.calls == 3, "the endpoint was probed once, so no waiting happened"
    assert all(not name.startswith("create_") for name, _value in sdk.calls)


def test_adoption_of_an_uncleaned_deployment_probes_instead_of_burning_the_deadline() -> None:
    """the solo case: nobody else is reclaiming, so the artifact stays put until this run acts.

    Waiting on the *cleaned* phase here could never build a proof while the secret was still
    there, so the endpoint was never probed at all -- the wait spun to the deadline and handed an
    exhausted one to the cleanup, turning a healthy app that needed nothing but artifact reclaim
    into `outcome_unknown`. The target has to be the phase the observation actually showed.
    """

    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    _seed_exact(sdk, artifact=True)
    probe = _Probe()
    clock = _Clock()
    started_at = clock()

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "ready"
    assert probe.calls, "the endpoint was never probed, so the wait could not prove readiness"
    # the real symptom: spinning to the deadline before the artifact is ever reclaimed.
    assert clock() - started_at < 90.0, "the wait burned the deadline instead of probing"
    assert sdk.artifact == [], "the artifact was left behind for the next run to trip over"


def test_adoption_of_a_cold_bootstrap_app_waits_instead_of_probing_once() -> None:
    """the bootstrap half of the cold-container fix above.

    A prior invocation can leave its app in the *bootstrap* phase -- artifact secret present,
    hydrating, not yet finalized. Probing that once capped the attempt at the probe's 30-second
    ceiling and returned `outcome_unknown` with nearly the whole deadline unspent, so a rerun could
    never follow an in-progress invocation through bootstrap readiness into its finalized
    transition. The finalized branch already waited; this one did not.
    """

    class _ColdProbe:
        def __init__(self, accept_on: int) -> None:
            self.accept_on = accept_on
            self.calls = 0

        def __call__(
            self,
            _url: str,
            _token: str,
            _bundle: DeploymentBundle,
            _timeout_seconds: float,
        ) -> bool:
            self.calls += 1
            return self.calls >= self.accept_on

    bundle = _bundle()
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    sdk = _FakeSdk(bootstrap_plan)
    _seed_exact(sdk, artifact=True)
    # more probes than a single attempt can make: one probe call is the whole pre-fix budget.
    probe = _ColdProbe(accept_on=3)
    clock = _Clock()

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=600.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
    )

    assert probe.calls >= 3, "the cold bootstrap app was probed once, so no waiting happened"
    assert result.status == "ready"
    assert ("deploy_app", "finalized") in sdk.calls


def test_fresh_readiness_timeout_aborts_confirmed_resources() -> None:
    factory = _Factory()

    result, _probe = _provision(_bundle(), factory, probe=_Probe(False))

    sdk = factory.sdk
    assert sdk is not None
    assert result.status == "failed"
    assert result.error_code == "readiness_failed"
    assert sdk.apps[0].state == "stopped"
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []


def test_readiness_read_uses_work_deadline_and_cleanup_keeps_its_reserve() -> None:
    clock = _Clock()

    class _DeadlineAwareSdk(_FakeSdk):
        def __init__(self, plan) -> None:
            super().__init__(plan)
            self.read_deadlines: list[float | None] = []
            self.timed_out_read = False

        def observe(self, plan, *, app_id_hint=None, deadline_at=None) -> ModalObservation:
            self.read_deadlines.append(deadline_at)
            if self.apps and deadline_at is not None and not self.timed_out_read:
                self.timed_out_read = True
                clock.now = deadline_at
                raise ModalSdkFailure("transport_failed")
            return super().observe(
                plan,
                app_id_hint=app_id_hint,
                deadline_at=deadline_at,
            )

    factory = _Factory()
    factory.sdk_class = _DeadlineAwareSdk
    result = provision_modal_deployment(
        _bundle(),
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=100.0,
        sdk_factory=factory,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )

    sdk = factory.sdk
    assert isinstance(sdk, _DeadlineAwareSdk)
    assert sdk.timed_out_read is True
    assert sdk.read_deadlines[:2] == [None, 70.0]
    assert sdk.read_deadlines[2:]
    assert all(value is None for value in sdk.read_deadlines[2:])
    assert clock() == 70.0
    assert result.status == "failed"
    assert result.error_code == "transport_failed"
    assert sdk.apps[0].state == "stopped"
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []


def test_fresh_phase_conflict_aborts_resources_confirmed_by_this_invocation() -> None:
    class _TransientArtifactOmissionSdk(_FakeSdk):
        omitted = False

        def observe(self, plan, *, app_id_hint=None, deadline_at=None) -> ModalObservation:
            observation = super().observe(plan, app_id_hint=app_id_hint)
            if self.apps and self.artifact and not self.omitted:
                self.omitted = True
                return replace(observation, artifact_secrets=())
            return observation

    factory = _Factory()
    factory.sdk_class = _TransientArtifactOmissionSdk

    result, _probe = _provision(_bundle(), factory)

    sdk = factory.sdk
    assert sdk is not None
    assert sdk.omitted is True, "the post-deploy conflict was not injected"
    assert result.status == "failed"
    assert result.error_code == "conflict"
    assert sdk.apps[0].state == "stopped"
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []


def test_fresh_phase_conflict_preserves_confirmed_handle_when_cleanup_is_unknown() -> None:
    class _RetainedVolumeAfterConflictSdk(_FakeSdk):
        omitted = False

        def observe(self, plan, *, app_id_hint=None, deadline_at=None) -> ModalObservation:
            observation = super().observe(plan, app_id_hint=app_id_hint)
            if self.apps and self.artifact and not self.omitted:
                self.omitted = True
                return replace(observation, artifact_secrets=())
            return observation

        def delete_volume(self, plan, volume_id: str, *, deadline_at=None) -> None:
            self.calls.append(("delete_volume", None))
            assert volume_id == VOLUME_ID

    factory = _Factory()
    factory.sdk_class = _RetainedVolumeAfterConflictSdk

    result, _probe = _provision(_bundle(), factory)

    sdk = factory.sdk
    assert sdk is not None
    assert result.status == "outcome_unknown"
    assert result.handle is not None
    assert result.handle.app_id == APP_ID
    assert result.handle.volume_id == VOLUME_ID
    assert result.handle.inference_secret_id == INFERENCE_SECRET_ID
    assert sdk.volumes, "the test did not retain the volume it is meant to report"


def test_fresh_readiness_timeout_leaves_time_for_asynchronous_stop() -> None:
    class _DelayedStopSdk(_FakeSdk):
        def __init__(self, plan) -> None:
            super().__init__(plan)
            self.stop_observations = 0

        def observe(self, plan, *, app_id_hint=None, deadline_at=None) -> ModalObservation:
            if self.apps and self.apps[0].state == "lifecycle_pending":
                self.stop_observations += 1
                if self.stop_observations >= 2:
                    self.apps = [
                        replace(
                            self.apps[0],
                            state="stopped",
                            running_containers=0,
                            tags=(),
                            function_id=None,
                            function_name=None,
                            public_url=None,
                        )
                    ]
            return super().observe(plan, app_id_hint=app_id_hint)

        def stop_app(self, plan, app_id: str, *, deadline_at=None) -> None:
            self.calls.append(("stop_app", None))
            self._fail("stop_app")
            self.apps = [
                replace(
                    self.apps[0],
                    state="lifecycle_pending",
                    running_containers=None,
                    function_id=None,
                    function_name=None,
                    public_url=None,
                )
            ]

    factory = _Factory()
    factory.sdk_class = _DelayedStopSdk

    result, _probe = _provision(_bundle(), factory, probe=_Probe(False))

    sdk = factory.sdk
    assert sdk is not None
    assert result.status == "failed"
    assert result.error_code == "readiness_failed"
    assert sdk.stop_observations >= 2
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []


def test_fresh_readiness_timeout_preserves_confirmed_handle_when_cleanup_is_unknown() -> None:
    class _RetainedVolumeSdk(_FakeSdk):
        def delete_volume(self, plan, volume_id: str, *, deadline_at=None) -> None:
            self.calls.append(("delete_volume", None))

    factory = _Factory()
    factory.sdk_class = _RetainedVolumeSdk

    result, _probe = _provision(_bundle(), factory, probe=_Probe(False))

    sdk = factory.sdk
    assert sdk is not None
    assert result.status == "outcome_unknown"
    assert result.handle is not None
    assert result.handle.app_id == APP_ID
    assert result.handle.volume_id == VOLUME_ID
    assert result.handle.inference_secret_id == INFERENCE_SECRET_ID
    assert sdk.volumes, "the test did not retain the volume it is meant to report"


def test_adoption_keeps_the_artifact_when_readiness_is_never_proven() -> None:
    """reclaim follows proof of readiness, never precedes it.

    A cold container that never answers the probe leaves the wait returning nothing. Deleting the
    artifact anyway strips the bootstrap credential from an app that has not proven it finished
    hydrating, and leaves nothing for the rerun that `outcome_unknown` invites -- a destructive
    mutation performed on the strength of a timeout.

    The verdict is `outcome_unknown` either way, so the deletion bought nothing: it was pure loss.
    """

    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    _seed_exact(sdk, artifact=True)
    clock = _Clock()

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(False),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "outcome_unknown"
    assert sdk.artifact, "the bootstrap credential was reclaimed without ever proving readiness"
    assert [name for name, _value in sdk.calls if name.startswith("delete_")] == [], (
        "a readiness timeout must not mutate provider state"
    )


def test_adoption_tolerates_a_concurrent_artifact_reclaim_while_waiting() -> None:
    """a racer finishing the reclaim mid-wait is the success state, not a conflict.

    Waiting re-observes the phase on every poll, so a concurrent `serve deploy` -- or the run that
    created this app completing its own reclaim -- can delete the artifact secret while this wait
    is still warming a cold container. Demanding the artifact still be present turned that into a
    definite `failed`/`conflict` for an app that was healthy, deployed, and billing.
    """

    class _ColdProbe:
        def __init__(self, accept_on: int) -> None:
            self.accept_on = accept_on
            self.calls = 0

        def __call__(
            self,
            _url: str,
            _token: str,
            _bundle: DeploymentBundle,
            _timeout_seconds: float,
        ) -> bool:
            self.calls += 1
            return self.calls >= self.accept_on

    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    _seed_exact(sdk, artifact=True)
    original_observe = sdk.observe
    observations = {"count": 0}

    def racing_observe(observed_plan, *, app_id_hint=None, deadline_at=None):
        # the reclaim lands after the adoption branch has already been chosen, which is the only
        # ordering that reaches the wait with the artifact disappearing underneath it.
        observations["count"] += 1
        result = original_observe(observed_plan, app_id_hint=app_id_hint)
        if observations["count"] >= 2:
            sdk.artifact.clear()
        return result

    sdk.observe = racing_observe  # type: ignore[assignment]
    probe = _ColdProbe(accept_on=3)
    clock = _Clock()

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=600.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "ready", "a healthy app was rejected because a racer reclaimed first"
    assert result.error_code is None
    assert sdk.apps, "the app must still be deployed"
    assert not sdk.artifact, "the artifact is reclaimed either way"


def test_exact_adoption_requires_authenticated_endpoint_provenance() -> None:
    bundle = _bundle()
    factory = _Factory()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk)

    def seeded_factory(credentials, received_plan, _deadline_at, _clock):
        assert received_plan == plan
        return sdk

    clock = _Clock()
    accepted = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=seeded_factory,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert accepted.status == "ready"
    assert accepted.handle == handle
    assert all(not name.startswith("create_") for name, _value in sdk.calls)

    sdk.calls.clear()
    refused = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=seeded_factory,
        probe=_Probe(False),
        clock=clock,
        sleep=clock.sleep,
    )
    assert refused.status == "outcome_unknown"
    assert refused.handle == handle
    assert all(not name.startswith("create_") for name, _value in sdk.calls)
    assert factory.calls == []


def test_existing_bootstrap_is_finalized_by_the_sole_retry() -> None:
    bundle = _bundle()
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    sdk = _FakeSdk(bootstrap_plan)
    _seed_exact(sdk, artifact=True)
    clock = _Clock()

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.status == "ready"
    assert [value for name, value in sdk.calls if name == "deploy_app"] == ["finalized"]
    assert sdk.artifact == []


@pytest.mark.parametrize("artifact_present", [True, False])
def test_adoption_accepts_a_pinned_concurrent_finalized_successor(artifact_present: bool) -> None:
    bundle = _bundle()
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    finalized_plan = build_modal_create_plan(bundle, phase="finalized")
    sdk = _FakeSdk(bootstrap_plan)
    _seed_exact(sdk, artifact=True)
    original_observe = sdk.observe
    observations = 0

    def racing_observe(observed_plan, *, app_id_hint=None, deadline_at=None):
        nonlocal observations
        observations += 1
        if observations == 2:
            sdk.apps[0] = replace(sdk.apps[0], tags=finalized_plan.tags)
            if not artifact_present:
                sdk.artifact.clear()
        return original_observe(observed_plan, app_id_hint=app_id_hint)

    sdk.observe = racing_observe  # type: ignore[assignment]
    clock = _Clock()
    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "ready"
    assert result.error_code is None
    assert [value for name, value in sdk.calls if name == "deploy_app"] == []
    assert sdk.artifact == []


def test_adoption_tolerates_the_artifact_flickering_back_in_a_cleaned_wait() -> None:
    """a cleaned successor whose artifact reappears for one poll is still the success state.

    The pinned-successor test clears the artifact atomically and permanently, so it never exercises
    a reading where the artifact comes back. Modal's list calls are not a snapshot: a concurrent
    finalize can be observed mid-flight, showing the artifact again after it looked gone. Waiting
    without the `with_artifact` transient turned that single reading into a definite `conflict` for
    an app that was deployed, healthy, and billing.
    """

    class _ColdProbe:
        """rejects the first probe so the cleaned wait polls again and meets the flicker."""

        def __init__(self, accept_on: int) -> None:
            self.accept_on = accept_on
            self.calls = 0

        def __call__(
            self,
            _url: str,
            _token: str,
            _bundle: DeploymentBundle,
            _timeout_seconds: float,
        ) -> bool:
            self.calls += 1
            return self.calls >= self.accept_on

    bundle = _bundle()
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    finalized_plan = build_modal_create_plan(bundle, phase="finalized")
    sdk = _FakeSdk(bootstrap_plan)
    _seed_exact(sdk, artifact=True)
    original_observe = sdk.observe
    artifact_secret = list(sdk.artifact)
    state = {"observations": 0, "flickered": False}

    def flickering_observe(observed_plan, *, app_id_hint=None, deadline_at=None):
        state["observations"] += 1
        if state["observations"] == 2:
            # the racer's finalize lands: tags flip and the artifact reads as reclaimed.
            sdk.apps[0] = replace(sdk.apps[0], tags=finalized_plan.tags)
            sdk.artifact.clear()
        elif state["observations"] == 4:
            # obs#4 is the cleaned wait's own poll -- obs#3 is the successor re-observe that picks
            # the branch. the artifact must be visible on *this* reading, not an earlier one, or
            # the wait proves on a clean view and the missing transient never matters.
            state["flickered"] = True
            sdk.artifact[:] = artifact_secret
        elif state["observations"] > 4:
            sdk.artifact.clear()
        return original_observe(observed_plan, app_id_hint=app_id_hint)

    sdk.observe = flickering_observe  # type: ignore[assignment]
    # the first probe is refused so the wait polls a second time and settles on a clean reading,
    # proving the flicker is tolerated rather than merely skipped.
    probe = _ColdProbe(accept_on=2)
    clock = _Clock()

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=600.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
    )

    assert state["flickered"], "the artifact never flickered back, so nothing was proven"
    assert result.status == "ready", "a transient artifact reading was treated as a real conflict"
    assert result.error_code is None
    assert sdk.apps, "the app must still be deployed"
    assert [value for name, value in sdk.calls if name == "deploy_app"] == []


def test_adoption_rejects_identity_drift_during_a_concurrent_transition() -> None:
    bundle = _bundle()
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    finalized_plan = build_modal_create_plan(bundle, phase="finalized")
    sdk = _FakeSdk(bootstrap_plan)
    _seed_exact(sdk, artifact=True)
    original_observe = sdk.observe
    observations = 0

    def drifting_observe(observed_plan, *, app_id_hint=None, deadline_at=None):
        nonlocal observations
        observations += 1
        if observations == 2:
            sdk.apps[0] = replace(sdk.apps[0], tags=finalized_plan.tags)
            sdk.volumes[0] = replace(sdk.volumes[0], id="vo-" + "D" * 22)
        return original_observe(observed_plan, app_id_hint=app_id_hint)

    sdk.observe = drifting_observe  # type: ignore[assignment]
    clock = _Clock()
    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "failed"
    assert result.error_code == "conflict"
    assert observations == 3, "the phase conflict was not re-observed exactly once"
    assert [name for name, _value in sdk.calls if name != "observe"] == []


def test_opaque_secret_or_name_only_resources_are_refused_without_mutation() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    sdk.inference = [ModalNamedResource(OPAQUE_SECRET_ID, plan.names.inference_secret)]

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )
    assert result.status == "failed"
    assert result.error_code == "conflict"
    assert all(not name.startswith("create_") for name, _value in sdk.calls)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda sdk: sdk.apps.append(sdk.apps[0]),
        lambda sdk: sdk.volumes.append(sdk.volumes[0]),
        lambda sdk: sdk.apps.__setitem__(
            0,
            replace(sdk.apps[0], tags=(("flash-spec", "wrong"),)),
        ),
        lambda sdk: sdk.apps.__setitem__(
            0,
            replace(sdk.apps[0], public_url="https://wrong.modal.run"),
        ),
        lambda sdk: sdk.apps.__setitem__(
            0,
            replace(sdk.apps[0], running_containers=2),
        ),
    ],
)
def test_duplicate_and_mismatched_resources_fail_closed(mutate) -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    _seed_exact(sdk)
    mutate(sdk)

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )
    assert result.status == "failed"
    assert result.error_code == "conflict"
    assert all(not name.startswith("create_") for name, _value in sdk.calls)


def test_exact_workspace_and_environment_binding_fail_closed() -> None:
    bundle = _bundle()
    workspace_factory = _Factory()
    workspace_factory.workspace_override = "other-workspace"
    result, _probe = _provision(bundle, workspace_factory, artifact_token=None)
    assert result.status == "failed"
    assert result.error_code == "authentication_failed"
    assert workspace_factory.sdk is not None
    assert workspace_factory.sdk.closed is True

    environment_factory = _Factory()
    environment_factory.environment_override = "other"
    result, _probe = _provision(bundle, environment_factory, artifact_token=None)
    assert result.status == "failed"
    assert result.error_code == "conflict"


def test_ambiguous_high_level_mutation_is_called_once_and_never_retried() -> None:
    bundle = _bundle()
    factory = _Factory()

    def failing_factory(credentials, plan, _deadline_at, _clock):
        sdk = factory(credentials, plan, _deadline_at, _clock)
        sdk.fail_operation = "create_volume"
        return sdk

    clock = _Clock()
    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=100.0,
        sdk_factory=failing_factory,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    sdk = factory.sdk
    assert sdk is not None
    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert [name for name, _value in sdk.calls].count("create_volume") == 1
    assert "deploy_app" not in [name for name, _value in sdk.calls]
    assert PROVIDER_SECRET not in repr(
        (result.spec, result.status, result.handle, result.error_code)
    )


def test_reconcile_is_read_only_for_ready_absent_and_lingering_artifact() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk)
    clock = _Clock()

    ready = reconcile_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert ready.status == "ready"
    assert ready.handle == handle
    assert all(name == "observe" for name, _value in sdk.calls)

    sdk.calls.clear()
    sdk.artifact = [ModalNamedResource(ARTIFACT_SECRET_ID, plan.names.artifact_secret)]
    clock.now = 0.0
    unknown = reconcile_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=4.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert unknown.status == "outcome_unknown"
    assert sdk.artifact
    assert all(name == "observe" for name, _value in sdk.calls)

    sdk.apps.clear()
    sdk.volumes.clear()
    sdk.inference.clear()
    sdk.artifact.clear()
    sdk.calls.clear()
    clock.now = 0.0
    absent = reconcile_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        None,
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert absent.status == "absent"
    assert all(name == "observe" for name, _value in sdk.calls)


def test_tokenless_reconcile_returns_observed_modal_handle() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk)
    probe = _Probe(True)

    result = reconcile_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        None,
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=probe,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )

    assert result.status == "outcome_unknown"
    assert result.handle is not None
    assert result.handle == handle
    assert result.error_reason == "readiness_deadline_unproven"
    assert probe.calls == []
    assert all(name == "observe" for name, _value in sdk.calls)


def test_unproven_readiness_reconcile_returns_observed_modal_handle() -> None:
    # the tokenless path above already reports its ids. this is the same deployment with a key,
    # where the probe never accepts: the app, volume, and secret are equally live and billing, so
    # losing the handle here would leave an operator with nothing to name in a teardown.
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk)
    clock = _Clock()

    result = reconcile_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(False),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "outcome_unknown"
    assert result.handle == handle
    assert result.error_reason == "readiness_deadline_unproven"


def test_artifact_cleanup_rejection_reports_the_live_modal_app() -> None:
    bundle = _bundle()
    factory = _Factory()

    class _RejectedDeleteSdk(_FakeSdk):
        def delete_secret(self, plan, secret_id: str, *, deadline_at=None) -> None:
            if secret_id == ARTIFACT_SECRET_ID:
                self.calls.append(("delete_artifact", None))
                raise ModalSdkFailure("provider_rejected")
            super().delete_secret(plan, secret_id, deadline_at=deadline_at)

    factory.sdk_class = _RejectedDeleteSdk
    result, _probe = _provision(bundle, factory)

    sdk = factory.sdk
    assert sdk is not None
    assert result.status == "failed"
    assert result.error_reason == "artifact_cleanup_delete_rejected"
    assert result.handle is not None
    assert result.handle.app_id == APP_ID
    assert sdk.apps[0].state == "deployed"
    assert sdk.artifact, "the rejected cleanup must leave the artifact secret visible"


def test_artifact_cleanup_ambiguity_keeps_handle_and_never_retries() -> None:
    bundle = _bundle()
    factory = _Factory()

    def failing_factory(credentials, plan, _deadline_at, _clock):
        sdk = factory(credentials, plan, _deadline_at, _clock)
        sdk.fail_operation = "delete_artifact"
        sdk.fail_with_sdk = True
        return sdk

    clock = _Clock()
    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=100.0,
        sdk_factory=failing_factory,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    sdk = factory.sdk
    assert sdk is not None
    assert result.status == "outcome_unknown"
    assert result.handle is not None
    assert result.error_reason == "artifact_cleanup_delete_unknown"
    assert result.error_reason.startswith("artifact_cleanup_")
    assert [name for name, _value in sdk.calls].count("delete_artifact") == 1


def test_artifact_cleanup_observation_failure_reports_the_live_modal_app() -> None:
    bundle = _bundle()
    factory = _Factory()

    class _ObservationFailureAfterDeleteSdk(_FakeSdk):
        artifact_deleted = False

        def observe(self, plan, *, app_id_hint=None, deadline_at=None) -> ModalObservation:
            if self.artifact_deleted:
                raise ModalSdkFailure("transport_failed")
            return super().observe(plan, app_id_hint=app_id_hint, deadline_at=deadline_at)

        def delete_secret(self, plan, secret_id: str, *, deadline_at=None) -> None:
            super().delete_secret(plan, secret_id, deadline_at=deadline_at)
            if secret_id == ARTIFACT_SECRET_ID:
                self.artifact_deleted = True

    factory.sdk_class = _ObservationFailureAfterDeleteSdk
    result, _probe = _provision(bundle, factory)

    assert result.status == "outcome_unknown"
    assert result.handle is not None
    assert result.error_reason == "artifact_cleanup_observation_failed"
    assert result.error_reason.startswith("artifact_cleanup_")


def test_finalization_ambiguity_is_reconciled_read_only_without_artifact_deletion() -> None:
    bundle = _bundle()
    factory = _Factory()

    def failing_factory(credentials, plan, _deadline_at, _clock):
        sdk = factory(credentials, plan, _deadline_at, _clock)
        sdk.fail_operation = "deploy_finalized"
        sdk.fail_with_sdk = True
        return sdk

    result, _probe = _provision(bundle, failing_factory)
    sdk = factory.sdk
    assert sdk is not None
    assert result.status == "outcome_unknown"
    assert result.handle is not None
    assert sdk.artifact == [ModalNamedResource(ARTIFACT_SECRET_ID, sdk.plan.names.artifact_secret)]
    assert [value for name, value in sdk.calls if name == "deploy_app"] == [
        "bootstrap",
        "finalized",
    ]
    assert "delete_artifact" not in [name for name, _value in sdk.calls]


def test_post_cleanup_core_resource_drift_is_outcome_unknown() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)

    class DriftingCleanupSdk(_FakeSdk):
        def delete_secret(self, received_plan, secret_id: str, *, deadline_at=None) -> None:
            super().delete_secret(received_plan, secret_id)
            if secret_id == ARTIFACT_SECRET_ID:
                self.volumes = [ModalNamedResource("vo-" + "D" * 22, received_plan.names.volume)]

    sdk = DriftingCleanupSdk(plan)
    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )
    assert result.status == "outcome_unknown"
    assert result.error_reason == "artifact_cleanup_conflict"
    assert result.error_reason.startswith("artifact_cleanup_")
    assert [name for name, _value in sdk.calls].count("delete_artifact") == 1


def test_teardown_stops_bootstrap_app_and_deletes_every_attached_resource() -> None:
    bundle = _bundle()
    factory = _Factory()

    def failing_factory(credentials, plan, _deadline_at, _clock):
        sdk = factory(credentials, plan, _deadline_at, _clock)
        sdk.fail_operation = "deploy_finalized"
        sdk.fail_with_sdk = True
        return sdk

    provisioned, _probe = _provision(bundle, failing_factory)
    sdk = factory.sdk
    assert sdk is not None
    assert provisioned.status == "outcome_unknown"
    assert provisioned.handle is not None
    assert sdk.apps[0].tags == build_modal_create_plan(bundle, phase="bootstrap").tags
    sdk.fail_operation = None
    sdk.calls.clear()
    clock = _Clock()

    result = teardown_modal_deployment(
        bundle,
        provisioned.handle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "absent"
    assert [name for name, _value in sdk.calls if name != "observe"] == [
        "stop_app",
        "delete_artifact",
        "delete_inference",
        "delete_volume",
    ]
    assert sdk.apps[0].state == "stopped"
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []


def test_handleless_teardown_reclaims_exact_deterministic_resources() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    _seed_exact(sdk, artifact=True)
    clock = _Clock()

    result = teardown_modal_deployment(
        bundle,
        None,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "absent"
    assert [name for name, _value in sdk.calls if name != "observe"] == [
        "stop_app",
        "delete_artifact",
        "delete_inference",
        "delete_volume",
    ]
    assert sdk.apps[0].state == "stopped"
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []


def test_handleless_teardown_reclaims_partial_resources_without_an_app() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    sdk.inference = [ModalNamedResource(INFERENCE_SECRET_ID, plan.names.inference_secret)]
    sdk.artifact = [ModalNamedResource(ARTIFACT_SECRET_ID, plan.names.artifact_secret)]
    sdk.volumes = [ModalNamedResource(VOLUME_ID, plan.names.volume)]

    result = teardown_modal_deployment(
        bundle,
        None,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )

    assert result.status == "absent"
    assert [name for name, _value in sdk.calls if name != "observe"] == [
        "delete_artifact",
        "delete_inference",
        "delete_volume",
    ]
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []


def test_handleless_teardown_delete_failure_remains_outcome_unknown() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    sdk.inference = [ModalNamedResource(INFERENCE_SECRET_ID, plan.names.inference_secret)]
    sdk.artifact = [ModalNamedResource(ARTIFACT_SECRET_ID, plan.names.artifact_secret)]
    sdk.volumes = [ModalNamedResource(VOLUME_ID, plan.names.volume)]
    sdk.fail_operation = "delete_artifact"

    result = teardown_modal_deployment(
        bundle,
        None,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert [name for name, _value in sdk.calls].count("delete_artifact") == 1
    assert sdk.artifact


def test_handleless_teardown_rejects_foreign_generation_resources() -> None:
    bundle = _bundle()
    foreign_bundle = _bundle(generation=bundle.spec.generation + 1)
    foreign_plan = build_modal_create_plan(foreign_bundle)
    sdk = _FakeSdk(foreign_plan)
    sdk.inference = [ModalNamedResource(INFERENCE_SECRET_ID, foreign_plan.names.inference_secret)]
    sdk.artifact = [ModalNamedResource(ARTIFACT_SECRET_ID, foreign_plan.names.artifact_secret)]
    sdk.volumes = [ModalNamedResource(VOLUME_ID, foreign_plan.names.volume)]

    result = teardown_modal_deployment(
        bundle,
        None,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )

    assert result.status == "failed"
    assert result.error_code == "conflict"
    assert [name for name, _value in sdk.calls if name != "observe"] == []
    assert sdk.apps == []
    assert sdk.volumes
    assert sdk.inference
    assert sdk.artifact


def test_teardown_stops_before_secret_and_volume_deletion_then_confirms_absence() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk, artifact=True)
    clock = _Clock()

    result = teardown_modal_deployment(
        bundle,
        handle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.status == "absent"
    mutations = [name for name, _value in sdk.calls if name != "observe"]
    assert mutations == [
        "stop_app",
        "delete_artifact",
        "delete_inference",
        "delete_volume",
    ]
    assert sdk.apps[0].state == "stopped"
    assert sdk.apps[0].running_containers == 0
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []

    sdk.calls.clear()
    clock.now = 0.0
    reconciled = reconcile_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=4.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert reconciled.status == "failed"
    assert reconciled.error_code == "conflict"

    sdk.apps.clear()
    sdk.calls.clear()
    clock.now = 0.0
    reconciled = reconcile_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert reconciled.status == "absent"
    assert all(name == "observe" for name, _value in sdk.calls)


def test_repeated_teardown_of_an_already_absent_generation_is_idempotent() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk, artifact=True)
    sdk.apps.clear()
    sdk.volumes.clear()
    sdk.inference.clear()
    sdk.artifact.clear()
    sdk.calls.clear()

    result = teardown_modal_deployment(
        bundle,
        handle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )

    assert result.status == "absent"
    assert all(name == "observe" for name, _value in sdk.calls)


def test_teardown_deletes_resources_after_lifecycle_only_stopped_app_observation() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)

    class LifecycleOnlyStoppedSdk(_FakeSdk):
        def __init__(self, received_plan) -> None:
            super().__init__(received_plan)
            self._client = object()
            self._modal = SimpleNamespace(
                experimental=SimpleNamespace(
                    get_app_lifecycle=SimpleNamespace(aio=self._get_app_lifecycle)
                ),
            )

        async def _get_app_lifecycle(self, app_id: str, *, client: object) -> object:
            self.calls.append(("get_app_lifecycle", app_id))
            assert app_id == APP_ID
            assert client is self._client
            return SimpleNamespace(stopped_at=object())

        def _read(self, operation, *, deadline_at):
            return asyncio.run(operation())

        def stop_app(self, received_plan, app_id: str, *, deadline_at=None) -> None:
            self.calls.append(("stop_app", None))
            self._fail("stop_app")
            self.apps.clear()

        def observe(self, received_plan, *, app_id_hint=None, deadline_at=None) -> ModalObservation:
            if self.apps or app_id_hint is None:
                return super().observe(received_plan, app_id_hint=app_id_hint)
            self.calls.append(("observe", app_id_hint))
            app = PinnedModalSdk._lifecycle_app(
                self,
                received_plan,
                app_id_hint,
                deadline_at=deadline_at,
            )
            return ModalObservation(
                workspace_name=self.workspace_name,
                environment_name=self.environment_name,
                apps=(app,),
                volumes=tuple(self.volumes),
                inference_secrets=tuple(self.inference),
                artifact_secrets=tuple(self.artifact),
            )

    sdk = LifecycleOnlyStoppedSdk(plan)
    handle = _seed_exact(sdk)
    clock = _Clock()

    result = teardown_modal_deployment(
        bundle,
        handle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "absent"
    assert sdk.apps == []
    assert ("delete_inference", None) in sdk.calls
    assert ("delete_volume", None) in sdk.calls
    assert [value for name, value in sdk.calls if name == "lookup"] == []
    assert [value for name, value in sdk.calls if name == "get_app_lifecycle"] == [APP_ID, APP_ID]


def test_teardown_post_stop_observation_failure_is_outcome_unknown() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)

    class FailingObservationSdk(_FakeSdk):
        def observe(self, plan, *, app_id_hint=None, deadline_at=None) -> ModalObservation:
            if self.apps and self.apps[0].state == "stopped":
                raise ModalSdkFailure("transport_failed")
            return super().observe(plan, app_id_hint=app_id_hint)

    sdk = FailingObservationSdk(plan)
    handle = _seed_exact(sdk)
    result = teardown_modal_deployment(
        bundle,
        handle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )
    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert [name for name, _value in sdk.calls].count("stop_app") == 1


def test_teardown_never_deletes_resources_without_explicit_terminal_app_proof() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)

    class PendingLifecycleSdk(_FakeSdk):
        def stop_app(self, received_plan, app_id: str, *, deadline_at=None) -> None:
            self.calls.append(("stop_app", None))
            app = self.apps[0]
            self.apps = [
                replace(
                    app,
                    state="lifecycle_pending",
                    running_containers=None,
                    tags=(),
                    function_id=None,
                    function_name=None,
                    public_url=None,
                )
            ]

    sdk = PendingLifecycleSdk(plan)
    handle = _seed_exact(sdk, artifact=True)
    clock = _Clock()
    result = teardown_modal_deployment(
        bundle,
        handle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=4.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.status == "outcome_unknown"
    assert [name for name, _value in sdk.calls if name.startswith("delete_")] == []

    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk, artifact=True)
    sdk.apps.clear()
    sdk.calls.clear()
    result = teardown_modal_deployment(
        bundle,
        handle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )
    assert result.status == "outcome_unknown"
    assert [name for name, _value in sdk.calls if name.startswith("delete_")] == []


def test_teardown_accepts_terminal_app_with_retained_finalized_tags() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)

    class RetainedTagsSdk(_FakeSdk):
        def stop_app(self, received_plan, app_id: str, *, deadline_at=None) -> None:
            self.calls.append(("stop_app", None))
            app = self.apps[0]
            self.apps = [
                replace(
                    app,
                    state="stopped",
                    running_containers=0,
                    function_id=None,
                    function_name=None,
                    public_url=None,
                )
            ]

    sdk = RetainedTagsSdk(plan)
    handle = _seed_exact(sdk, artifact=True)
    result = teardown_modal_deployment(
        bundle,
        handle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )

    assert result.status == "absent"
    assert [name for name, _value in sdk.calls if name.startswith("delete_")] == [
        "delete_artifact",
        "delete_inference",
        "delete_volume",
    ]
    assert sdk.apps[0].tags == plan.tags
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []


@pytest.mark.parametrize("state", ["stopped", "failed"])
def test_teardown_accepts_terminal_zero_container_app_states_without_second_stop(
    state: str,
) -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk)
    sdk.apps = [
        replace(
            sdk.apps[0],
            state=state,
            running_containers=0,
            tags=(),
            function_id=None,
            function_name=None,
            public_url=None,
        )
    ]

    result = teardown_modal_deployment(
        bundle,
        handle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan, _deadline_at, _clock: sdk,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )
    assert result.status == "absent"
    assert "stop_app" not in [name for name, _value in sdk.calls]


def test_teardown_wrong_generation_handle_fails_before_client_construction() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk)
    calls = 0

    def factory(_credentials, _plan, _deadline_at, _clock):
        nonlocal calls
        calls += 1
        return sdk

    with pytest.raises(ValueError, match="exact deployment generation"):
        teardown_modal_deployment(
            bundle,
            replace(handle, generation=handle.generation + 1),
            ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
            deadline_at=100.0,
            sdk_factory=factory,
            clock=_Clock(),
            sleep=lambda _seconds: None,
        )
    assert calls == 0


def test_provider_error_reprs_and_results_never_leak_credential_sentinels() -> None:
    bundle = _bundle()
    factory = _Factory()

    def failing_factory(credentials, plan, _deadline_at, _clock):
        sdk = factory(credentials, plan, _deadline_at, _clock)
        sdk.fail_operation = "deploy_finalized"
        return sdk

    result, _probe = _provision(bundle, failing_factory)
    rendered = repr((result.spec, result.status, result.handle, result.error_code))
    assert result.status == "outcome_unknown"
    for secret in (PROVIDER_ID, PROVIDER_SECRET, INFERENCE_SECRET):
        assert secret not in rendered
    failure = ModalSdkFailure("transport_failed")
    assert PROVIDER_SECRET not in repr(failure) + str(failure)


def test_modal_endpoint_probe_rejects_redirect_and_wrong_provenance() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    observed: list[dict[str, object]] = []

    class RedirectSource(__import__("urllib.request").request.BaseHandler):
        def default_open(self, request):
            observed.append(
                {
                    "url": request.full_url,
                    "auth_ok": request.get_header("Authorization") == f"Bearer {INFERENCE_SECRET}",
                }
            )
            headers = email.message.Message()
            headers["Location"] = "https://attacker.invalid/models"
            return urllib.response.addinfourl(
                io.BytesIO(b""),
                headers,
                request.full_url,
                code=302,
            )

    opener = __import__("urllib.request").request.build_opener(RedirectSource())
    probe = ModalEndpointProbe(opener=opener)
    assert probe(plan.expected_public_url, INFERENCE_SECRET, bundle, 3.0) is False
    assert observed == [
        {
            "url": plan.expected_public_url + "/v1/models",
            "auth_ok": True,
        }
    ]

    wrong = _models_payload(bundle)
    wrong["data"][0]["flash_provenance"]["engine_id"] = "0" * 64
    assert _provenance_matches(wrong, bundle) is False
    assert _provenance_matches(_models_payload(bundle), bundle) is True


def test_modal_endpoint_provenance_requires_exact_checkpoint_ids_and_full_mapping() -> None:
    bundle = _bundle()
    valid = _models_payload(bundle)
    assert _provenance_matches(valid, bundle) is True

    cases = []
    missing = json.loads(json.dumps(valid))
    missing["data"].pop()
    cases.append(missing)

    duplicate = json.loads(json.dumps(valid))
    duplicate["data"].append(dict(duplicate["data"][0]))
    cases.append(duplicate)

    extra = json.loads(json.dumps(valid))
    extra_entry = json.loads(json.dumps(extra["data"][0]))
    extra_entry["id"] = "unexpected"
    extra_entry["flash_provenance"]["requested_model"] = "unexpected"
    extra["data"].append(extra_entry)
    cases.append(extra)

    wrong_requested = json.loads(json.dumps(valid))
    wrong_requested["data"][0]["flash_provenance"]["requested_model"] = "wrong"
    cases.append(wrong_requested)

    wrong_checkpoint = json.loads(json.dumps(valid))
    wrong_checkpoint["data"][0]["flash_provenance"]["checkpoint_id"] = "wrong"
    cases.append(wrong_checkpoint)

    wrong_global = json.loads(json.dumps(valid))
    wrong_global["data"][0]["flash_provenance"]["logical_base_revision"] = "0" * 40
    cases.append(wrong_global)

    unexpected_provenance = json.loads(json.dumps(valid))
    unexpected_provenance["data"][0]["flash_provenance"]["extra"] = True
    cases.append(unexpected_provenance)

    assert all(_provenance_matches(case, bundle) is False for case in cases)


def test_modal_handle_rejects_malformed_role_ids_at_creation() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk)
    for field, malformed in (
        ("app_id", "ap-short"),
        ("volume_id", "vo-bad/value"),
        ("inference_secret_id", "xx-" + "S" * 22),
    ):
        with pytest.raises(ValueError, match="pinned provider contract"):
            replace(handle, **{field: malformed})


def test_import_purity_blocks_modal_for_control_app_manifest_and_materialize() -> None:
    program = r"""
import builtins
import sys

real_import = builtins.__import__
intercepted = []


def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "modal" or name.startswith("modal."):
        intercepted.append(name)
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded
from flash.serve.app.manifest import ServingManifest
from flash.serve.app.materialize import hydrate_manifest
from flash.serve.provisioning import ServingImage
from flash.serve.provisioning.modal.planning.plan import build_modal_create_plan
from flash.serve.provisioning.modal.execution.operations import provision_modal_deployment

assert ServingManifest
assert hydrate_manifest
assert ServingImage
assert build_modal_create_plan
assert provision_modal_deployment
assert intercepted == []
assert "modal" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", program],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


class _InterruptingProbe(_Probe):
    """Ctrl-C while polling a slow app, which is when a user is most likely to press it."""

    def __call__(self, url, token, bundle, timeout_seconds):
        raise KeyboardInterrupt


class _DelayedAbortSdk(_FakeSdk):
    """modal acknowledges stop before the app releases its volume mount."""

    def __init__(self, plan) -> None:
        super().__init__(plan)
        self.terminal_polls = 0

    def stop_app(self, plan, app_id: str, *, deadline_at=None) -> None:
        self.calls.append(("stop_app", None))
        app = self.apps[0]
        self.apps = [
            replace(
                app,
                state="lifecycle_pending",
                running_containers=None,
                tags=(),
                function_id=None,
                function_name=None,
                public_url=None,
            )
        ]

    def observe(self, plan, *, app_id_hint=None, deadline_at=None) -> ModalObservation:
        if app_id_hint is not None and self.apps[0].state == "lifecycle_pending":
            self.terminal_polls += 1
            if self.terminal_polls >= 2:
                self.apps = [replace(self.apps[0], state="stopped", running_containers=0)]
        observation = super().observe(plan, app_id_hint=app_id_hint)
        if app_id_hint is None and self.apps and self.apps[0].state != "deployed":
            return replace(observation, apps=())
        return observation

    def delete_volume(self, plan, volume_id: str, *, deadline_at=None) -> None:
        self.calls.append(("delete_volume", None))
        if self.apps and self.apps[0].state != "stopped":
            return
        self.volumes.clear()


def test_abort_declines_a_plan_identical_app_with_a_different_confirmed_id() -> None:
    plan = build_modal_create_plan(_bundle(), phase="bootstrap")
    sdk = _FakeSdk(plan)
    _seed_exact(sdk, artifact=True)
    sdk.apps = [replace(sdk.apps[0], app_id=OTHER_APP_ID)]
    expected = ExpectedResources(
        app_id=APP_ID,
        volume_id=VOLUME_ID,
        inference_secret_id=INFERENCE_SECRET_ID,
        artifact_secret_id=ARTIFACT_SECRET_ID,
    )
    created = _CreatedResources(
        inference=True,
        artifact=True,
        volume=True,
        app_deployed=True,
        confirmed=expected,
    )
    clock = _Clock()

    absent = _abort_created_resources(
        plan,
        sdk,
        created,
        expected=expected,
        deadline_at=100.0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert absent is False
    operations = [name for name, _value in sdk.calls]
    assert operations == ["observe"], "abort mutated a plan-identical concurrent app"
    assert sdk.apps[0].state == "deployed"
    assert sdk.volumes
    assert sdk.inference
    assert sdk.artifact


def test_abort_without_an_app_declines_unconfirmed_race_winner_resources() -> None:
    plan = build_modal_create_plan(_bundle(), phase="bootstrap")
    sdk = _FakeSdk(plan)
    sdk.inference = [ModalNamedResource("st-" + "R" * 22, plan.names.inference_secret)]
    sdk.artifact = [ModalNamedResource("st-" + "B" * 22, plan.names.artifact_secret)]
    sdk.volumes = [ModalNamedResource("vo-" + "R" * 22, plan.names.volume)]
    created = _CreatedResources(inference=True, artifact=True, volume=True)
    clock = _Clock()

    absent = _abort_created_resources(
        plan,
        sdk,
        created,
        expected=None,
        deadline_at=100.0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert absent is False
    operations = [name for name, _value in sdk.calls]
    assert operations == ["observe"], "abort deleted resources without confirmed ownership"
    assert sdk.volumes
    assert sdk.inference
    assert sdk.artifact


def test_abort_without_an_app_declines_an_unconfirmed_app_attempt() -> None:
    plan = build_modal_create_plan(_bundle(), phase="finalized")
    sdk = _FakeSdk(plan)
    created = _CreatedResources(app_deployed=True)
    clock = _Clock()

    absent = _abort_created_resources(
        plan,
        sdk,
        created,
        expected=None,
        deadline_at=100.0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert absent is False
    assert [name for name, _value in sdk.calls] == ["observe"]


def test_abort_without_an_app_deletes_resources_with_matching_confirmed_ids() -> None:
    plan = build_modal_create_plan(_bundle(), phase="bootstrap")
    sdk = _FakeSdk(plan)
    _seed_exact(sdk, artifact=True)
    sdk.apps = []
    expected = ExpectedResources(
        app_id=None,
        volume_id=VOLUME_ID,
        inference_secret_id=INFERENCE_SECRET_ID,
        artifact_secret_id=ARTIFACT_SECRET_ID,
    )
    created = _CreatedResources(
        inference=True,
        artifact=True,
        volume=True,
        confirmed=expected,
    )
    clock = _Clock()

    absent = _abort_created_resources(
        plan,
        sdk,
        created,
        expected=expected,
        deadline_at=100.0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert absent is True
    operations = [name for name, _value in sdk.calls]
    assert operations == [
        "observe",
        "delete_artifact",
        "delete_inference",
        "delete_volume",
        "observe",
    ]
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []


def test_abort_with_confirmed_ids_stops_waits_deletes_and_confirms_absence() -> None:
    plan = build_modal_create_plan(_bundle(), phase="bootstrap")
    sdk = _DelayedAbortSdk(plan)
    _seed_exact(sdk, artifact=True)
    expected = ExpectedResources(
        app_id=APP_ID,
        volume_id=VOLUME_ID,
        inference_secret_id=INFERENCE_SECRET_ID,
        artifact_secret_id=ARTIFACT_SECRET_ID,
    )
    created = _CreatedResources(
        inference=True,
        artifact=True,
        volume=True,
        app_deployed=True,
        confirmed=expected,
    )
    clock = _Clock()

    absent = _abort_created_resources(
        plan,
        sdk,
        created,
        expected=expected,
        deadline_at=100.0,
        clock=clock,
        sleep=clock.sleep,
    )

    assert absent is True
    operations = [name for name, _value in sdk.calls]
    assert operations == [
        "observe",
        "stop_app",
        "observe",
        "observe",
        "delete_artifact",
        "delete_inference",
        "delete_volume",
        "observe",
    ]
    assert sdk.apps[0].state == "stopped"
    assert sdk.volumes == []
    assert sdk.inference == []
    assert sdk.artifact == []


def test_interrupt_before_create_returns_declines_race_winner_teardown() -> None:
    class _InterruptBeforeCreateReturnsSdk(_FakeSdk):
        def create_inference_secret(
            self, plan, value: str, *, deadline_at=None
        ) -> ModalNamedResource:
            assert value == INFERENCE_SECRET
            self.plan = plan
            _seed_exact(self, artifact=True)
            raise KeyboardInterrupt

    factory = _Factory()
    factory.sdk_class = _InterruptBeforeCreateReturnsSdk

    interruption: KeyboardInterrupt | None = None
    try:
        _provision(_bundle(), factory)
    except KeyboardInterrupt as exc:
        interruption = exc
    except Exception as exc:
        pytest.fail(f"abort crashed before handling missing ownership evidence: {exc!r}")
    else:
        pytest.fail("the injected interrupt did not propagate")

    sdk = factory.sdk
    assert sdk is not None
    operations = [name for name, _value in sdk.calls]
    assert operations == ["observe"], "abort mutated without confirmed provider ids"
    assert isinstance(interruption, InterruptedProvisioning)
    assert sdk.apps[0].state == "deployed"
    assert sdk.volumes
    assert sdk.inference
    assert sdk.artifact


def test_abort_does_not_report_confirmed_cleanup_while_the_volume_remains() -> None:
    class _RetainedVolumeSdk(_FakeSdk):
        def delete_volume(self, plan, volume_id: str, *, deadline_at=None) -> None:
            self.calls.append(("delete_volume", None))

    factory = _Factory()
    factory.sdk_class = _RetainedVolumeSdk

    with pytest.raises(InterruptedProvisioning):
        _provision(_bundle(), factory, probe=_InterruptingProbe())

    sdk = factory.sdk
    assert sdk is not None
    assert sdk.volumes, "the test did not retain the volume it is meant to detect"
    operations = [name for name, _value in sdk.calls]
    assert operations.count("observe") >= 3
    assert "stop_app" in operations
    assert "delete_volume" in operations


def test_abort_waits_for_terminal_state_and_confirms_absence_before_success() -> None:
    factory = _Factory()
    factory.sdk_class = _DelayedAbortSdk
    clock = _Clock()

    with pytest.raises(KeyboardInterrupt) as raised:
        provision_modal_deployment(
            _bundle(),
            ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
            ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
            deadline_at=100.0,
            sdk_factory=factory,
            probe=_InterruptingProbe(),
            clock=clock,
            sleep=clock.sleep,
        )

    sdk = factory.sdk
    assert sdk is not None
    assert not isinstance(raised.value, InterruptedProvisioning)
    assert sdk.terminal_polls == 2, "abort deleted before modal reported a terminal app"
    assert clock.now == 2.0
    assert sdk.volumes == []
    operations = [name for name, _value in sdk.calls]
    assert operations.index("delete_volume") < len(operations) - 1
    assert operations[-1] == "observe", "abort trusted delete acknowledgement without re-observing"


def test_provision_succeeds_after_abort_reclaims_the_delayed_volume() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _DelayedAbortSdk(plan)

    def factory(credentials, received_plan, _deadline_at, _clock):
        assert credentials.reveal() == (PROVIDER_ID, PROVIDER_SECRET)
        assert received_plan.names == plan.names
        return sdk

    clock = _Clock()
    with pytest.raises(KeyboardInterrupt):
        provision_modal_deployment(
            bundle,
            ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
            ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
            deadline_at=100.0,
            sdk_factory=factory,
            probe=_InterruptingProbe(),
            clock=clock,
            sleep=clock.sleep,
        )

    assert sdk.volumes == []
    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=100.0,
        sdk_factory=factory,
        probe=_Probe(),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "ready", "a leftover abort volume blocked the next provision"
    assert [name for name, _value in sdk.calls].count("create_volume") == 2


def test_interrupting_a_slow_modal_readiness_poll_cleans_up_created_resources() -> None:
    """Ctrl-C after the creates succeed must not strand a live, billable Modal deployment.

    `KeyboardInterrupt` derives from BaseException, so neither `except ModalResourceConflict` nor
    `except ModalSdkFailure` sees it. Without an explicit handler the app, its volume, and its
    secrets stay live in the customer's own Modal account and keep billing, while the user reads
    the traceback as "it didn't happen".
    """
    factory = _Factory()

    with pytest.raises(KeyboardInterrupt):
        _provision(_bundle(), factory, probe=_InterruptingProbe())

    sdk = factory.sdk
    assert sdk is not None
    operations = [name for name, _payload in sdk.calls]
    assert "delete_volume" in operations, operations
    assert "delete_inference" in operations, operations
    assert not sdk.volumes, sdk.volumes
    assert not sdk.inference, sdk.inference
    assert not sdk.artifact, sdk.artifact


def test_interrupting_a_slow_modal_readiness_poll_stops_the_billing_app() -> None:
    """the deployed app is the gpu, and deleting only secrets and the volume leaves it running.

    `deploy_app` returns well before the readiness probe answers, so by the time a user gives up
    and presses Ctrl-C the compute is already live and billing. Cleanup that walks only the named
    resources it created hands the customer a running gpu app with no record that it exists.
    Compute must be stopped first, which is also the order canonical teardown uses.
    """
    factory = _Factory()

    with pytest.raises(KeyboardInterrupt):
        _provision(_bundle(), factory, probe=_InterruptingProbe())

    sdk = factory.sdk
    assert sdk is not None
    operations = [name for name, _payload in sdk.calls]
    assert "stop_app" in operations, operations
    assert operations.index("stop_app") < operations.index("delete_volume"), operations
    assert sdk.apps, "the deployed app must still be observable after the interrupt"
    assert sdk.apps[0].state == "stopped", sdk.apps


def test_a_failed_abort_delete_does_not_replace_the_interrupt() -> None:
    """cleanup runs from the interrupt handler, so its own failure must not become the exception.

    Modal refuses to delete a volume an app still holds attached, so a delete failing during abort
    is the expected case rather than an exotic one. If that error propagated it would replace the
    `KeyboardInterrupt` with what reads like an unrelated provider bug, and it would also skip
    whatever cleanup steps came after it.
    """

    class _VolumeDeleteFailsSdk(_FakeSdk):
        def __init__(self, plan) -> None:
            super().__init__(plan)
            self.fail_operation = "delete_volume"

    factory = _Factory()
    factory.sdk_class = _VolumeDeleteFailsSdk

    # the interrupt survives: pytest.raises would report the provider error instead.
    with pytest.raises(KeyboardInterrupt):
        _provision(_bundle(), factory, probe=_InterruptingProbe())

    sdk = factory.sdk
    assert sdk is not None
    operations = [name for name, _payload in sdk.calls]
    # the failing delete does not stop the app from being stopped, nor the secrets from going.
    assert "stop_app" in operations, operations
    assert "delete_inference" in operations, operations
    assert not sdk.inference, sdk.inference


def test_interrupting_after_the_modal_app_is_ready_leaves_the_deployment_standing() -> None:
    """Ctrl-C once the app has answered the probe must not delete a working deployment.

    After readiness the only work left is swapping the bootstrap phase for the finalized one.
    Tearing down there would destroy an app the user just waited to warm up, whereas a
    half-finalized deployment is recoverable by re-running the command.
    """

    class _InterruptOnFinalizeSdk(_FakeSdk):
        def deploy_app(self, plan, *, deadline_at=None) -> str:
            # the finalized redeploy only runs after the bootstrap phase probed ready.
            if plan.phase == "finalized":
                raise KeyboardInterrupt
            return super().deploy_app(plan)

    factory = _Factory()
    factory.sdk_class = _InterruptOnFinalizeSdk

    with pytest.raises(KeyboardInterrupt):
        _provision(_bundle(), factory)

    sdk = factory.sdk
    assert sdk is not None
    operations = [name for name, _payload in sdk.calls]
    assert "delete_volume" not in operations, operations
    assert "delete_inference" not in operations, operations
    assert sdk.volumes, "the ready deployment's volume must survive the interrupt"


def test_environment_web_suffix_is_part_of_the_expected_public_url() -> None:
    """Modal builds web urls as `<workspace>-<web_suffix>--<label>.modal.run`.

    The suffix is a per-environment field the operator sets; it is NOT the environment name and is
    not derivable from it (one environment per workspace may have none). Deriving the url from the
    workspace alone made every suffixed environment's expected url wrong, and because
    `_modal_resources.exact_core_resources` matches on `public_url`, the app deploys and is then
    rejected as not ours -- an identity mismatch, not an obviously bad url.
    """
    plan = build_modal_create_plan(_bundle(web_suffix="dev"))

    assert plan.expected_public_url == f"https://workspace-dev--{plan.endpoint_label}.modal.run"
    # the suffix also spends the 63-char dns budget, so the label must shrink to make room.
    unsuffixed = build_modal_create_plan(_bundle())
    assert unsuffixed.expected_public_url == (
        f"https://workspace--{unsuffixed.endpoint_label}.modal.run"
    )
    assert len(f"workspace-dev--{plan.endpoint_label}") <= 63


@pytest.mark.parametrize("bad", ["dev_test", "Dev", "-dev", "dev-", "dev.test", "dev test"])
def test_a_malformed_web_suffix_is_rejected_before_any_resource_is_created(bad: str) -> None:
    """A suffix that cannot appear in a hostname must fail while planning, not after billing.

    `web_suffix` is concatenated into the same hostname label as the workspace
    (`<workspace>-<suffix>--<label>.modal.run`). `validate_modal_placement` only required it to be
    nonempty, so `dev_test` produced an `expected_public_url` that no real Modal hostname can ever
    equal -- and `validate_modal_plan` accepted it. The mismatch would surface only when the probe
    compared against Modal's actual URL, by which point the secrets, volume, and app exist and the
    GPU is billing, and the run ends `outcome_unknown` with live resources.
    """
    with pytest.raises(ValueError, match="web_suffix"):
        build_modal_create_plan(_bundle(web_suffix=bad), phase="finalized")


def test_a_wellformed_web_suffix_still_builds_its_plan() -> None:
    """The guard must not reject the suffixes Modal actually issues."""
    plan = build_modal_create_plan(_bundle(web_suffix="dev-2"), phase="finalized")

    assert "-dev-2--" in plan.expected_public_url


def test_loopback_image_registries_are_rejected_before_any_modal_call() -> None:
    # `ServingImage` accepts a loopback registry on purpose: pulling from a local registry is
    # legitimate when the pull happens on the operator's machine. modal resolves the reference
    # inside its own build infrastructure instead, and nothing here uploads the image or opens a
    # tunnel back, so the pull would fail only after the app and its resources already exist and
    # bill. the plan builder is the last point where rejecting it costs nothing.
    original = _bundle()
    for registry in (
        "localhost",
        "localhost:5000",
        "registry.localhost",
        "localhost.localdomain",
        "local",
        "registry.local",
        "registry.local:5000",
        "127.0.0.1",
        "10.20.30.40",
        "169.254.1.2",
    ):
        bundle = DeploymentBundle(
            spec=original.spec,
            manifest=original.manifest,
            image=ServingImage(
                reference=f"{registry}/flash/serve@{original.image.digest}",
                digest=original.image.digest,
            ),
        )
        with pytest.raises(ValueError, match=r"loopback|private"):
            build_modal_create_plan(bundle)

    assert build_modal_create_plan(original)

    reachable = DeploymentBundle(
        spec=original.spec,
        manifest=original.manifest,
        image=ServingImage(
            reference=f"notlocalhost.example/flash/serve@{original.image.digest}",
            digest=original.image.digest,
        ),
    )
    assert build_modal_create_plan(reachable)


def test_a_failed_abort_delete_reports_that_cleanup_was_not_confirmed() -> None:
    """a suppressed teardown failure must still reach the caller as ambiguity.

    every abort step is suppressed individually so one failure cannot stop the rest, which used to
    mean nobody held the knowledge that a step failed. the cli then printed "aborted", which reads
    as "nothing was created" while the gpu is still live and billing. the interrupt still
    propagates -- `InterruptedProvisioning` subclasses `KeyboardInterrupt` -- but it now names the
    provider so the cli can warn before the generic handler exits 130.
    """

    class _StopAppFailsSdk(_FakeSdk):
        def __init__(self, plan) -> None:
            super().__init__(plan)
            self.fail_operation = "stop_app"

    factory = _Factory()
    factory.sdk_class = _StopAppFailsSdk

    with pytest.raises(InterruptedProvisioning) as raised:
        _provision(_bundle(), factory, probe=_InterruptingProbe())

    assert raised.value.provider == "modal"
    # still an interrupt, so the existing cli handler keeps exiting 130 rather than tracebacking.
    assert isinstance(raised.value, KeyboardInterrupt)


@pytest.mark.parametrize(
    ("failing_call", "expected_operations"),
    [
        (
            "create_artifact_secret",
            ["observe", "create_inference", "create_artifact", "observe"],
        ),
        (
            "create_volume",
            ["observe", "create_inference", "create_artifact", "create_volume", "observe"],
        ),
    ],
)
def test_a_create_failure_after_acceptance_stays_ambiguous_without_returned_ids(
    failing_call: str,
    expected_operations: list[str],
) -> None:
    sdk_holder: list[_FakeSdk] = []

    class _FailAfterAcceptSdk(_FakeSdk):
        def __init__(self, plan) -> None:
            super().__init__(plan)
            sdk_holder.append(self)

    original = getattr(_FailAfterAcceptSdk, failing_call)

    def fail_after_accept(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise ModalSdkFailure("provider_rejected")

    setattr(_FailAfterAcceptSdk, failing_call, fail_after_accept)
    factory = _Factory()
    factory.sdk_class = _FailAfterAcceptSdk

    result, _probe = _provision(_bundle(), factory)
    sdk = sdk_holder[0]

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert [name for name, _payload in sdk.calls] == expected_operations
    assert sdk.inference
    assert sdk.artifact
    if failing_call == "create_volume":
        assert sdk.volumes


def test_volume_create_interrupt_cleans_the_previously_confirmed_secrets() -> None:
    class _InterruptBeforeVolumeCreateSdk(_FakeSdk):
        def create_volume(self, plan, *, deadline_at=None) -> ModalNamedResource:
            self.calls.append(("create_volume", None))
            raise KeyboardInterrupt

    factory = _Factory()
    factory.sdk_class = _InterruptBeforeVolumeCreateSdk

    with pytest.raises(KeyboardInterrupt) as raised:
        _provision(_bundle(), factory)

    assert not isinstance(raised.value, InterruptedProvisioning)
    sdk = factory.sdk
    assert sdk is not None
    operations = [name for name, _payload in sdk.calls]
    assert operations == [
        "observe",
        "create_inference",
        "create_artifact",
        "create_volume",
        "observe",
        "delete_artifact",
        "delete_inference",
        "observe",
    ]
    assert "delete_volume" not in operations
    assert sdk.inference == []
    assert sdk.artifact == []
    assert sdk.volumes == []


@pytest.mark.parametrize("interrupted_call", ["create_volume", "create_inference_secret"])
def test_a_create_interrupted_after_the_provider_accepted_it_stays_ambiguous_without_ownership(
    interrupted_call: str,
) -> None:
    """the window between Modal accepting a create and the caller seeing its handle.

    marking before the call preserves knowledge that a create may have landed, so abort cannot
    report clean absence. the missing returned id also means ownership is unproved: deleting the
    deterministic name could destroy a same-generation race winner, so the resource stays for later
    proof-based reclaim and the user receives an ambiguous interruption.
    """

    factory = _Factory()
    sdk_holder: list[_FakeSdk] = []

    class _InterruptAfterAcceptSdk(_FakeSdk):
        def __init__(self, plan) -> None:
            super().__init__(plan)
            sdk_holder.append(self)

    def _interrupt_after(name: str) -> None:
        original = getattr(_InterruptAfterAcceptSdk, name)

        def wrapper(self, *args, **kwargs):
            original(self, *args, **kwargs)  # the resource now exists in the provider
            raise KeyboardInterrupt  # ...before its handle reaches the caller

        setattr(_InterruptAfterAcceptSdk, name, wrapper)

    _interrupt_after(interrupted_call)
    factory.sdk_class = _InterruptAfterAcceptSdk

    with pytest.raises(InterruptedProvisioning):
        _provision(_bundle(), factory)

    sdk = sdk_holder[0]
    operations = [name for name, _payload in sdk.calls]
    assert operations[-1] == "observe"
    assert "delete_volume" not in operations
    assert "delete_inference" not in operations
    if interrupted_call == "create_volume":
        assert sdk.volumes, "the accepted volume was not retained as the ambiguous resource"
    else:
        assert sdk.inference, "the accepted secret was not retained as the ambiguous resource"


def test_a_fully_confirmed_abort_leaves_the_interrupt_unchanged() -> None:
    # the carrier must mean "cleanup could not be confirmed", not merely "an interrupt happened".
    # when every teardown step succeeds there is nothing ambiguous to report, and raising the
    # carrier anyway would warn about billing resources that were provably removed.
    factory = _Factory()

    with pytest.raises(KeyboardInterrupt) as raised:
        _provision(_bundle(), factory, probe=_InterruptingProbe())

    assert not isinstance(raised.value, InterruptedProvisioning)
