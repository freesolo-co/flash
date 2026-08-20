"""offline modal serving lifecycle, identity, and secret-boundary coverage."""

from __future__ import annotations

import email.message
import io
import json
import re
import subprocess
import sys
import urllib.response
from dataclasses import replace
from pathlib import Path

import pytest

from flash.serve.app.manifest import build_serving_manifest
from flash.serve.control import (
    DeploymentSpec,
    ModalCredentials,
    ModalPlacement,
    ModalProviderHandle,
    sanitized_dict,
)
from flash.serve.provisioning import DeploymentBundle, ServingImage, ServingRuntimeSecrets
from flash.serve.provisioning._modal_plan import (
    MODAL_APP_TAG_LIMIT,
    MODAL_DEPLOYMENT_TAG_LIMIT,
    build_modal_create_plan,
)
from flash.serve.provisioning._modal_probe import ModalEndpointProbe, _provenance_matches
from flash.serve.provisioning._modal_sdk import (
    ModalAppObservation,
    ModalNamedResource,
    ModalObservation,
    ModalSdkFailure,
)
from flash.serve.provisioning.modal import (
    confirm_modal_absence,
    provision_modal_deployment,
    reconcile_modal_deployment,
    resize_modal_volume,
    teardown_modal_deployment,
)
from tests.test_serve_app_manifest import _spec_and_inputs

PROVIDER_ID = "provider-id-sentinel"
PROVIDER_SECRET = "provider-secret-sentinel"
INFERENCE_SECRET = "inference-secret-sentinel"
ARTIFACT_SECRET = "artifact-secret-sentinel"
APP_ID = "ap-" + "A" * 22
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
) -> DeploymentBundle:
    original, inputs = _spec_and_inputs()
    spec = DeploymentSpec(
        deployment_id=original.deployment_id,
        generation=original.generation,
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


def _models_payload(bundle: DeploymentBundle) -> dict[str, object]:
    manifest = bundle.manifest
    by_revision = {adapter.adapter_revision: adapter for adapter in manifest.adapters}
    mapping = {revision: revision for revision in by_revision}
    mapping.update(manifest.aliases)
    data = []
    for model_id, revision in sorted(mapping.items()):
        adapter = by_revision[revision]
        data.append(
            {
                "id": model_id,
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
                    "requested_model": model_id,
                    "adapter_revision": revision,
                    "checkpoint": adapter.checkpoint,
                    "source_revision": adapter.source_revision,
                    "source_subfolder": adapter.source_subfolder,
                    "aggregate_sha256": adapter.aggregate_sha256,
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

    def observe(self, plan, *, app_id_hint=None) -> ModalObservation:
        self.calls.append(("observe", app_id_hint))
        return ModalObservation(
            workspace_name=self.workspace_name,
            environment_name=self.environment_name,
            apps=tuple(self.apps),
            volumes=tuple(self.volumes),
            inference_secrets=tuple(self.inference),
            artifact_secrets=tuple(self.artifact),
        )

    def create_inference_secret(self, plan, value: str) -> ModalNamedResource:
        self.calls.append(("create_inference", value == INFERENCE_SECRET))
        self._fail("create_inference")
        resource = ModalNamedResource(INFERENCE_SECRET_ID, plan.names.inference_secret)
        self.inference.append(resource)
        return resource

    def create_artifact_secret(self, plan, value: str) -> ModalNamedResource:
        self.calls.append(("create_artifact", value == ARTIFACT_SECRET))
        self._fail("create_artifact")
        resource = ModalNamedResource(ARTIFACT_SECRET_ID, plan.names.artifact_secret)
        self.artifact.append(resource)
        return resource

    def create_volume(self, plan) -> ModalNamedResource:
        self.calls.append(("create_volume", None))
        self._fail("create_volume")
        resource = ModalNamedResource(VOLUME_ID, plan.names.volume)
        self.volumes.append(resource)
        return resource

    def deploy_app(self, plan) -> str:
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

    def stop_app(self, plan) -> None:
        self.calls.append(("stop_app", None))
        self._fail("stop_app")
        app = self.apps[0]
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

    def delete_secret(self, plan, name: str) -> None:
        role = "artifact" if name == plan.names.artifact_secret else "inference"
        self.calls.append((f"delete_{role}", None))
        self._fail(f"delete_{role}")
        if role == "artifact":
            self.artifact.clear()
        else:
            self.inference.clear()

    def delete_volume(self, plan) -> None:
        self.calls.append(("delete_volume", None))
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

    def __call__(self, credentials: ModalCredentials, plan) -> _FakeSdk:
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

    assert "flash-launcher" not in dict(build_modal_create_plan(_bundle()).tags)


def test_plan_is_complete_secret_free_and_binds_launcher_abi(monkeypatch) -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    bootstrap = build_modal_create_plan(bundle, phase="bootstrap")
    rendered = repr(plan)
    assert plan.phase == "finalized"
    assert bootstrap.phase == "bootstrap"
    assert dict(plan.tags)["flash-phase"] == "finalized"
    assert dict(bootstrap.tags)["flash-phase"] == "bootstrap"
    assert plan.names == bootstrap.names
    assert plan.gpu_request == "B200:1"
    assert plan.expected_public_url == (f"https://workspace--{plan.endpoint_label}.modal.run")
    assert plan.include_source is False
    assert plan.min_containers == 0
    assert plan.max_containers == 1
    assert plan.buffer_containers == 0
    assert plan.scaledown_window_seconds > 0
    assert plan.environment
    assert "FLASH_SERVING_MANIFEST" in dict(plan.environment)
    assert "FLASH_SERVING_CACHE_ROOT" in dict(plan.environment)
    assert all(
        secret not in rendered
        for secret in (PROVIDER_ID, PROVIDER_SECRET, INFERENCE_SECRET, ARTIFACT_SECRET)
    )

    from flash.serve.provisioning import _modal_plan

    original = plan.names
    original_topology = dict(plan.tags)["flash-topology"]
    monkeypatch.setattr(_modal_plan, "LAUNCHER_ABI_ID", "fsla1-" + "a" * 43)
    changed = build_modal_create_plan(bundle)
    assert changed.names != original
    # the abi no longer travels as its own tag, so `flash-topology` is what carries it into the
    # tag set. without this, dropping `flash-launcher` would let two different launcher abis
    # produce byte-identical tags, and `deployed_app_matches` compares tags to decide whether an
    # existing app may be adopted rather than replaced.
    assert dict(changed.tags)["flash-topology"] != original_topology


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

    rendered = json.dumps(sanitized_dict(result), sort_keys=True) + repr(result) + repr(sdk.calls)
    assert factory.calls == [(True, True)]
    assert probe.calls[0][1] is True
    assert ("create_inference", True) in sdk.calls
    assert ("create_artifact", True) in sdk.calls
    for secret in (PROVIDER_ID, PROVIDER_SECRET, INFERENCE_SECRET, ARTIFACT_SECRET):
        assert secret not in rendered
    assert "FREESOLO_INTERNAL_KEY" not in rendered


def test_exact_adoption_requires_authenticated_endpoint_provenance() -> None:
    bundle = _bundle()
    factory = _Factory()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk)

    def seeded_factory(credentials, received_plan):
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


def test_existing_bootstrap_is_only_reconciled_and_never_blindly_finalized() -> None:
    bundle = _bundle()
    bootstrap_plan = build_modal_create_plan(bundle, phase="bootstrap")
    sdk = _FakeSdk(bootstrap_plan)
    handle = _seed_exact(sdk, artifact=True)
    clock = _Clock()

    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=4.0,
        sdk_factory=lambda _credentials, _plan: sdk,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.status == "outcome_unknown"
    assert result.handle == handle
    assert all(name == "observe" for name, _value in sdk.calls)
    assert sdk.artifact


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
        sdk_factory=lambda _credentials, _plan: sdk,
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
        sdk_factory=lambda _credentials, _plan: sdk,
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

    def failing_factory(credentials, plan):
        sdk = factory(credentials, plan)
        sdk.fail_operation = "create_volume"
        return sdk

    clock = _Clock()
    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
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
    assert PROVIDER_SECRET not in json.dumps(sanitized_dict(result))


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
        sdk_factory=lambda _credentials, _plan: sdk,
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
        sdk_factory=lambda _credentials, _plan: sdk,
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
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan: sdk,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert absent.status == "absent"
    assert all(name == "observe" for name, _value in sdk.calls)


def test_artifact_cleanup_ambiguity_keeps_handle_and_never_retries() -> None:
    bundle = _bundle()
    factory = _Factory()

    def failing_factory(credentials, plan):
        sdk = factory(credentials, plan)
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
    assert [name for name, _value in sdk.calls].count("delete_artifact") == 1


def test_finalization_ambiguity_is_reconciled_read_only_without_artifact_deletion() -> None:
    bundle = _bundle()
    factory = _Factory()

    def failing_factory(credentials, plan):
        sdk = factory(credentials, plan)
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
        def delete_secret(self, received_plan, name: str) -> None:
            super().delete_secret(received_plan, name)
            if name == received_plan.names.artifact_secret:
                self.volumes = [ModalNamedResource("vo-" + "D" * 22, received_plan.names.volume)]

    sdk = DriftingCleanupSdk(plan)
    result = provision_modal_deployment(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan: sdk,
        probe=_Probe(True),
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )
    assert result.status == "outcome_unknown"
    assert [name for name, _value in sdk.calls].count("delete_artifact") == 1


def test_modal_resize_is_fixed_invalid_request_before_sdk_construction() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)
    sdk = _FakeSdk(plan)
    handle = _seed_exact(sdk)
    calls = 0

    def forbidden_factory(_credentials, _plan):
        nonlocal calls
        calls += 1
        raise AssertionError("modal resize must not construct a client")

    for size in (1, 100, 10_000):
        result = resize_modal_volume(
            bundle,
            handle,
            ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
            size,
            sdk_factory=forbidden_factory,
        )
        assert result.status == "failed"
        assert result.error_code == "invalid_request"
    assert calls == 0


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
        sdk_factory=lambda _credentials, _plan: sdk,
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
    confirmed = confirm_modal_absence(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan: sdk,
        clock=clock,
    )
    assert confirmed.status == "failed"
    assert confirmed.error_code == "conflict"

    sdk.apps.clear()
    sdk.calls.clear()
    confirmed = confirm_modal_absence(
        bundle,
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        deadline_at=100.0,
        sdk_factory=lambda _credentials, _plan: sdk,
        clock=clock,
    )
    assert confirmed.status == "absent"
    assert all(name == "observe" for name, _value in sdk.calls)


def test_teardown_post_stop_observation_failure_is_outcome_unknown() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)

    class FailingObservationSdk(_FakeSdk):
        def observe(self, plan, *, app_id_hint=None) -> ModalObservation:
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
        sdk_factory=lambda _credentials, _plan: sdk,
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
        def stop_app(self, received_plan) -> None:
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
        sdk_factory=lambda _credentials, _plan: sdk,
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
        sdk_factory=lambda _credentials, _plan: sdk,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )
    assert result.status == "outcome_unknown"
    assert [name for name, _value in sdk.calls if name.startswith("delete_")] == []


def test_teardown_refuses_terminal_app_with_retained_finalized_tags() -> None:
    bundle = _bundle()
    plan = build_modal_create_plan(bundle)

    class RetainedTagsSdk(_FakeSdk):
        def stop_app(self, received_plan) -> None:
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
        sdk_factory=lambda _credentials, _plan: sdk,
        clock=_Clock(),
        sleep=lambda _seconds: None,
    )

    assert result.status == "outcome_unknown"
    assert result.status != "absent"
    assert [name for name, _value in sdk.calls if name.startswith("delete_")] == []
    assert sdk.apps[0].tags == plan.tags
    assert sdk.volumes
    assert sdk.inference
    assert sdk.artifact


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
        sdk_factory=lambda _credentials, _plan: sdk,
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

    def factory(_credentials, _plan):
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

    def failing_factory(credentials, plan):
        sdk = factory(credentials, plan)
        sdk.fail_operation = "deploy_finalized"
        return sdk

    result, _probe = _provision(bundle, failing_factory, artifact_token=None)
    rendered = repr(result) + json.dumps(sanitized_dict(result))
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


def test_modal_endpoint_provenance_requires_exact_ids_aliases_and_full_mapping() -> None:
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

    wrong_revision = json.loads(json.dumps(valid))
    wrong_revision["data"][0]["flash_provenance"]["adapter_revision"] = "wrong"
    cases.append(wrong_revision)

    wrong_global = json.loads(json.dumps(valid))
    wrong_global["data"][0]["flash_provenance"]["logical_base_revision"] = "0" * 40
    cases.append(wrong_global)

    unexpected_provenance = json.loads(json.dumps(valid))
    unexpected_provenance["data"][0]["flash_provenance"]["extra"] = True
    cases.append(unexpected_provenance)

    assert all(_provenance_matches(case, bundle) is False for case in cases)


def test_modal_handle_rejects_malformed_role_ids_at_creation_and_serialization() -> None:
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

    object.__setattr__(handle, "app_id", "ap-short")
    with pytest.raises(ValueError, match="pinned provider contract"):
        sanitized_dict(handle)


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
from flash.serve.provisioning._modal_plan import build_modal_create_plan
from flash.serve.provisioning.modal import provision_modal_deployment

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


def test_interrupting_after_the_modal_app_is_ready_leaves_the_deployment_standing() -> None:
    """Ctrl-C once the app has answered the probe must not delete a working deployment.

    After readiness the only work left is swapping the bootstrap phase for the finalized one.
    Tearing down there would destroy an app the user just waited to warm up, whereas a
    half-finalized deployment is recoverable by re-running the command.
    """

    class _InterruptOnFinalizeSdk(_FakeSdk):
        def deploy_app(self, plan) -> str:
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
