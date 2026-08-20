"""offline persistent runpod pod lifecycle and secret-boundary coverage."""

from __future__ import annotations

import email.message
import io
import json
import urllib.error
import urllib.response
from dataclasses import replace

import pytest

from flash.serve.app.manifest import build_serving_manifest
from flash.serve.control import (
    DeploymentSpec,
    RunPodCredentials,
    RunPodPlacement,
    RunPodProviderHandle,
    sanitized_dict,
)
from flash.serve.provisioning import DeploymentBundle, ServingImage, ServingRuntimeSecrets, _common
from flash.serve.provisioning._common import serving_resource_names
from flash.serve.provisioning._runpod_plan import build_runpod_create_plan
from flash.serve.provisioning._runpod_probe import RunPodEndpointProbe
from flash.serve.provisioning._runpod_protocol import (
    CREATE_SECRET,
    DELETE_SECRET,
    PROXY_PORT_SPEC,
    parse_deleted_secret,
    parse_pods,
    parse_templates,
    pod_payload,
    template_payload,
    volume_payload,
)
from flash.serve.provisioning._runpod_transport import (
    GRAPHQL_URL,
    REST_BASE_URL,
    USER_AGENT,
    RunPodTransportFailure,
    StdlibRunPodTransport,
    build_no_redirect_opener,
)
from flash.serve.provisioning.runpod import (
    confirm_runpod_absence,
    grow_runpod_volume,
    provision_runpod_deployment,
    reconcile_runpod_deployment,
    teardown_runpod_deployment,
)
from tests.test_serve_app_manifest import _spec_and_inputs

PROVIDER_SECRET = "provider-api-secret-sentinel"
INFERENCE_SECRET = "inference-secret-sentinel"
ARTIFACT_SECRET = "artifact-secret-sentinel"
POD_ID = "abc123def4567"


def _bundle() -> DeploymentBundle:
    modal_spec, inputs = _spec_and_inputs()
    spec = DeploymentSpec(
        deployment_id=modal_spec.deployment_id,
        generation=modal_spec.generation,
        provider="runpod",
        placement=RunPodPlacement(
            account_id="account-01",
            gpu_type_id="NVIDIA B200",
            gpu_count=1,
            data_center_id="US-KS-2",
            container_disk_gb=50,
            volume_size_gb=100,
        ),
        engine=modal_spec.engine,
        adapters=modal_spec.adapters,
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


def _oversized_bundle() -> DeploymentBundle:
    modal_spec, inputs = _spec_and_inputs()
    oversized_default = json.dumps(
        {"payload": "x" * (_common.MAX_CANONICAL_MANIFEST_BYTES + 1)},
        separators=(",", ":"),
        sort_keys=True,
    )
    adapter = replace(
        modal_spec.adapters[0],
        structured_outputs_default_json=oversized_default,
    )
    spec = DeploymentSpec(
        deployment_id=modal_spec.deployment_id,
        generation=modal_spec.generation,
        provider="runpod",
        placement=RunPodPlacement(
            account_id="account-01",
            gpu_type_id="NVIDIA B200",
            gpu_count=1,
            data_center_id="US-KS-2",
            container_disk_gb=50,
            volume_size_gb=100,
        ),
        engine=modal_spec.engine,
        adapters=(adapter,),
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


def _names(bundle: DeploymentBundle):
    return serving_resource_names(
        bundle.spec.deployment_id,
        bundle.spec.generation,
        bundle.spec.engine.engine_id,
        workload_role="pod",
    )


def _models_payload(bundle: DeploymentBundle) -> dict[str, object]:
    return {
        "data": [
            {
                "id": bundle.spec.adapters[0].adapter_revision,
                "flash_provenance": {
                    "deployment_id": bundle.spec.deployment_id,
                    "spec_id": bundle.spec.spec_id,
                    "manifest_id": bundle.manifest.manifest_id,
                    "engine_id": bundle.spec.engine.engine_id,
                    "image_digest": bundle.image.digest,
                },
            }
        ]
    }


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


class _Factory:
    def __init__(self, transport: _FakeTransport) -> None:
        self.transport = transport
        self.accepted_keys: list[bool] = []

    def __call__(self, api_key: str) -> _FakeTransport:
        self.accepted_keys.append(api_key == PROVIDER_SECRET)
        return self.transport


class _FakeTransport:
    def __init__(self, account_id: str = "account-01") -> None:
        self.account_id = account_id
        self.secrets: list[dict[str, object]] = []
        self.templates: list[dict[str, object]] = []
        self.volumes: list[dict[str, object]] = []
        self.pods: list[dict[str, object]] = []
        self.calls: list[tuple[str, str, bool, object]] = []
        self.queries: list[tuple[str, dict[str, str]]] = []
        self.mutation_count = 0
        self.fail_mutation_at: int | None = None
        self.failure_mode = "ambiguous_before"
        self.malformed_pod_id = False

    def graphql(
        self,
        document: str,
        variables: dict[str, object],
        *,
        mutation: bool,
        deadline_at: float,
    ) -> object:
        # match runpod's real mutation names. keying off the old ones made this double answer a
        # document runpod itself rejects, so the whole suite passed against a schema that does
        # not exist -- the create branch is selected by name, so a renamed field would silently
        # fall through to the delete branch here rather than failing.
        operation = (
            "secretCreate"
            if "secretCreate" in document
            else "secretDelete"
            if "secretDelete" in document
            else "myself"
        )
        self._record("graphql", operation, mutation, dict(variables))
        if not mutation:
            return {
                "data": {
                    "myself": {
                        "id": self.account_id,
                        "secrets": [dict(item) for item in self.secrets],
                    }
                }
            }
        self._begin_mutation()
        if operation == "secretCreate":
            created = {
                "id": f"secret{len(self.secrets) + 1:02d}",
                "name": variables["name"],
            }
            self.secrets.append(created)
            response = {"data": {"secretCreate": dict(created)}}
        else:
            secret_id = variables["id"]
            self.secrets = [item for item in self.secrets if item["id"] != secret_id]
            # secretDelete's type is Void, so a success is null rather than true.
            response = {"data": {"secretDelete": None}}
        self._end_mutation()
        return response

    def rest(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None,
        *,
        mutation: bool,
        deadline_at: float,
        query: dict[str, str] | None = None,
    ) -> object:
        # the real transport takes query parameters as a mapping, and runpod gates whole response
        # objects behind them (includeMachine). a double without this parameter raises TypeError
        # on every call, which the lifecycle then classifies as transport_failed -- so each such
        # test would still "fail closed" and look like it was exercising its own scenario.
        self.queries.append((f"{method} {path}", dict(query or {})))
        self._record(
            "rest", f"{method} {path}", mutation, None if payload is None else dict(payload)
        )
        if method == "GET":
            return self._list(path)
        self._begin_mutation()
        if method == "POST":
            assert payload is not None
            response = self._create(path, payload)
        elif method == "PATCH":
            assert payload is not None
            volume_id = path.rsplit("/", 1)[-1]
            response = None
            for volume in self.volumes:
                if volume["id"] == volume_id:
                    volume["size"] = payload["size"]
                    response = dict(volume)
                    break
            if response is None:
                raise AssertionError("unknown fake volume")
        elif method == "DELETE":
            self._delete(path)
            response = None
        else:
            raise AssertionError("unexpected fake rest operation")
        self._end_mutation()
        return response

    def _record(self, kind: str, operation: str, mutation: bool, payload: object) -> None:
        recorded = payload
        if operation == "secretCreate" and type(payload) is dict:
            value = payload.get("value")
            recorded = {
                "name": payload.get("name"),
                "value_present": type(value) is str and bool(value),
                "value_is_expected": value in {INFERENCE_SECRET, ARTIFACT_SECRET},
            }
        self.calls.append((kind, operation, mutation, recorded))

    def _begin_mutation(self) -> None:
        self.mutation_count += 1
        if self.mutation_count != self.fail_mutation_at:
            return
        if self.failure_mode == "definite_before":
            raise RunPodTransportFailure("provider_rejected")
        if self.failure_mode == "ambiguous_before":
            raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)

    def _end_mutation(self) -> None:
        if self.mutation_count != self.fail_mutation_at:
            return
        if self.failure_mode == "definite_after":
            raise RunPodTransportFailure("provider_rejected")
        if self.failure_mode == "ambiguous_after":
            raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)

    def _list(self, path: str) -> list[dict[str, object]]:
        selected = {
            "/templates": self.templates,
            "/networkvolumes": self.volumes,
            "/pods": self.pods,
        }[path]
        return [dict(item) for item in selected]

    def _create(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        if path == "/networkvolumes":
            created = {"id": "volume01", **payload}
            self.volumes.append(created)
            return dict(created)
        if path == "/templates":
            created = {"id": "template01", **payload}
            self.templates.append(created)
            return dict(created)
        if path == "/pods":
            created = {
                "id": POD_ID,
                "name": payload["name"],
                "desiredStatus": "RUNNING",
                "imageName": payload["imageName"],
                "gpuTypeId": payload["gpuTypeIds"][0],
                "gpuCount": payload["gpuCount"],
                "dataCenterId": payload["dataCenterIds"][0],
                "containerDiskInGb": payload["containerDiskInGb"],
                "networkVolumeId": payload["networkVolumeId"],
                "templateId": payload["templateId"],
                "ports": payload["ports"],
            }
            self.pods.append(created)
            response = dict(created)
            if self.malformed_pod_id:
                response["id"] = "malformed"
            return response
        raise AssertionError("unexpected fake create path")

    def _delete(self, path: str) -> None:
        resource, resource_id = path.strip("/").split("/", 1)
        if resource == "pods":
            self.pods = [item for item in self.pods if item["id"] != resource_id]
        elif resource == "templates":
            self.templates = [item for item in self.templates if item["id"] != resource_id]
        elif resource == "networkvolumes":
            if self.pods:
                raise AssertionError("network volume deleted while a pod remained attached")
            self.volumes = [item for item in self.volumes if item["id"] != resource_id]
        else:
            raise AssertionError("unexpected fake delete path")


def _seed_exact(
    transport: _FakeTransport,
    bundle: DeploymentBundle,
    *,
    status: str = "RUNNING",
    artifact_secret: bool = False,
) -> RunPodProviderHandle:
    names = _names(bundle)
    transport.secrets = [{"id": "secret01", "name": names.inference_secret}]
    if artifact_secret:
        transport.secrets.append({"id": "secret02", "name": names.artifact_secret})
    volume = {"id": "volume01", **volume_payload(bundle, names)}
    template = {
        "id": "template01",
        **template_payload(
            bundle,
            names,
            include_artifact_secret=artifact_secret,
        ),
    }
    pod_request = pod_payload(
        bundle,
        names,
        template_id="template01",
        volume_id="volume01",
    )
    pod = {
        "id": POD_ID,
        "name": pod_request["name"],
        "desiredStatus": status,
        "imageName": pod_request["imageName"],
        "gpuTypeId": pod_request["gpuTypeIds"][0],
        "gpuCount": pod_request["gpuCount"],
        "dataCenterId": pod_request["dataCenterIds"][0],
        "containerDiskInGb": pod_request["containerDiskInGb"],
        "networkVolumeId": pod_request["networkVolumeId"],
        "templateId": pod_request["templateId"],
        "ports": pod_request["ports"],
    }
    transport.volumes = [volume]
    transport.templates = [template]
    transport.pods = [pod]
    placement = bundle.spec.placement
    assert type(placement) is RunPodPlacement
    return RunPodProviderHandle(
        deployment_id=bundle.spec.deployment_id,
        generation=bundle.spec.generation,
        engine_id=bundle.spec.engine.engine_id,
        account_id=placement.account_id,
        pod_id=POD_ID,
        pod_name=names.app_or_pod,
        network_volume_id="volume01",
        network_volume_name=names.volume,
        template_id="template01",
        template_name=names.template,
        inference_secret_id="secret01",
        inference_secret_name=names.inference_secret,
        data_center_id=placement.data_center_id,
        image_digest=bundle.image.digest,
        public_url=f"https://{POD_ID}-8000.proxy.runpod.net",
    )


def _mutation_calls(transport: _FakeTransport) -> list[tuple[str, str, bool, object]]:
    return [call for call in transport.calls if call[2]]


def _provision(
    bundle: DeploymentBundle,
    transport: _FakeTransport,
    *,
    artifact_token: str | None = ARTIFACT_SECRET,
    probe: _Probe | None = None,
):
    clock = _Clock()
    selected_probe = probe or _Probe()
    factory = _Factory(transport)
    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, artifact_token),
        deadline_at=100.0,
        transport_factory=factory,
        probe=selected_probe,
        clock=clock,
        sleep=clock.sleep,
    )
    return result, factory, selected_probe


def test_invalid_inputs_fail_before_client_construction() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    factory = _Factory(transport)
    clock = _Clock()

    with pytest.raises(ValueError, match="credential type"):
        provision_runpod_deployment(
            bundle,
            object(),
            ServingRuntimeSecrets(INFERENCE_SECRET),
            deadline_at=100.0,
            transport_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        )
    with pytest.raises(ValueError, match="future"):
        provision_runpod_deployment(
            bundle,
            RunPodCredentials(PROVIDER_SECRET),
            ServingRuntimeSecrets(INFERENCE_SECRET),
            deadline_at=0.0,
            transport_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        )
    assert factory.accepted_keys == []
    assert transport.calls == []


def test_oversized_manifest_fails_before_secret_reveal_or_provider_construction(
    monkeypatch,
) -> None:
    bundle = _oversized_bundle()
    transport = _FakeTransport()
    factory = _Factory(transport)
    clock = _Clock()
    revealed = False

    def reveal(_self):
        nonlocal revealed
        revealed = True
        raise AssertionError("secret revelation must not occur")

    monkeypatch.setattr(ServingRuntimeSecrets, "_reveal_for_launch", reveal)
    with pytest.raises(ValueError, match=r"manifest.*byte limit"):
        provision_runpod_deployment(
            bundle,
            RunPodCredentials(PROVIDER_SECRET),
            ServingRuntimeSecrets(INFERENCE_SECRET),
            deadline_at=100.0,
            transport_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        )
    assert revealed is False
    assert factory.accepted_keys == []
    assert transport.calls == []


def test_pod_creation_constrains_the_host_cuda_version() -> None:
    # the serving image's vllm ships a compiled extension linked against libcudart.so.13. runpod
    # documents allowedCudaVersions as "if not set, any CUDA version is acceptable", so omitting it
    # let it place a pod on an L4 host reporting driver 12080: the container died at engine init and
    # restarted forever, which externally is indistinguishable from a slow image pull. asking for a
    # gpu type does not ask for a driver, so the constraint has to be stated on every create path.
    bundle = _bundle()
    names = serving_resource_names(
        bundle.spec.deployment_id,
        bundle.spec.generation,
        bundle.spec.engine.engine_id,
        workload_role="pod",
    )

    direct = pod_payload(bundle, names, template_id="template01", volume_id="volume01")
    planned = json.loads(build_runpod_create_plan(bundle).pod_static_json)

    assert direct["allowedCudaVersions"] == ["13.0"]
    assert planned["allowedCudaVersions"] == ["13.0"]


def test_exact_happy_create_uses_one_pod_digest_volume_and_proxy_url() -> None:
    bundle = _bundle()
    transport = _FakeTransport()

    result, factory, probe = _provision(bundle, transport)

    assert result.status == "ready"
    assert result.handle is not None
    assert result.handle.pod_id == POD_ID
    # the pod listing must ask for the nested objects. runpod returns no gpu type or data center
    # without these flags, and the fake serves flat rows either way, so nothing else here would
    # notice the caller dropping them -- while against the real api observation would fail.
    assert ("GET /pods", {"includeMachine": "true", "includeNetworkVolume": "true"}) in (
        transport.queries
    )
    assert result.handle.public_url == f"https://{POD_ID}-8000.proxy.runpod.net"
    assert result.handle.image_digest == bundle.image.digest
    assert result.handle.account_id == bundle.spec.placement.account_id
    assert factory.accepted_keys == [True]
    assert probe.calls == [(result.handle.public_url, True, bundle, 30.0)]
    assert transport.pods[0]["ports"] == [PROXY_PORT_SPEC]
    assert transport.pods[0]["imageName"] == bundle.image.reference
    assert transport.templates[0]["imageName"] == bundle.image.reference
    # argv, not a joined string: runpod types dockerStartCmd as array<string> in its rest schema
    # and returns it that way, and env as an object rather than a list of {key, value} rows.
    assert transport.templates[0]["dockerStartCmd"] == ["python", "/app/serve_launch.py"]
    template_env = dict(transport.templates[0]["env"])
    names = _names(bundle)
    assert template_env["FLASH_INFERENCE_TOKEN"] == (
        f"{{{{ RUNPOD_SECRET_{names.inference_secret} }}}}"
    )
    assert template_env["FLASH_ARTIFACT_TOKEN"] == (
        f"{{{{ RUNPOD_SECRET_{names.artifact_secret} }}}}"
    )
    assert transport.volumes[0]["dataCenterId"] == bundle.spec.placement.data_center_id
    assert all(item["name"] != _names(bundle).artifact_secret for item in transport.secrets)
    assert not hasattr(result.handle, "artifact_secret_id")

    mutations = [
        (kind, operation) for kind, operation, mutation, _payload in transport.calls if mutation
    ]
    assert mutations == [
        ("graphql", "secretCreate"),
        ("graphql", "secretCreate"),
        ("rest", "POST /networkvolumes"),
        ("rest", "POST /templates"),
        ("rest", "POST /pods"),
        ("graphql", "secretDelete"),
    ]


def test_secret_sentinels_are_confined_to_exact_request_sinks() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    result, factory, probe = _provision(bundle, transport)

    encoded = json.dumps(sanitized_dict(result), sort_keys=True)
    assert all(
        secret not in encoded for secret in (PROVIDER_SECRET, INFERENCE_SECRET, ARTIFACT_SECRET)
    )
    assert all(
        secret not in repr(result)
        for secret in (PROVIDER_SECRET, INFERENCE_SECRET, ARTIFACT_SECRET)
    )
    assert factory.accepted_keys == [True]
    assert [call[1] for call in probe.calls] == [True]

    secret_mutations = [
        payload
        for kind, operation, mutation, payload in transport.calls
        if kind == "graphql" and operation == "secretCreate" and mutation
    ]
    assert [payload["value_present"] for payload in secret_mutations] == [True, True]
    assert [payload["value_is_expected"] for payload in secret_mutations] == [True, True]
    non_secret_calls = [
        payload
        for kind, operation, _mutation, payload in transport.calls
        if not (kind == "graphql" and operation == "secretCreate")
    ]
    serialized_calls = repr(non_secret_calls)
    assert INFERENCE_SECRET not in serialized_calls
    assert ARTIFACT_SECRET not in serialized_calls
    assert PROVIDER_SECRET not in serialized_calls
    assert "FREESOLO_INTERNAL_KEY" not in repr(transport.calls)


def test_existing_resources_require_exact_match_and_direct_endpoint_proof() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle)
    transport.calls.clear()

    accepted, _factory, probe = _provision(
        bundle,
        transport,
        artifact_token=None,
        probe=_Probe(True),
    )
    assert accepted.status == "ready"
    assert accepted.handle == handle
    assert _mutation_calls(transport) == []
    assert probe.calls[0][1] is True

    transport.calls.clear()
    refused, _factory, _probe = _provision(
        bundle,
        transport,
        artifact_token=None,
        probe=_Probe(False),
    )
    assert refused.status == "outcome_unknown"
    assert refused.error_code == "resource_ambiguous"
    assert refused.handle == handle
    assert _mutation_calls(transport) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda transport: transport.pods.append(dict(transport.pods[0])),
        lambda transport: transport.pods[0].update(ports=["22/tcp"]),
        lambda transport: transport.templates[0].update(
            imageName="registry.example/other@sha256:" + "0" * 64
        ),
        lambda transport: transport.volumes[0].update(dataCenterId="EU-RO-1"),
    ],
)
def test_duplicate_or_mismatched_resources_fail_closed_without_mutation(mutate) -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    _seed_exact(transport, bundle)
    mutate(transport)
    transport.calls.clear()

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "failed"
    assert result.error_code == "conflict"
    assert _mutation_calls(transport) == []


def test_wrong_account_fails_closed_with_sanitized_error() -> None:
    bundle = _bundle()
    transport = _FakeTransport(account_id="account-other")

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "failed"
    assert result.error_code == "authentication_failed"
    assert result.handle is None
    assert PROVIDER_SECRET not in json.dumps(sanitized_dict(result))


def test_ambiguous_mutation_is_not_retried_and_returns_outcome_unknown() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    transport.fail_mutation_at = 1

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert transport.mutation_count == 1
    assert len(_mutation_calls(transport)) == 1


@pytest.mark.parametrize("boundary", range(1, 6))
@pytest.mark.parametrize(
    ("failure_mode", "expected_status"),
    [("definite_before", "failed"), ("ambiguous_after", "outcome_unknown")],
)
def test_create_failure_boundaries_abort_all_partial_resources(
    boundary: int,
    failure_mode: str,
    expected_status: str,
) -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    transport.fail_mutation_at = boundary
    transport.failure_mode = failure_mode

    result, _factory, _probe = _provision(bundle, transport)

    assert result.status == expected_status
    if failure_mode == "definite_before":
        assert result.error_code == "provider_rejected"
    else:
        assert result.error_code == "resource_ambiguous"
    assert transport.secrets == []
    assert transport.templates == []
    assert transport.volumes == []
    assert transport.pods == []


def test_malformed_success_id_binds_no_invalid_proxy_url() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    transport.malformed_pod_id = True

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert result.handle is None
    assert len([call for call in _mutation_calls(transport) if call[1] == "POST /pods"]) == 1
    assert transport.secrets == []
    assert transport.templates == []
    assert transport.volumes == []
    assert transport.pods == []


def test_reconcile_is_read_only_and_reports_ready_or_absent() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle)
    factory = _Factory(transport)
    clock = _Clock()

    ready = reconcile_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        transport_factory=factory,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert ready.status == "ready"
    assert ready.handle == handle
    assert _mutation_calls(transport) == []

    transport.secrets.clear()
    transport.templates.clear()
    transport.volumes.clear()
    transport.pods.clear()
    transport.calls.clear()
    absent = reconcile_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        transport_factory=factory,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert absent.status == "absent"
    assert _mutation_calls(transport) == []


def test_read_only_reconcile_never_reports_ready_with_transient_artifact_secret() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    factory = _Factory(transport)
    clock = _Clock()

    result = reconcile_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=4.0,
        transport_factory=factory,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.status == "outcome_unknown"
    assert result.handle == handle
    assert _mutation_calls(transport) == []
    assert len(transport.secrets) == 2


def test_adoption_deletes_one_lingering_artifact_only_after_endpoint_proof() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    transport.calls.clear()

    result, _factory, probe = _provision(
        bundle,
        transport,
        artifact_token=None,
        probe=_Probe(True),
    )
    assert result.status == "ready"
    assert result.handle == handle
    assert probe.calls
    assert probe.calls[0][1] is True
    assert [call[1] for call in _mutation_calls(transport)] == ["secretDelete"]
    assert [item["name"] for item in transport.secrets] == [_names(bundle).inference_secret]


def test_production_probe_overrun_never_accepts_readiness_or_cleans_artifact() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    factory = _Factory(transport)
    clock = _Clock()
    observed: list[dict[str, object]] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(_models_payload(bundle)).encode()

    def opener(request, *, timeout: float):
        observed.append(
            {
                "url_ok": request.full_url == handle.public_url + "/v1/models",
                "method": request.get_method(),
                "auth_ok": request.get_header("Authorization") == f"Bearer {INFERENCE_SECRET}",
                "timeout": timeout,
            }
        )
        clock.now = 6.0
        return Response()

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=5.0,
        transport_factory=factory,
        probe=RunPodEndpointProbe(opener=opener),
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.status == "outcome_unknown"
    assert result.handle == handle
    assert observed == [{"url_ok": True, "method": "GET", "auth_ok": True, "timeout": 5.0}]
    assert _mutation_calls(transport) == []
    assert len(transport.secrets) == 2
    assert INFERENCE_SECRET not in repr(observed)


def test_production_probe_rejects_redirect_without_second_request() -> None:
    bundle = _bundle()
    handle = _seed_exact(_FakeTransport(), bundle)
    observed: list[dict[str, object]] = []

    class RedirectSource(urllib.request.BaseHandler):
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

    probe = RunPodEndpointProbe(opener=build_no_redirect_opener(RedirectSource()))
    assert probe(handle.public_url, INFERENCE_SECRET, bundle, 3.0) is False
    assert observed == [{"url": handle.public_url + "/v1/models", "auth_ok": True}]
    assert "attacker.invalid" not in repr(observed)


def test_artifact_secret_duplicates_are_conflict_without_cleanup() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    _seed_exact(transport, bundle, artifact_secret=True)
    transport.secrets.append({"id": "secret03", "name": _names(bundle).artifact_secret})
    transport.calls.clear()

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)
    assert result.status == "failed"
    assert result.error_code == "conflict"
    assert _mutation_calls(transport) == []


def test_ambiguous_adoption_artifact_cleanup_is_outcome_unknown() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    transport.fail_mutation_at = 1
    transport.failure_mode = "ambiguous_after"
    transport.calls.clear()

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)
    assert result.status == "outcome_unknown"
    assert result.handle == handle
    assert len([call for call in _mutation_calls(transport) if call[1] == "secretDelete"]) == 1


def test_volume_resize_only_grows_and_mutates_once() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle)
    factory = _Factory(transport)
    clock = _Clock()
    transport.calls.clear()

    grown = grow_runpod_volume(
        bundle,
        handle,
        RunPodCredentials(PROVIDER_SECRET),
        150,
        deadline_at=100.0,
        transport_factory=factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert grown.status == "ready"
    assert transport.volumes[0]["size"] == 150
    patches = [call for call in _mutation_calls(transport) if call[1].startswith("PATCH ")]
    assert len(patches) == 1
    assert patches[0][3] == {"size": 150}

    fresh_factory = _Factory(transport)
    with pytest.raises(ValueError, match="cannot shrink"):
        grow_runpod_volume(
            bundle,
            handle,
            RunPodCredentials(PROVIDER_SECRET),
            99,
            deadline_at=100.0,
            transport_factory=fresh_factory,
            clock=clock,
            sleep=clock.sleep,
        )
    assert fresh_factory.accepted_keys == []

    transport.calls.clear()
    shrink = grow_runpod_volume(
        bundle,
        handle,
        RunPodCredentials(PROVIDER_SECRET),
        125,
        deadline_at=100.0,
        transport_factory=factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert shrink.status == "failed"
    assert shrink.error_code == "invalid_request"
    assert _mutation_calls(transport) == []


def test_teardown_deletes_pod_before_attached_volume_and_confirms_absence() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle)
    factory = _Factory(transport)
    clock = _Clock()
    transport.calls.clear()

    result = teardown_runpod_deployment(
        bundle,
        handle,
        RunPodCredentials(PROVIDER_SECRET),
        deadline_at=100.0,
        transport_factory=factory,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "absent"
    mutations = [call[1] for call in _mutation_calls(transport)]
    assert mutations == [
        f"DELETE /pods/{POD_ID}",
        "DELETE /templates/template01",
        "DELETE /networkvolumes/volume01",
        "secretDelete",
    ]
    assert transport.pods == []
    assert transport.templates == []
    assert transport.volumes == []
    assert transport.secrets == []

    transport.calls.clear()
    confirmed = confirm_runpod_absence(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        deadline_at=100.0,
        transport_factory=factory,
        clock=clock,
    )
    assert confirmed.status == "absent"
    assert _mutation_calls(transport) == []


@pytest.mark.parametrize("status", ["STOPPED", "FAILED"])
def test_teardown_accepts_exact_pods_in_nonready_statuses(status: str) -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, status=status)
    factory = _Factory(transport)
    clock = _Clock()
    transport.calls.clear()

    result = teardown_runpod_deployment(
        bundle,
        handle,
        RunPodCredentials(PROVIDER_SECRET),
        deadline_at=100.0,
        transport_factory=factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.status == "absent"
    assert transport.pods == []
    assert transport.volumes == []


def test_teardown_rejects_wrong_generation_handle_before_client() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle)
    factory = _Factory(transport)
    clock = _Clock()

    with pytest.raises(ValueError, match="exact deployment generation"):
        teardown_runpod_deployment(
            bundle,
            replace(handle, generation=handle.generation + 1),
            RunPodCredentials(PROVIDER_SECRET),
            deadline_at=100.0,
            transport_factory=factory,
            clock=clock,
            sleep=clock.sleep,
        )
    assert factory.accepted_keys == []
    assert transport.calls == []


def test_load_bearing_guards_are_sabotage_sensitive(monkeypatch) -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    _seed_exact(transport, bundle)
    transport.pods[0]["ports"] = ["22/tcp"]

    guarded, _factory, _probe = _provision(bundle, transport, artifact_token=None)
    assert guarded.error_code == "conflict"

    monkeypatch.setattr(
        "flash.serve.provisioning._runpod_plan.PROXY_PORT_SPEC",
        "22/tcp",
    )
    monkeypatch.setattr(
        "flash.serve.provisioning._runpod_resources.PROXY_PORT_SPEC",
        "22/tcp",
    )
    transport.templates[0]["ports"] = ["22/tcp"]
    unguarded, _factory, _probe = _provision(bundle, transport, artifact_token=None)
    assert unguarded.status == "ready"


def test_probe_exceptions_are_sanitized_as_unproven_adoption() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle)

    def explode(
        _url: str,
        _token: str,
        _bundle: DeploymentBundle,
        _timeout_seconds: float,
    ) -> bool:
        raise RuntimeError(INFERENCE_SECRET)

    result, _factory, _probe = _provision(
        bundle,
        transport,
        artifact_token=None,
        probe=explode,
    )
    assert result.status == "outcome_unknown"
    assert result.handle == handle
    assert INFERENCE_SECRET not in json.dumps(sanitized_dict(result))


def test_teardown_refuses_mismatched_exact_resource_before_deletion() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle)
    transport.templates[0]["dockerStartCmd"] = "python -c bad"
    factory = _Factory(transport)
    clock = _Clock()
    transport.calls.clear()

    result = teardown_runpod_deployment(
        bundle,
        handle,
        RunPodCredentials(PROVIDER_SECRET),
        deadline_at=100.0,
        transport_factory=factory,
        clock=clock,
        sleep=clock.sleep,
    )
    assert result.status == "failed"
    assert result.error_code == "conflict"
    assert _mutation_calls(transport) == []


def test_artifact_cleanup_ambiguity_keeps_handle_and_never_retries() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    transport.fail_mutation_at = 6

    result, _factory, _probe = _provision(bundle, transport)

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert result.handle is not None
    assert transport.mutation_count == 6
    deletes = [call for call in _mutation_calls(transport) if call[1] == "secretDelete"]
    assert len(deletes) == 1


def test_stdlib_transport_attempts_mutation_once_and_sanitizes_malformed_success() -> None:
    requests: list[dict[str, object]] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"not-json"

    def opener(request, *, timeout: float):
        requests.append(
            {
                "url_ok": request.full_url == GRAPHQL_URL,
                "method": request.get_method(),
                "auth_ok": request.get_header("Authorization") == f"Bearer {PROVIDER_SECRET}",
                "content_type": request.get_header("Content-type"),
                "timeout": timeout,
            }
        )
        return Response()

    transport = StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)
    with pytest.raises(RunPodTransportFailure) as exc_info:
        transport.graphql(
            "mutation Test { mutate }",
            {},
            mutation=True,
            deadline_at=10.0,
        )
    assert exc_info.value.code == "resource_ambiguous"
    assert exc_info.value.outcome_unknown is True
    assert requests == [
        {
            "url_ok": True,
            "method": "POST",
            "auth_ok": True,
            "content_type": "application/json",
            "timeout": 10.0,
        }
    ]
    assert PROVIDER_SECRET not in repr(requests)
    assert PROVIDER_SECRET not in repr(transport)
    assert PROVIDER_SECRET not in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"errors": [{"message": "rejected"}]},
        {"data": {"secretCreate": {"id": "secret01"}}, "errors": [{"message": "partial"}]},
    ],
)
def test_graphql_mutation_errors_are_always_outcome_unknown(payload: dict[str, object]) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode()

    calls = 0

    def opener(_request, *, timeout: float):
        nonlocal calls
        calls += 1
        return Response()

    transport = StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)
    with pytest.raises(RunPodTransportFailure) as exc_info:
        transport.graphql("mutation Test { mutate }", {}, mutation=True, deadline_at=10.0)
    assert exc_info.value.code == "resource_ambiguous"
    assert exc_info.value.outcome_unknown is True
    assert calls == 1


@pytest.mark.parametrize("status", [408, 500, 503])
def test_ambiguous_http_mutation_statuses_are_not_retried(status: int) -> None:
    calls = 0

    def opener(_request, *, timeout: float):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            GRAPHQL_URL,
            status,
            "sanitized",
            None,
            io.BytesIO(b"provider body"),
        )

    transport = StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)
    with pytest.raises(RunPodTransportFailure) as exc_info:
        transport.graphql("mutation Test { mutate }", {}, mutation=True, deadline_at=10.0)
    assert exc_info.value.code == "resource_ambiguous"
    assert exc_info.value.outcome_unknown is True
    assert calls == 1


@pytest.mark.parametrize("mutation", [False, True])
def test_no_redirect_opener_rejects_redirect_without_forwarding_authorization(
    mutation: bool,
) -> None:
    observed: list[dict[str, object]] = []

    class RedirectSource(urllib.request.BaseHandler):
        def default_open(self, request):
            observed.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "auth_ok": request.get_header("Authorization") == f"Bearer {PROVIDER_SECRET}",
                }
            )
            headers = email.message.Message()
            headers["Location"] = "https://attacker.invalid/steal"
            return urllib.response.addinfourl(
                io.BytesIO(b""),
                headers,
                request.full_url,
                code=302,
            )

    opener = build_no_redirect_opener(RedirectSource())
    transport = StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)
    with pytest.raises(RunPodTransportFailure) as exc_info:
        transport.rest(
            "POST" if mutation else "GET",
            "/pods",
            {} if mutation else None,
            mutation=mutation,
            deadline_at=10.0,
        )
    assert len(observed) == 1
    assert observed[0]["url"] == "https://rest.runpod.io/v1/pods"
    assert observed[0]["auth_ok"] is True
    assert "attacker.invalid" not in repr(observed)
    if mutation:
        assert exc_info.value.code == "resource_ambiguous"
        assert exc_info.value.outcome_unknown is True
    else:
        assert exc_info.value.code == "transport_failed"
        assert exc_info.value.outcome_unknown is False


def test_stdlib_transport_does_not_reflect_provider_error_bodies() -> None:
    calls = 0

    def opener(_request, *, timeout: float):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            "https://api.runpod.io/graphql",
            400,
            ARTIFACT_SECRET,
            None,
            io.BytesIO(ARTIFACT_SECRET.encode()),
        )

    transport = StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)
    with pytest.raises(RunPodTransportFailure) as exc_info:
        transport.graphql(
            "mutation Test { mutate }",
            {},
            mutation=True,
            deadline_at=10.0,
        )
    assert exc_info.value.code == "provider_rejected"
    assert exc_info.value.outcome_unknown is False
    assert calls == 1
    assert ARTIFACT_SECRET not in str(exc_info.value)


def test_requests_send_an_explicit_user_agent_on_both_apis() -> None:
    # runpod's graphql edge answers 403 to urllib's default "Python-urllib/x.y" agent while
    # accepting the identical authenticated request under any other agent. the rest api does not,
    # so an unset agent failed only on graphql and surfaced as authentication_failed -- which
    # reads as a bad credential and sends debugging to the wrong place entirely.
    seen: list[str | None] = []

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def opener(request, *, timeout: float):
        seen.append(request.get_header("User-agent"))
        return Response()

    transport = StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)
    transport.graphql("query Test { probe }", {}, mutation=False, deadline_at=10.0)
    transport.rest("GET", "/pods", None, mutation=False, deadline_at=10.0)

    assert seen == [USER_AGENT, USER_AGENT]
    for agent in seen:
        assert agent, "an unset agent falls back to urllib's default, which runpod rejects"
        assert "urllib" not in agent.lower()


def _capturing_transport(seen: list[str]) -> StdlibRunPodTransport:
    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def opener(request, *, timeout: float):
        seen.append(request.full_url)
        return Response()

    return StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)


def test_rest_encodes_query_parameters_without_reopening_the_path_injection_hole() -> None:
    # runpod hides whole response objects behind query flags, so the transport has to send them.
    # they are encoded from a mapping rather than spliced into `path` precisely so the "no ?"
    # check on the path cannot be bypassed by a caller passing a url fragment as the path.
    seen: list[str] = []
    transport = _capturing_transport(seen)
    transport.rest(
        "GET",
        "/pods",
        None,
        mutation=False,
        deadline_at=10.0,
        query={"includeNetworkVolume": "true", "includeMachine": "true"},
    )
    transport.rest("GET", "/templates", None, mutation=False, deadline_at=10.0)

    assert seen == [
        f"{REST_BASE_URL}/pods?includeMachine=true&includeNetworkVolume=true",
        f"{REST_BASE_URL}/templates",
    ]

    for path in ("/pods?includeMachine=true", "/pods#frag", "https://evil.example/pods"):
        with pytest.raises(ValueError, match="absolute path"):
            transport.rest("GET", path, None, mutation=False, deadline_at=10.0)
    for bad in ({"": "true"}, {"includeMachine": ""}, {"includeMachine": True}):
        with pytest.raises(ValueError, match="query"):
            transport.rest("GET", "/pods", None, mutation=False, deadline_at=10.0, query=bad)


def test_secret_documents_name_the_mutations_runpods_schema_actually_defines() -> None:
    # runpod's graphql schema has secretCreate / secretDelete. the old createSecret / deleteSecret
    # names are rejected outright with "Cannot query field ... on type Mutation", so provisioning
    # died at its first secret. only the fake answered them, which is why every offline test
    # passed against a schema that does not exist -- pinning the names here is what stops a
    # matching rename in the fake from keeping the suite green on its own.
    assert "secretCreate(input: {name: $name, value: $value})" in CREATE_SECRET
    assert "secretDelete(id: $id)" in DELETE_SECRET
    # secretDelete takes a bare ID! and returns Void: an input-object argument or a selection set
    # is a schema error, not a stylistic difference.
    assert "$id: ID!" in DELETE_SECRET
    assert "input: {id:" not in DELETE_SECRET
    for document in (CREATE_SECRET, DELETE_SECRET):
        assert "createSecret" not in document
        assert "deleteSecret" not in document

    # Void returns null on success, so a bare `true` is not what runpod sends back.
    assert parse_deleted_secret({"data": {"secretDelete": None}}) is True
    for malformed in (
        {"data": {}},
        {"data": {"secretDelete": True}},
        {"data": {"deleteSecret": None}},
    ):
        with pytest.raises(ValueError, match="deleted secret response is malformed"):
            parse_deleted_secret(malformed)


def test_pod_observation_reads_the_nested_shape_runpods_rest_api_returns() -> None:
    # runpod's rest Pod schema has no flat gpuTypeId / dataCenterId / networkVolumeId: they live
    # in the nested machine, gpu, and networkVolume objects. reading only the flat keys left
    # gpu_type_id absent on every real pod, which fails the whole observation pass -- so the
    # lifecycle could never even look at its own pod.
    nested = parse_pods(
        [
            {
                "id": "abc123def45678",
                "name": "pod-a",
                "desiredStatus": "RUNNING",
                "imageName": "ghcr.io/org/image@sha256:" + "0" * 64,
                "gpuCount": 1,
                "containerDiskInGb": 60,
                "ports": ["8000/http"],
                "templateId": "tpl0000001",
                "machine": {"gpuTypeId": "NVIDIA L4", "dataCenterId": "US-KS-2"},
                "networkVolume": {"id": "vol0000001"},
            }
        ]
    )
    assert nested[0].gpu_type_id == "NVIDIA L4"
    assert nested[0].data_center_id == "US-KS-2"
    assert nested[0].network_volume_id == "vol0000001"

    # a pod with no volume and no template is a real state for pods flash did not create. every
    # pod in the account is parsed, so rejecting those aborts observation before reaching ours.
    bare = parse_pods(
        [
            {
                "id": "abc123def45679",
                "name": "foreign",
                "desiredStatus": "EXITED",
                "imageName": "pytorch/pytorch:2.6.0",
                "gpuCount": 1,
                "containerDiskInGb": 300,
                "ports": ["8000/http"],
                "templateId": "",
                "machine": {"gpuTypeId": "NVIDIA A100 80GB PCIe", "dataCenterId": "CA-MTL-3"},
            }
        ]
    )
    assert bare[0].network_volume_id is None
    assert bare[0].template_id is None


def test_observation_survives_a_resource_that_exposes_no_ports() -> None:
    # runpod sends `ports: null` for a resource created without exposed ports rather than an
    # empty list. observed live: a foreign pod in the account came back that way, `_ports` rejected
    # null, and the whole listing collapsed into an opaque transport failure -- so flash could
    # never see its own pods. an empty string was always accepted and means the same thing.
    pod = parse_pods(
        [
            {
                "id": "abc123def45680",
                "name": "portless",
                "desiredStatus": "RUNNING",
                "imageName": "pytorch/pytorch:2.6.0",
                "gpuCount": 1,
                "containerDiskInGb": 60,
                "ports": None,
                "machine": {"gpuTypeId": "NVIDIA H200", "dataCenterId": "EUR-IS-4"},
            }
        ]
    )
    assert pod[0].ports == ()

    template = parse_templates(
        [
            {
                "id": "tpl0000009",
                "name": "portless-template",
                "imageName": "pytorch/pytorch:2.6.0",
                "dockerStartCmd": ["sleep", "infinity"],
                "containerDiskInGb": 60,
                "volumeMountPath": "/workspace",
                "ports": None,
                "env": {},
            }
        ]
    )
    assert template[0].ports == ()

    # absence must stay absence, not become a wildcard that lets any port shape through: a
    # genuinely malformed value is still rejected, so this cannot mask a real schema change.
    for malformed in (0, 12.5, {"8000": "http"}, True):
        with pytest.raises(ValueError, match="pod ports must be a string or list"):
            parse_pods(
                [
                    {
                        "id": "abc123def45681",
                        "name": "malformed",
                        "desiredStatus": "RUNNING",
                        "imageName": "pytorch/pytorch:2.6.0",
                        "gpuCount": 1,
                        "containerDiskInGb": 60,
                        "ports": malformed,
                        "machine": {"gpuTypeId": "NVIDIA L4", "dataCenterId": "US-KS-2"},
                    }
                ]
            )


def test_template_observation_reads_argv_and_omitted_defaults() -> None:
    # dockerStartCmd comes back as argv and volumeInGb / isServerless are omitted at their
    # defaults rather than returned as 0 / false. demanding a string and a present key failed on
    # a foreign template and took the whole observation pass with it.
    parsed = parse_templates(
        [
            {
                "id": "tpl0000002",
                "name": "foreign-tpl",
                "imageName": "pytorch/pytorch:2.6.0",
                "dockerStartCmd": ["bash", "-lc", "run.sh"],
                "containerDiskInGb": 300,
                "volumeMountPath": "/workspace",
                "ports": ["8000/http"],
                "env": {"A": "1"},
            }
        ]
    )
    assert parsed[0].docker_start_cmd == "bash -lc run.sh"
    assert parsed[0].volume_gb == 0
    assert parsed[0].is_serverless is False

    for broken in ([], "", 5, [""], ["ok", 3]):
        with pytest.raises(ValueError, match="dockerStartCmd"):
            parse_templates(
                [
                    {
                        "id": "tpl0000002",
                        "name": "t",
                        "imageName": "img",
                        "dockerStartCmd": broken,
                        "containerDiskInGb": 300,
                        "volumeMountPath": "/workspace",
                        "ports": ["8000/http"],
                        "env": {},
                    }
                ]
            )


def test_foreign_templates_are_filtered_before_strict_parsing() -> None:
    # the customer's account holds templates flash did not create. those rows legitimately omit
    # dockerStartCmd (they use the image default) and env (no overrides), which the strict field
    # parsing rejects. parsing every row meant one unrelated template failed the whole
    # observation with transport_failed and blocked deployments with no conflicting resource.
    foreign = {
        "id": "tpl0000123",
        "name": "someone-elses-template",
        "imageName": "nginx:latest",
        "dockerStartCmd": None,
        "containerDiskInGb": 5,
        "volumeMountPath": "/workspace",
        "ports": None,
        "env": None,
    }
    ours = {
        "id": "tpl0000124",
        "name": "flash-owned-tpl",
        "imageName": "img:1",
        "dockerStartCmd": ["python", "-m", "flash.serve.app"],
        "containerDiskInGb": 300,
        "volumeMountPath": "/workspace",
        "ports": ["8000/http"],
        "env": {"A": "1"},
    }

    # unfiltered, the foreign row still fails -- the strictness is intact.
    with pytest.raises(ValueError, match="dockerStartCmd"):
        parse_templates([foreign, ours])

    kept = parse_templates([foreign, ours], keep_name="flash-owned-tpl")
    assert [item.name for item in kept] == ["flash-owned-tpl"]
    assert kept[0].docker_start_cmd == "python -m flash.serve.app"

    # an account with only foreign templates observes no flash template rather than failing.
    assert parse_templates([foreign], keep_name="flash-owned-tpl") == ()

    # filtering must not weaken validation of flash's own row: a malformed template that carries
    # the kept name is still rejected.
    broken_ours = dict(ours, dockerStartCmd=None)
    with pytest.raises(ValueError, match="dockerStartCmd"):
        parse_templates([broken_ours], keep_name="flash-owned-tpl")


def test_observation_survives_a_foreign_template_in_the_customers_account() -> None:
    # the parser-level test above proves `keep_name` filters, but not that the observation path
    # passes it. filtering after strict parsing type-checks and reads fine, so only driving the
    # real lifecycle catches a call site that parses the whole account first: one unrelated
    # template in the customer's account then fails observation with `transport_failed` and
    # blocks a deployment that has no conflicting flash resource at all.
    bundle = _bundle()
    transport = _FakeTransport()
    transport.templates.append(
        {
            "id": "tpl0000999",
            "name": "someone-elses-template",
            "imageName": "nginx:latest",
            # a template that relies on its image's default command and sets no overrides. both
            # are legitimate on runpod and both are rejected by flash's strict field parsing.
            "dockerStartCmd": None,
            "containerDiskInGb": 5,
            "volumeMountPath": "/workspace",
            "ports": None,
            "env": None,
        }
    )
    clock = _Clock()

    confirmed = confirm_runpod_absence(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        deadline_at=100.0,
        transport_factory=_Factory(transport),
        clock=clock,
    )

    # absent, not transport_failed: flash owns nothing here, and the foreign row is not flash's
    # business to validate.
    assert confirmed.status == "absent"
    assert confirmed.error_code is None


class _InterruptingProbe:
    """Ctrl-C while polling a slow pod, which is when a user is most likely to press it."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, url: str, token: str, bundle: DeploymentBundle, timeout: float) -> bool:
        self.calls += 1
        raise KeyboardInterrupt


def test_interrupting_a_slow_readiness_poll_still_cleans_up_created_resources() -> None:
    """Ctrl-C after the creates succeed must not strand live, billable RunPod resources.

    `KeyboardInterrupt` derives from BaseException, so an `except RunPodTransportFailure` never
    sees it. The pod, its network volume, and its secrets stay live with no cleanup and no outcome
    warning -- the user reads the traceback as "it didn't happen" while the pod keeps billing.
    """
    bundle = _bundle()
    transport = _FakeTransport()
    probe = _InterruptingProbe()

    with pytest.raises(KeyboardInterrupt):
        _provision(bundle, transport, probe=probe)

    assert probe.calls == 1
    # everything the ledger recorded must have been torn back down before the interrupt propagates.
    operations = [operation for _kind, operation, mutation, _payload in transport.calls if mutation]
    assert "DELETE /pods" in " ".join(operations), operations
    assert not transport.pods, transport.pods
    assert not transport.volumes, transport.volumes
    assert not transport.secrets, transport.secrets


class _InterruptOnArtifactDeleteTransport(_FakeTransport):
    """Ctrl-C in the one window where the pod is already live and serving.

    `secretDelete` for the artifact secret is only reached after the probe reports ready, so
    raising here reproduces an interrupt during the final hydration-secret cleanup and nowhere
    else.
    """

    def graphql(self, document, variables, *, mutation: bool, deadline_at: float):
        if mutation and "secretDelete" in document:
            raise KeyboardInterrupt
        return super().graphql(document, variables, mutation=mutation, deadline_at=deadline_at)


def test_interrupting_after_the_pod_is_ready_does_not_tear_down_the_live_pod() -> None:
    """Ctrl-C once the pod probes healthy must leave the deployment standing.

    The abort cleanup exists for an interrupt during a slow readiness poll, where the resources are
    half-built and worthless. After readiness the only work left is deleting the hydration secret,
    and treating an interrupt there as "abandon it" destroys a live, warmed, billable pod the user
    just waited for. A leftover artifact secret is recoverable by re-running the command; a deleted
    pod is not, so the cleanup must not reach into this window.
    """
    bundle = _bundle()
    transport = _InterruptOnArtifactDeleteTransport()

    with pytest.raises(KeyboardInterrupt):
        _provision(bundle, transport)

    assert transport.pods, "the ready pod must survive the interrupt"
    assert transport.volumes, "the volume backing the ready pod must survive the interrupt"
    operations = [operation for _kind, operation, mutation, _payload in transport.calls if mutation]
    assert "DELETE /pods" not in " ".join(operations), operations


def test_loopback_image_registries_are_rejected_before_any_runpod_call() -> None:
    # the same reasoning as the modal case: `ServingImage` allows a loopback registry because a
    # local pull is legitimate, but a runpod pod resolves `imageName` on the provider side. no
    # code path uploads the image or tunnels to the operator's host, so the pull fails only once
    # the pod exists and bills. reject it while the plan is still local.
    original = _bundle()
    for registry in (
        "localhost",
        "localhost:5000",
        "registry.localhost",
        "localhost.localdomain",
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
            build_runpod_create_plan(bundle)

    assert build_runpod_create_plan(original)

    reachable = DeploymentBundle(
        spec=original.spec,
        manifest=original.manifest,
        image=ServingImage(
            reference=f"notlocalhost.example/flash/serve@{original.image.digest}",
            digest=original.image.digest,
        ),
    )
    assert build_runpod_create_plan(reachable)
