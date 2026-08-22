"""offline persistent runpod pod lifecycle and secret-boundary coverage."""

from __future__ import annotations

import email.message
import io
import json
import urllib.error
import urllib.response
from collections.abc import Callable
from dataclasses import replace

import pytest

from flash.serve.app.manifest import build_serving_manifest
from flash.serve.control import (
    DeploymentSpec,
    RunPodCredentials,
    RunPodPlacement,
    RunPodProviderHandle,
)
from flash.serve.provisioning import (
    DeploymentBundle,
    InterruptedProvisioning,
    ServingImage,
    ServingRuntimeSecrets,
    _common,
)
from flash.serve.provisioning._common import serving_resource_names
from flash.serve.provisioning._runpod_plan import build_runpod_create_plan
from flash.serve.provisioning._runpod_probe import RunPodEndpointProbe, _provenance_matches
from flash.serve.provisioning._runpod_protocol import (
    CREATE_SECRET,
    DELETE_SECRET,
    PROXY_PORT_SPEC,
    parse_deleted_secret,
    parse_pods,
    parse_templates,
    parse_volumes,
)
from flash.serve.provisioning._runpod_resources import (
    RunPodResourceConflict,
    _one,
    pod_identity_matches,
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
    _delete_tolerating_ambiguity,
    _observe,
    _work_deadline,
    provision_runpod_deployment,
    reconcile_runpod_deployment,
    teardown_runpod_deployment,
)
from tests.test_serve_app_manifest import _spec_and_inputs

PROVIDER_SECRET = "provider-api-secret-sentinel"
INFERENCE_SECRET = "inference-secret-sentinel"
ARTIFACT_SECRET = "artifact-secret-sentinel"
PROVIDER_PUBLIC_KEY = "ssh-rsa provider-managed"
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


class _SequenceProbe(_Probe):
    def __init__(self, *accepted: bool) -> None:
        if not accepted:
            raise ValueError("at least one result is required")
        super().__init__()
        self.accepted_results = accepted

    def __call__(
        self,
        url: str,
        token: str,
        bundle: DeploymentBundle,
        timeout_seconds: float,
    ) -> bool:
        super().__call__(url, token, bundle, timeout_seconds)
        index = min(len(self.calls) - 1, len(self.accepted_results) - 1)
        return self.accepted_results[index]


class _Factory:
    def __init__(self, transport: _FakeTransport) -> None:
        self.transport = transport
        self.accepted_keys: list[bool] = []

    def __call__(self, api_key: str) -> _FakeTransport:
        self.accepted_keys.append(api_key == PROVIDER_SECRET)
        return self.transport


class _FakeTransport:
    """provider double that enforces the deadline, and owns the clock that measures it.

    The clock lives here rather than beside each caller so the transport and the lifecycle under
    test cannot be handed different ones: tests pass `transport.clock` to `provision_...`, and a
    deadline the lifecycle believes is live is one this double agrees is live.
    """

    def __init__(self, account_id: str = "account-01") -> None:
        self.clock = _Clock()
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
        self.on_first_mutation: Callable[[], None] | None = None

    def _honour_deadline(self, deadline_at: float) -> None:
        # the real transport refuses a call once the deadline has passed, and that refusal is what
        # decides whether teardown gets to issue its deletes. a double that accepted an expired
        # deadline let the whole suite pass over a create path that, in production, spent the
        # deadline on readiness and then leaked the pod, volume, template, and both secrets --
        # every cleanup call rejected before it was sent. see `_RunPodTransport._timeout`.
        if deadline_at - self.clock() <= 0:
            raise RunPodTransportFailure("transport_failed")

    def graphql(
        self,
        document: str,
        variables: dict[str, object],
        *,
        mutation: bool,
        deadline_at: float,
    ) -> object:
        self._honour_deadline(deadline_at)
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
        self._honour_deadline(deadline_at)
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
            resource, resource_id = path.strip("/").split("/", 1)
            rows = {
                "templates": self.templates,
                "networkvolumes": self.volumes,
                "pods": self.pods,
            }[resource]
            response = None
            for index, row in enumerate(rows):
                if row["id"] == resource_id:
                    updated = (
                        {"id": resource_id, **payload}
                        if resource == "templates"
                        else {**row, **payload}
                    )
                    rows[index] = updated
                    response = dict(updated)
                    break
            if response is None:
                raise AssertionError("unknown fake patch target")
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
        if self.mutation_count == 1 and self.on_first_mutation is not None:
            # lets a test model a concurrent racer whose resources appear after this run's initial
            # observation but before its first create returns.
            self.on_first_mutation()
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
        rows = [dict(item) for item in selected]
        if path == "/pods":
            # runpod adds PUBLIC_KEY to live pod observations even though flash never authored it.
            # keeping that provider-owned entry on every read prevents the fake from teaching strict
            # equality against a pod environment that can never equal the authored template map.
            for row in rows:
                row["env"] = {**dict(row.get("env", {})), "PUBLIC_KEY": PROVIDER_PUBLIC_KEY}
        return rows

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
            template = next(item for item in self.templates if item["id"] == payload["templateId"])
            created = {
                "id": POD_ID,
                "name": payload["name"],
                "desiredStatus": "RUNNING",
                "imageName": payload["imageName"],
                "machine": {
                    "gpuTypeId": payload["gpuTypeIds"][0],
                    "dataCenterId": payload["dataCenterIds"][0],
                },
                "gpuCount": payload["gpuCount"],
                "containerDiskInGb": payload["containerDiskInGb"],
                "networkVolume": {"id": payload["networkVolumeId"]},
                "templateId": payload["templateId"],
                "ports": payload["ports"],
                # runpod copies the template environment into the pod at creation; later template
                # patches do not mutate this independent live-pod copy.
                "env": {**dict(template["env"]), "PUBLIC_KEY": PROVIDER_PUBLIC_KEY},
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


class _LateVisibleAmbiguousVolumeTransport(_FakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_volume_observations = 0

    def _list(self, path: str) -> list[dict[str, object]]:
        if (
            path == "/networkvolumes"
            and self.mutation_count >= 3
            and self.volumes
            and self.hidden_volume_observations < 3
        ):
            # the live 500 landed the volume, but runpod's immediate abort observations did not
            # expose it. a later reclaim listing does, matching the measured consistency window.
            self.hidden_volume_observations += 1
            return []
        return super()._list(path)


class _PartialThenCompleteTransport(_FakeTransport):
    def __init__(self, *, post_create: bool = False, fully_empty: bool = False) -> None:
        super().__init__()
        self.post_create = post_create
        self.fully_empty = fully_empty
        self.observation_number = 0
        self.delete_observations: list[int] = []

    def _hide_first_observation(self) -> bool:
        hidden_count = 1 if self.post_create else 2
        return self.observation_number < hidden_count and (
            not self.post_create or self.mutation_count >= 5
        )

    def graphql(self, document, variables, *, mutation: bool, deadline_at: float):
        response = super().graphql(
            document,
            variables,
            mutation=mutation,
            deadline_at=deadline_at,
        )
        if not mutation and self._hide_first_observation():
            response["data"]["myself"]["secrets"] = []
        return response

    def _list(self, path: str) -> list[dict[str, object]]:
        rows = super()._list(path)
        if self._hide_first_observation():
            rows = (
                []
                if self.post_create or self.fully_empty or path in {"/networkvolumes", "/pods"}
                else rows
            )
        if path == "/pods" and (not self.post_create or self.mutation_count >= 5):
            self.observation_number += 1
        return rows

    def _delete(self, path: str) -> None:
        self.delete_observations.append(self.observation_number)
        super()._delete(path)


class _AmbiguousAbortDeleteTransport(_FakeTransport):
    def __init__(self, resource_kind: str, *, delete_lands: bool) -> None:
        super().__init__()
        self.resource_kind = resource_kind
        self.delete_lands = delete_lands

    def graphql(self, document, variables, *, mutation: bool, deadline_at: float):
        if mutation and "secretDelete" in document and self.resource_kind == "secret":
            if self.delete_lands:
                super().graphql(
                    document,
                    variables,
                    mutation=mutation,
                    deadline_at=deadline_at,
                )
            else:
                self._honour_deadline(deadline_at)
                self._record("graphql", "secretDelete", True, dict(variables))
                self._begin_mutation()
            raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)
        return super().graphql(
            document,
            variables,
            mutation=mutation,
            deadline_at=deadline_at,
        )

    def _delete(self, path: str) -> None:
        if self.resource_kind == "template" and path == "/templates/template01":
            if self.delete_lands:
                super()._delete(path)
            raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)
        super()._delete(path)


class _IgnoredPostReadyPodPatchTransport(_FakeTransport):
    def rest(
        self,
        method,
        path,
        payload,
        *,
        mutation: bool,
        deadline_at: float,
        query=None,
    ):
        if method == "PATCH" and path == f"/pods/{POD_ID}":
            original = dict(self.pods[0])
            response = super().rest(
                method,
                path,
                payload,
                mutation=mutation,
                deadline_at=deadline_at,
                query=query,
            )
            self.pods[0] = original
            return response
        return super().rest(
            method,
            path,
            payload,
            mutation=mutation,
            deadline_at=deadline_at,
            query=query,
        )


class _PostReadyObservationFailureTransport(_FakeTransport):
    def __init__(self, failures: int | None = None) -> None:
        super().__init__()
        self.fail_observation = False
        self.failures = failures

    def graphql(self, document, variables, *, mutation: bool, deadline_at: float):
        if self.fail_observation and not mutation:
            if self.failures is None:
                raise RunPodTransportFailure("transport_failed")
            if self.failures > 0:
                self.failures -= 1
                raise RunPodTransportFailure("transport_failed")
        return super().graphql(
            document,
            variables,
            mutation=mutation,
            deadline_at=deadline_at,
        )

    def rest(
        self,
        method,
        path,
        payload,
        *,
        mutation: bool,
        deadline_at: float,
        query=None,
    ):
        response = super().rest(
            method,
            path,
            payload,
            mutation=mutation,
            deadline_at=deadline_at,
            query=query,
        )
        if method == "PATCH" and path == f"/pods/{POD_ID}":
            self.fail_observation = True
        return response


class _TerminalThenEventuallyAbsentTransport(_FakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.stale_template: dict[str, object] | None = None

    def rest(
        self,
        method,
        path,
        payload,
        *,
        mutation: bool,
        deadline_at: float,
        query=None,
    ):
        if method == "GET" and path == "/pods" and self.pods:
            self.pods[0]["desiredStatus"] = "EXITED"
        return super().rest(
            method,
            path,
            payload,
            mutation=mutation,
            deadline_at=deadline_at,
            query=query,
        )

    def _delete(self, path: str) -> None:
        if path == "/templates/template01":
            self.stale_template = dict(self.templates[0])
        super()._delete(path)

    def _list(self, path: str) -> list[dict[str, object]]:
        if path == "/templates" and self.stale_template is not None:
            stale = self.stale_template
            self.stale_template = None
            return [stale]
        return super()._list(path)


class _TerminalThenUnconfirmedCleanupTransport(_TerminalThenEventuallyAbsentTransport):
    def _list(self, path: str) -> list[dict[str, object]]:
        if path == "/templates" and self.stale_template is not None:
            return [dict(self.stale_template)]
        return super()._list(path)


class _PostReadyIdentityDriftTransport(_FakeTransport):
    def rest(
        self,
        method,
        path,
        payload,
        *,
        mutation: bool,
        deadline_at: float,
        query=None,
    ):
        response = super().rest(
            method,
            path,
            payload,
            mutation=mutation,
            deadline_at=deadline_at,
            query=query,
        )
        if method == "PATCH" and path == f"/pods/{POD_ID}":
            self.templates[0]["id"] = "template02"
            self.pods[0]["templateId"] = "template02"
        return response


class _PostReadyResourceConflictTransport(_FakeTransport):
    def rest(
        self,
        method,
        path,
        payload,
        *,
        mutation: bool,
        deadline_at: float,
        query=None,
    ):
        response = super().rest(
            method,
            path,
            payload,
            mutation=mutation,
            deadline_at=deadline_at,
            query=query,
        )
        if method == "PATCH" and path == f"/pods/{POD_ID}":
            self.pods.append(dict(self.pods[0]))
        return response


class _PostDeleteObservationFailureTransport(_FakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.fail_observation = False

    def graphql(self, document, variables, *, mutation: bool, deadline_at: float):
        if self.fail_observation and not mutation:
            raise RunPodTransportFailure("transport_failed")
        response = super().graphql(
            document,
            variables,
            mutation=mutation,
            deadline_at=deadline_at,
        )
        if mutation and "secretDelete" in document:
            self.fail_observation = True
        return response


class _PostDeleteResourceConflictTransport(_FakeTransport):
    def graphql(self, document, variables, *, mutation: bool, deadline_at: float):
        response = super().graphql(
            document,
            variables,
            mutation=mutation,
            deadline_at=deadline_at,
        )
        if mutation and "secretDelete" in document:
            self.pods.append(dict(self.pods[0]))
        return response


class _PersistentArtifactAfterDeleteTransport(_FakeTransport):
    def graphql(self, document, variables, *, mutation: bool, deadline_at: float):
        response = super().graphql(
            document,
            variables,
            mutation=mutation,
            deadline_at=deadline_at,
        )
        if mutation and "secretDelete" in document:
            self.secrets.append(
                {
                    "id": variables["id"],
                    "name": _names(_bundle()).artifact_secret,
                }
            )
        return response


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
    # seed from the plan the production path actually sends. building these rows from a second
    # payload implementation let the fixtures agree with a shape runpod rejects -- `dockerStartCmd`
    # as a joined string and `env` as {key, value} rows -- while production sent argv and an object.
    plan = build_runpod_create_plan(bundle)
    volume = {"id": "volume01", **plan.volume_payload()}
    template = {"id": "template01", **plan.template_payload(artifact_secret)}
    pod_request = plan.pod_payload(template_id="template01", volume_id="volume01")
    pod = {
        "id": POD_ID,
        "name": pod_request["name"],
        "desiredStatus": status,
        "imageName": pod_request["imageName"],
        "machine": {
            "gpuTypeId": pod_request["gpuTypeIds"][0],
            "dataCenterId": pod_request["dataCenterIds"][0],
        },
        "gpuCount": pod_request["gpuCount"],
        "containerDiskInGb": pod_request["containerDiskInGb"],
        "networkVolume": {"id": pod_request["networkVolumeId"]},
        "templateId": pod_request["templateId"],
        "ports": pod_request["ports"],
        "env": {**dict(template["env"]), "PUBLIC_KEY": PROVIDER_PUBLIC_KEY},
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
    clock = transport.clock
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
    clock = transport.clock

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
    clock = transport.clock
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
    planned = json.loads(build_runpod_create_plan(_bundle()).pod_static_json)

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
    assert probe.calls == [
        (result.handle.public_url, True, bundle, 30.0),
        (result.handle.public_url, True, bundle, 30.0),
    ]
    assert transport.pods[0]["ports"] == [PROXY_PORT_SPEC]
    assert transport.pods[0]["imageName"] == bundle.image.reference
    assert transport.templates[0]["imageName"] == bundle.image.reference
    # argv, not a joined string: runpod types dockerStartCmd as array<string> in its rest schema
    # and returns it that way, and env as an object rather than a list of {key, value} rows.
    assert transport.templates[0]["dockerStartCmd"] == ["python", "/app/serve_launch.py"]
    template_env = dict(transport.templates[0]["env"])
    names = _names(bundle)
    # the literal `{{ RUNPOD_SECRET_<name> }}` shape, not an f-string rebuilt from the same
    # `names.inference_secret` the plan used: rebuilding it asserts the sentinel equals itself and
    # passes for any format. runpod only substitutes this exact spelling, and an unsubstituted
    # placeholder reaches the container as literal text, so the launcher dies on a bogus token.
    expected_token_reference = "{{ RUNPOD_SECRET_" + names.inference_secret + " }}"
    assert template_env["FLASH_INFERENCE_TOKEN"] == expected_token_reference
    assert "FLASH_ARTIFACT_TOKEN" not in template_env
    assert "FLASH_ARTIFACT_TOKEN" not in transport.pods[0]["env"]
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
        ("rest", "PATCH /templates/template01"),
        ("rest", f"PATCH /pods/{POD_ID}"),
        ("graphql", "secretDelete"),
    ]


def test_secret_sentinels_are_confined_to_exact_request_sinks() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    result, factory, probe = _provision(bundle, transport)

    encoded = repr(
        (result.spec, result.status, result.handle, result.error_code, result.error_reason)
    )
    assert all(
        secret not in encoded for secret in (PROVIDER_SECRET, INFERENCE_SECRET, ARTIFACT_SECRET)
    )
    assert all(
        secret
        not in repr(
            (result.spec, result.status, result.handle, result.error_code, result.error_reason)
        )
        for secret in (PROVIDER_SECRET, INFERENCE_SECRET, ARTIFACT_SECRET)
    )
    assert factory.accepted_keys == [True]
    assert [call[1] for call in probe.calls] == [True, True]

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


def test_partial_adoption_waits_for_the_complete_resource_set() -> None:
    bundle = _bundle()
    transport = _PartialThenCompleteTransport()
    handle = _seed_exact(transport, bundle)
    transport.calls.clear()

    result, _factory, probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "ready"
    assert result.handle == handle
    assert transport.observation_number >= 2
    assert _mutation_calls(transport) == []
    assert len(probe.calls) == 1


def test_empty_then_complete_adoption_never_creates_a_duplicate_pod() -> None:
    bundle = _bundle()
    transport = _PartialThenCompleteTransport(fully_empty=True)
    handle = _seed_exact(transport, bundle)
    transport.calls.clear()

    result, _factory, probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "ready"
    assert result.handle == handle
    assert transport.observation_number >= 2
    assert _mutation_calls(transport) == [], "an invisible existing pod triggered a second create"
    assert len(probe.calls) == 1


def test_genuinely_empty_account_creates_after_the_discovery_window() -> None:
    bundle = _bundle()
    transport = _FakeTransport()

    result, _factory, _probe = _provision(bundle, transport)

    assert result.status == "ready"
    assert transport.clock.now == 30.0
    assert len([call for call in _mutation_calls(transport) if call[1] == "POST /pods"]) == 1


def test_fresh_create_waits_through_an_empty_first_readiness_observation() -> None:
    bundle = _bundle()
    transport = _PartialThenCompleteTransport(post_create=True)

    result, _factory, probe = _provision(bundle, transport)

    assert result.status == "ready"
    assert result.handle is not None
    assert result.handle.pod_id == POD_ID
    assert transport.observation_number >= 2
    assert transport.pods, "the live pod was mistaken for an absent deployment"
    assert transport.volumes, "the live volume was mistaken for an absent deployment"
    assert len(probe.calls) == 2


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((), "inference secret is missing"),
        ((object(), object()), "inference secret is duplicated"),
    ],
)
def test_exact_resource_diagnostic_distinguishes_missing_from_duplicated(
    values: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(RunPodResourceConflict, match=message):
        _one(values, "inference secret")


def test_wrong_account_fails_closed_with_sanitized_error() -> None:
    bundle = _bundle()
    transport = _FakeTransport(account_id="account-other")

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "failed"
    assert result.error_code == "authentication_failed"
    assert result.handle is None
    assert PROVIDER_SECRET not in repr(
        (result.spec, result.status, result.handle, result.error_code, result.error_reason)
    )


def test_delete_tolerating_ambiguity_rethrows_definite_failure() -> None:
    def fail() -> None:
        raise RunPodTransportFailure("provider_rejected")

    with pytest.raises(RunPodTransportFailure) as exc_info:
        _delete_tolerating_ambiguity(fail)

    assert exc_info.value.code == "provider_rejected"
    assert exc_info.value.outcome_unknown is False


def test_ambiguous_mutation_is_not_retried_and_returns_outcome_unknown() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    transport.fail_mutation_at = 1

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert result.error_reason == "mutation_outcome_unknown"
    assert transport.mutation_count == 1
    assert len(_mutation_calls(transport)) == 1


@pytest.mark.parametrize("boundary", range(1, 6))
@pytest.mark.parametrize(
    ("failure_mode", "expected_status"),
    [("definite_before", "failed"), ("ambiguous_after", "outcome_unknown")],
)
def test_create_failure_boundaries_reclaim_only_fully_confirmed_resources(
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
        assert transport.secrets == []
        assert transport.templates == []
        assert transport.volumes == []
        assert transport.pods == []
    else:
        assert result.error_code == "resource_ambiguous"
        # the failed mutation landed but its response never reached the ledger, so cleanup cannot
        # prove which matching resource is ours. preserve the whole connected set for later reclaim.
        assert len(transport.secrets) == min(boundary, 2)
        assert len(transport.volumes) == int(boundary >= 3)
        assert len(transport.templates) == int(boundary >= 4)
        assert len(transport.pods) == int(boundary >= 5)
        assert not any(
            call[1] == "secretDelete" or "DELETE" in call[1] for call in _mutation_calls(transport)
        )


def test_identity_reclaim_preserves_an_incomplete_late_visible_volume() -> None:
    bundle = _bundle()
    transport = _LateVisibleAmbiguousVolumeTransport()
    transport.fail_mutation_at = 3
    transport.failure_mode = "ambiguous_after"

    failed, _factory, _probe = _provision(bundle, transport)

    assert failed.status == "outcome_unknown"
    assert failed.error_code == "resource_ambiguous"
    assert failed.handle is None
    assert transport.hidden_volume_observations == 2
    assert len(transport.volumes) == 1
    assert not any(call[1] == "DELETE /networkvolumes/volume01" for call in transport.calls)

    transport.calls.clear()
    reclaimed = teardown_runpod_deployment(
        bundle,
        None,
        RunPodCredentials(PROVIDER_SECRET),
        deadline_at=transport.clock() + 100.0,
        transport_factory=_Factory(transport),
        clock=transport.clock,
        sleep=transport.clock.sleep,
    )

    assert reclaimed.status == "outcome_unknown"
    assert reclaimed.error_reason == "teardown_cleanup_unconfirmed"
    assert transport.hidden_volume_observations == 3
    assert len(transport.volumes) == 1
    assert _mutation_calls(transport) == []


def test_identity_reclaim_waits_for_complete_visibility_before_any_delete() -> None:
    bundle = _bundle()
    transport = _PartialThenCompleteTransport()
    _seed_exact(transport, bundle, artifact_secret=True)
    transport.calls.clear()

    reclaimed = teardown_runpod_deployment(
        bundle,
        None,
        RunPodCredentials(PROVIDER_SECRET),
        deadline_at=100.0,
        transport_factory=_Factory(transport),
        clock=transport.clock,
        sleep=transport.clock.sleep,
    )

    assert reclaimed.status == "absent"
    assert transport.pods == [], "a later-visible pod survived after its template was deleted"
    assert transport.volumes == [], "a live volume survived the partial reclaim"
    assert transport.templates == []
    assert transport.secrets == []
    assert transport.delete_observations
    assert min(transport.delete_observations) >= 2, (
        "reclaim deleted dependencies before the partial listing settled"
    )
    assert [call[1] for call in _mutation_calls(transport)] == [
        f"DELETE /pods/{POD_ID}",
        "DELETE /templates/template01",
        "DELETE /networkvolumes/volume01",
        "secretDelete",
        "secretDelete",
    ]


def test_identity_reclaim_deletes_duplicate_exact_name_volumes() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    _seed_exact(transport, bundle)
    plan = build_runpod_create_plan(bundle)
    transport.volumes.append({"id": "volume02", **plan.volume_payload()})

    reclaimed = teardown_runpod_deployment(
        bundle,
        None,
        RunPodCredentials(PROVIDER_SECRET),
        deadline_at=100.0,
        transport_factory=_Factory(transport),
        clock=transport.clock,
        sleep=transport.clock.sleep,
    )

    assert reclaimed.status == "absent"
    assert transport.volumes == []
    assert [call[1] for call in _mutation_calls(transport)] == [
        f"DELETE /pods/{POD_ID}",
        "DELETE /templates/template01",
        "DELETE /networkvolumes/volume01",
        "DELETE /networkvolumes/volume02",
        "secretDelete",
    ]


@pytest.mark.parametrize(("resource_kind", "failure_boundary"), [("template", 4), ("secret", 2)])
def test_abort_uses_observed_absence_after_ambiguous_delete(
    resource_kind: str,
    failure_boundary: int,
) -> None:
    bundle = _bundle()
    transport = _AmbiguousAbortDeleteTransport(resource_kind, delete_lands=True)
    transport.fail_mutation_at = failure_boundary
    transport.failure_mode = "definite_before"

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "failed"
    assert result.error_code == "provider_rejected"
    assert transport.secrets == []
    assert transport.templates == []
    assert transport.volumes == []
    assert transport.pods == []


@pytest.mark.parametrize(("resource_kind", "failure_boundary"), [("template", 4), ("secret", 2)])
def test_abort_stays_unknown_when_ambiguous_delete_leaves_resource_present(
    resource_kind: str,
    failure_boundary: int,
) -> None:
    bundle = _bundle()
    transport = _AmbiguousAbortDeleteTransport(resource_kind, delete_lands=False)
    transport.fail_mutation_at = failure_boundary
    transport.failure_mode = "definite_before"

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    remaining = transport.templates if resource_kind == "template" else transport.secrets
    assert len(remaining) == 1


def test_malformed_success_id_binds_no_invalid_proxy_url() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    transport.malformed_pod_id = True

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert result.handle is None
    assert len([call for call in _mutation_calls(transport) if call[1] == "POST /pods"]) == 1
    # the malformed response cannot confirm the live pod's id. keep its connected resources for
    # proof-based reclaim rather than deleting a pod that this attempt cannot establish it owns.
    assert len(transport.secrets) == 1
    assert len(transport.templates) == 1
    assert len(transport.volumes) == 1
    assert len(transport.pods) == 1
    assert not any(
        call[1] == "secretDelete" or "DELETE" in call[1] for call in _mutation_calls(transport)
    )


def test_losing_racer_never_deletes_the_winners_resources() -> None:
    """a second `serve deploy` for one generation must not tear down the first one's deployment.

    both runs build byte-identical names and identities, so the loser's plan matches the winner's
    live resources exactly. the loser conflicts on its very first secret create, which leaves that
    kind attempted but unconfirmed -- and cleanup keyed only on plan identity would then delete the
    winner's secret, pod, template, and volume, destroying a running deployment the user is paying
    for. cleanup must refuse and report the outcome as unknown instead.
    """

    bundle = _bundle()
    transport = _FakeTransport()
    # build the winner's resources, then hide them: both racers observe an empty account first, so
    # the loser proceeds to create rather than adopting. the winner lands in between.
    handle = _seed_exact(transport, bundle)
    winner_secrets = [dict(item) for item in transport.secrets]
    winner_volumes = [dict(item) for item in transport.volumes]
    winner_templates = [dict(item) for item in transport.templates]
    winner_pods = [dict(item) for item in transport.pods]
    transport.secrets, transport.volumes, transport.templates, transport.pods = [], [], [], []

    def _winner_lands() -> None:
        # only the secret: the winner is mid-provision, exactly one create ahead of the loser.
        # this is the case plan identity alone cannot survive -- with the later kinds absent,
        # nothing else betrays the resources as another run's. without a provider-confirmed id,
        # the loser has no ownership proof and must leave the secret for reclaim.
        transport.secrets = [dict(item) for item in winner_secrets]

    transport.on_first_mutation = _winner_lands
    # the loser's first mutation -- creating the inference secret -- then collides with it.
    transport.fail_mutation_at = 1
    transport.failure_mode = "definite_before"

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "outcome_unknown"
    assert transport.secrets == winner_secrets
    assert handle.pod_id == winner_pods[0]["id"]
    # and it must not have issued a single delete against them.
    assert [call for call in _mutation_calls(transport) if "DELETE" in call[1]] == []
    assert [call for call in _mutation_calls(transport) if call[1] == "secretDelete"] == []


def test_ambiguous_race_loser_leaves_unconfirmed_winner_secret_for_reclaim() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    _seed_exact(transport, bundle)
    winner_secrets = [dict(item) for item in transport.secrets]
    transport.secrets, transport.volumes, transport.templates, transport.pods = [], [], [], []

    def _winner_lands() -> None:
        transport.secrets = [dict(item) for item in winner_secrets]

    transport.on_first_mutation = _winner_lands
    transport.fail_mutation_at = 1
    transport.failure_mode = "ambiguous_before"

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    # the provider did not return this secret's id to the loser, so matching deterministic identity
    # cannot authorize deletion. declining cleanup must stay unknown so a later reclaim can prove it.
    assert result.status == "outcome_unknown"
    assert transport.secrets == winner_secrets, (
        "ambiguous cleanup deleted a secret without a confirmed provider id"
    )
    assert [call for call in _mutation_calls(transport) if "DELETE" in call[1]] == []
    assert [call for call in _mutation_calls(transport) if call[1] == "secretDelete"] == []


def test_reconcile_is_read_only_and_reports_ready_or_absent() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle)
    factory = _Factory(transport)
    clock = transport.clock

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
    clock = transport.clock

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
    assert [call[1] for call in _mutation_calls(transport)] == [
        "PATCH /templates/template01",
        f"PATCH /pods/{POD_ID}",
        "secretDelete",
    ]
    assert [item["name"] for item in transport.secrets] == [_names(bundle).inference_secret]


def test_artifact_cleanup_tolerates_provider_owned_pod_environment_entries() -> None:
    bundle = _bundle()
    plan = build_runpod_create_plan(bundle)
    transport = _FakeTransport()
    _seed_exact(transport, bundle, artifact_secret=True)
    assert transport.pods[0]["env"]["PUBLIC_KEY"] == PROVIDER_PUBLIC_KEY
    transport.calls.clear()

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "ready"
    observed_environment = dict(parse_pods(transport._list("/pods"))[0].environment)
    assert observed_environment["PUBLIC_KEY"] == PROVIDER_PUBLIC_KEY
    assert "FLASH_ARTIFACT_TOKEN" not in observed_environment
    assert all(
        observed_environment[key] == value for key, value in plan.environment_without_artifact
    )
    assert [call[1] for call in _mutation_calls(transport)] == [
        "PATCH /templates/template01",
        f"PATCH /pods/{POD_ID}",
        "secretDelete",
    ]


def test_artifact_cleanup_does_not_repatch_an_already_stripped_pod() -> None:
    bundle = _bundle()
    plan = build_runpod_create_plan(bundle)
    transport = _FakeTransport()
    _seed_exact(transport, bundle, artifact_secret=True)
    transport.pods[0]["env"] = dict(plan.environment_without_artifact)
    transport.calls.clear()

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "ready"
    assert parse_pods(transport.pods)[0].environment == plan.environment_without_artifact
    assert [call[1] for call in _mutation_calls(transport)] == [
        "PATCH /templates/template01",
        "secretDelete",
    ]


def test_artifact_pod_environment_patch_retries_transient_failure() -> None:
    # `ambiguous_before`, not `definite_before`: a 4xx `provider_rejected` answers identically on
    # every attempt, so it is terminal rather than transient. only an unknown outcome is re-observed.
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    transport.fail_mutation_at = 2
    transport.failure_mode = "ambiguous_before"
    transport.calls.clear()

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "ready"
    assert result.handle == handle
    assert transport.clock.now == 2.0
    assert [call[1] for call in _mutation_calls(transport)] == [
        "PATCH /templates/template01",
        f"PATCH /pods/{POD_ID}",
        f"PATCH /pods/{POD_ID}",
        "secretDelete",
    ]
    assert all(item["name"] != _names(bundle).artifact_secret for item in transport.secrets)


def test_artifact_environment_patch_rejection_fails_without_retrying() -> None:
    """a definite rejection answers identically every attempt, so retrying it only burns the clock.

    a live deploy spent its whole deadline re-sending a payload runpod had already rejected with
    400, then reported `outcome_unknown` for a failure the provider had stated definitively.
    """

    bundle = _bundle()

    class _PersistentPatchFailureTransport(_FakeTransport):
        def _begin_mutation(self) -> None:
            super()._begin_mutation()
            raise RunPodTransportFailure("provider_rejected")

    transport = _PersistentPatchFailureTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    transport.calls.clear()

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)
    patch_calls = [call for call in _mutation_calls(transport) if call[1].startswith("PATCH ")]

    assert result.status == "failed"
    assert result.error_code == "provider_rejected"
    assert result.error_reason == "artifact_cleanup_patch_rejected"
    assert result.handle == handle
    assert len(patch_calls) == 1, "a definite rejection must not be retried"
    assert transport.clock.now < 100.0, "the deadline must not be spent on a stated failure"
    assert any(item["name"] == _names(bundle).artifact_secret for item in transport.secrets)


def test_artifact_environment_patch_still_retries_an_unknown_outcome() -> None:
    """the fast-fail above must not swallow a genuinely ambiguous mutation, which may have landed."""

    bundle = _bundle()

    class _AmbiguousPatchFailureTransport(_FakeTransport):
        def _begin_mutation(self) -> None:
            super()._begin_mutation()
            raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)

    transport = _AmbiguousPatchFailureTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    transport.calls.clear()

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)
    patch_calls = [call for call in _mutation_calls(transport) if call[1].startswith("PATCH ")]

    assert result.status == "outcome_unknown"
    assert result.error_reason == "artifact_cleanup_patch_unknown"
    assert result.handle == handle
    assert len(patch_calls) > 1, "an unknown outcome must still be retried"


def test_template_update_payload_drops_the_create_only_key() -> None:
    """`isServerless` is accepted by template create and rejected by template update.

    runpod's update schema refuses unknown keys, so reusing the create body made every artifact
    cleanup attempt fail with 400 "Extra input keys". the create payload must still carry it.
    """

    plan = build_runpod_create_plan(_bundle())
    create = plan.template_payload(False)
    update = plan.template_update_payload()

    assert "isServerless" in create
    assert "isServerless" not in update
    assert update == {key: value for key, value in create.items() if key != "isServerless"}


def test_lost_artifact_pod_patch_response_is_confirmed_before_secret_delete() -> None:
    events: list[str] = []

    class _OrderedProbe(_Probe):
        def __call__(
            self,
            url: str,
            token: str,
            bundle: DeploymentBundle,
            timeout_seconds: float,
        ) -> bool:
            accepted = super().__call__(url, token, bundle, timeout_seconds)
            events.append("probe")
            return accepted

    class _SecretDeleteOrderingTransport(_FakeTransport):
        def graphql(
            self,
            document: str,
            variables: dict[str, object],
            *,
            mutation: bool,
            deadline_at: float,
        ) -> object:
            if mutation and "secretDelete" in document:
                events.append("secretDelete")
            return super().graphql(
                document,
                variables,
                mutation=mutation,
                deadline_at=deadline_at,
            )

    bundle = _bundle()
    plan = build_runpod_create_plan(bundle)
    transport = _SecretDeleteOrderingTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    transport.templates[0]["env"] = dict(plan.environment_without_artifact)
    transport.fail_mutation_at = 1
    transport.failure_mode = "ambiguous_after"
    transport.calls.clear()
    probe = _OrderedProbe()

    result, _factory, _probe = _provision(
        bundle,
        transport,
        artifact_token=None,
        probe=probe,
    )

    assert result.status == "ready"
    assert result.handle == handle
    assert transport.clock.now == 2.0
    assert events == ["probe", "probe", "secretDelete"]
    assert [call[1] for call in _mutation_calls(transport)] == [
        f"PATCH /pods/{POD_ID}",
        "secretDelete",
    ]
    assert all(item["name"] != _names(bundle).artifact_secret for item in transport.secrets)


def test_lost_artifact_template_patch_response_is_confirmed_without_endpoint_probe() -> None:
    bundle = _bundle()
    plan = build_runpod_create_plan(bundle)
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    transport.pods[0]["env"] = dict(plan.environment_without_artifact)
    transport.fail_mutation_at = 1
    transport.failure_mode = "ambiguous_after"
    transport.calls.clear()
    probe = _Probe()

    result, _factory, _probe = _provision(
        bundle,
        transport,
        artifact_token=None,
        probe=probe,
    )

    assert result.status == "ready"
    assert result.handle == handle
    assert transport.clock.now == 2.0
    assert len(probe.calls) == 1, "a template-only patch must not trigger an endpoint probe"
    patch_index = next(
        index
        for index, call in enumerate(transport.calls)
        if call[1] == "PATCH /templates/template01"
    )
    delete_index = next(
        index for index, call in enumerate(transport.calls) if call[1] == "secretDelete"
    )
    assert [
        call[1]
        for call in transport.calls[patch_index + 1 : delete_index]
        if call[1] == "GET /pods"
    ] == ["GET /pods", "GET /pods"]
    assert [call[1] for call in _mutation_calls(transport)] == [
        "PATCH /templates/template01",
        "secretDelete",
    ]
    assert all(item["name"] != _names(bundle).artifact_secret for item in transport.secrets)


def test_artifact_secret_survives_until_a_patched_pod_is_running_again() -> None:
    bundle = _bundle()

    class _RestartingPodAfterPatchTransport(_FakeTransport):
        def rest(
            self,
            method,
            path,
            payload,
            *,
            mutation: bool,
            deadline_at: float,
            query=None,
        ):
            response = super().rest(
                method,
                path,
                payload,
                mutation=mutation,
                deadline_at=deadline_at,
                query=query,
            )
            if method == "PATCH" and path == f"/pods/{POD_ID}":
                # runpod may restart a pod while applying its new specification. an updated env is
                # not enough to reclaim the secret until the live workload is running again.
                self.pods[0]["desiredStatus"] = "RESTARTING"
            return response

    transport = _RestartingPodAfterPatchTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    transport.calls.clear()

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=4.0,
        transport_factory=_Factory(transport),
        probe=_Probe(True),
        clock=transport.clock,
        sleep=transport.clock.sleep,
    )

    assert result.status == "failed"
    assert result.error_code == "artifact_cleanup_timeout"
    assert result.handle == handle
    assert [call[1] for call in _mutation_calls(transport)] == [
        "PATCH /templates/template01",
        f"PATCH /pods/{POD_ID}",
    ]
    assert [item["name"] for item in transport.secrets] == [
        _names(bundle).inference_secret,
        _names(bundle).artifact_secret,
    ]


def test_patched_running_pod_is_not_ready_until_its_endpoint_serves_again() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    probe = _SequenceProbe(True, False)

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=2.0,
        transport_factory=_Factory(transport),
        probe=probe,
        clock=transport.clock,
        sleep=transport.clock.sleep,
    )

    assert transport.pods[0]["desiredStatus"] == "RUNNING"
    assert result.status == "failed", (
        "desiredStatus RUNNING was accepted without post-restart endpoint proof"
    )
    assert result.error_code == "artifact_cleanup_timeout"
    assert result.error_reason == "artifact_cleanup_unproven"
    assert result.handle == handle
    assert len(probe.calls) == 2, "the endpoint was not re-probed after the pod patch"
    assert any(item["name"] == _names(bundle).artifact_secret for item in transport.secrets)


def test_patched_pod_is_ready_after_its_endpoint_serves_again() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    probe = _SequenceProbe(True, True)

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=4.0,
        transport_factory=_Factory(transport),
        probe=probe,
        clock=transport.clock,
        sleep=transport.clock.sleep,
    )

    assert result.status == "ready"
    assert result.handle == handle
    assert len(probe.calls) == 2, "the restarted pod was not proven through the endpoint"
    assert all(item["name"] != _names(bundle).artifact_secret for item in transport.secrets)


def test_initial_readiness_transient_observation_failure_uses_the_remaining_budget() -> None:
    bundle = _bundle()
    transport = _PostReadyObservationFailureTransport(failures=1)

    class _FailOneReadWhileLoading(_SequenceProbe):
        def __call__(self, url, token, probed_bundle, timeout_seconds):
            accepted = super().__call__(url, token, probed_bundle, timeout_seconds)
            if len(self.calls) == 1:
                transport.fail_observation = True
            return accepted

    result, _factory, probe = _provision(
        bundle,
        transport,
        probe=_FailOneReadWhileLoading(False, True),
    )

    assert result.status == "ready"
    assert result.handle is not None
    assert result.handle.pod_id == POD_ID
    assert transport.clock.now == 34.0, "discovery and the transient readiness read did not retry"
    assert len(probe.calls) == 3


def test_post_ready_cleanup_deadline_is_definite_without_weakening_ambiguity() -> None:
    bundle = _bundle()
    transport = _IgnoredPostReadyPodPatchTransport()
    result, _factory, probe = _provision(bundle, transport, probe=_Probe(True))

    assert result.status == "failed"
    assert result.error_code == "artifact_cleanup_timeout"
    assert result.error_reason == "artifact_cleanup_unproven"
    assert result.handle is not None
    assert result.handle.pod_id == POD_ID
    assert transport.clock.now == 70.0, "the post-ready wait exceeded its work deadline"
    assert len(probe.calls) == 1, "an unstripped pod must not be endpoint-probed"
    assert transport.pods, "a tracked ready pod must remain available for explicit teardown"
    assert any(item["name"] == _names(bundle).artifact_secret for item in transport.secrets)

    ambiguous_cases = (
        (_PostReadyObservationFailureTransport(), "artifact_cleanup_observation_failed"),
        (_PostReadyResourceConflictTransport(), "artifact_cleanup_conflict"),
        (_PostReadyIdentityDriftTransport(), "artifact_cleanup_identity_drift"),
        (_PostDeleteObservationFailureTransport(), "artifact_cleanup_observation_failed"),
        (_PostDeleteResourceConflictTransport(), "artifact_cleanup_conflict"),
    )
    for ambiguous_transport, expected_reason in ambiguous_cases:
        ambiguous, _factory, _probe = _provision(bundle, ambiguous_transport)
        assert ambiguous.status == "outcome_unknown"
        assert ambiguous.error_code == "resource_ambiguous"
        assert ambiguous.error_reason == expected_reason
        assert ambiguous.handle is not None
        assert ambiguous.handle.pod_id == POD_ID


def test_post_ready_transient_observation_failure_uses_the_remaining_budget() -> None:
    bundle = _bundle()
    transport = _PostReadyObservationFailureTransport(failures=1)

    result, _factory, _probe = _provision(bundle, transport, probe=_Probe(True))

    assert result.status == "ready"
    assert result.handle is not None
    assert result.handle.pod_id == POD_ID
    assert transport.clock.now == 32.0, "discovery and the transient read did not retry"
    assert all(item["name"] != _names(bundle).artifact_secret for item in transport.secrets)


def test_artifact_absence_deadline_is_definite_and_keeps_the_handle() -> None:
    bundle = _bundle()
    transport = _PersistentArtifactAfterDeleteTransport()

    result, _factory, _probe = _provision(bundle, transport, probe=_Probe(True))

    assert result.status == "failed"
    assert result.error_code == "artifact_cleanup_timeout"
    assert result.error_reason == "artifact_cleanup_delete_unknown"
    assert result.handle is not None
    assert result.handle.pod_id == POD_ID
    assert transport.clock.now == 70.0, "artifact confirmation exceeded its work deadline"
    assert transport.pods, "the observable pod must remain available for explicit teardown"
    assert any(item["name"] == _names(bundle).artifact_secret for item in transport.secrets)


def test_post_restart_probe_uses_the_established_user_agent() -> None:
    from flash.serve.provisioning._modal_probe import _expected_models

    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    observed: list[str | None] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {
                    "data": [
                        {"id": model_id, "flash_provenance": provenance}
                        for model_id, provenance in _expected_models(bundle).items()
                    ]
                }
            ).encode()

    def opener(request, *, timeout: float):
        observed.append(request.get_header("User-agent"))
        return Response()

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=4.0,
        transport_factory=_Factory(transport),
        probe=RunPodEndpointProbe(opener=opener),
        clock=transport.clock,
        sleep=transport.clock.sleep,
    )

    assert result.status == "ready"
    assert result.handle == handle
    assert observed == [USER_AGENT, USER_AGENT], (
        "the post-restart proof did not reuse the established RunPod endpoint probe"
    )
    assert not any((agent or "").startswith("Python-urllib") for agent in observed)


def test_artifact_reference_is_absent_before_its_secret_is_deleted() -> None:
    bundle = _bundle()
    plan = build_runpod_create_plan(bundle)

    class _ObserveTemplateAtDeleteTransport(_FakeTransport):
        def __init__(self) -> None:
            super().__init__()
            self.template_environment_at_delete: tuple[tuple[str, str], ...] | None = None
            self.pod_environment_at_delete: tuple[tuple[str, str], ...] | None = None
            self.pending_template: dict[str, object] | None = None

        def graphql(self, document, variables, *, mutation: bool, deadline_at: float):
            if mutation and "secretDelete" in document:
                self.template_environment_at_delete = tuple(
                    sorted(self.templates[0]["env"].items())
                )
                self.pod_environment_at_delete = tuple(sorted(self.pods[0]["env"].items()))
            return super().graphql(
                document,
                variables,
                mutation=mutation,
                deadline_at=deadline_at,
            )

        def rest(
            self,
            method,
            path,
            payload,
            *,
            mutation: bool,
            deadline_at: float,
            query=None,
        ):
            if method == "GET" and path == "/templates" and self.pending_template is not None:
                self.templates[0] = self.pending_template
                self.pending_template = None
            if method == "PATCH" and path == "/templates/template02":
                original = dict(self.templates[0])
                response = super().rest(
                    method,
                    path,
                    payload,
                    mutation=mutation,
                    deadline_at=deadline_at,
                    query=query,
                )
                self.pending_template = dict(self.templates[0])
                self.templates[0] = original
                return response
            return super().rest(
                method,
                path,
                payload,
                mutation=mutation,
                deadline_at=deadline_at,
                query=query,
            )

    transport = _ObserveTemplateAtDeleteTransport()
    _seed_exact(transport, bundle, artifact_secret=True)
    transport.calls.clear()

    class _ReplaceTemplateBeforeCleanupProbe(_Probe):
        def __call__(self, url, token, probed_bundle, timeout_seconds):
            accepted = super().__call__(url, token, probed_bundle, timeout_seconds)
            transport.templates[0]["id"] = "template02"
            transport.pods[0]["templateId"] = "template02"
            return accepted

    result, _factory, _probe = _provision(
        bundle,
        transport,
        artifact_token=None,
        probe=_ReplaceTemplateBeforeCleanupProbe(True),
    )
    mutations = [call[1] for call in _mutation_calls(transport)]

    assert (
        result.status,
        transport.template_environment_at_delete,
        transport.pod_environment_at_delete,
        mutations,
    ) == (
        "ready",
        plan.environment_without_artifact,
        plan.environment_without_artifact,
        ["PATCH /templates/template02", f"PATCH /pods/{POD_ID}", "secretDelete"],
    )


def test_artifact_secret_survives_if_template_reference_returns_during_pod_patch() -> None:
    bundle = _bundle()
    plan = build_runpod_create_plan(bundle)

    class _TemplateRepatchedDuringPodPatchTransport(_FakeTransport):
        def rest(
            self,
            method,
            path,
            payload,
            *,
            mutation: bool,
            deadline_at: float,
            query=None,
        ):
            response = super().rest(
                method,
                path,
                payload,
                mutation=mutation,
                deadline_at=deadline_at,
                query=query,
            )
            if method == "PATCH" and path == f"/pods/{POD_ID}":
                # a concurrent deploy can adopt and repatch the deterministic template while this
                # cleanup is waiting for its independent live-pod patch to become observable.
                self.templates[0]["env"] = dict(plan.environment_with_artifact)
            return response

    transport = _TemplateRepatchedDuringPodPatchTransport()
    _seed_exact(transport, bundle, artifact_secret=True)
    transport.calls.clear()

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=4.0,
        transport_factory=_Factory(transport),
        probe=_Probe(True),
        clock=transport.clock,
        sleep=transport.clock.sleep,
    )

    assert result.status == "failed"
    assert result.error_code == "artifact_cleanup_timeout"
    assert "FLASH_ARTIFACT_TOKEN" in transport.templates[0]["env"]
    assert any(item["name"] == _names(bundle).artifact_secret for item in transport.secrets)
    assert [call[1] for call in _mutation_calls(transport)] == [
        "PATCH /templates/template01",
        f"PATCH /pods/{POD_ID}",
    ]


def test_artifact_secret_survives_until_the_template_patch_is_observed() -> None:
    bundle = _bundle()

    class _IgnoredTemplatePatchTransport(_FakeTransport):
        def rest(
            self,
            method,
            path,
            payload,
            *,
            mutation: bool,
            deadline_at: float,
            query=None,
        ):
            if method == "PATCH" and path == "/templates/template01":
                original = dict(self.templates[0])
                response = super().rest(
                    method,
                    path,
                    payload,
                    mutation=mutation,
                    deadline_at=deadline_at,
                    query=query,
                )
                self.templates[0] = original
                return response
            return super().rest(
                method,
                path,
                payload,
                mutation=mutation,
                deadline_at=deadline_at,
                query=query,
            )

    transport = _IgnoredTemplatePatchTransport()
    _seed_exact(transport, bundle, artifact_secret=True)
    transport.calls.clear()

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=4.0,
        transport_factory=_Factory(transport),
        probe=_Probe(True),
        clock=transport.clock,
        sleep=transport.clock.sleep,
    )

    assert (
        result.status,
        [call[1] for call in _mutation_calls(transport)],
        [item["name"] for item in transport.secrets],
    ) == (
        "failed",
        ["PATCH /templates/template01", f"PATCH /pods/{POD_ID}"],
        [_names(bundle).inference_secret, _names(bundle).artifact_secret],
    )
    assert result.error_code == "artifact_cleanup_timeout"


def test_artifact_secret_survives_until_the_pod_patch_is_observed() -> None:
    bundle = _bundle()

    class _IgnoredPodPatchTransport(_FakeTransport):
        def rest(
            self,
            method,
            path,
            payload,
            *,
            mutation: bool,
            deadline_at: float,
            query=None,
        ):
            if method == "PATCH" and path == f"/pods/{POD_ID}":
                original = dict(self.pods[0])
                response = super().rest(
                    method,
                    path,
                    payload,
                    mutation=mutation,
                    deadline_at=deadline_at,
                    query=query,
                )
                # runpod may accept an update before its next observation reflects the new pod
                # specification. deleting the secret on the patch response alone leaves a live pod
                # carrying a reference to a secret that no longer exists when that update is lost.
                self.pods[0] = original
                return response
            return super().rest(
                method,
                path,
                payload,
                mutation=mutation,
                deadline_at=deadline_at,
                query=query,
            )

    transport = _IgnoredPodPatchTransport()
    _seed_exact(transport, bundle, artifact_secret=True)
    transport.calls.clear()

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=4.0,
        transport_factory=_Factory(transport),
        probe=_Probe(True),
        clock=transport.clock,
        sleep=transport.clock.sleep,
    )

    assert (
        result.status,
        [call[1] for call in _mutation_calls(transport)],
        [item["name"] for item in transport.secrets],
    ) == (
        "failed",
        ["PATCH /templates/template01", f"PATCH /pods/{POD_ID}"],
        [_names(bundle).inference_secret, _names(bundle).artifact_secret],
    )
    assert result.error_code == "artifact_cleanup_timeout"


def test_artifact_cleanup_does_not_patch_a_template_that_drifted_after_readiness() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    _seed_exact(transport, bundle, artifact_secret=True)
    transport.calls.clear()

    class _DriftingProbe(_Probe):
        def __call__(self, url, token, probed_bundle, timeout_seconds):
            accepted = super().__call__(url, token, probed_bundle, timeout_seconds)
            transport.templates[0]["imageName"] = "registry.example/other@sha256:" + "0" * 64
            return accepted

    result, _factory, _probe = _provision(
        bundle,
        transport,
        artifact_token=None,
        probe=_DriftingProbe(True),
    )

    assert (
        result.status,
        _mutation_calls(transport),
        [item["name"] for item in transport.secrets],
    ) == (
        "outcome_unknown",
        [],
        [_names(bundle).inference_secret, _names(bundle).artifact_secret],
    )


def test_production_probe_overrun_never_accepts_readiness_or_cleans_artifact() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    factory = _Factory(transport)
    clock = transport.clock
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


def test_production_probe_sends_a_non_default_user_agent() -> None:
    """cloudflare fronts the runpod pod proxy and 403s urllib's default agent.

    verified against a live pod: the same authenticated GET returned 403 error 1010
    (browser_signature_banned) under "Python-urllib/3.11" and 200 under any other agent. the probe
    maps that 403 to `False`, so without an explicit agent readiness never converges -- a pod that
    is serving correctly burns the whole deadline and the deployment reports `outcome_unknown`.
    """

    from flash.serve.provisioning._modal_probe import _expected_models

    bundle = _bundle()
    handle = _seed_exact(_FakeTransport(), bundle)
    observed: list[str | None] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            # the exact payload a correctly serving pod returns, derived from the same expectation
            # the probe checks -- so this asserts on the agent rather than on a provenance mismatch.
            return json.dumps(
                {
                    "data": [
                        {"id": model_id, "flash_provenance": provenance}
                        for model_id, provenance in _expected_models(bundle).items()
                    ]
                }
            ).encode()

    def opener(request, *, timeout: float):
        observed.append(request.get_header("User-agent"))
        return Response()

    probe = RunPodEndpointProbe(opener=opener)
    assert probe(handle.public_url, INFERENCE_SECRET, bundle, 3.0) is True
    assert observed == [USER_AGENT]
    assert not any((agent or "").startswith("Python-urllib") for agent in observed)


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


def test_landed_adoption_artifact_patch_is_confirmed_after_lost_response() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, artifact_secret=True)
    transport.fail_mutation_at = 1
    transport.failure_mode = "ambiguous_after"
    transport.calls.clear()

    result, _factory, _probe = _provision(bundle, transport, artifact_token=None)

    assert result.status == "ready"
    assert result.handle == handle
    assert transport.clock.now == 2.0
    assert [call[1] for call in _mutation_calls(transport)] == [
        "PATCH /templates/template01",
        f"PATCH /pods/{POD_ID}",
        "secretDelete",
    ]
    assert all(item["name"] != _names(bundle).artifact_secret for item in transport.secrets)


def test_teardown_deletes_pod_before_attached_volume_and_confirms_absence() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle)
    factory = _Factory(transport)
    clock = transport.clock
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
    reconciled = reconcile_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        transport_factory=factory,
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )
    assert reconciled.status == "absent"
    assert _mutation_calls(transport) == []


class _PodOutlivesDeleteTransport(_FakeTransport):
    """runpod accepts the pod delete but keeps listing the pod, as its teardown is asynchronous."""

    def _delete(self, path: str) -> None:
        if path.startswith("/pods/"):
            return
        super()._delete(path)


def test_teardown_that_cannot_prove_the_pod_is_gone_is_unknown_not_failed() -> None:
    """an unproved deletion must not be reported as a confirmed failure.

    the pod may already be gone, or may still be live and billing. `failed` reads as "nothing
    changed" and suppresses the cli's reconcile warning, so the user retries and double-provisions
    a gpu they are already paying for. the modal teardown returns unknown for exactly this case.
    """

    bundle = _bundle()
    transport = _PodOutlivesDeleteTransport()
    handle = _seed_exact(transport, bundle)
    factory = _Factory(transport)
    clock = transport.clock
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

    assert result.status == "outcome_unknown"
    # the delete was issued, and the pod outliving it is exactly what could not be confirmed.
    assert f"DELETE /pods/{POD_ID}" in [call[1] for call in _mutation_calls(transport)]
    assert transport.pods != []


@pytest.mark.parametrize("status", ["STOPPED", "FAILED"])
def test_teardown_accepts_exact_pods_in_nonready_statuses(status: str) -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, status=status)
    factory = _Factory(transport)
    clock = transport.clock
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
    clock = transport.clock

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
    assert INFERENCE_SECRET not in repr(
        (result.spec, result.status, result.handle, result.error_code, result.error_reason)
    )


def test_teardown_refuses_mismatched_exact_resource_before_deletion() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle)
    transport.templates[0]["dockerStartCmd"] = ["python", "-c", "bad"]
    factory = _Factory(transport)
    clock = transport.clock
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


def test_artifact_template_patch_ambiguity_retries_within_deadline() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    transport.fail_mutation_at = 6

    result, _factory, _probe = _provision(bundle, transport)

    assert result.status == "ready"
    assert result.handle is not None
    assert transport.clock.now == 32.0
    assert transport.mutation_count == 9
    mutations = [call[1] for call in _mutation_calls(transport)]
    assert mutations.count("PATCH /templates/template01") == 2
    assert mutations[-1] == "secretDelete"
    assert all(item["name"] != _names(bundle).artifact_secret for item in transport.secrets)


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
                "env": {"B": "2", "A": "1"},
                "templateId": "tpl0000001",
                "machine": {"gpuTypeId": "NVIDIA L4", "dataCenterId": "US-KS-2"},
                "networkVolume": {"id": "vol0000001"},
            }
        ]
    )
    assert nested[0].gpu_type_id == "NVIDIA L4"
    assert nested[0].data_center_id == "US-KS-2"
    assert nested[0].network_volume_id == "vol0000001"
    assert nested[0].environment == (("A", "1"), ("B", "2"))

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
    assert bare[0].environment == ()


def test_a_pod_awaiting_placement_is_still_recognized_as_ours() -> None:
    """absent placement means "not assigned yet", which cannot contradict the plan.

    RunPod omits `machine` until it schedules the pod -- including in the response to the create
    call itself, and in every listing while the pod is CREATED or PENDING, even with
    `includeMachine=true`. Comparing those fields for equality turned `None != "NVIDIA B200"` into
    a mismatch, and `exact_core_resources` runs *before* `readiness_state`, so the pod was rejected
    as a permanent conflict instead of being waited on: a fresh creation that had not been
    scheduled yet failed, and a rerun could not adopt one either.

    The wrong-value cases are asserted alongside because "ignore it when absent" is one edit away
    from "ignore it always", which would let this adopt a pod running on hardware nobody asked for.
    """

    plan = build_runpod_create_plan(_bundle())
    row = {
        "id": "abc123def45682",
        "name": plan.names.app_or_pod,
        "imageName": plan.bundle.image.reference,
        "gpuCount": plan.placement.gpu_count,
        "containerDiskInGb": plan.placement.container_disk_gb,
        "ports": ["8000/http"],
        "templateId": "tpl0000001",
        "networkVolume": {"id": "vol0000001"},
    }

    def _matches(**overrides) -> bool:
        pod = parse_pods([{**row, **overrides}])[0]
        return pod_identity_matches(
            plan,
            pod,
            template_id="tpl0000001",
            volume_id="vol0000001",
        )

    placed = {
        "gpuTypeId": plan.placement.gpu_type_id,
        "dataCenterId": plan.placement.data_center_id,
    }
    for status in ("CREATED", "PENDING"):
        assert _matches(desiredStatus=status), f"our own {status} pod was rejected as a conflict"
    assert _matches(desiredStatus="RUNNING", machine=placed)

    # tolerating absence is scoped to the placement window. a RUNNING pod is on real hardware, so
    # missing placement is no longer "not yet decided" -- and `exact_core_resources` runs right
    # before the probe reports `ready`, so accepting it would declare the deployment ready without
    # ever confirming the customer got the gpu and data center they asked for.
    assert not _matches(desiredStatus="RUNNING"), (
        "a running pod with no placement was accepted; readiness would confirm nothing"
    )
    # a value that *is* present must still match while pending: absence is the only tolerance.
    assert not _matches(desiredStatus="PENDING", machine={**placed, "gpuTypeId": "NVIDIA L4"})
    assert not _matches(
        desiredStatus="RUNNING",
        machine={**placed, "gpuTypeId": "NVIDIA L4"},
    )
    assert not _matches(
        desiredStatus="RUNNING",
        machine={**placed, "dataCenterId": "CA-MTL-3"},
    )


def test_a_pod_that_has_released_its_machine_still_matches_for_teardown() -> None:
    """A terminal pod stops reporting its attachments, and teardown must not read that as conflict.

    Runpod reports `networkVolumeId` and `templateId` only while the pod holds its machine. The
    create response carries neither, and a released pod stops carrying them even when the listing
    asks for `includeNetworkVolume=true` -- confirmed against the live api, where an `EXITED` pod
    still reported `machine.gpuTypeId` and `machine.dataCenterId` but neither attachment id.

    Comparing an absent id against the handle's real one raised a conflict inside
    `exact_teardown_resources`, which runs *before* the first delete -- so the pod, its volume, its
    template and its secrets were all left behind, still billing, for exactly the terminal pods
    teardown exists to remove.

    The wrong-value cases are asserted alongside for the same reason as the placement test above:
    "ignore it when absent" is one edit away from "ignore it always", which would let teardown
    delete resources attached to something other than this deployment.
    """

    plan = build_runpod_create_plan(_bundle())
    row = {
        "id": "abc123def45682",
        "name": plan.names.app_or_pod,
        "imageName": plan.bundle.image.reference,
        "gpuCount": plan.placement.gpu_count,
        "containerDiskInGb": plan.placement.container_disk_gb,
        "ports": ["8000/http"],
        "machine": {
            "gpuTypeId": plan.placement.gpu_type_id,
            "dataCenterId": plan.placement.data_center_id,
        },
    }

    def _matches(**overrides) -> bool:
        pod = parse_pods([{**row, **overrides}])[0]
        return pod_identity_matches(
            plan,
            pod,
            template_id="tpl0000001",
            volume_id="vol0000001",
        )

    attached = {"templateId": "tpl0000001", "networkVolume": {"id": "vol0000001"}}
    for status in ("EXITED", "STOPPED", "TERMINATED", "DEAD", "FAILED"):
        assert _matches(desiredStatus=status), (
            f"a {status} pod was rejected over attachments it can no longer report, so teardown "
            "would strand it and its volume, template and secrets while they keep billing"
        )
    assert _matches(desiredStatus="EXITED", **attached)

    # a RUNNING pod holds its machine, so it does report both. absence there is a real mismatch,
    # not a released resource, and must stay a conflict.
    assert not _matches(desiredStatus="RUNNING")
    # a value that *is* present must still match in every state: absence is the only tolerance.
    assert not _matches(desiredStatus="EXITED", **{**attached, "templateId": "tpl0000002"})
    assert not _matches(
        desiredStatus="EXITED",
        **{**attached, "networkVolume": {"id": "vol0000002"}},
    )


def _flip_to_running_after(transport: _FakeTransport, reads: int) -> dict[str, int]:
    """let the seeded pod finish provisioning after `reads` pod listings."""

    seen = {"n": 0}
    original = transport.rest

    def counting(method: str, path: str, payload, **kwargs):
        if method == "GET" and path.startswith("/pods"):
            seen["n"] += 1
            if seen["n"] > reads:
                for pod in transport.pods:
                    pod["desiredStatus"] = "RUNNING"
        return original(method, path, payload, **kwargs)

    transport.rest = counting  # type: ignore[assignment]
    return seen


def test_rerunning_deploy_follows_its_own_pending_pod_to_ready() -> None:
    """adoption must wait out the pod it already created, not give up on the first look.

    `serve deploy` returns `provisioning` while the pod is still CREATED/PENDING, and rerunning it
    is the documented way to continue. That rerun lands in the adoption branch, which used to
    perform a single readiness check and answer `outcome_unknown` with the entire new timeout
    unspent -- so the retry was structurally incapable of ever finishing the deployment, and a
    RUNNING pod still loading its engine failed identically. It now shares the create path's wait.
    """

    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, status="PENDING")
    # the adoption branch reads pods once itself before delegating, so the pod must stay pending
    # past that first read -- otherwise a single-check implementation would pass this test too.
    reads = _flip_to_running_after(transport, 2)
    clock = transport.clock
    probe = _Probe(True)

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=600.0,
        transport_factory=_Factory(transport),
        probe=probe,
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "ready", "adoption abandoned a pod that became ready in time"
    assert result.handle == handle
    assert reads["n"] > 1, "the pod was read once, so no waiting happened"
    assert _mutation_calls(transport) == [], "adoption must not mutate to reach readiness"


def test_adoption_reports_unproven_rather_than_failed_when_the_deadline_expires() -> None:
    """a pod adoption could not prove is unknown, not failed.

    The create path may answer `failed` because it tears down what it made in the same breath.
    Adoption owns resources it did not create and has no ledger to undo, so `failed` would invite
    the supervisor to drop the deployment record while a customer's pod keeps billing.
    """

    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, status="RUNNING")
    clock = transport.clock

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=6.0,
        transport_factory=_Factory(transport),
        probe=_Probe(False),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert result.handle == handle
    assert _mutation_calls(transport) == [], "adoption must never tear down a pod it did not create"


def test_terminal_create_waits_for_eventual_cleanup_absence() -> None:
    bundle = _bundle()
    transport = _TerminalThenEventuallyAbsentTransport()

    result, _factory, _probe = _provision(bundle, transport, probe=_Probe(False))

    assert result.status == "failed"
    assert result.error_code == "readiness_failed"
    assert result.error_reason == "readiness_terminal"
    assert transport.clock.now == 32.0
    assert transport.pods == []
    assert transport.templates == []
    assert transport.volumes == []
    assert transport.secrets == []


def test_terminal_create_with_unconfirmed_cleanup_stays_outcome_unknown() -> None:
    bundle = _bundle()
    transport = _TerminalThenUnconfirmedCleanupTransport()

    result, _factory, _probe = _provision(bundle, transport, probe=_Probe(False))

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert result.error_reason == "readiness_terminal_cleanup_unconfirmed"
    assert result.handle is not None
    assert result.handle.pod_id == POD_ID


def test_a_create_whose_pod_never_proves_readiness_is_still_torn_down() -> None:
    """readiness must not spend the deadline that teardown needs.

    A pod that runs but never answers the probe polls until the deadline is gone. Cleanup then
    inherits an expired deadline, and every transport call refuses before it is sent -- so no
    delete is issued and the pod, its template, volume, and both secrets stay live and billing,
    while the result claims this path aborted its own creation ledger.

    The probe here never accepts, which is the whole failure mode: the create succeeds, readiness
    cannot be proven, and what matters is that the provider is empty afterwards.
    """

    bundle = _bundle()
    transport = _FakeTransport()
    clock = transport.clock

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=100.0,
        transport_factory=_Factory(transport),
        probe=_Probe(False),
        clock=clock,
        sleep=clock.sleep,
    )

    assert transport.pods == [], "the pod outlived the create that could not prove it"
    assert transport.templates == []
    assert transport.volumes == []
    assert transport.secrets == [], "both secrets outlived the create"
    assert result.status == "failed", (
        f"cleanup emptied the provider, so the outcome is known: got {result.status}"
    )
    deleted = [call[1] for call in _mutation_calls(transport) if "DELETE" in call[1]]
    assert f"DELETE /pods/{POD_ID}" in deleted, "teardown never reached the pod delete"


def test_a_create_deadline_shorter_than_the_cleanup_reserve_still_creates_and_cleans_up() -> None:
    """every window is a share of the budget, not a fixed subtraction from it.

    Holding back a flat 30s from a deadline shorter than that would leave readiness starting
    already expired -- the create would fail its first call and no pod would ever exist. Halving
    the remaining budget instead gives both phases room at any deadline. Adoption's discovery
    window takes its share the same way, so confirming absence can never consume the whole budget
    and leave a short-deadline deploy structurally unable to provision.
    """

    bundle = _bundle()
    transport = _FakeTransport()
    clock = transport.clock

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET, ARTIFACT_SECRET),
        deadline_at=10.0,
        transport_factory=_Factory(transport),
        probe=_Probe(False),
        clock=clock,
        sleep=clock.sleep,
    )

    created = [call[1] for call in _mutation_calls(transport) if "POST" in call[1]]
    assert "POST /pods" in created, "the create never ran: a window consumed the whole budget"
    assert transport.pods == [], "the pod outlived a create with a deadline under the reserve"
    assert transport.volumes == []
    assert transport.secrets == []
    assert result.status == "failed"


def test_read_only_reconcile_reports_unproven_rather_than_failed_for_a_live_pod() -> None:
    """`reconcile` mutates nothing, so it can never be the caller that undoes what it doubts.

    A definite `failed` is reserved for the create path, which aborts its own ledger in the same
    breath. Reporting it from a read-only reconcile tells the supervisor to discard the deployment
    record while the pod is still there and still billing.
    """

    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, status="RUNNING")
    clock = transport.clock

    result = reconcile_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=6.0,
        transport_factory=_Factory(transport),
        probe=_Probe(False),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    assert result.handle == handle
    assert transport.pods, "the pod is still live, so the verdict must not read as discardable"
    assert _mutation_calls(transport) == [], "a read-only reconcile must not mutate provider state"


def test_read_only_reconcile_reports_unproven_rather_than_failed_for_a_terminal_pod() -> None:
    """A terminal pod is not discardable either: its resources are still in the customer account.

    `EXITED`/`FAILED` took a branch that returned a definite `failed` regardless of the caller's
    `unproven_is_failure` policy, so a read-only reconcile -- which mutates nothing and therefore
    cannot undo what it doubts -- told the supervisor the deployment was finished while the pod,
    template, volume, and secrets stayed in the account, and the cli omitted its reconciliation
    warning.
    """

    bundle = _bundle()
    transport = _FakeTransport()
    handle = _seed_exact(transport, bundle, status="EXITED")
    clock = transport.clock

    result = reconcile_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=6.0,
        transport_factory=_Factory(transport),
        probe=_Probe(False),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "outcome_unknown"
    assert result.error_reason == "readiness_terminal"
    assert result.handle == handle
    assert transport.pods, "the resources are still there, so the verdict must not read as done"
    assert _mutation_calls(transport) == [], "a read-only reconcile must not mutate provider state"


def test_a_deployment_that_vanishes_during_artifact_cleanup_is_not_reported_ready() -> None:
    """artifact absence alone cannot distinguish "token cleaned up" from "everything is gone".

    An empty observation has no artifact secret either, so a pod deleted between the readiness
    probe and this confirmation used to return `ready` with a handle whose url resolves to nothing.
    """

    bundle = _bundle()
    transport = _FakeTransport()
    _seed_exact(transport, bundle, artifact_secret=True, status="RUNNING")
    original = transport.graphql

    def wipe_after_delete(document: str, variables, *, mutation: bool, deadline_at: float):
        response = original(document, variables, mutation=mutation, deadline_at=deadline_at)
        if mutation and "secretDelete" in document:
            transport.pods.clear()
            transport.templates.clear()
            transport.volumes.clear()
            transport.secrets.clear()
        return response

    transport.graphql = wipe_after_delete  # type: ignore[assignment]
    clock = transport.clock

    result = provision_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=600.0,
        transport_factory=_Factory(transport),
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status != "ready", "a deployment that no longer exists was reported ready"
    assert result.status == "outcome_unknown"
    assert not transport.pods


def test_observation_survives_a_pod_waiting_for_its_machine() -> None:
    # a created or pending selected pod can omit machine placement. the parser preserves that
    # absence so readiness can classify it as pending rather than treating it as malformed.
    for status in ("CREATED", "PENDING"):
        pending = parse_pods(
            [
                {
                    "id": "abc123def45682",
                    "name": "queued",
                    "desiredStatus": status,
                    "imageName": "pytorch/pytorch:2.6.0",
                    "gpuCount": 1,
                    "containerDiskInGb": 60,
                    "ports": ["8000/http"],
                }
            ]
        )
        assert pending[0].gpu_type_id is None
        assert pending[0].data_center_id is None

    # a *foreign* queued pod still must not match: this one differs in name and image, which is
    # what rejects it. placement is deliberately not what does the rejecting -- see
    # `test_a_pod_awaiting_placement_is_still_recognized_as_ours` for our own unplaced pod.
    plan = build_runpod_create_plan(_bundle())
    assert not pod_identity_matches(
        plan,
        pending[0],
        template_id="tpl0000001",
        volume_id="vol0000001",
    )

    # absence must stay absence rather than become a wildcard: a placed pod is still parsed
    # strictly, so this cannot mask a real schema change that starts sending malformed placement.
    for malformed in ("", "  ", 7, True):
        with pytest.raises(ValueError, match="pod gpuTypeId must be a nonempty unpadded string"):
            parse_pods(
                [
                    {
                        "id": "abc123def45683",
                        "name": "malformed-placement",
                        "desiredStatus": "RUNNING",
                        "imageName": "pytorch/pytorch:2.6.0",
                        "gpuCount": 1,
                        "containerDiskInGb": 60,
                        "ports": ["8000/http"],
                        "machine": {"gpuTypeId": malformed, "dataCenterId": "US-KS-2"},
                    }
                ]
            )


def test_pod_observation_rejects_non_list_ports() -> None:
    for malformed in (None, "", "8000/http", 0, 12.5, {"8000": "http"}, True):
        with pytest.raises(ValueError, match="pod ports must be a list"):
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


def test_pod_observation_accepts_an_empty_environment_value() -> None:
    pod = parse_pods(
        [
            {
                "id": "abc123def45689",
                "name": "chalk-hive-sm80",
                "desiredStatus": "RUNNING",
                "imageName": "pytorch/pytorch:2.6.0",
                "gpuCount": 1,
                "containerDiskInGb": 60,
                "ports": ["8000/http"],
                "env": {"PUBLIC_KEY": "", "NORMAL_VAR": "populated"},
                "machine": {"gpuTypeId": "NVIDIA H100", "dataCenterId": "US-KS-2"},
            }
        ]
    )

    assert pod[0].environment == (("NORMAL_VAR", "populated"), ("PUBLIC_KEY", ""))


def test_a_foreign_empty_env_pod_does_not_fail_the_account_observation() -> None:
    # the whole account pod list is parsed during the pre-create observe, so a single unrelated
    # pod carrying an empty env value used to raise and surface as transport_failed, failing the
    # deploy before anything was provisioned. parsing one pod in isolation cannot catch that:
    # the property is that a foreign pod never removes ours from the observation.
    pods = parse_pods(
        [
            {
                "id": "foreignpod1234",
                "name": "chalk-hive-sm80",
                "desiredStatus": "EXITED",
                "imageName": "pytorch/pytorch:2.6.0",
                "gpuCount": 1,
                "containerDiskInGb": 60,
                "ports": ["8000/http"],
                "env": {"PUBLIC_KEY": ""},
                "machine": {"gpuTypeId": "NVIDIA H100", "dataCenterId": "US-KS-2"},
            },
            {
                "id": "ourpod98765432",
                "name": "flash-owned-pod",
                "desiredStatus": "RUNNING",
                "imageName": "ghcr.io/x/y@sha256:" + "a" * 64,
                "gpuCount": 1,
                "containerDiskInGb": 60,
                "ports": ["8000/http"],
                "env": {"REAL": "value"},
                "machine": {"gpuTypeId": "NVIDIA H100", "dataCenterId": "US-KS-2"},
            },
        ]
    )

    assert [pod.name for pod in pods] == ["chalk-hive-sm80", "flash-owned-pod"]


def test_template_observation_preserves_opaque_environment_values() -> None:
    template = parse_templates(
        [
            {
                "id": "tpl0000011",
                "name": "opaque-env-template",
                "imageName": "pytorch/pytorch:2.6.0",
                "dockerStartCmd": ["sleep", "infinity"],
                "containerDiskInGb": 60,
                "volumeMountPath": "/workspace",
                "ports": ["8000/http"],
                "env": {"EMPTY": "", "PADDED": "  keep exactly  "},
            }
        ]
    )

    assert template[0].environment == (("EMPTY", ""), ("PADDED", "  keep exactly  "))


def test_environment_still_rejects_an_empty_key() -> None:
    with pytest.raises(ValueError, match="env key must be a nonempty unpadded string"):
        parse_pods(
            [
                {
                    "id": "abc123def45688",
                    "name": "empty-env-key",
                    "desiredStatus": "RUNNING",
                    "imageName": "pytorch/pytorch:2.6.0",
                    "gpuCount": 1,
                    "containerDiskInGb": 60,
                    "ports": ["8000/http"],
                    "env": {"": "value"},
                    "machine": {"gpuTypeId": "NVIDIA L4", "dataCenterId": "US-KS-2"},
                }
            ]
        )


def test_observation_survives_a_resource_that_sets_no_environment() -> None:
    # a retained resource with an absent env carries no observed overrides. present values remain
    # strict objects, so absence does not broaden the accepted environment shape.
    pod = parse_pods(
        [
            {
                "id": "abc123def45690",
                "name": "envless",
                "desiredStatus": "RUNNING",
                "imageName": "pytorch/pytorch:2.6.0",
                "gpuCount": 1,
                "containerDiskInGb": 60,
                "ports": ["8000/http"],
                "env": None,
                "machine": {"gpuTypeId": "NVIDIA H200", "dataCenterId": "EUR-IS-4"},
            }
        ]
    )
    assert pod[0].environment == ()

    template = parse_templates(
        [
            {
                "id": "tpl0000010",
                "name": "envless-template",
                "imageName": "pytorch/pytorch:2.6.0",
                "dockerStartCmd": ["sleep", "infinity"],
                "containerDiskInGb": 60,
                "volumeMountPath": "/workspace",
                "ports": ["8000/http"],
                "env": None,
            }
        ]
    )
    assert template[0].environment == ()

    # absence must stay absence rather than becoming a wildcard that lets any env shape through: a
    # genuinely malformed value is still rejected, so this cannot mask a real schema change.
    for malformed in (0, 12.5, "FOO=bar", [{"key": "FOO", "value": "bar"}], True):
        with pytest.raises(ValueError, match="template env must be an object"):
            parse_pods(
                [
                    {
                        "id": "abc123def45691",
                        "name": "malformed-env",
                        "desiredStatus": "RUNNING",
                        "imageName": "pytorch/pytorch:2.6.0",
                        "gpuCount": 1,
                        "containerDiskInGb": 60,
                        "ports": ["8000/http"],
                        "env": malformed,
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

    for broken in ([], "", "bash -lc run.sh", 5, [""], ["ok", 3]):
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


@pytest.mark.parametrize(
    ("parser", "wrapped"),
    [
        (parse_templates, {"templates": []}),
        (parse_volumes, {"networkVolumes": []}),
        (parse_pods, {"pods": []}),
    ],
)
def test_rest_collection_parsers_reject_object_wrappers(parser, wrapped) -> None:
    with pytest.raises(ValueError, match="response must be a list"):
        parser(wrapped)


def test_pod_observation_ignores_flat_identity_fields() -> None:
    pod = parse_pods(
        [
            {
                "id": "abc123def45684",
                "name": "flat-identity",
                "desiredStatus": "RUNNING",
                "imageName": "pytorch/pytorch:2.6.0",
                "gpuTypeId": "NVIDIA L4",
                "gpuCount": 1,
                "dataCenterId": "US-KS-2",
                "containerDiskInGb": 60,
                "networkVolumeId": "vol0000001",
                "ports": ["8000/http"],
            }
        ]
    )[0]

    assert pod.gpu_type_id is None
    assert pod.data_center_id is None
    assert pod.network_volume_id is None


def test_malformed_rest_collection_is_classified_as_transport_failure() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    original = transport.rest

    def wrapped_templates(method: str, path: str, payload, **kwargs):
        response = original(method, path, payload, **kwargs)
        if method == "GET" and path == "/templates":
            return {"templates": response}
        return response

    transport.rest = wrapped_templates  # type: ignore[assignment]
    clock = transport.clock
    result = reconcile_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        transport_factory=_Factory(transport),
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )

    assert result.status == "failed"
    assert result.error_code == "transport_failed"


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
    clock = transport.clock

    reconciled = reconcile_runpod_deployment(
        bundle,
        RunPodCredentials(PROVIDER_SECRET),
        ServingRuntimeSecrets(INFERENCE_SECRET),
        deadline_at=100.0,
        transport_factory=_Factory(transport),
        probe=_Probe(True),
        clock=clock,
        sleep=clock.sleep,
    )

    # absent, not transport_failed: flash owns nothing here, and the foreign row is not flash's
    # business to validate.
    assert reconciled.status == "absent"
    assert reconciled.error_code is None


def test_observation_filters_foreign_cpu_pods_before_strict_field_parsing() -> None:
    bundle = _bundle()
    transport = _FakeTransport()
    expected = _seed_exact(transport, bundle)
    transport.pods.insert(
        0,
        {
            "id": "cpu123def4567",
            "name": "someone-elses-cpu-pod",
            "desiredStatus": "RUNNING",
            "imageName": "ubuntu:24.04",
            "gpuCount": 0,
            # cpu-only account pods need neither gpu placement nor flash's container disk field.
        },
    )
    plan = build_runpod_create_plan(bundle)

    observed = _observe(plan, transport, deadline_at=100.0)

    assert [pod.id for pod in observed.pods] == [expected.pod_id]
    assert [pod.name for pod in observed.pods] == [plan.names.app_or_pod]


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


class _CleanupFailsAfterInterruptTransport(_FakeTransport):
    """Ctrl-C during the readiness poll, then a cleanup delete that cannot be confirmed.

    the interrupt fires from the probe, and the teardown delete this drives then fails ambiguously
    -- the shape a real expired deadline produces, since the abort reuses the original deadline.
    """

    def __init__(self, account_id: str = "account-01") -> None:
        super().__init__(account_id)
        self.interrupted = False

    def rest(
        self,
        method,
        path,
        payload,
        *,
        mutation: bool,
        deadline_at: float,
        query: dict[str, str] | None = None,
    ):
        # `query` must be accepted even though this override ignores it: the listing reads pass it,
        # and omitting it makes them fail with a TypeError long before the interrupt.
        #
        # only the pod delete that runs AFTER the interrupt fails, and ambiguously. an
        # unconditional failure would fire during creation rollback instead, so provisioning would
        # return `transport_failed` and the probe -- and therefore the interrupt path -- would
        # never run at all. failing reads would break the observation the abort depends on and
        # make it bail out before reaching the branch under test.
        if self.interrupted and mutation and method == "DELETE" and path.startswith("/pods/"):
            raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)
        return super().rest(
            method, path, payload, mutation=mutation, deadline_at=deadline_at, query=query
        )


def test_unconfirmed_runpod_interrupt_cleanup_is_reported_not_discarded() -> None:
    """the cleanup result must reach the caller when it cannot prove the pod is gone.

    `_failure_after_create_attempt` already distinguishes "confirmed absent" from
    "outcome_unknown", but the interrupt path discarded its return value and re-raised, so the cli
    printed only "aborted" while the pod and volume could still be live and billing. the interrupt
    still propagates -- `InterruptedProvisioning` subclasses `KeyboardInterrupt` -- and now names
    the provider so the cli can warn first.
    """
    bundle = _bundle()
    transport = _CleanupFailsAfterInterruptTransport()

    def _interrupt(url: str, token: str, probed, timeout: float) -> bool:
        transport.interrupted = True
        raise KeyboardInterrupt

    with pytest.raises(InterruptedProvisioning) as raised:
        _provision(bundle, transport, probe=_interrupt)

    assert raised.value.provider == "runpod"
    assert isinstance(raised.value, KeyboardInterrupt)


def test_confirmed_runpod_interrupt_cleanup_stays_a_plain_interrupt() -> None:
    # when teardown proves every resource is gone there is no ambiguity to report, and raising the
    # carrier would warn about billing resources that were provably removed.
    bundle = _bundle()
    transport = _FakeTransport()

    with pytest.raises(KeyboardInterrupt) as raised:
        _provision(bundle, transport, probe=_InterruptingProbe())

    assert not isinstance(raised.value, InterruptedProvisioning)
    assert not transport.pods, transport.pods


def _exact_models_payload(bundle: DeploymentBundle) -> dict[str, object]:
    """the payload a correct runpod pod returns: every revision AND every run alias."""

    from flash.serve.provisioning._modal_probe import _expected_models

    return {
        "data": [
            {"id": model_id, "flash_provenance": provenance}
            for model_id, provenance in _expected_models(bundle).items()
        ]
    }


def test_runpod_readiness_requires_the_same_exact_provenance_as_modal() -> None:
    """readiness must mean the same thing on both providers.

    the runpod check compared only five deployment-wide fields and accepted any superset of the
    revision ids (`expected <= observed`). a pod that omitted the run alias, served stale extra
    models, or reported wrong per-adapter provenance therefore probed "ready", and the customer
    got a successful deployment whose documented alias request could 404 or route to the wrong
    adapter.
    """
    bundle = _bundle()
    exact = _exact_models_payload(bundle)
    assert _provenance_matches(exact, bundle), "a correct pod must still be accepted"

    # the alias is what the documented request uses, so dropping it cannot read as ready.
    aliases = set(bundle.manifest.aliases)
    assert aliases, "the fixture must carry an alias for this to test anything"
    missing_alias = {"data": [entry for entry in exact["data"] if entry["id"] not in aliases]}
    assert not _provenance_matches(missing_alias, bundle)

    # a stale extra model means the pod is not serving exactly this deployment.
    extra = {"data": [*exact["data"], {"id": "stale-model", "flash_provenance": {}}]}
    assert not _provenance_matches(extra, bundle)

    # per-adapter provenance must match, not merely the deployment-wide fields.
    wrong = json.loads(json.dumps(exact))
    wrong["data"][0]["flash_provenance"]["requested_model"] = "someone-elses-model"
    assert not _provenance_matches(wrong, bundle)


class _NoCapacityPodTransport(_FakeTransport):
    """the real shape runpod returns when it has no gpu of the planned type to give.

    measured live: `POST /pods` answers HTTP 500 with
    `{"error":"create pod: There are no instances currently available"}`, which the transport now
    reads as a definite `capacity_unavailable`. this covers what the lifecycle does with it: the
    four resources created before the pod must be torn down and the failure reported plainly.
    """

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
        if method == "POST" and path == "/pods":
            raise RunPodTransportFailure("capacity_unavailable")
        return super().rest(
            method, path, payload, mutation=mutation, deadline_at=deadline_at, query=query
        )


def test_proven_empty_abort_reports_failure_not_an_unknown_outcome() -> None:
    """a create that leaves nothing behind must not warn about resources that may be billing.

    `_abort_created_resources` only returns True after re-observing the account and finding
    `resource_count == 0`, which is an authoritative proof that nothing exists to bill. discarding
    that proof sends the user to `serve status` and `serve undeploy` for resources that provably
    do not exist. the interrupt path already makes exactly this distinction.
    """
    bundle = _bundle()
    transport = _NoCapacityPodTransport()

    result, _factory, _probe = _provision(bundle, transport)

    assert result.status == "failed", result.status
    assert result.error_code == "capacity_unavailable"
    assert result.handle is None
    # nothing survived the abort, which is what makes the plain failure honest.
    assert not transport.pods, transport.pods
    assert not transport.templates, transport.templates
    assert not transport.volumes, transport.volumes


def test_no_capacity_500_is_a_definite_rejection_not_an_ambiguous_outcome() -> None:
    """runpod reports a plain capacity shortfall as a 500, which must not read as ambiguous.

    measured live against `POST /pods`:
    `{"error":"create pod: There are no instances currently available","status":500}`.
    nothing is created, so the outcome is not in doubt -- but a blanket 5xx-on-mutation rule
    reports `outcome_unknown`, telling the user resources may be billing and sending them to
    `serve status` and `serve undeploy` for a pod that never existed. runpod already answers 402
    and 429 with `capacity_unavailable`; this is the same condition wearing a different status.
    """
    body = b'{"error":"create pod: There are no instances currently available","status":500}'

    def opener(_request, *, timeout: float):
        raise urllib.error.HTTPError(
            "https://rest.runpod.io/v1/pods", 500, "sanitized", None, io.BytesIO(body)
        )

    transport = StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)
    with pytest.raises(RunPodTransportFailure) as exc_info:
        transport.rest("POST", "/pods", {}, mutation=True, deadline_at=10.0)
    assert exc_info.value.code == "capacity_unavailable"
    assert exc_info.value.outcome_unknown is False


def test_no_matching_machine_500_is_a_definite_rejection_not_an_ambiguous_outcome() -> None:
    """`POST /pods` states the same capacity shortfall in different words, and it must classify alike.

    measured live in US-KS-2 against the real account:
    `{"error":"create pod: could not find any pods with required specifications","status":500}`.
    the pod was never created -- the instrumented run showed the volume and template that preceded
    it being deleted cleanly -- yet the deploy reported `mutation_outcome_unknown` and told the
    user to go hunt resources that provably did not exist. matching only the `no instances
    currently available` wording left this second phrasing in the ambiguous bucket.
    """
    body = (
        b'{"error":"create pod: could not find any pods with required specifications","status":500}'
    )

    def opener(_request, *, timeout: float):
        raise urllib.error.HTTPError(
            "https://rest.runpod.io/v1/pods", 500, "sanitized", None, io.BytesIO(body)
        )

    transport = StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)
    with pytest.raises(RunPodTransportFailure) as exc_info:
        transport.rest("POST", "/pods", {}, mutation=True, deadline_at=10.0)
    assert exc_info.value.code == "capacity_unavailable"
    assert exc_info.value.outcome_unknown is False


def test_unsupported_datacenter_500_is_a_definite_rejection_not_an_ambiguous_outcome() -> None:
    """an unsupported datacenter is a config error runpod also reports as a 500.

    measured live against `POST /networkvolumes` with `US-TX-4`, which has L40S capacity but no
    network-volume support: `{"error":"create network volume: Data center \\"US-TX-4\\" not found
    or does not support network volumes. Available data centers: ...","status":500}`. the provider
    refused before creating anything and its body names the datacenters that would work, so
    reporting `outcome_unknown` both invents a cleanup problem and buries the fix. `capacity` is
    the wrong code here -- waiting will never help, the request itself has to change.
    """
    body = (
        b'{"error":"create network volume: Data center \\"US-TX-4\\" not found or does not '
        b'support network volumes. Available data centers: US-TX-3, US-WA-1.","status":500}'
    )

    def opener(_request, *, timeout: float):
        raise urllib.error.HTTPError(
            "https://rest.runpod.io/v1/networkvolumes", 500, "sanitized", None, io.BytesIO(body)
        )

    transport = StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)
    with pytest.raises(RunPodTransportFailure) as exc_info:
        transport.rest("POST", "/networkvolumes", {}, mutation=True, deadline_at=10.0)
    assert exc_info.value.code == "provider_rejected"
    assert exc_info.value.outcome_unknown is False


def test_unrelated_500_stays_ambiguous_on_a_mutation() -> None:
    """only the provider's own stated refusals are definite; every other 5xx stays unknown.

    a 500 whose cause is not stated may well have landed, so narrowing must not leak into the
    general case that protects a possibly-created resource from being forgotten.
    """

    def opener(_request, *, timeout: float):
        raise urllib.error.HTTPError(
            "https://rest.runpod.io/v1/pods",
            500,
            "sanitized",
            None,
            io.BytesIO(b'{"error":"internal server error"}'),
        )

    transport = StdlibRunPodTransport(PROVIDER_SECRET, opener=opener, clock=lambda: 0.0)
    with pytest.raises(RunPodTransportFailure) as exc_info:
        transport.rest("POST", "/pods", {}, mutation=True, deadline_at=10.0)
    assert exc_info.value.code == "resource_ambiguous"
    assert exc_info.value.outcome_unknown is True
