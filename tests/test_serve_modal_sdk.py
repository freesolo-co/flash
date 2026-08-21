"""exact controlled seams for the pinned modal sdk adapter."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.serve.control import ModalCredentials
from flash.serve.provisioning._modal_plan import (
    MODAL_VOLUME_MOUNT,
    MODAL_WRAPPER_REMOTE_PATH,
    build_modal_create_plan,
)
from flash.serve.provisioning._modal_sdk import ModalSdkFailure, PinnedModalSdk
from flash.serve.provisioning._modal_wrapper import launch_modal_server
from tests.test_serve_provisioning_modal import (
    APP_ID,
    ARTIFACT_SECRET,
    ARTIFACT_SECRET_ID,
    FUNCTION_ID,
    INFERENCE_SECRET,
    INFERENCE_SECRET_ID,
    PROVIDER_ID,
    PROVIDER_SECRET,
    VOLUME_ID,
    _bundle,
)

ROOT = Path(__file__).resolve().parents[1]


class _NamedHandle:
    def __init__(
        self,
        *,
        name: str,
        object_id: str,
        hydrate_error: bool = False,
    ) -> None:
        self.name = name
        self.object_id = object_id
        self.hydrate_error = hydrate_error

    def hydrate(self):
        if self.hydrate_error:
            raise RuntimeError(PROVIDER_SECRET)
        return self


class _FunctionHandle:
    def __init__(self, module, object_id: str) -> None:
        self.module = module
        self.object_id = object_id

    def hydrate(self, client):
        assert client is self.module.client
        self.module.calls["function_hydrate_client"] = client
        return self

    def get_web_url(self):
        return self.module.plan.expected_public_url


class _Image:
    def __init__(self, calls: dict[str, object]) -> None:
        self.calls = calls

    def add_local_file(self, local_path: str, remote_path: str, *, copy: bool):
        self.calls["add_local_file"] = (local_path, remote_path, copy)
        return self


class _LookupApp:
    def __init__(self, module) -> None:
        self.module = module
        self.app_id = APP_ID

    def get_tags(self, *, client):
        assert client is self.module.client
        self.module.calls["get_tags_client"] = client
        return dict(self.module.deployed_tags)


class _DeployedApp:
    def __init__(self, module, name: str, *, tags: dict[str, str], include_source: bool) -> None:
        self.module = module
        self.name = name
        self.tags = tags
        self.app_id = None
        module.calls["app"] = {
            "name": name,
            "tags": tags,
            "include_source": include_source,
        }

    def function(self, **kwargs):
        self.module.calls["function"] = kwargs

        def decorate(value):
            self.module.calls["function_target"] = value
            return value

        return decorate

    def deploy(
        self,
        *,
        name: str,
        environment_name: str,
        tag: str,
        client,
        strategy: str,
    ):
        assert client is self.module.client
        self.module.calls["deploy"] = {
            "name": name,
            "environment_name": environment_name,
            "tag": tag,
            "client": client,
            "strategy": strategy,
        }
        self.app_id = APP_ID
        self.module.deployed_tags = dict(self.tags)
        self.module.deployed_apps = [SimpleNamespace(app_id=APP_ID, name=name, containers=0)]
        return self


class _AppApi:
    def __init__(self, module) -> None:
        self.module = module

    def __call__(self, name: str, *, tags: dict[str, str], include_source: bool):
        return _DeployedApp(self.module, name, tags=tags, include_source=include_source)

    def lookup(
        self,
        name: str,
        *,
        client,
        environment_name: str,
        create_if_missing: bool,
    ):
        assert client is self.module.client
        self.module.calls["app_lookup"] = (
            name,
            environment_name,
            create_if_missing,
            client,
        )
        return _LookupApp(self.module)


class _SecretObjects:
    def __init__(self, module) -> None:
        self.module = module

    def create(
        self,
        name: str,
        values: dict[str, str],
        *,
        allow_existing: bool,
        environment_name: str,
        client,
    ) -> None:
        assert client is self.module.client
        object_id = INFERENCE_SECRET_ID if "FLASH_INFERENCE_TOKEN" in values else ARTIFACT_SECRET_ID
        self.module.resources[("secret", name)] = _NamedHandle(
            name=name,
            object_id=object_id,
        )
        self.module.mutations.append(
            (
                "create_secret",
                name,
                {
                    "keys": tuple(sorted(values)),
                    "values_expected": all(
                        value in {INFERENCE_SECRET, ARTIFACT_SECRET} for value in values.values()
                    ),
                },
                {
                    "allow_existing": allow_existing,
                    "environment_name": environment_name,
                    "client": client,
                },
            )
        )

    def list(self, *, environment_name: str, client):
        assert client is self.module.client
        self.module.calls["list_secret"] = {
            "environment_name": environment_name,
            "client": client,
        }
        return [
            resource
            for (kind, _name), resource in self.module.resources.items()
            if kind == "secret"
        ]

    def delete(
        self,
        name: str,
        *,
        allow_missing: bool,
        environment_name: str,
        client,
    ) -> None:
        assert client is self.module.client
        self.module.mutations.append(
            (
                "delete_secret",
                name,
                None,
                {
                    "allow_missing": allow_missing,
                    "environment_name": environment_name,
                    "client": client,
                },
            )
        )
        self.module.resources.pop(("secret", name), None)


class _VolumeObjects:
    def __init__(self, module) -> None:
        self.module = module

    def create(
        self,
        name: str,
        *,
        allow_existing: bool,
        environment_name: str,
        client,
    ) -> None:
        assert client is self.module.client
        self.module.resources[("volume", name)] = _NamedHandle(
            name=name,
            object_id=VOLUME_ID,
        )
        self.module.mutations.append(
            (
                "create_volume",
                name,
                None,
                {
                    "allow_existing": allow_existing,
                    "environment_name": environment_name,
                    "client": client,
                },
            )
        )

    def list(self, *, environment_name: str, client):
        assert client is self.module.client
        self.module.calls["list_volume"] = {
            "environment_name": environment_name,
            "client": client,
        }
        return [
            resource
            for (kind, _name), resource in self.module.resources.items()
            if kind == "volume"
        ]

    def delete(
        self,
        name: str,
        *,
        allow_missing: bool,
        environment_name: str,
        client,
    ) -> None:
        assert client is self.module.client
        self.module.mutations.append(
            (
                "delete_volume",
                name,
                None,
                {
                    "allow_missing": allow_missing,
                    "environment_name": environment_name,
                    "client": client,
                },
            )
        )
        self.module.resources.pop(("volume", name), None)


class _SecretApi:
    def __init__(self, module) -> None:
        self.module = module
        self.objects = _SecretObjects(module)

    def from_name(
        self,
        name: str,
        *,
        environment_name: str,
        required_keys: list[str],
        client,
    ):
        assert client is self.module.client
        self.module.calls.setdefault("secret_from_name", []).append(
            (name, environment_name, tuple(required_keys), client)
        )
        return self.module.resources[("secret", name)]


class _VolumeApi:
    def __init__(self, module) -> None:
        self.module = module
        self.objects = _VolumeObjects(module)

    def from_name(
        self,
        name: str,
        *,
        environment_name: str,
        create_if_missing: bool,
        client,
    ):
        assert client is self.module.client
        self.module.calls.setdefault("volume_from_name", []).append(
            (name, environment_name, create_if_missing, client)
        )
        return self.module.resources[("volume", name)]


class _Client:
    def __init__(self, module) -> None:
        self.module = module
        self.close_count = 0
        self.close_error = False

    def _close(self) -> None:
        self.close_count += 1
        if self.close_error:
            raise RuntimeError(PROVIDER_SECRET)


class _Experimental:
    def __init__(self, module) -> None:
        self.module = module

    def list_deployed_apps(self, *, environment_name: str, client):
        assert client is self.module.client
        self.module.calls["list_deployed_apps"] = (environment_name, client)
        return list(self.module.deployed_apps)

    def get_app_objects(self, name: str, *, environment_name: str, client):
        assert client is self.module.client
        self.module.calls["get_app_objects"] = (name, environment_name, client)
        return {
            self.module.plan.function_name: _FunctionHandle(
                self.module,
                self.module.function_id,
            )
        }

    def get_app_lifecycle(self, app_id: str, *, client):
        assert client is self.module.client
        self.module.calls["get_app_lifecycle"] = (app_id, client)
        return SimpleNamespace(stopped_at=self.module.lifecycle_stopped_at)

    def stop_app(self, name: str, *, environment_name: str, client) -> None:
        assert client is self.module.client
        self.module.mutations.append(
            (
                "stop_app",
                name,
                None,
                {"environment_name": environment_name, "client": client},
            )
        )
        self.module.deployed_apps.clear()


class _ModalModule:
    def __init__(self, plan) -> None:
        self.plan = plan
        self.calls: dict[str, object] = {}
        self.mutations: list[tuple[str, str, object, dict[str, object]]] = []
        self.resources: dict[tuple[str, str], _NamedHandle] = {}
        self.deployed_apps: list[object] = []
        self.deployed_tags: dict[str, str] = {}
        self.function_id = FUNCTION_ID
        self.lifecycle_stopped_at = None
        self.workspace_error = False
        self.environment_error = False
        self.workspace_name = plan.placement.workspace_name
        self.environment_name = plan.placement.environment
        self.client = _Client(self)
        self.Secret = _SecretApi(self)
        self.Volume = _VolumeApi(self)
        self.experimental = _Experimental(self)
        self.Client = SimpleNamespace(from_credentials=self._from_credentials)
        self.Workspace = SimpleNamespace(from_context=self._workspace)
        self.Environment = SimpleNamespace(from_name=self._environment)
        self.Image = SimpleNamespace(from_registry=self._image)
        self.App = _AppApi(self)

    def _from_credentials(self, token_id: str, token_secret: str):
        self.calls["client_credentials"] = (
            token_id == PROVIDER_ID,
            token_secret == PROVIDER_SECRET,
        )
        return self.client

    def _workspace(self, *, client):
        assert client is self.client
        self.calls["workspace_client"] = client
        return _NamedHandle(
            name=self.workspace_name,
            object_id="ac-workspace",
            hydrate_error=self.workspace_error,
        )

    def _environment(
        self,
        name: str,
        *,
        create_if_missing: bool,
        client,
    ):
        assert client is self.client
        self.calls["environment"] = (name, create_if_missing, client)
        return _NamedHandle(
            name=self.environment_name,
            object_id="en-main",
            hydrate_error=self.environment_error,
        )

    def _image(self, reference: str):
        self.calls["image_reference"] = reference
        return _Image(self.calls)

    def web_server(self, port: int, *, startup_timeout: int, label: str):
        self.calls["web_server"] = (port, startup_timeout, label)

        def decorate(value):
            self.calls["web_server_target"] = value
            return value

        return decorate


def _sdk(plan, modal: _ModalModule) -> PinnedModalSdk:
    return PinnedModalSdk(
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        plan,
        module_loader=lambda: modal,
    )


def test_lazy_loader_binds_pinned_experimental_surface_without_provider_calls() -> None:
    program = r"""
import importlib
import importlib.metadata
import socket

import modal

provider_calls = []
network_calls = []


def forbidden_provider_call(*args, **kwargs):
    provider_calls.append((args, kwargs))
    raise AssertionError("provider operation attempted")


def forbidden_network_call(*args, **kwargs):
    network_calls.append((args, kwargs))
    raise AssertionError("network operation attempted")


assert importlib.metadata.version("modal") == "1.5.4"
assert not hasattr(modal, "experimental")
modal.Client.from_credentials = forbidden_provider_call
modal.Client.from_env = forbidden_provider_call
socket.create_connection = forbidden_network_call

from flash.serve.provisioning._modal_sdk import _load_modal_module

loaded = _load_modal_module()
assert loaded is modal
assert loaded.experimental is importlib.import_module("modal.experimental")
for name in ("list_deployed_apps", "get_app_objects", "get_app_lifecycle", "stop_app"):
    assert callable(getattr(loaded.experimental, name))
assert provider_calls == []
assert network_calls == []
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fixed_wrapper_uses_packaged_secret_scrubbing_child_boundary(monkeypatch) -> None:
    from flash.serve.app import launch

    calls = []
    monkeypatch.setattr(launch, "start_launcher_process", lambda: calls.append(True))
    launch_modal_server()
    assert calls == [True]


def test_wrapper_returns_while_the_child_still_runs(monkeypatch) -> None:
    """modal probes the port only *after* this call returns, so it must not block.

    `user_code_imports.py` runs the `web_server` callable, then `asgi.wait_for_web_server` and
    `asgi.web_server_proxy`. Waiting on the child here never returns, so neither the readiness
    probe nor the proxy is ever built and the endpoint serves nothing. A blocking wrapper looks
    like a tidier lifetime but is a total outage; this pins the spawn-and-return contract.
    """

    from flash.serve.app import launch

    class _NeverExits:
        def wait(self, timeout: float | None = None) -> int:
            raise AssertionError("wrapper must not wait on the child; modal needs it to return")

    monkeypatch.setattr(launch, "start_launcher_process", lambda: _NeverExits())
    launch_modal_server()


def test_pinned_sdk_binds_exact_client_workspace_environment_and_secret_sinks() -> None:
    plan = build_modal_create_plan(_bundle(), phase="bootstrap")
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal)

    assert sdk.workspace_name == plan.placement.workspace_name
    assert sdk.environment_name == plan.placement.environment
    assert modal.calls["client_credentials"] == (True, True)
    assert modal.calls["workspace_client"] is modal.client
    assert modal.calls["environment"] == (
        plan.placement.environment,
        False,
        modal.client,
    )

    inference = sdk.create_inference_secret(plan, INFERENCE_SECRET)
    artifact = sdk.create_artifact_secret(plan, ARTIFACT_SECRET)
    volume = sdk.create_volume(plan)
    assert inference.id == INFERENCE_SECRET_ID
    assert artifact.id == ARTIFACT_SECRET_ID
    assert volume.id == VOLUME_ID
    secret_creates = [entry for entry in modal.mutations if entry[0] == "create_secret"]
    assert [entry[2]["keys"] for entry in secret_creates] == [
        ("FLASH_INFERENCE_TOKEN",),
        ("FLASH_ARTIFACT_TOKEN",),
    ]
    assert all(entry[2]["values_expected"] is True for entry in secret_creates)
    assert all(entry[3]["allow_existing"] is False for entry in secret_creates)
    assert PROVIDER_SECRET not in repr(modal.calls) + repr(modal.mutations)

    sdk.close()
    assert modal.client.close_count == 1


@pytest.mark.parametrize(
    "failure",
    [
        "workspace_hydration",
        "environment_hydration",
        "workspace_validation",
        "environment_validation",
    ],
)
def test_client_closes_once_when_workspace_or_environment_binding_fails(failure: str) -> None:
    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)
    if failure == "workspace_hydration":
        modal.workspace_error = True
    elif failure == "environment_hydration":
        modal.environment_error = True
    elif failure == "workspace_validation":
        modal.workspace_name = "other-workspace"
    else:
        modal.environment_name = "other-environment"
    modal.client.close_error = True

    with pytest.raises(ModalSdkFailure) as exc_info:
        _sdk(plan, modal)

    assert exc_info.value.code == "authentication_failed"
    assert modal.client.close_count == 1
    assert PROVIDER_SECRET not in str(exc_info.value) + repr(exc_info.value)


def test_pinned_sdk_deploys_exact_bootstrap_and_finalized_phases() -> None:
    bootstrap = build_modal_create_plan(_bundle(), phase="bootstrap")
    modal = _ModalModule(bootstrap)
    sdk = _sdk(bootstrap, modal)
    sdk.create_inference_secret(bootstrap, INFERENCE_SECRET)
    sdk.create_artifact_secret(bootstrap, ARTIFACT_SECRET)
    sdk.create_volume(bootstrap)

    app_id = sdk.deploy_app(bootstrap)

    assert app_id == APP_ID
    assert modal.calls["image_reference"] == bootstrap.bundle.image.reference
    local_path, remote_path, copy = modal.calls["add_local_file"]
    assert local_path == bootstrap.wrapper_local_path
    assert remote_path == MODAL_WRAPPER_REMOTE_PATH
    assert copy is True
    assert modal.calls["app"] == {
        "name": bootstrap.names.app_or_pod,
        "tags": dict(bootstrap.tags),
        "include_source": False,
    }
    function = modal.calls["function"]
    assert function["env"] == dict(bootstrap.environment)
    assert len(function["secrets"]) == 2
    assert function["gpu"] == "B200:1"
    assert function["serialized"] is False
    assert function["volumes"] == {
        MODAL_VOLUME_MOUNT: modal.resources[("volume", bootstrap.names.volume)]
    }
    assert function["min_containers"] == 0
    assert function["max_containers"] == 1
    assert function["buffer_containers"] == 0
    assert function["scaledown_window"] == bootstrap.scaledown_window_seconds
    assert function["startup_timeout"] == bootstrap.startup_timeout_seconds
    assert function["name"] == bootstrap.function_name
    assert function["region"] == "us-east-1"
    assert function["include_source"] is False
    assert modal.calls["web_server"] == (
        8000,
        bootstrap.startup_timeout_seconds,
        bootstrap.endpoint_label,
    )
    assert modal.calls["deploy"] == {
        "name": bootstrap.names.app_or_pod,
        "environment_name": bootstrap.placement.environment,
        "tag": bootstrap.deployment_tag,
        "client": modal.client,
        "strategy": "recreate",
    }

    finalized = build_modal_create_plan(_bundle(), phase="finalized")
    modal.plan = finalized
    sdk.deploy_app(finalized)
    assert modal.calls["app"]["tags"] == dict(finalized.tags)
    assert len(modal.calls["function"]["secrets"]) == 1


def test_observe_lifecycle_stop_and_deletes_use_exact_pinned_signatures() -> None:
    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal)
    sdk.create_inference_secret(plan, INFERENCE_SECRET)
    sdk.create_artifact_secret(plan, ARTIFACT_SECRET)
    sdk.create_volume(plan)
    sdk.deploy_app(plan)
    modal.mutations.clear()

    deployed = sdk.observe(plan)
    assert deployed.apps[0].app_id == APP_ID
    assert deployed.apps[0].function_id == FUNCTION_ID
    assert modal.calls["list_secret"] == {
        "environment_name": plan.placement.environment,
        "client": modal.client,
    }
    assert modal.calls["list_volume"] == {
        "environment_name": plan.placement.environment,
        "client": modal.client,
    }
    assert modal.calls["app_lookup"] == (
        plan.names.app_or_pod,
        plan.placement.environment,
        False,
        modal.client,
    )
    assert modal.calls["get_app_objects"] == (
        plan.names.app_or_pod,
        plan.placement.environment,
        modal.client,
    )

    # a lifecycle-only app is resolved by id: modal cannot look one up by name once it leaves the
    # deployed listing, and exposes no tag surface for it, so tags stay empty whatever the app kept.
    modal.deployed_apps.clear()
    pending = sdk.observe(plan, app_id_hint=APP_ID)
    assert pending.apps[0].state == "lifecycle_pending"
    assert pending.apps[0].running_containers is None
    assert pending.apps[0].tags == ()
    assert modal.calls["get_app_lifecycle"] == (APP_ID, modal.client)

    sdk.stop_app(plan)
    modal.lifecycle_stopped_at = object()
    stopped = sdk.observe(plan, app_id_hint=APP_ID)
    assert stopped.apps[0].state == "stopped"
    assert stopped.apps[0].running_containers == 0
    assert stopped.apps[0].tags == ()
    assert modal.deployed_tags == dict(plan.tags)
    sdk.delete_secret(plan, plan.names.artifact_secret)
    sdk.delete_secret(plan, plan.names.inference_secret)
    sdk.delete_volume(plan)
    assert modal.mutations == [
        (
            "stop_app",
            plan.names.app_or_pod,
            None,
            {"environment_name": plan.placement.environment, "client": modal.client},
        ),
        (
            "delete_secret",
            plan.names.artifact_secret,
            None,
            {
                "allow_missing": True,
                "environment_name": plan.placement.environment,
                "client": modal.client,
            },
        ),
        (
            "delete_secret",
            plan.names.inference_secret,
            None,
            {
                "allow_missing": True,
                "environment_name": plan.placement.environment,
                "client": modal.client,
            },
        ),
        (
            "delete_volume",
            plan.names.volume,
            None,
            {
                "allow_missing": True,
                "environment_name": plan.placement.environment,
                "client": modal.client,
            },
        ),
    ]


@pytest.mark.parametrize(
    ("role", "malformed"),
    [
        ("app", "ap-short"),
        ("function", "fu-short"),
        ("secret", "st-bad/value"),
        ("volume", "xx-" + "V" * 22),
    ],
)
def test_observation_rejects_malformed_role_specific_provider_ids(
    role: str,
    malformed: str,
) -> None:
    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal)
    sdk.create_inference_secret(plan, INFERENCE_SECRET)
    sdk.create_volume(plan)
    sdk.deploy_app(plan)
    if role == "app":
        modal.deployed_apps[0].app_id = malformed
    elif role == "function":
        modal.function_id = malformed
    elif role == "secret":
        modal.resources[("secret", plan.names.inference_secret)].object_id = malformed
    else:
        modal.resources[("volume", plan.names.volume)].object_id = malformed

    with pytest.raises(ModalSdkFailure) as exc_info:
        sdk.observe(plan)
    assert exc_info.value.code == "transport_failed"


def test_post_create_malformed_observation_is_outcome_unknown_without_retry() -> None:
    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal)

    def malformed_lookup(
        name: str,
        *,
        environment_name: str,
        required_keys: list[str],
        client,
    ):
        raise RuntimeError(PROVIDER_SECRET)

    modal.Secret.from_name = malformed_lookup
    with pytest.raises(ModalSdkFailure) as exc_info:
        sdk.create_inference_secret(plan, INFERENCE_SECRET)
    assert exc_info.value.code == "resource_ambiguous"
    assert exc_info.value.outcome_unknown is True
    creates = [entry for entry in modal.mutations if entry[0] == "create_secret"]
    assert len(creates) == 1
    assert PROVIDER_SECRET not in str(exc_info.value) + repr(exc_info.value)


def test_include_source_topology_sabotage_is_rejected_before_sdk_calls() -> None:
    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal)
    sabotaged = replace(plan, include_source=True)
    with pytest.raises(ValueError, match="source inclusion"):
        sdk.deploy_app(sabotaged)
    assert "image_reference" not in modal.calls
