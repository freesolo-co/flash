"""exact controlled seams for the pinned modal sdk adapter."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from synchronicity import Synchronizer

from flash.serve.app.manifest import build_serving_manifest
from flash.serve.control import ModalCredentials
from flash.serve.deployment.profiles import get_profile, placement_for
from flash.serve.provisioning import DeploymentBundle, ServingImage
from flash.serve.provisioning.modal.execution.sdk import ModalSdkFailure, PinnedModalSdk
from flash.serve.provisioning.modal.planning.plan import MODAL_VOLUME_MOUNT, build_modal_create_plan
from flash.serve.provisioning.modal.planning.wrapper import launch_modal_server
from tests.test_serve_app_manifest import _profile_spec_and_inputs
from tests.test_serve_provisioning_modal import (
    APP_ID,
    ARTIFACT_SECRET,
    ARTIFACT_SECRET_ID,
    FUNCTION_ID,
    INFERENCE_SECRET,
    INFERENCE_SECRET_ID,
    OTHER_APP_ID,
    PROVIDER_ID,
    PROVIDER_SECRET,
    VOLUME_ID,
    _bundle,
)

_MODAL_SYNCHRONIZER = Synchronizer()


@pytest.fixture(scope="session", autouse=True)
def _close_modal_synchronizer():
    yield
    _MODAL_SYNCHRONIZER._close_loop()


class _AioCallable:
    def __init__(self, operation) -> None:
        self.operation = operation

    def __call__(self, *args, **kwargs):
        return self.operation(*args, **kwargs)

    async def aio(self, *args, **kwargs):
        return self.operation(*args, **kwargs)


class _AioMethod:
    def __init__(self, operation) -> None:
        self.operation = operation

    def __get__(self, instance, owner):
        return _AioCallable(self.operation.__get__(instance, owner))


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

    hydrate = _AioMethod(hydrate)


class _FunctionHandle:
    def __init__(self, module, object_id: str) -> None:
        self.module = module
        self.object_id = object_id

    def hydrate(self, client):
        assert client is self.module.client
        self.module.calls["function_hydrate_client"] = client
        return self

    hydrate = _AioMethod(hydrate)

    def get_web_url(self):
        return self.module.plan.expected_public_url

    get_web_url = _AioMethod(get_web_url)


class _Image:
    def add_local_file(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("digest-pinned image must not be mutated with local wrapper code")


class _LookupApp:
    def __init__(self, module) -> None:
        self.module = module
        self.app_id = APP_ID

    def get_tags(self, *, client):
        assert client is self.module.client
        self.module.calls["get_tags_client"] = client
        return dict(self.module.deployed_tags)

    get_tags = _AioMethod(get_tags)


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

    deploy = _AioMethod(deploy)


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

    lookup = _AioMethod(lookup)


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

    create = _AioMethod(create)

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

    create = _AioMethod(create)

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


class _IdMutationStub:
    def __init__(self, module) -> None:
        self.module = module

    async def _before_mutation(self) -> None:
        self.module.id_mutation_loops.append(asyncio.get_running_loop())
        if self.module.block_id_mutation:
            await asyncio.Event().wait()

    async def AppStop(self, request) -> None:
        await self._before_mutation()
        self.module.mutations.append(("stop_app_by_id", request.app_id, request.source, {}))
        self.module.deployed_apps = [
            app for app in self.module.deployed_apps if app.app_id != request.app_id
        ]

    async def SecretDelete(self, request) -> None:
        await self._before_mutation()
        self.module.mutations.append(("delete_secret_by_id", request.secret_id, None, {}))
        self.module.resources = {
            key: resource
            for key, resource in self.module.resources.items()
            if resource.object_id != request.secret_id
        }

    async def VolumeDelete(self, request) -> None:
        await self._before_mutation()
        self.module.mutations.append(("delete_volume_by_id", request.volume_id, None, {}))
        self.module.resources = {
            key: resource
            for key, resource in self.module.resources.items()
            if resource.object_id != request.volume_id
        }


class _Client:
    def __init__(self, module) -> None:
        self.module = module
        self.close_count = 0
        self.close_error = False
        self.stub = _IdMutationStub(module)
        self._sync_synchronizer = _MODAL_SYNCHRONIZER

    def _close(self) -> None:
        self.close_count += 1
        if self.close_error:
            raise RuntimeError(PROVIDER_SECRET)

    _close = _AioMethod(_close)


class _Experimental:
    def __init__(self, module) -> None:
        self.module = module

    def list_deployed_apps(self, *, environment_name: str, client):
        assert client is self.module.client
        self.module.calls["list_deployed_apps"] = (environment_name, client)
        return list(self.module.deployed_apps)

    list_deployed_apps = _AioMethod(list_deployed_apps)

    def get_app_objects(self, name: str, *, environment_name: str, client):
        assert client is self.module.client
        self.module.calls["get_app_objects"] = (name, environment_name, client)
        return {
            self.module.plan.function_name: _FunctionHandle(
                self.module,
                self.module.function_id,
            )
        }

    get_app_objects = _AioMethod(get_app_objects)

    def get_app_lifecycle(self, app_id: str, *, client):
        assert client is self.module.client
        self.module.calls["get_app_lifecycle"] = (app_id, client)
        return SimpleNamespace(stopped_at=self.module.lifecycle_stopped_at)

    get_app_lifecycle = _AioMethod(get_app_lifecycle)

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
        self.client_owner_loop = None
        self.id_mutation_loops: list[asyncio.AbstractEventLoop] = []
        self.block_id_mutation = False
        self.client = _Client(self)
        self.Secret = _SecretApi(self)
        self.Secret.objects.list = _AioCallable(self.Secret.objects.list)
        self.Volume = _VolumeApi(self)
        self.Volume.objects.list = _AioCallable(self.Volume.objects.list)
        self.experimental = _Experimental(self)

        async def from_credentials(token_id: str, token_secret: str):
            self.client_owner_loop = asyncio.get_running_loop()
            return self._from_credentials(token_id, token_secret)

        self.Client = SimpleNamespace(
            from_credentials=_MODAL_SYNCHRONIZER.create_blocking(from_credentials)
        )
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
        image = _Image()
        self.calls["image"] = image
        return image

    def web_server(self, port: int, *, startup_timeout: int, label: str):
        self.calls["web_server"] = (port, startup_timeout, label)

        def decorate(value):
            self.calls["web_server_target"] = value
            return value

        return decorate


def _sdk(
    plan, modal: _ModalModule, *, deadline_at: float = 60.0, clock=lambda: 0.0
) -> PinnedModalSdk:
    return PinnedModalSdk(
        ModalCredentials(PROVIDER_ID, PROVIDER_SECRET),
        plan,
        deadline_at,
        clock,
        module_loader=lambda: modal,
    )


def test_blocking_provider_wrapper_is_bypassed_for_deadline_bound_aio() -> None:
    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)

    class _BlockingFromCredentials:
        sync_calls = 0

        def __call__(self, _token_id: str, _token_secret: str):
            self.sync_calls += 1
            time.sleep(0.2)
            return modal.client

        async def aio(self, _token_id: str, _token_secret: str):
            await __import__("asyncio").Event().wait()

    blocking = _BlockingFromCredentials()
    modal.Client = SimpleNamespace(from_credentials=blocking)
    started_at = time.monotonic()

    with pytest.raises(ModalSdkFailure) as exc_info:
        _sdk(plan, modal, deadline_at=started_at + 0.01, clock=time.monotonic)

    assert exc_info.value.code == "transport_failed"
    assert time.monotonic() - started_at < 0.1
    assert blocking.sync_calls == 0


class _DeferredThread:
    def __init__(self, *, target, daemon: bool) -> None:
        self.target = target
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True

    def run(self) -> None:
        self.target()


def _defer_modal_watcher(monkeypatch) -> list[_DeferredThread]:
    from flash.serve.provisioning.modal.planning import wrapper as _modal_wrapper

    watchers = []

    def make_thread(*, target, daemon: bool) -> _DeferredThread:
        watcher = _DeferredThread(target=target, daemon=daemon)
        watchers.append(watcher)
        return watcher

    monkeypatch.setattr(_modal_wrapper, "Thread", make_thread)
    return watchers


def test_fixed_wrapper_uses_packaged_secret_scrubbing_child_boundary(monkeypatch) -> None:
    from flash.serve.app import launch

    class _Exited:
        def wait(self) -> int:
            return 0

    calls = []

    def start():
        calls.append(True)
        return _Exited()

    _defer_modal_watcher(monkeypatch)
    monkeypatch.setattr(launch, "start_launcher_process", start)
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

    class _Running:
        wait_calls = 0

        def wait(self) -> int:
            self.wait_calls += 1
            raise AssertionError("the caller thread waited on a live child")

    process = _Running()
    watchers = _defer_modal_watcher(monkeypatch)
    monkeypatch.setattr(launch, "start_launcher_process", lambda: process)
    launch_modal_server()

    assert process.wait_calls == 0
    assert len(watchers) == 1
    assert watchers[0].started is True
    assert watchers[0].daemon is True


def _assert_modal_wrapper_signals_parent(monkeypatch, exit_code: int) -> None:
    import signal

    from flash.serve.app import launch
    from flash.serve.provisioning.modal.planning import wrapper as _modal_wrapper

    parent_pid = 2468
    signals = []

    class _Exited:
        def wait(self) -> int:
            return exit_code

    watchers = _defer_modal_watcher(monkeypatch)
    monkeypatch.setattr(launch, "start_launcher_process", lambda: _Exited())
    monkeypatch.setattr(_modal_wrapper, "getpid", lambda: parent_pid)
    monkeypatch.setattr(_modal_wrapper, "kill", lambda pid, signum: signals.append((pid, signum)))
    launch_modal_server()
    watchers[0].run()

    assert signals == [(parent_pid, signal.SIGTERM)]


def test_wrapper_signals_parent_after_nonzero_child_exit(monkeypatch) -> None:
    _assert_modal_wrapper_signals_parent(monkeypatch, 17)


def test_wrapper_signals_parent_after_zero_child_exit(monkeypatch) -> None:
    _assert_modal_wrapper_signals_parent(monkeypatch, 0)


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


def test_client_close_interruption_does_not_escape_cleanup() -> None:
    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal)

    def interrupt_close() -> None:
        modal.client.close_count += 1
        raise KeyboardInterrupt

    modal.client._close = _AioCallable(interrupt_close)

    sdk.close()

    assert modal.client.close_count == 1


def _profile_modal_bundle(model_id: str) -> DeploymentBundle:
    spec, inputs = _profile_spec_and_inputs(model_id)
    profile = get_profile(model_id)
    placement = placement_for(
        profile,
        "modal",
        workspace_name="workspace",
        environment="main",
        region="us-east",
    )
    spec = replace(spec, placement=placement)
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
    ("model_id", "expected_gpu"),
    [
        ("Qwen/Qwen3.5-9B", "L40S:1"),
        ("Qwen/Qwen3.8-27B", "H100!:1"),
        ("Qwen/Qwen3.6-35B-A3B", "H200:1"),
    ],
)
def test_profile_gpu_request_reaches_the_literal_modal_function_payload(
    model_id: str, expected_gpu: str
) -> None:
    plan = build_modal_create_plan(_profile_modal_bundle(model_id), phase="bootstrap")
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal)
    sdk.create_inference_secret(plan, INFERENCE_SECRET)
    sdk.create_artifact_secret(plan, ARTIFACT_SECRET)
    sdk.create_volume(plan)

    sdk.deploy_app(plan)

    assert modal.calls["function"]["gpu"] == expected_gpu


def test_pinned_sdk_deploys_exact_image_without_local_source_overlays() -> None:
    bootstrap = build_modal_create_plan(_bundle(), phase="bootstrap")
    modal = _ModalModule(bootstrap)
    sdk = _sdk(bootstrap, modal)
    sdk.create_inference_secret(bootstrap, INFERENCE_SECRET)
    sdk.create_artifact_secret(bootstrap, ARTIFACT_SECRET)
    sdk.create_volume(bootstrap)

    app_id = sdk.deploy_app(bootstrap)

    assert app_id == APP_ID
    assert modal.calls["image_reference"] == bootstrap.bundle.image.reference
    assert modal.calls["function"]["image"] is modal.calls["image"]
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
    assert modal.calls["function_target"] is launch_modal_server
    assert launch_modal_server.__module__ == "flash.serve.provisioning.modal.planning.wrapper"
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


def test_destructive_mutations_bind_confirmed_provider_ids() -> None:
    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal)
    sdk.create_inference_secret(plan, INFERENCE_SECRET)
    sdk.create_artifact_secret(plan, ARTIFACT_SECRET)
    sdk.create_volume(plan)
    sdk.deploy_app(plan)
    modal.mutations.clear()
    modal.deployed_apps = [
        SimpleNamespace(app_id=OTHER_APP_ID, name=plan.names.app_or_pod, containers=0)
    ]
    newer_artifact_id = "st-" + "R" * 22
    newer_inference_id = "st-" + "N" * 22
    newer_volume_id = "vo-" + "R" * 22
    modal.resources[("secret", plan.names.artifact_secret)] = _NamedHandle(
        name=plan.names.artifact_secret, object_id=newer_artifact_id
    )
    modal.resources[("secret", plan.names.inference_secret)] = _NamedHandle(
        name=plan.names.inference_secret, object_id=newer_inference_id
    )
    modal.resources[("volume", plan.names.volume)] = _NamedHandle(
        name=plan.names.volume, object_id=newer_volume_id
    )

    sdk.stop_app(plan, APP_ID)
    sdk.delete_secret(plan, ARTIFACT_SECRET_ID)
    sdk.delete_secret(plan, INFERENCE_SECRET_ID)
    sdk.delete_volume(plan, VOLUME_ID)

    assert modal.deployed_apps[0].app_id == OTHER_APP_ID
    assert {resource.object_id for resource in modal.resources.values()} == {
        newer_artifact_id,
        newer_inference_id,
        newer_volume_id,
    }
    assert modal.mutations == [
        ("stop_app_by_id", APP_ID, 2, {}),
        ("delete_secret_by_id", ARTIFACT_SECRET_ID, None, {}),
        ("delete_secret_by_id", INFERENCE_SECRET_ID, None, {}),
        ("delete_volume_by_id", VOLUME_ID, None, {}),
    ]


def test_id_mutation_runs_on_the_modal_client_owning_loop() -> None:
    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal)

    try:
        sdk.delete_secret(plan, INFERENCE_SECRET_ID)

        assert modal.client_owner_loop is not None
        assert modal.id_mutation_loops == [modal.client_owner_loop]
    finally:
        sdk.close()


def test_id_mutation_deadline_is_ambiguous_and_cancels_the_owning_loop_rpc() -> None:
    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal, deadline_at=time.monotonic() + 1.0, clock=time.monotonic)
    modal.block_id_mutation = True
    started_at = time.monotonic()

    try:
        with pytest.raises(ModalSdkFailure) as exc_info:
            sdk.delete_secret(plan, INFERENCE_SECRET_ID, deadline_at=started_at + 0.02)

        assert time.monotonic() - started_at < 0.2
        assert exc_info.value.code == "resource_ambiguous"
        assert exc_info.value.outcome_unknown is True
        assert modal.id_mutation_loops == [modal.client_owner_loop]
        assert modal.mutations == []
    finally:
        sdk.close()


def test_id_mutation_declines_when_the_generated_request_is_unavailable(monkeypatch) -> None:
    from flash.serve.provisioning.modal.execution import sdk as _modal_sdk

    plan = build_modal_create_plan(_bundle())
    modal = _ModalModule(plan)
    sdk = _sdk(plan, modal)
    monkeypatch.setattr(_modal_sdk.importlib, "import_module", lambda _name: SimpleNamespace())

    with pytest.raises(ModalSdkFailure) as exc_info:
        sdk.stop_app(plan, APP_ID)

    assert exc_info.value.code == "resource_ambiguous"
    assert exc_info.value.outcome_unknown is True
    assert modal.mutations == []


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

    sdk.stop_app(plan, APP_ID)
    modal.lifecycle_stopped_at = object()
    stopped = sdk.observe(plan, app_id_hint=APP_ID)
    assert stopped.apps[0].state == "stopped"
    assert stopped.apps[0].running_containers == 0
    assert stopped.apps[0].tags == ()
    assert modal.deployed_tags == dict(plan.tags)
    sdk.delete_secret(plan, ARTIFACT_SECRET_ID)
    sdk.delete_secret(plan, INFERENCE_SECRET_ID)
    sdk.delete_volume(plan, VOLUME_ID)
    assert modal.mutations == [
        ("stop_app_by_id", APP_ID, 2, {}),
        ("delete_secret_by_id", ARTIFACT_SECRET_ID, None, {}),
        ("delete_secret_by_id", INFERENCE_SECRET_ID, None, {}),
        ("delete_volume_by_id", VOLUME_ID, None, {}),
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
