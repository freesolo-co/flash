"""lazy import-light adapter for the pinned modal sdk surface."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from flash.serve.control import DeploymentErrorCode, ModalCredentials
from flash.serve.control._urls import validate_modal_public_url
from flash.serve.control.types import validate_modal_provider_id
from flash.serve.provisioning.common.records import Clock
from flash.serve.provisioning.modal.planning.plan import (
    MODAL_VOLUME_MOUNT,
    ModalCreatePlan,
    validate_modal_plan,
)

ModalAppState = Literal["deployed", "lifecycle_pending", "stopped", "failed"]

_FAILURE_MESSAGES = {
    "authentication_failed": "modal authentication failed",
    "conflict": "modal resource conflict",
    "provider_rejected": "modal rejected the operation",
    "resource_ambiguous": "modal resource outcome is ambiguous",
    "transport_failed": "modal transport failed",
}


class ModalSdkFailure(RuntimeError):
    """one sanitized modal sdk failure without retained provider data."""

    __slots__ = ("code", "outcome_unknown")

    def __init__(
        self,
        code: DeploymentErrorCode,
        *,
        outcome_unknown: bool = False,
    ) -> None:
        if code not in _FAILURE_MESSAGES:
            raise ValueError("modal sdk failure code is not allowlisted")
        self.code = code
        self.outcome_unknown = outcome_unknown
        super().__init__(_FAILURE_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class ModalNamedResource:
    """one opaque modal resource identity with no secret value or sdk object."""

    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ModalAppObservation:
    """one allowlisted deployed or terminal modal app observation."""

    app_id: str
    app_name: str
    state: ModalAppState
    running_containers: int | None
    tags: tuple[tuple[str, str], ...]
    function_id: str | None
    function_name: str | None
    public_url: str | None


@dataclass(frozen=True, slots=True)
class ModalObservation:
    """one workspace and environment scoped view of deterministic resources."""

    workspace_name: str
    environment_name: str
    apps: tuple[ModalAppObservation, ...]
    volumes: tuple[ModalNamedResource, ...]
    inference_secrets: tuple[ModalNamedResource, ...]
    artifact_secrets: tuple[ModalNamedResource, ...]

    @property
    def resource_count(self) -> int:
        return sum(
            len(values)
            for values in (
                self.apps,
                self.volumes,
                self.inference_secrets,
                self.artifact_secrets,
            )
        )


class ModalSdk(Protocol):
    workspace_name: str
    environment_name: str

    def observe(
        self,
        plan: ModalCreatePlan,
        *,
        app_id_hint: str | None = None,
        deadline_at: float | None = None,
    ) -> ModalObservation: ...

    def create_inference_secret(
        self,
        plan: ModalCreatePlan,
        value: str,
        *,
        deadline_at: float | None = None,
    ) -> ModalNamedResource: ...

    def create_artifact_secret(
        self,
        plan: ModalCreatePlan,
        value: str,
        *,
        deadline_at: float | None = None,
    ) -> ModalNamedResource: ...

    def create_volume(
        self,
        plan: ModalCreatePlan,
        *,
        deadline_at: float | None = None,
    ) -> ModalNamedResource: ...

    def deploy_app(self, plan: ModalCreatePlan, *, deadline_at: float | None = None) -> str: ...

    def stop_app(
        self,
        plan: ModalCreatePlan,
        app_id: str,
        *,
        deadline_at: float | None = None,
    ) -> None: ...

    def delete_secret(
        self,
        plan: ModalCreatePlan,
        secret_id: str,
        *,
        deadline_at: float | None = None,
    ) -> None: ...

    def delete_volume(
        self,
        plan: ModalCreatePlan,
        volume_id: str,
        *,
        deadline_at: float | None = None,
    ) -> None: ...

    def close(self) -> None: ...


ModalSdkFactory = Callable[[ModalCredentials, ModalCreatePlan, float, Clock], ModalSdk]
ModuleLoader = Callable[[], object]


def _load_modal_module() -> object:
    modal = importlib.import_module("modal")
    experimental = importlib.import_module("modal.experimental")
    modal.experimental = experimental
    return modal


AsyncOperation = Callable[[], Awaitable[object]]


async def _wait_for_operation(operation: AsyncOperation, timeout_seconds: float) -> object:
    return await asyncio.wait_for(operation(), timeout=timeout_seconds)


class _AsyncBridge:
    """run every request-local modal async operation on one event loop."""

    __slots__ = ("_clock", "_loop")

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._loop = asyncio.new_event_loop()

    def run(self, operation: AsyncOperation, *, deadline_at: float) -> object:
        timeout_seconds = deadline_at - self._clock()
        if timeout_seconds <= 0:
            raise TimeoutError
        return self._loop.run_until_complete(_wait_for_operation(operation, timeout_seconds))

    def close(self) -> None:
        self._loop.close()


def _sync_value(
    operation: AsyncOperation,
    *,
    deadline_at: float,
    clock: Clock = time.monotonic,
) -> object:
    bridge = _AsyncBridge(clock)
    try:
        return bridge.run(operation, deadline_at=deadline_at)
    finally:
        bridge.close()


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ModalSdkFailure("transport_failed")
    return value


def _provider_id(value: object, role: str) -> str:
    try:
        return validate_modal_provider_id(value, role)
    except ValueError:
        raise ModalSdkFailure("transport_failed") from None


def _resource(value: object, role: Literal["secret", "volume"]) -> ModalNamedResource:
    return ModalNamedResource(
        id=_provider_id(getattr(value, "object_id", None), role),
        name=_text(getattr(value, "name", None), f"modal {role} name"),
    )


def _run_async(
    operation: AsyncOperation,
    *,
    deadline_at: float,
    clock: Clock,
    bridge: _AsyncBridge | None,
) -> object:
    if bridge is not None:
        return bridge.run(operation, deadline_at=deadline_at)
    return _sync_value(operation, deadline_at=deadline_at, clock=clock)


def _call_read(
    operation: AsyncOperation,
    *,
    deadline_at: float,
    clock: Clock = time.monotonic,
    bridge: _AsyncBridge | None = None,
) -> object:
    try:
        return _run_async(
            operation,
            deadline_at=deadline_at,
            clock=clock,
            bridge=bridge,
        )
    except ModalSdkFailure:
        raise
    except Exception:
        raise ModalSdkFailure("transport_failed") from None


def _call_mutation(
    operation: AsyncOperation,
    *,
    deadline_at: float,
    clock: Clock = time.monotonic,
    bridge: _AsyncBridge | None = None,
) -> object:
    try:
        return _run_async(
            operation,
            deadline_at=deadline_at,
            clock=clock,
            bridge=bridge,
        )
    except ModalSdkFailure:
        raise
    except Exception:
        raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True) from None


def _post_mutation_read(operation: Callable[[], object]) -> object:
    try:
        return operation()
    except Exception:
        raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True) from None


def _close_client(
    client: object,
    *,
    deadline_at: float,
    clock: Clock = time.monotonic,
    bridge: _AsyncBridge | None = None,
) -> None:
    close = getattr(getattr(client, "_close", None), "aio", None)
    if callable(close):
        with contextlib.suppress(BaseException):
            _run_async(
                close,
                deadline_at=deadline_at,
                clock=clock,
                bridge=bridge,
            )


class PinnedModalSdk:
    """request-scoped adapter for the locally pinned modal 1.5 sdk contract."""

    __slots__ = (
        "_bridge",
        "_client",
        "_clock",
        "_deadline_at",
        "_modal",
        "environment_name",
        "workspace_name",
    )

    def __init__(
        self,
        credentials: ModalCredentials,
        plan: ModalCreatePlan,
        deadline_at: float,
        clock: Clock = time.monotonic,
        *,
        module_loader: ModuleLoader = _load_modal_module,
    ) -> None:
        if type(credentials) is not ModalCredentials:
            raise ValueError("modal credentials must use the exact credential type")
        validate_modal_plan(plan)
        token_id, token_secret = credentials.reveal()
        bridge = _AsyncBridge(clock)
        client: object | None = None
        try:
            modal = module_loader()
            client = bridge.run(
                lambda: modal.Client.from_credentials.aio(token_id, token_secret),
                deadline_at=deadline_at,
            )
            workspace_handle = modal.Workspace.from_context(client=client)
            workspace = bridge.run(
                workspace_handle.hydrate.aio,
                deadline_at=deadline_at,
            )
            environment_handle = modal.Environment.from_name(
                plan.placement.environment,
                create_if_missing=False,
                client=client,
            )
            environment = bridge.run(
                environment_handle.hydrate.aio,
                deadline_at=deadline_at,
            )
            workspace_name = _text(getattr(workspace, "name", None), "modal workspace name")
            environment_name = _text(getattr(environment, "name", None), "modal environment name")
            if (
                workspace_name != plan.placement.workspace_name
                or environment_name != plan.placement.environment
            ):
                raise ModalSdkFailure("authentication_failed")
        except TimeoutError:
            if client is not None:
                _close_client(client, deadline_at=deadline_at, clock=clock, bridge=bridge)
            bridge.close()
            raise ModalSdkFailure("transport_failed") from None
        except ModalSdkFailure:
            if client is not None:
                _close_client(client, deadline_at=deadline_at, clock=clock, bridge=bridge)
            bridge.close()
            raise
        except Exception:
            if client is not None:
                _close_client(client, deadline_at=deadline_at, clock=clock, bridge=bridge)
            bridge.close()
            raise ModalSdkFailure("authentication_failed") from None
        except BaseException:
            if client is not None:
                _close_client(client, deadline_at=deadline_at, clock=clock, bridge=bridge)
            bridge.close()
            raise
        self._modal = modal
        self._client = client
        self._bridge = bridge
        self._deadline_at = deadline_at
        self._clock = clock
        self.workspace_name = workspace_name
        self.environment_name = environment_name

    def __repr__(self) -> str:
        return "PinnedModalSdk(<request-scoped>)"

    def _operation_deadline(self, deadline_at: float | None) -> float:
        return self._deadline_at if deadline_at is None else min(deadline_at, self._deadline_at)

    def _read(
        self,
        operation: AsyncOperation,
        *,
        deadline_at: float | None,
    ) -> object:
        return _call_read(
            operation,
            deadline_at=self._operation_deadline(deadline_at),
            clock=self._clock,
            bridge=self._bridge,
        )

    def _mutate(
        self,
        operation: AsyncOperation,
        *,
        deadline_at: float | None,
    ) -> object:
        return _call_mutation(
            operation,
            deadline_at=self._operation_deadline(deadline_at),
            clock=self._clock,
            bridge=self._bridge,
        )

    def _list_named(
        self,
        manager: object,
        plan: ModalCreatePlan,
        role: Literal["secret", "volume"],
        *,
        deadline_at: float | None,
    ) -> tuple[ModalNamedResource, ...]:
        values = self._read(
            lambda: manager.list.aio(
                environment_name=plan.placement.environment,
                client=self._client,
            ),
            deadline_at=deadline_at,
        )
        if type(values) is not list:
            raise ModalSdkFailure("transport_failed")
        return tuple(_resource(value, role) for value in values)

    def _deployed_apps(
        self,
        plan: ModalCreatePlan,
        *,
        deadline_at: float | None,
    ) -> list[object]:
        values = self._read(
            lambda: self._modal.experimental.list_deployed_apps.aio(
                environment_name=plan.placement.environment,
                client=self._client,
            ),
            deadline_at=deadline_at,
        )
        if type(values) is not list:
            raise ModalSdkFailure("transport_failed")
        return [value for value in values if getattr(value, "name", None) == plan.names.app_or_pod]

    def _deployed_app(
        self,
        plan: ModalCreatePlan,
        value: object,
        *,
        deadline_at: float | None,
    ) -> ModalAppObservation:
        app_id = _provider_id(getattr(value, "app_id", None), "app")
        containers = getattr(value, "containers", None)
        if type(containers) is not int or containers < 0:
            raise ModalSdkFailure("transport_failed")
        app = self._read(
            lambda: self._modal.App.lookup.aio(
                plan.names.app_or_pod,
                client=self._client,
                environment_name=plan.placement.environment,
                create_if_missing=False,
            ),
            deadline_at=deadline_at,
        )
        if _provider_id(getattr(app, "app_id", None), "app") != app_id:
            raise ModalSdkFailure("transport_failed")
        tags = self._read(
            lambda: app.get_tags.aio(client=self._client),
            deadline_at=deadline_at,
        )
        if type(tags) is not dict or any(
            type(key) is not str or type(item) is not str for key, item in tags.items()
        ):
            raise ModalSdkFailure("transport_failed")
        objects = self._read(
            lambda: self._modal.experimental.get_app_objects.aio(
                plan.names.app_or_pod,
                environment_name=plan.placement.environment,
                client=self._client,
            ),
            deadline_at=deadline_at,
        )
        if type(objects) is not dict or len(objects) != 1 or plan.function_name not in objects:
            raise ModalSdkFailure("transport_failed")
        function = self._read(
            lambda: objects[plan.function_name].hydrate.aio(self._client),
            deadline_at=deadline_at,
        )
        function_id = _provider_id(getattr(function, "object_id", None), "function")
        public_url = self._read(function.get_web_url.aio, deadline_at=deadline_at)
        return ModalAppObservation(
            app_id=app_id,
            app_name=_text(getattr(value, "name", None), "modal app name"),
            state="deployed",
            running_containers=containers,
            tags=tuple(sorted(tags.items())),
            function_id=function_id,
            function_name=plan.function_name,
            public_url=validate_modal_public_url(public_url),
        )

    def _lifecycle_app(
        self,
        plan: ModalCreatePlan,
        app_id: str,
        *,
        deadline_at: float | None,
    ) -> ModalAppObservation:
        validated_app_id = _provider_id(app_id, "app")
        lifecycle = self._read(
            lambda: self._modal.experimental.get_app_lifecycle.aio(
                validated_app_id,
                client=self._client,
            ),
            deadline_at=deadline_at,
        )
        stopped = getattr(lifecycle, "stopped_at", None) is not None
        return ModalAppObservation(
            app_id=validated_app_id,
            app_name=plan.names.app_or_pod,
            state="stopped" if stopped else "lifecycle_pending",
            running_containers=0 if stopped else None,
            tags=(),
            function_id=None,
            function_name=None,
            public_url=None,
        )

    def observe(
        self,
        plan: ModalCreatePlan,
        *,
        app_id_hint: str | None = None,
        deadline_at: float | None = None,
    ) -> ModalObservation:
        validate_modal_plan(plan)
        secrets = self._list_named(
            self._modal.Secret.objects,
            plan,
            "secret",
            deadline_at=deadline_at,
        )
        volumes = self._list_named(
            self._modal.Volume.objects,
            plan,
            "volume",
            deadline_at=deadline_at,
        )
        apps = tuple(
            self._deployed_app(plan, value, deadline_at=deadline_at)
            for value in self._deployed_apps(plan, deadline_at=deadline_at)
        )
        if not apps and app_id_hint is not None:
            apps = (self._lifecycle_app(plan, app_id_hint, deadline_at=deadline_at),)
        return ModalObservation(
            workspace_name=self.workspace_name,
            environment_name=self.environment_name,
            apps=apps,
            volumes=tuple(value for value in volumes if value.name == plan.names.volume),
            inference_secrets=tuple(
                value for value in secrets if value.name == plan.names.inference_secret
            ),
            artifact_secrets=tuple(
                value for value in secrets if value.name == plan.names.artifact_secret
            ),
        )

    def _created_secret(
        self,
        plan: ModalCreatePlan,
        name: str,
        key: str,
        value: str,
        *,
        deadline_at: float | None,
    ) -> ModalNamedResource:
        self._mutate(
            lambda: self._modal.Secret.objects.create.aio(
                name,
                {key: value},
                allow_existing=False,
                environment_name=plan.placement.environment,
                client=self._client,
            ),
            deadline_at=deadline_at,
        )
        handle = _post_mutation_read(
            lambda: self._modal.Secret.from_name(
                name,
                environment_name=plan.placement.environment,
                required_keys=[key],
                client=self._client,
            )
        )
        hydrated = _post_mutation_read(
            lambda: self._read(
                handle.hydrate.aio,
                deadline_at=deadline_at,
            )
        )
        resource = _post_mutation_read(lambda: _resource(hydrated, "secret"))
        if resource.name != name:
            raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True)
        return resource

    def create_inference_secret(
        self,
        plan: ModalCreatePlan,
        value: str,
        *,
        deadline_at: float | None = None,
    ) -> ModalNamedResource:
        return self._created_secret(
            plan,
            plan.names.inference_secret,
            "FLASH_INFERENCE_TOKEN",
            value,
            deadline_at=deadline_at,
        )

    def create_artifact_secret(
        self,
        plan: ModalCreatePlan,
        value: str,
        *,
        deadline_at: float | None = None,
    ) -> ModalNamedResource:
        return self._created_secret(
            plan,
            plan.names.artifact_secret,
            "FLASH_ARTIFACT_TOKEN",
            value,
            deadline_at=deadline_at,
        )

    def create_volume(
        self,
        plan: ModalCreatePlan,
        *,
        deadline_at: float | None = None,
    ) -> ModalNamedResource:
        self._mutate(
            lambda: self._modal.Volume.objects.create.aio(
                plan.names.volume,
                allow_existing=False,
                environment_name=plan.placement.environment,
                client=self._client,
            ),
            deadline_at=deadline_at,
        )
        handle = _post_mutation_read(
            lambda: self._modal.Volume.from_name(
                plan.names.volume,
                environment_name=plan.placement.environment,
                create_if_missing=False,
                client=self._client,
            )
        )
        hydrated = _post_mutation_read(
            lambda: self._read(
                handle.hydrate.aio,
                deadline_at=deadline_at,
            )
        )
        resource = _post_mutation_read(lambda: _resource(hydrated, "volume"))
        if resource.name != plan.names.volume:
            raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True)
        return resource

    def deploy_app(
        self,
        plan: ModalCreatePlan,
        *,
        deadline_at: float | None = None,
    ) -> str:
        validate_modal_plan(plan)
        image = self._modal.Image.from_registry(plan.bundle.image.reference)
        volume = self._modal.Volume.from_name(
            plan.names.volume,
            environment_name=plan.placement.environment,
            create_if_missing=False,
            client=self._client,
        )
        secrets = [
            self._modal.Secret.from_name(
                plan.names.inference_secret,
                environment_name=plan.placement.environment,
                required_keys=["FLASH_INFERENCE_TOKEN"],
                client=self._client,
            )
        ]
        if plan.phase == "bootstrap":
            secrets.append(
                self._modal.Secret.from_name(
                    plan.names.artifact_secret,
                    environment_name=plan.placement.environment,
                    required_keys=["FLASH_ARTIFACT_TOKEN"],
                    client=self._client,
                )
            )
        app = self._modal.App(
            plan.names.app_or_pod,
            tags=dict(plan.tags),
            include_source=False,
        )
        from flash.serve.provisioning.modal.planning.wrapper import launch_modal_server

        web_function = self._modal.web_server(
            plan.web_port,
            startup_timeout=plan.startup_timeout_seconds,
            label=plan.endpoint_label,
        )(launch_modal_server)
        app.function(
            image=image,
            env=dict(plan.environment),
            secrets=secrets,
            gpu=plan.gpu_request,
            serialized=False,
            volumes={MODAL_VOLUME_MOUNT: volume},
            min_containers=plan.min_containers,
            max_containers=plan.max_containers,
            buffer_containers=plan.buffer_containers,
            scaledown_window=plan.scaledown_window_seconds,
            startup_timeout=plan.startup_timeout_seconds,
            name=plan.function_name,
            region=plan.placement.region,
            include_source=False,
        )(web_function)
        deployed = self._mutate(
            lambda: app.deploy.aio(
                name=plan.names.app_or_pod,
                environment_name=plan.placement.environment,
                tag=plan.deployment_tag,
                client=self._client,
                strategy="recreate",
            ),
            deadline_at=deadline_at,
        )
        return _post_mutation_read(lambda: _provider_id(getattr(deployed, "app_id", None), "app"))

    def _id_mutation(
        self,
        request_name: str,
        rpc_name: str,
        *,
        deadline_at: float | None,
        **fields: object,
    ) -> None:
        try:
            api_pb2 = importlib.import_module("modal_proto.api_pb2")
            request_type = getattr(api_pb2, request_name)
            rpc = getattr(self._client.stub, rpc_name)
            synchronizer = vars(self._client)["_sync_synchronizer"]
            if request_name == "AppStopRequest":
                fields["source"] = api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT
            request = request_type(**fields)

            async def invoke() -> object:
                return await rpc(request)

            synchronized_rpc = synchronizer.create_blocking(invoke)
            operation = synchronized_rpc.aio
        except Exception:
            # name mutations can target a newer same-generation deployment after ownership proof.
            # if the pinned generated id rpc is unavailable, decline cleanup as ambiguous instead.
            raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True) from None
        self._mutate(operation, deadline_at=deadline_at)

    def stop_app(
        self,
        plan: ModalCreatePlan,
        app_id: str,
        *,
        deadline_at: float | None = None,
    ) -> None:
        validate_modal_plan(plan)
        self._id_mutation(
            "AppStopRequest",
            "AppStop",
            deadline_at=deadline_at,
            app_id=_provider_id(app_id, "app"),
        )

    def delete_secret(
        self,
        plan: ModalCreatePlan,
        secret_id: str,
        *,
        deadline_at: float | None = None,
    ) -> None:
        validate_modal_plan(plan)
        self._id_mutation(
            "SecretDeleteRequest",
            "SecretDelete",
            deadline_at=deadline_at,
            secret_id=_provider_id(secret_id, "secret"),
        )

    def delete_volume(
        self,
        plan: ModalCreatePlan,
        volume_id: str,
        *,
        deadline_at: float | None = None,
    ) -> None:
        validate_modal_plan(plan)
        self._id_mutation(
            "VolumeDeleteRequest",
            "VolumeDelete",
            deadline_at=deadline_at,
            volume_id=_provider_id(volume_id, "volume"),
        )

    def close(self) -> None:
        client = self._client
        self._client = None
        try:
            _close_client(
                client,
                deadline_at=self._deadline_at,
                clock=self._clock,
                bridge=self._bridge,
            )
        finally:
            self._bridge.close()


def create_modal_sdk(
    credentials: ModalCredentials,
    plan: ModalCreatePlan,
    deadline_at: float,
    clock: Clock = time.monotonic,
) -> ModalSdk:
    """construct one request-local pinned modal sdk adapter after plan validation."""

    return PinnedModalSdk(credentials, plan, deadline_at, clock)
