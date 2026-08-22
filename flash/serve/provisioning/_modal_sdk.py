"""lazy import-light adapter for the pinned modal sdk surface."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from flash.serve.control import DeploymentErrorCode, ModalCredentials
from flash.serve.control._urls import validate_modal_public_url
from flash.serve.control.types import validate_modal_provider_id

from ._modal_plan import MODAL_VOLUME_MOUNT, ModalCreatePlan, validate_modal_plan

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
    ) -> ModalObservation: ...

    def create_inference_secret(self, plan: ModalCreatePlan, value: str) -> ModalNamedResource: ...

    def create_artifact_secret(self, plan: ModalCreatePlan, value: str) -> ModalNamedResource: ...

    def create_volume(self, plan: ModalCreatePlan) -> ModalNamedResource: ...

    def deploy_app(self, plan: ModalCreatePlan) -> str: ...

    def stop_app(self, plan: ModalCreatePlan) -> None: ...

    def delete_secret(self, plan: ModalCreatePlan, name: str) -> None: ...

    def delete_volume(self, plan: ModalCreatePlan) -> None: ...

    def close(self) -> None: ...


ModalSdkFactory = Callable[[ModalCredentials, ModalCreatePlan], ModalSdk]
ModuleLoader = Callable[[], object]


def _load_modal_module() -> object:
    modal = importlib.import_module("modal")
    experimental = importlib.import_module("modal.experimental")
    modal.experimental = experimental
    return modal


def _sync_value(value: object) -> object:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


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


def _call_read(operation: Callable[[], object]) -> object:
    try:
        return _sync_value(operation())
    except ModalSdkFailure:
        raise
    except Exception:
        raise ModalSdkFailure("transport_failed") from None


def _call_mutation(operation: Callable[[], object]) -> object:
    try:
        return _sync_value(operation())
    except ModalSdkFailure:
        raise
    except Exception:
        raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True) from None


def _post_mutation_read(operation: Callable[[], object]) -> object:
    try:
        return operation()
    except Exception:
        raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True) from None


def _close_client(client: object) -> None:
    close = getattr(client, "_close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            _sync_value(close())


class PinnedModalSdk:
    """request-scoped adapter for the locally pinned modal 1.5 sdk contract."""

    __slots__ = ("_client", "_modal", "environment_name", "workspace_name")

    def __init__(
        self,
        credentials: ModalCredentials,
        plan: ModalCreatePlan,
        *,
        module_loader: ModuleLoader = _load_modal_module,
    ) -> None:
        if type(credentials) is not ModalCredentials:
            raise ValueError("modal credentials must use the exact credential type")
        validate_modal_plan(plan)
        token_id, token_secret = credentials.reveal()
        client: object | None = None
        try:
            modal = module_loader()
            client = modal.Client.from_credentials(token_id, token_secret)
            workspace = modal.Workspace.from_context(client=client).hydrate()
            environment = modal.Environment.from_name(
                plan.placement.environment,
                create_if_missing=False,
                client=client,
            ).hydrate()
            workspace_name = _text(getattr(workspace, "name", None), "modal workspace name")
            environment_name = _text(getattr(environment, "name", None), "modal environment name")
            if (
                workspace_name != plan.placement.workspace_name
                or environment_name != plan.placement.environment
            ):
                raise ModalSdkFailure("authentication_failed")
        except Exception:
            if client is not None:
                _close_client(client)
            raise ModalSdkFailure("authentication_failed") from None
        self._modal = modal
        self._client = client
        self.workspace_name = workspace_name
        self.environment_name = environment_name

    def __repr__(self) -> str:
        return "PinnedModalSdk(<request-scoped>)"

    def _list_named(
        self,
        manager: object,
        plan: ModalCreatePlan,
        role: Literal["secret", "volume"],
    ) -> tuple[ModalNamedResource, ...]:
        values = _call_read(
            lambda: manager.list(
                environment_name=plan.placement.environment,
                client=self._client,
            )
        )
        if type(values) is not list:
            raise ModalSdkFailure("transport_failed")
        return tuple(_resource(value, role) for value in values)

    def _deployed_apps(self, plan: ModalCreatePlan) -> list[object]:
        values = _call_read(
            lambda: self._modal.experimental.list_deployed_apps(
                environment_name=plan.placement.environment,
                client=self._client,
            )
        )
        if type(values) is not list:
            raise ModalSdkFailure("transport_failed")
        return [value for value in values if getattr(value, "name", None) == plan.names.app_or_pod]

    def _deployed_app(self, plan: ModalCreatePlan, value: object) -> ModalAppObservation:
        app_id = _provider_id(getattr(value, "app_id", None), "app")
        containers = getattr(value, "containers", None)
        if type(containers) is not int or containers < 0:
            raise ModalSdkFailure("transport_failed")
        app = _call_read(
            lambda: self._modal.App.lookup(
                plan.names.app_or_pod,
                client=self._client,
                environment_name=plan.placement.environment,
                create_if_missing=False,
            )
        )
        if _provider_id(getattr(app, "app_id", None), "app") != app_id:
            raise ModalSdkFailure("transport_failed")
        tags = _call_read(lambda: app.get_tags(client=self._client))
        if type(tags) is not dict or any(
            type(key) is not str or type(item) is not str for key, item in tags.items()
        ):
            raise ModalSdkFailure("transport_failed")
        objects = _call_read(
            lambda: self._modal.experimental.get_app_objects(
                plan.names.app_or_pod,
                environment_name=plan.placement.environment,
                client=self._client,
            )
        )
        if type(objects) is not dict or len(objects) != 1 or plan.function_name not in objects:
            raise ModalSdkFailure("transport_failed")
        function = _call_read(lambda: objects[plan.function_name].hydrate(self._client))
        function_id = _provider_id(getattr(function, "object_id", None), "function")
        public_url = _call_read(lambda: function.get_web_url())
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

    def _lifecycle_app(self, plan: ModalCreatePlan, app_id: str) -> ModalAppObservation:
        validated_app_id = _provider_id(app_id, "app")
        lifecycle = _call_read(
            lambda: self._modal.experimental.get_app_lifecycle(
                validated_app_id,
                client=self._client,
            )
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
    ) -> ModalObservation:
        validate_modal_plan(plan)
        secrets = self._list_named(self._modal.Secret.objects, plan, "secret")
        volumes = self._list_named(self._modal.Volume.objects, plan, "volume")
        apps = tuple(self._deployed_app(plan, value) for value in self._deployed_apps(plan))
        if not apps and app_id_hint is not None:
            apps = (self._lifecycle_app(plan, app_id_hint),)
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
    ) -> ModalNamedResource:
        _call_mutation(
            lambda: self._modal.Secret.objects.create(
                name,
                {key: value},
                allow_existing=False,
                environment_name=plan.placement.environment,
                client=self._client,
            )
        )
        handle = _post_mutation_read(
            lambda: _call_read(
                lambda: self._modal.Secret.from_name(
                    name,
                    environment_name=plan.placement.environment,
                    required_keys=[key],
                    client=self._client,
                ).hydrate()
            )
        )
        resource = _post_mutation_read(lambda: _resource(handle, "secret"))
        if resource.name != name:
            raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True)
        return resource

    def create_inference_secret(self, plan: ModalCreatePlan, value: str) -> ModalNamedResource:
        return self._created_secret(
            plan, plan.names.inference_secret, "FLASH_INFERENCE_TOKEN", value
        )

    def create_artifact_secret(self, plan: ModalCreatePlan, value: str) -> ModalNamedResource:
        return self._created_secret(plan, plan.names.artifact_secret, "FLASH_ARTIFACT_TOKEN", value)

    def create_volume(self, plan: ModalCreatePlan) -> ModalNamedResource:
        _call_mutation(
            lambda: self._modal.Volume.objects.create(
                plan.names.volume,
                allow_existing=False,
                environment_name=plan.placement.environment,
                client=self._client,
            )
        )
        handle = _post_mutation_read(
            lambda: _call_read(
                lambda: self._modal.Volume.from_name(
                    plan.names.volume,
                    environment_name=plan.placement.environment,
                    create_if_missing=False,
                    client=self._client,
                ).hydrate()
            )
        )
        resource = _post_mutation_read(lambda: _resource(handle, "volume"))
        if resource.name != plan.names.volume:
            raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True)
        return resource

    def deploy_app(self, plan: ModalCreatePlan) -> str:
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
        from ._modal_wrapper import launch_modal_server

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
        deployed = _call_mutation(
            lambda: app.deploy(
                name=plan.names.app_or_pod,
                environment_name=plan.placement.environment,
                tag=plan.deployment_tag,
                client=self._client,
                strategy="recreate",
            )
        )
        return _post_mutation_read(lambda: _provider_id(getattr(deployed, "app_id", None), "app"))

    def stop_app(self, plan: ModalCreatePlan) -> None:
        _call_mutation(
            lambda: self._modal.experimental.stop_app(
                plan.names.app_or_pod,
                environment_name=plan.placement.environment,
                client=self._client,
            )
        )

    def delete_secret(self, plan: ModalCreatePlan, name: str) -> None:
        _call_mutation(
            lambda: self._modal.Secret.objects.delete(
                name,
                allow_missing=True,
                environment_name=plan.placement.environment,
                client=self._client,
            )
        )

    def delete_volume(self, plan: ModalCreatePlan) -> None:
        _call_mutation(
            lambda: self._modal.Volume.objects.delete(
                plan.names.volume,
                allow_missing=True,
                environment_name=plan.placement.environment,
                client=self._client,
            )
        )

    def close(self) -> None:
        client = self._client
        self._client = None
        _close_client(client)


def create_modal_sdk(credentials: ModalCredentials, plan: ModalCreatePlan) -> ModalSdk:
    """construct one request-local pinned modal sdk adapter after plan validation."""

    return PinnedModalSdk(credentials, plan)
