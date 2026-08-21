"""input validation, sdk construction, and the modal call wrapper.

The mirror of `_runpod_lifecycle`: the wrapper decides whether a failed provider call is merely
failed or leaves the outcome unknown. This module imports nothing from `modal.py`, so the
dependency stays one-way.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from flash.serve.control import ModalCredentials

from ._common import Clock, ServingRuntimeSecrets, validate_deadline
from ._modal_plan import ModalCreatePlan
from ._modal_sdk import ModalObservation, ModalSdk, ModalSdkFactory, ModalSdkFailure


def validate_runtime_inputs(
    credentials: ModalCredentials,
    runtime_secrets: ServingRuntimeSecrets,
    deadline_at: float,
    clock: Clock,
) -> None:
    if type(credentials) is not ModalCredentials:
        raise ValueError("modal credentials must use the exact credential type")
    if type(runtime_secrets) is not ServingRuntimeSecrets:
        raise ValueError("runtime secrets must use the exact secret boundary")
    validate_deadline(deadline_at, clock)


def validate_control_inputs(
    credentials: ModalCredentials,
    deadline_at: float,
    clock: Clock,
) -> None:
    if type(credentials) is not ModalCredentials:
        raise ValueError("modal credentials must use the exact credential type")
    validate_deadline(deadline_at, clock)


def open_sdk(
    factory: ModalSdkFactory,
    credentials: ModalCredentials,
    plan: ModalCreatePlan,
) -> ModalSdk:
    """open one sdk proven to be pointed at the workspace and environment the plan names.

    a client authenticated against the wrong workspace would otherwise read as an empty account and
    provision a second copy of everything there.
    """

    try:
        sdk = factory(credentials, plan)
    except ModalSdkFailure:
        raise
    except Exception:
        raise ModalSdkFailure("transport_failed") from None
    if sdk.workspace_name != plan.placement.workspace_name:
        with contextlib.suppress(Exception):
            sdk.close()
        raise ModalSdkFailure("authentication_failed")
    if sdk.environment_name != plan.placement.environment:
        with contextlib.suppress(Exception):
            sdk.close()
        raise ModalSdkFailure("conflict")
    return sdk


def observe(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    *,
    app_id_hint: str | None = None,
) -> ModalObservation:
    try:
        observation = sdk.observe(plan, app_id_hint=app_id_hint)
    except ModalSdkFailure:
        raise
    except Exception:
        raise ModalSdkFailure("transport_failed") from None
    if type(observation) is not ModalObservation:
        raise ModalSdkFailure("transport_failed")
    if (
        observation.workspace_name != plan.placement.workspace_name
        or observation.environment_name != plan.placement.environment
    ):
        raise ModalSdkFailure("authentication_failed")
    return observation


def mutation(operation: Callable[[], object]) -> object:
    """run one mutation, whose failure may still have applied.

    `resource_ambiguous` rather than `transport_failed`: a create or delete that raised locally may
    already have reached modal, so the caller must reconcile rather than retry blindly.
    """

    try:
        return operation()
    except ModalSdkFailure:
        raise
    except Exception:
        raise ModalSdkFailure("resource_ambiguous", outcome_unknown=True) from None
