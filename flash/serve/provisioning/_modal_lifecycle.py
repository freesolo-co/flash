"""shared primitives for the modal serving deployment lifecycle.

`modal.py` reached the 1000-line file limit. The group extracted here is the same one taken out of
`runpod.py` into `_runpod_lifecycle`: input validation, sdk construction, and the call wrapper that
decides whether a failed provider call is merely failed or leaves the outcome unknown. Keeping the
two providers structured alike means a reader who has learned one can navigate the other.

The split is one-way -- this module imports nothing from `modal.py` -- so there is no cycle.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from flash.serve.control import ModalCredentials

# provider-neutral, so they live in `_common` rather than once per provider. re-exported here for
# the same reason `_runpod_lifecycle` re-exports them: callers import lifecycle names from their
# own provider's lifecycle module.
from ._common import (
    Clock,
    LifecycleFailure,
    ServingRuntimeSecrets,
    Sleeper,
    validate_deadline,
)
from ._modal_plan import ModalCreatePlan
from ._modal_sdk import ModalObservation, ModalSdk, ModalSdkFactory, ModalSdkFailure

__all__ = [
    "Clock",
    "LifecycleFailure",
    "Sleeper",
    "mutation",
    "observe",
    "open_sdk",
    "validate_control_inputs",
    "validate_deadline",
    "validate_runtime_inputs",
]


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
