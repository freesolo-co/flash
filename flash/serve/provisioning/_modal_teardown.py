"""shared terminal-state and absence proofs for modal resource teardown."""

from __future__ import annotations

from collections.abc import Callable

from flash.serve.control import ModalProviderHandle

from ._common import Clock, Sleeper
from ._modal_lifecycle import mutation, observe
from ._modal_plan import ModalCreatePlan
from ._modal_readiness import sleep_until_poll
from ._modal_resources import exact_teardown_resources, resources_are_absent
from ._modal_sdk import ModalNamedResource, ModalObservation, ModalSdk


def wait_for_terminal_app(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    handle: ModalProviderHandle,
    *,
    deadline_at: float,
    clock: Clock,
    sleep: Sleeper,
) -> ModalObservation | None:
    while True:
        observation = observe(plan, sdk, app_id_hint=handle.app_id)
        app, _volume, _inference, _artifact = exact_teardown_resources(
            plan,
            handle,
            observation,
        )
        if app is not None and app.state in {"stopped", "failed"}:
            return observation if app.running_containers == 0 else None
        if not sleep_until_poll(deadline_at, clock, sleep):
            return None


def suppressed(step: Callable[[], object]) -> bool:
    """run one teardown step without replacing the exception that initiated abort cleanup."""

    try:
        step()
    except Exception:
        return False
    return True


def delete_teardown_resources(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    volume: ModalNamedResource | None,
    inference: ModalNamedResource | None,
    artifact: ModalNamedResource | None,
    *,
    suppress_failures: bool = False,
) -> None:
    def run(step: Callable[[], object]) -> None:
        if suppress_failures:
            suppressed(step)
        else:
            step()

    if artifact is not None:
        run(lambda: mutation(lambda: sdk.delete_secret(plan, artifact.name)))
    if inference is not None:
        run(lambda: mutation(lambda: sdk.delete_secret(plan, inference.name)))
    if volume is not None:
        run(lambda: mutation(lambda: sdk.delete_volume(plan)))


def confirm_teardown_absence(
    plan: ModalCreatePlan,
    sdk: ModalSdk,
    handle: ModalProviderHandle,
) -> bool:
    final = observe(plan, sdk, app_id_hint=handle.app_id)
    exact_teardown_resources(plan, handle, final)
    return resources_are_absent(final, allow_terminal_app=True)
