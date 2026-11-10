"""Lambda takes over a shared pool only after its workers are ready."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from flash.serving.lambda_cloud.config import NoCapacityError
from flash.serving.lambda_cloud.pool_capacity import SharedCapacity, sequence_capacity

MODEL = "Qwen/Qwen3.5-9B-Base"
REVISION = "a" * 40


@pytest.fixture
def anyio_backend():
    return "asyncio"


def pool(runners=0, running=0, backlog=0):
    stats = SimpleNamespace(num_total_runners=runners, num_running_inputs=running, backlog=backlog)
    return SimpleNamespace(
        health=SimpleNamespace(
            get_current_stats=SimpleNamespace(aio=AsyncMock(return_value=stats))
        ),
        update_autoscaler=SimpleNamespace(aio=AsyncMock()),
    )


def controller(ready=0, modal_runners=0, running=0):
    primary, fallback = pool(), pool(modal_runners, running)
    fleet = SimpleNamespace(
        ready_count=AsyncMock(return_value=ready), allocated_count=AsyncMock(return_value=ready)
    )
    state = SimpleNamespace(
        get=AsyncMock(
            side_effect=lambda key: (
                None
                if key.startswith("demand:")
                else REVISION
                if key == "active_revision"
                else {"modal_slots": max(0, 4 - ready)}
            )
        ),
        put=AsyncMock(),
    )
    control = SharedCapacity(fleet, lambda _: (primary, fallback), state, REVISION)
    return control, primary, fallback


@pytest.mark.anyio
async def test_no_lambda_capacity_leaves_all_four_slots_for_modal():
    control, primary, fallback = controller()
    await control.reconcile_model(MODEL)
    assert await control.choose(MODEL) == "modal"
    primary.update_autoscaler.aio.assert_awaited_once_with(
        min_containers=2, max_containers=4, buffer_containers=0
    )
    fallback.update_autoscaler.aio.assert_awaited_once_with(
        min_containers=0, max_containers=4, buffer_containers=0, scaledown_window=60
    )


@pytest.mark.anyio
async def test_four_ready_lambda_workers_route_all_requests_and_drain_modal():
    control, _primary, fallback = controller(ready=4, modal_runners=1)
    await control.reconcile_model(MODEL)
    assert await control.choose(MODEL) == "lambda"
    options = fallback.update_autoscaler.aio.await_args.kwargs
    assert options["min_containers"] == options["buffer_containers"] == 0
    assert options["scaledown_window"] == 2


@pytest.mark.anyio
async def test_modal_demand_keeps_requesting_lambda_scaleup():
    control, primary, _ = controller(running=4 * sequence_capacity(MODEL))
    await control.reconcile_model(MODEL)
    assert primary.update_autoscaler.aio.await_args.kwargs["min_containers"] == 4
    # Minute reconciliation retries the desired floor even if no worker started.
    await control.reconcile_model(MODEL)
    assert primary.update_autoscaler.aio.await_count == 2


@pytest.mark.anyio
async def test_handoff_boots_one_then_waits_for_old_modal_gpu_to_drain():
    control, _, fallback = controller(ready=0, modal_runners=4)
    await control.before_launch(MODEL, REVISION, allocated=0)
    with pytest.raises(NoCapacityError):
        await control.before_launch(MODEL, REVISION, allocated=1)
    control.fleet.ready_count.return_value = 1
    with pytest.raises(NoCapacityError):
        await control.before_launch(MODEL, REVISION, allocated=1)
    fallback.health.get_current_stats.aio.return_value.num_total_runners = 3
    await control.before_launch(MODEL, REVISION, allocated=1)
    assert fallback.update_autoscaler.aio.await_args.kwargs["max_containers"] == 3


@pytest.mark.anyio
async def test_busy_partial_lambda_pool_uses_modal_but_full_pool_does_not_overflow():
    control, primary, _ = controller(ready=2)
    primary.health.get_current_stats.aio.return_value.num_running_inputs = 2 * sequence_capacity(
        MODEL
    )
    assert await control.choose(MODEL) == "modal"
    control.fleet.ready_count.return_value = 4
    assert await control.choose(MODEL) == "lambda"


@pytest.mark.anyio
async def test_old_controller_cannot_reconfigure_new_deployment():
    control, primary, fallback = controller()
    control.state.get.side_effect = None
    control.state.get.return_value = "b" * 40
    await control.reconcile()
    primary.update_autoscaler.aio.assert_not_awaited()
    fallback.update_autoscaler.aio.assert_not_awaited()
    with pytest.raises(NoCapacityError):
        await control.before_launch(MODEL, REVISION, 0)
    with pytest.raises(RuntimeError):
        await control.activate("b" * 40)


@pytest.mark.anyio
async def test_each_handoff_preserves_four_slots_and_at_most_one_booting_gpu():
    for ready in range(4):
        control, _, fallback = controller(ready=ready, modal_runners=4 - ready)
        await control.before_launch(MODEL, REVISION, allocated=ready)
        assert fallback.update_autoscaler.aio.await_args.kwargs["max_containers"] == 4 - ready
        # The replacement has started, but cannot take traffic before readiness.
        control.fleet.allocated_count.return_value = ready + 1
        with pytest.raises(NoCapacityError):
            await control.before_launch(MODEL, REVISION, allocated=ready + 1)
        assert (await control.reconcile_model(MODEL))[0] == ready
        assert fallback.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0


@pytest.mark.anyio
async def test_unconfirmed_lambda_allocations_reserve_budget_until_cleanup():
    control, _, fallback = controller()
    control.fleet.allocated_count.return_value = 4
    await control.reconcile_model(MODEL)
    assert fallback.update_autoscaler.aio.await_args.kwargs["max_containers"] == 1
    with pytest.raises(NoCapacityError):
        await control.before_launch(MODEL, REVISION, allocated=4)


@pytest.mark.anyio
async def test_short_requests_boot_lambda_for_zero_floor_models_then_return_to_zero():
    model = "Qwen/Qwen3.5-4B"
    control, primary, fallback = controller()
    now = [1000]
    state = {"active_revision": REVISION, model: {"modal_slots": 2}}

    async def get(key):
        return state.get(key)

    async def put(key, value):
        state[key] = value

    control.state = SimpleNamespace(get=get, put=put)
    control.clock = lambda: now[0]
    await control.reconcile_model(model)
    assert primary.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0
    assert await control.choose(model) == "modal"
    # The request finishes before the next sample; remembered demand still
    # starts Lambda, instead of leaving all short requests on Modal forever.
    now[0] += 60
    await control.reconcile_model(model)
    assert primary.update_autoscaler.aio.await_args.kwargs["min_containers"] == 1
    control.fleet.ready_count.return_value = 1
    control.fleet.allocated_count.return_value = 1
    assert await control.choose(model) == "lambda"
    now[0] += 121
    await control.reconcile_model(model)
    assert primary.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0
    assert fallback.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0

    with pytest.raises(NoCapacityError, match="idle"):
        await control.before_launch(model, REVISION, allocated=1)
