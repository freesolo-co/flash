"""Lambda takes over a shared pool only after its workers are ready."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from flash.serving.app.inference_provider import lambda_models, policy_for
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
        environment="main",
        ready_count=AsyncMock(return_value=ready),
        allocated_count=AsyncMock(return_value=ready),
    )
    fleet.allocated_count.side_effect = lambda model, exclude_revision=None: (
        0 if exclude_revision is not None else fleet.allocated_count.return_value
    )
    data = {"active_revision": REVISION, MODEL: {"modal_slots": max(0, 4 - ready)}}
    state = SimpleNamespace(
        get=AsyncMock(side_effect=data.get),
        put=AsyncMock(side_effect=lambda key, value: data.__setitem__(key, value)),
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
    control.state.get.side_effect = lambda key: (
        {"revision": "b" * 40, "retired": [REVISION]} if key == "serving_release" else None
    )
    await control.reconcile()
    primary.update_autoscaler.aio.assert_not_awaited()
    fallback.update_autoscaler.aio.assert_not_awaited()
    with pytest.raises(NoCapacityError):
        await control.before_launch(MODEL, REVISION, 0)
    with pytest.raises(RuntimeError):
        await control.activate(REVISION)


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


@pytest.mark.anyio
@pytest.mark.parametrize(("allocated", "expected"), [(4, "modal"), (5, "lambda")])
async def test_health_loss_refreshes_zero_slots_without_exceeding_handoff_budget(
    allocated, expected
):
    control, _, fallback = controller()
    control.fleet.allocated_count.return_value = allocated
    state = {MODEL: {"modal_slots": 0}}

    async def get(key):
        return state.get(key)

    async def put(key, value):
        state[key] = value

    control.state = SimpleNamespace(get=get, put=put)
    # Mimic the independent routing function asking the serialized controller
    # to refresh, without accessing provider credentials on the request path.
    control.refresh = AsyncMock(side_effect=control.reconcile_model)
    assert await control.choose(MODEL) == expected
    control.refresh.assert_awaited_once_with(MODEL)
    assert state[MODEL]["modal_slots"] == 5 - allocated
    assert fallback.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0


class ReleaseState:
    def __init__(self, revision):
        self.data = {"active_revision": revision}

    async def get(self, key):
        return self.data.get(key)

    async def put(self, key, value):
        self.data[key] = value


@pytest.mark.anyio
async def test_controller_upgrade_preserves_serving_until_replacement_activation():
    control, primary, _ = controller()
    state = ReleaseState(REVISION)
    control.state = state
    control.revision = "b" * 40  # New controller is deployed before serving.
    await control.activate(REVISION)  # Existing serving router cold start.
    await control.before_launch(MODEL, REVISION, allocated=0)  # Existing pool replenishment.
    assert (await control.serving_release())["revision"] == REVISION
    assert await control.active()
    primary.update_autoscaler.aio.reset_mock()
    await control.reconcile()  # Minute autoscaling still works after a failed serving deploy.
    assert primary.update_autoscaler.aio.await_count > 0

    await control.activate(control.revision)
    assert (await control.serving_release())["revision"] == control.revision
    with pytest.raises(RuntimeError):
        await control.activate(REVISION)
    with pytest.raises(NoCapacityError):
        await control.before_launch(MODEL, REVISION, allocated=0)
    await control.before_launch(MODEL, control.revision, allocated=0)
    old = SharedCapacity(control.fleet, control.pools, state, REVISION)
    assert not await old.active()
    with pytest.raises(RuntimeError):
        await old.activate(REVISION)


@pytest.mark.anyio
async def test_failed_activation_keeps_old_serving_revision_usable():
    control, primary, _ = controller()
    control.state = ReleaseState(REVISION)
    control.revision = "b" * 40
    primary.update_autoscaler.aio.side_effect = RuntimeError("pool update failed")
    with pytest.raises(ExceptionGroup):
        await control.activate(control.revision)
    assert (await control.serving_release()) == {"revision": REVISION, "retired": []}
    await control.activate(REVISION)  # Admission does not depend on retrying the failed update.
    primary.update_autoscaler.aio.side_effect = None
    await control.before_launch(MODEL, REVISION, allocated=0)


@pytest.mark.anyio
async def test_first_controller_upgrade_keeps_legacy_lambda_pools_usable():
    control, primary, fallback = controller()
    control.state = ReleaseState(None)
    await control.before_launch(MODEL, "b" * 40, allocated=0)
    # Legacy serving has no fallback class: do not look it up before activation.
    primary.update_autoscaler.aio.assert_not_awaited()
    fallback.update_autoscaler.aio.assert_not_awaited()
    with pytest.raises(NoCapacityError):
        await control.before_launch(MODEL, REVISION, allocated=0)
    await control.activate(REVISION)
    with pytest.raises(NoCapacityError):
        await control.before_launch(MODEL, "b" * 40, allocated=0)


@pytest.mark.anyio
async def test_rolling_activation_routes_by_new_revision_but_budgets_all_workers():
    from flash.serving.lambda_cloud.fleet import Fleet
    from tests.serving.test_lambda_fleet import Ledger

    old, new = REVISION, "b" * 40
    ledger = Ledger()
    for i in range(4):
        await ledger.put(str(i), {"model": MODEL, "revision": old, "ready_at": 100, "gpu": "H100"})
    api = SimpleNamespace(call=AsyncMock(return_value=[]))
    fleet = Fleet(api, ledger, "dev", clock=lambda: 100)
    primary, fallback = pool(), pool()
    state = ReleaseState(old)
    control = SharedCapacity(fleet, lambda _: (primary, fallback), state, new)
    assert await fleet.ready_count(MODEL) == 4
    assert await fleet.ready_count(MODEL, revision=new) == 0

    await control.activate(new)
    # Candidate reconciliation happens before the release switch, so it must
    # explicitly use the candidate SHA instead of the previous active SHA.
    assert state.data[MODEL]["modal_slots"] == 1
    await control.reconcile_model(MODEL)
    assert await fleet.allocated_count(MODEL) == 4
    assert await control.choose(MODEL) == "modal"
    assert fallback.update_autoscaler.aio.await_args.kwargs["max_containers"] == 1
    # Do not allocate a sixth GPU while four old GPUs and a fallback slot exist.
    with pytest.raises(NoCapacityError, match="handoff"):
        await control.before_launch(MODEL, new, allocated=4)

    # As old workers drain, the budget returns; they never became routing-ready
    # just because a still newer controller was deployed.
    ledger.data["0"]["released"] = True
    ledger.data["1"]["released"] = True
    await control.before_launch(MODEL, new, allocated=2)
    await ledger.put("new", {"model": MODEL, "revision": new, "ready_at": 100, "gpu": "H100"})
    control.revision = "c" * 40
    await control.reconcile_model(MODEL)
    assert await control.choose(MODEL) == "lambda"
    assert await fleet.allocated_count(MODEL) == 3
    assert await fleet.ready_count(MODEL, revision=new) == 1
    assert state.data[MODEL]["modal_slots"] == 1

    ledger.data["2"]["ready_at"] = 0  # An unhealthy old worker still consumes budget.
    await control.before_launch(MODEL, new, allocated=3)
    assert await fleet.allocated_count(MODEL) == 3


@pytest.mark.anyio
@pytest.mark.parametrize("model", lambda_models())
async def test_dev_cold_start_demand_and_return_to_zero(model):
    control, primary, fallback = controller()
    control.fleet.environment = "dev"
    control.clock = lambda: 1000
    await control.activate(REVISION)
    await control.reconcile_model(model)
    assert primary.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0
    with pytest.raises(NoCapacityError, match="idle"):
        await control.before_launch(model, REVISION, allocated=0)

    assert await control.choose(model) == "modal"
    await control.before_launch(model, REVISION, allocated=0)
    assert primary.update_autoscaler.aio.await_args.kwargs["min_containers"] == 1
    maximum = policy_for(model).max_capacity
    fallback.health.get_current_stats.aio.return_value.backlog = maximum * sequence_capacity(model)
    control.clock = lambda: 1201
    await control.reconcile_model(model)
    assert primary.update_autoscaler.aio.await_args.kwargs["min_containers"] == maximum
    # Capacity retries continue while demand exists, even after the recent-request window.
    await control.reconcile_model(model)
    assert primary.update_autoscaler.aio.await_args.kwargs["min_containers"] == maximum
    fallback.health.get_current_stats.aio.return_value.backlog = 0
    await control.reconcile_model(model)
    assert primary.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0
    assert fallback.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0
