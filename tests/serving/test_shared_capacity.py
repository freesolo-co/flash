"""The Lambda-only pool scales from demand, one booting GPU at a time, within its budget."""

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


@pytest.fixture(autouse=True)
def warm_floor(monkeypatch):
    """Every production model scales from 0; pin a floor of 2 so floor paths stay covered."""
    from flash.serving.app import inference_provider

    monkeypatch.setitem(
        inference_provider.MODEL_POLICIES,
        MODEL,
        inference_provider.ModelPolicy(min_capacity=2, max_capacity=4),
    )


def pool(runners=0, running=0, backlog=0):
    stats = SimpleNamespace(num_total_runners=runners, num_running_inputs=running, backlog=backlog)
    return SimpleNamespace(
        health=SimpleNamespace(
            get_current_stats=SimpleNamespace(aio=AsyncMock(return_value=stats))
        ),
        update_autoscaler=SimpleNamespace(aio=AsyncMock()),
    )


def controller(ready=0, runners=0, running=0, backlog=0):
    lambda_pool = pool(runners, running, backlog)
    fleet = SimpleNamespace(
        environment="main",
        ready_count=AsyncMock(return_value=ready),
        allocated_count=AsyncMock(return_value=ready),
    )
    fleet.allocated_count.side_effect = lambda model, exclude_revision=None: (
        0 if exclude_revision is not None else fleet.allocated_count.return_value
    )
    data = {"active_revision": REVISION}
    state = SimpleNamespace(
        get=AsyncMock(side_effect=data.get),
        put=AsyncMock(side_effect=lambda key, value: data.__setitem__(key, value)),
    )
    control = SharedCapacity(fleet, lambda _: lambda_pool, state, REVISION)
    return control, lambda_pool


@pytest.mark.anyio
async def test_pool_floor_and_ceiling_come_from_policy_without_a_modal_fallback():
    control, lambda_pool = controller()
    assert await control.reconcile_model(MODEL) == (0, 2)
    lambda_pool.update_autoscaler.aio.assert_awaited_once_with(
        min_containers=2, max_containers=4, buffer_containers=0
    )


@pytest.mark.anyio
async def test_demand_raises_desired_gpus_up_to_the_ceiling():
    control, _ = controller(runners=1, backlog=10 * sequence_capacity(MODEL))
    assert (await control.reconcile_model(MODEL))[1] == 4


@pytest.mark.anyio
async def test_scale_up_boots_one_gpu_at_a_time_up_to_four():
    for ready in range(4):
        control, _ = controller(ready=ready, runners=ready + 1, backlog=1)
        await control.before_launch(MODEL, REVISION, allocated=ready)
        # The next GPU has started; another cannot boot before it is ready.
        with pytest.raises(NoCapacityError, match="already starting"):
            await control.before_launch(MODEL, REVISION, allocated=ready + 1)


@pytest.mark.anyio
async def test_full_pool_plus_handoff_surge_blocks_another_gpu():
    control, _ = controller(ready=4, runners=4, backlog=1)
    with pytest.raises(NoCapacityError, match="handoff"):
        await control.before_launch(MODEL, REVISION, allocated=5)


@pytest.mark.anyio
async def test_old_controller_cannot_reconfigure_new_deployment():
    control, lambda_pool = controller()
    control.state.get.side_effect = lambda key: (
        {"revision": "b" * 40, "retired": [REVISION]} if key == "serving_release" else None
    )
    await control.reconcile()
    lambda_pool.update_autoscaler.aio.assert_not_awaited()
    with pytest.raises(NoCapacityError):
        await control.before_launch(MODEL, REVISION, 0)
    with pytest.raises(RuntimeError):
        await control.activate(REVISION)


@pytest.mark.anyio
async def test_unconfirmed_lambda_allocations_reserve_budget_until_cleanup():
    control, _ = controller(runners=1, backlog=1)
    control.fleet.allocated_count.return_value = 5
    with pytest.raises(NoCapacityError):
        await control.before_launch(MODEL, REVISION, allocated=5)


@pytest.mark.anyio
async def test_idle_zero_floor_model_does_not_allocate_lambda():
    model = "Qwen/Qwen3.5-4B"
    control, lambda_pool = controller()
    await control.reconcile_model(model)
    assert lambda_pool.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0
    with pytest.raises(NoCapacityError, match="idle"):
        await control.before_launch(model, REVISION, allocated=0)


class ReleaseState:
    def __init__(self, revision):
        self.data = {"active_revision": revision}

    async def get(self, key):
        return self.data.get(key)

    async def put(self, key, value):
        self.data[key] = value


@pytest.mark.anyio
async def test_controller_upgrade_preserves_serving_until_replacement_activation():
    control, primary = controller()
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
    control, primary = controller()
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
    control, primary = controller()
    control.state = ReleaseState(None)
    await control.before_launch(MODEL, "b" * 40, allocated=0)
    # Legacy serving pools are not reconfigured before activation.
    primary.update_autoscaler.aio.assert_not_awaited()
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
    lambda_pool = pool(runners=1, backlog=1)
    state = ReleaseState(old)
    control = SharedCapacity(fleet, lambda _: lambda_pool, state, new)
    assert await fleet.ready_count(MODEL) == 4
    assert await fleet.ready_count(MODEL, revision=new) == 0

    await control.activate(new)
    assert await fleet.allocated_count(MODEL) == 4
    # Four old GPUs plus one handoff GPU is the ceiling during a rolling deploy.
    await control.before_launch(MODEL, new, allocated=4)
    with pytest.raises(NoCapacityError, match="handoff"):
        await control.before_launch(MODEL, new, allocated=5)

    # As old workers drain, the budget returns.
    ledger.data["0"]["released"] = True
    ledger.data["1"]["released"] = True
    await control.before_launch(MODEL, new, allocated=2)
    await ledger.put("new", {"model": MODEL, "revision": new, "ready_at": 100, "gpu": "H100"})
    assert await fleet.allocated_count(MODEL) == 3
    assert await fleet.ready_count(MODEL, revision=new) == 1

    ledger.data["2"]["ready_at"] = 0  # An unhealthy old worker still consumes budget.
    await control.before_launch(MODEL, new, allocated=3)
    assert await fleet.allocated_count(MODEL) == 3


@pytest.mark.anyio
@pytest.mark.parametrize("model", lambda_models())
async def test_dev_cold_start_demand_and_return_to_zero(model):
    control, lambda_pool = controller()
    control.fleet.environment = "dev"
    await control.activate(REVISION)
    await control.reconcile_model(model)
    assert lambda_pool.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0
    with pytest.raises(NoCapacityError, match="idle"):
        await control.before_launch(model, REVISION, allocated=0)

    stats = lambda_pool.health.get_current_stats.aio.return_value
    stats.num_total_runners, stats.backlog = 1, 1
    await control.before_launch(model, REVISION, allocated=0)
    maximum = policy_for(model).max_capacity
    stats.backlog = maximum * sequence_capacity(model) * 2
    assert (await control.reconcile_model(model))[1] == maximum
    # Dev never holds an idle floor, whatever the demand.
    assert lambda_pool.update_autoscaler.aio.await_args.kwargs["min_containers"] == 0
    stats.num_total_runners, stats.backlog = 0, 0
    assert (await control.reconcile_model(model))[1] == 0
