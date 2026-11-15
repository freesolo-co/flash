import json
from unittest.mock import AsyncMock

import pytest

from flash.serving.lambda_cloud import reserved
from flash.serving.lambda_cloud.config import NoCapacityError
from flash.serving.lambda_cloud.fleet import Fleet

MODEL = "Qwen/Qwen3.5-9B-Base"
SMALL = "Qwen/Qwen3.5-0.8B-Base"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def workers():
    return [
        {"slot": i, "instance_id": str(i), "ip": f"8.8.8.{i}", "revision": "a" * 40, "tls": {}}
        for i in range(1, 5)
    ]


def test_reject_duplicate_reserved_instance():
    data = workers()
    data[1]["instance_id"] = data[0]["instance_id"]
    with pytest.raises(ValueError, match="distinct"):
        reserved.worker_config(1, json.dumps(data))


@pytest.mark.anyio
async def test_reserved_shutdown_does_not_release_gpu():
    proxy = reserved.ReservedLambdaEngineProxy()
    proxy._transport = AsyncMock()
    proxy._fleet = AsyncMock()
    await proxy._shutdown()
    proxy._transport.aclose.assert_awaited_once()
    proxy._fleet.release.remote.aio.assert_not_awaited()


@pytest.mark.anyio
async def test_wrong_worker_identity_fails_closed(monkeypatch):
    monkeypatch.setenv("FLASH_RESERVED_H100_WORKERS", json.dumps(workers()))
    transport = AsyncMock()
    transport.call.return_value = {
        "ok": True,
        "base_model": MODEL,
        "deployment_sha": "b" * 40,
        "configured_gpu": "H100",
    }
    monkeypatch.setattr(reserved, "WorkerTransport", lambda *a: transport)
    proxy = reserved.ReservedLambdaEngineProxy()
    proxy.base_model = MODEL
    proxy.reserved_slot = 1
    with pytest.raises(RuntimeError, match="identity"):
        await proxy._load()
    transport.aclose.assert_awaited_once()


@pytest.mark.anyio
async def test_dynamic_fleet_cannot_allocate_more_9b_base_gpus():
    api = AsyncMock()
    ledger = AsyncMock()
    fleet = Fleet(api, ledger, "main")
    with pytest.raises(NoCapacityError, match="reserved"):
        await fleet._check_capacity(MODEL, "a" * 40)
    api.call.assert_not_awaited()


@pytest.mark.anyio
async def test_shared_reconciler_skips_reserved_model(monkeypatch):
    from types import SimpleNamespace

    from flash.serving.lambda_cloud.pool_capacity import SharedCapacity

    monkeypatch.setattr("flash.serving.lambda_cloud.pool_capacity.lambda_models", lambda: [MODEL])
    state = SimpleNamespace(get=AsyncMock(return_value={"revision": "a" * 40, "retired": []}))
    control = SharedCapacity(AsyncMock(), AsyncMock(), state, "a" * 40)
    control.reconcile_model = AsyncMock()
    await control.reconcile()
    control.reconcile_model.assert_not_awaited()


@pytest.mark.anyio
async def test_development_keeps_lambda_only_capacity_without_modal_fallback():
    from types import SimpleNamespace

    async def empty_items():
        if False:
            yield None

    api = SimpleNamespace(call=AsyncMock(return_value=[]))
    ledger = SimpleNamespace(items=empty_items)
    fleet = Fleet(api, ledger, "dev")
    fleet.shared_capacity = AsyncMock()
    await fleet._check_capacity(MODEL, "a" * 40)
    fleet.shared_capacity.before_launch.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("old_count", "current_count", "allowed"), [(4, 0, True), (4, 1, False), (0, 4, False)]
)
@pytest.mark.parametrize("ownership", ["leased", "draining", "uncertain"])
async def test_dev_lambda_only_budget_counts_all_revisions(
    old_count, current_count, allowed, ownership
):
    from types import SimpleNamespace

    now = 10000
    records = []
    instances = []
    for i in range(old_count + current_count):
        owner = f"{i:032x}"
        records.append(
            (
                owner,
                {
                    "model": MODEL,
                    "revision": "old" if i < old_count else "new",
                    "instance_id": None if ownership == "uncertain" else str(i),
                    "released": ownership == "draining",
                    "heartbeat": now,
                },
            )
        )
        if ownership == "draining":
            instances.append(
                {
                    "name": f"flash-dev-{owner}",
                    "status": "terminating",
                    "tags": [
                        {"key": "flash-owner", "value": owner},
                        {"key": "flash-environment", "value": "dev"},
                    ],
                }
            )

    async def items():
        for item in records:
            yield item

    fleet = Fleet(
        SimpleNamespace(call=AsyncMock(return_value=instances)),
        SimpleNamespace(items=items),
        "dev",
        clock=lambda: now,
    )
    fleet.shared_capacity = AsyncMock()
    if allowed:
        await fleet._check_capacity(MODEL, "new")
    else:
        with pytest.raises(NoCapacityError, match="replica limit"):
            await fleet._check_capacity(MODEL, "new")
    fleet.shared_capacity.before_launch.assert_not_awaited()


def test_each_reserved_model_has_its_own_hardware_slots_and_secret():
    assert reserved.MODELS == (MODEL, SMALL)
    assert (reserved.fleet_for(MODEL).gpu, reserved.fleet_for(MODEL).slots) == ("H100", 4)
    assert (reserved.fleet_for(SMALL).gpu, reserved.fleet_for(SMALL).slots) == ("H200", 1)
    assert reserved.engine_name_suffix(MODEL, 3) == "_ReservedH1003"
    assert reserved.engine_name_suffix(SMALL, 1) == "_ReservedH2001"
    assert reserved.fleet_for(MODEL).secret_name != reserved.fleet_for(SMALL).secret_name
    assert reserved.enabled(SMALL, "main")
    assert not reserved.enabled(SMALL, "dev")
    with pytest.raises(ValueError, match="no reserved serving fleet"):
        reserved.fleet_for("Qwen/Qwen3.5-0.8B")


def test_single_h200_slot_rejects_a_four_slot_inventory():
    single = [workers()[0]]
    assert reserved.worker_config(1, json.dumps(single), model=SMALL)["instance_id"] == "1"
    with pytest.raises(ValueError, match="1 unique slots"):
        reserved.worker_config(1, json.dumps(workers()), model=SMALL)


@pytest.mark.anyio
@pytest.mark.parametrize(("gpu", "ok"), [("H200", True), ("H100", False)])
async def test_h200_worker_identity_requires_its_own_model_and_hardware(monkeypatch, gpu, ok):
    monkeypatch.setenv("FLASH_RESERVED_H200_WORKERS", json.dumps([workers()[0]]))
    transport = AsyncMock()
    transport.call.return_value = {
        "ok": True,
        "base_model": SMALL,
        "deployment_sha": "a" * 40,
        "configured_gpu": gpu,
    }
    monkeypatch.setattr(reserved, "WorkerTransport", lambda *a: transport)
    proxy = reserved.ReservedLambdaEngineProxy()
    proxy.base_model = SMALL
    proxy.reserved_slot = 1
    if ok:
        await proxy._load()
        assert proxy.engine is transport
        transport.aclose.assert_not_awaited()
    else:
        with pytest.raises(RuntimeError, match="H200"):
            await proxy._load()
        transport.aclose.assert_awaited_once()


@pytest.mark.anyio
async def test_dynamic_fleet_cannot_allocate_main_0_8b_base_gpus():
    api = AsyncMock()
    fleet = Fleet(api, AsyncMock(), "main")
    with pytest.raises(NoCapacityError, match="reserved"):
        await fleet._check_capacity(SMALL, "a" * 40)
    api.call.assert_not_awaited()
