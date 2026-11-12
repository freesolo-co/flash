import json
from unittest.mock import AsyncMock

import pytest

from flash.serving.lambda_cloud import reserved
from flash.serving.lambda_cloud.config import NoCapacityError
from flash.serving.lambda_cloud.fleet import Fleet


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
        "base_model": reserved.MODEL,
        "deployment_sha": "b" * 40,
        "configured_gpu": "H100",
    }
    monkeypatch.setattr(reserved, "WorkerTransport", lambda *a: transport)
    proxy = reserved.ReservedLambdaEngineProxy()
    proxy.base_model = reserved.MODEL
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
        await fleet._check_capacity(reserved.MODEL, "a" * 40)
    api.call.assert_not_awaited()


@pytest.mark.anyio
async def test_shared_reconciler_skips_reserved_model(monkeypatch):
    from types import SimpleNamespace

    from flash.serving.lambda_cloud.pool_capacity import SharedCapacity

    monkeypatch.setattr(
        "flash.serving.lambda_cloud.pool_capacity.lambda_models", lambda: [reserved.MODEL]
    )
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
    await fleet._check_capacity(reserved.MODEL, "a" * 40)
    fleet.shared_capacity.before_launch.assert_not_awaited()
