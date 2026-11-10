"""Deployment waits for the entire requested fleet without mutating it."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from flash.serving.lambda_cloud import capacity_gate as gate
from tests.serving.test_lambda_capacity_wait import Clock
from tests.serving.test_lambda_placement import offer

MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-9B-Base"]
AVAILABLE = {"a10": offer("A10 (24 GB PCIe)"), "h100": offer("H100 (80 GB SXM5)")}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_waits_for_all_shapes_and_reports_each_five_minute_check(caplog):
    clock = Clock()
    api = SimpleNamespace(call=AsyncMock(side_effect=[{}, {"a10": AVAILABLE["a10"]}, AVAILABLE]))
    await gate.wait_for_capacity(api, MODELS, clock=clock, sleep=clock.sleep)
    assert clock.waits == [300, 300]
    assert "elapsed 5 min, remaining 295 min" in caplog.text
    assert "checking again in 5 min" in caplog.text
    assert "proceeding" in caplog.text
    assert all(call.args == ("GET", "instance-types") for call in api.call.call_args_list)


@pytest.mark.anyio
async def test_no_capacity_stops_after_five_hours_without_a_late_check(caplog):
    clock = Clock()
    api = SimpleNamespace(call=AsyncMock(return_value={}))
    with pytest.raises(TimeoutError, match="five hours"):
        await gate.wait_for_capacity(api, MODELS, clock=clock, sleep=clock.sleep)
    assert clock.now == 18000
    assert clock.waits == [300] * 60
    assert api.call.await_count == 60
    assert len(caplog.records) == 60


@pytest.mark.anyio
async def test_authentication_errors_fail_without_waiting():
    clock = Clock()
    api = SimpleNamespace(call=AsyncMock(side_effect=RuntimeError("auth")))
    with pytest.raises(RuntimeError, match="auth"):
        await gate.wait_for_capacity(api, MODELS, clock=clock, sleep=clock.sleep)
    assert clock.waits == []
    assert api.call.await_count == 1


@pytest.mark.anyio
async def test_all_hosted_deployments_require_lambda_credentials(monkeypatch):
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LAMBDA_API_KEY"):
        await gate.main()


@pytest.mark.anyio
async def test_capacity_on_last_poll_still_allows_deployment():
    clock = Clock()
    api = SimpleNamespace(call=AsyncMock(side_effect=[{}] * 59 + [AVAILABLE]))
    await gate.wait_for_capacity(api, MODELS, clock=clock, sleep=clock.sleep)
    assert clock.now == 17700
    assert api.call.await_count == 60


@pytest.mark.anyio
async def test_stalled_catalog_call_is_bounded(monkeypatch):
    import asyncio

    monkeypatch.setattr(gate, "DEPLOY_CAPACITY_WAIT_SECONDS", 0.01)

    async def call(*args):
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        await gate.wait_for_capacity(SimpleNamespace(call=call), MODELS)
