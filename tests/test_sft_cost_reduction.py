"""Cost optimizations preserve model identity, bounded resources, and safe memory fallback."""

import asyncio
import json
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flash.core.spec import JobSpec
from flash.providers.core.base import UnreconciledCreateError, get_gpu_info
from flash.providers.modal import api, prefetch
from flash.providers.modal.api import call as modal_call
from flash.providers.modal.handles import ModalJobHandle
from flash.providers.modal.pricing import (
    CPU_HOURLY,
    CPU_PER_GPU,
    MEMORY_GIB_HOURLY,
    MEMORY_MIB_PER_GPU,
    hourly_rate,
)
from tests.test_modal_provider import AsyncMethod, remote

PIN = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def test_rate_matches_actual_minimum_reservations():
    assert hourly_rate("L40S") == pytest.approx(
        get_gpu_info("L40S").hourly_usd
        + CPU_PER_GPU * CPU_HOURLY
        + MEMORY_MIB_PER_GPU / 1024 * MEMORY_GIB_HOURLY
    )


@pytest.fixture
def warmer(monkeypatch):
    class Missing(Exception):
        pass

    state = SimpleNamespace(
        existing=None, creates=[], events=[], fail_create=False, fail_reconcile=False, exit=0
    )
    sandbox = SimpleNamespace(
        object_id="sb-weights",
        wait=AsyncMethod(lambda: state.events.append("wait")),
        poll=AsyncMethod(lambda: state.exit),
        terminate=AsyncMethod(lambda **kw: state.events.append("committed-and-stopped")),
    )

    def find(*args, **kwargs):
        if state.creates and state.fail_reconcile == "lookup":
            raise OSError("private transport detail")
        if state.existing is None:
            raise Missing
        return state.existing

    def create(*args, **kwargs):
        state.creates.append((args, kwargs))

        def tags():
            if state.fail_reconcile == "tags":
                raise OSError("private transport detail")
            return kwargs["tags"]

        sandbox.get_tags = AsyncMethod(tags)
        state.existing = sandbox
        if state.fail_create:
            raise OSError("lost response")
        return sandbox

    modal = SimpleNamespace(
        App=SimpleNamespace(lookup=AsyncMethod(lambda *a, **kw: object())),
        Volume=SimpleNamespace(from_name=lambda *a, **kw: object()),
        Image=SimpleNamespace(
            debian_slim=lambda **kw: SimpleNamespace(pip_install=lambda *a: object())
        ),
        Secret=SimpleNamespace(from_dict=lambda values: values),
        Sandbox=SimpleNamespace(create=AsyncMethod(create), from_name=AsyncMethod(find)),
        exception=SimpleNamespace(NotFoundError=Missing),
    )
    state.modal = modal
    monkeypatch.setattr(
        api, "call", lambda operation, **kw: asyncio.run(operation(modal, object()))
    )
    return state


def prepare(**changes):
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B", model_revision=PIN, algorithm="sft", run_id="test-cpu-stage"
    )
    return prefetch.prepare(
        replace(spec, **changes),
        attempt=0,
        deadline_at=time.time() + 600,
        payload={
            "env": {
                "FLASH_WEIGHT_CACHE_DIR": "/weight-cache/hf-cache/hub",
                "HF_TOKEN": "test-private",
            }
        },
    )


def test_cpu_stage_is_bounded_pinned_and_stopped_before_return(warmer):
    result = prepare()
    args, kw = warmer.creates[0]
    assert "gpu" not in kw
    assert kw["cpu"] == (2, 4)
    assert kw["memory"] == (4096, 8192)
    assert 0 < kw["timeout"] <= 600
    secret = kw["secrets"][0]
    assert json.loads(secret["FLASH_BASE_MODEL"]) == ["Qwen/Qwen3.5-9B", PIN]
    assert "test-private" not in str(args) + str(kw["tags"])
    assert warmer.events == ["wait", "committed-and-stopped"]
    assert result["sandbox_id"] == "sb-weights"
    assert result["cost_upper_estimate_usd"] >= 0
    data = remote(cpu_prefetch=result)
    assert ModalJobHandle.from_dict(data).to_dict() == data


def test_lost_cpu_create_is_adopted_without_duplicate(warmer):
    warmer.fail_create = True
    assert prepare()["sandbox_id"] == "sb-weights"
    assert len(warmer.creates) == 1


def test_cpu_failure_stops_and_aborts_before_gpu(warmer):
    warmer.exit = 1
    with pytest.raises(api.ModalApiError, match="before GPU"):
        prepare()
    assert warmer.events[-1] == "committed-and-stopped"


def test_cpu_stage_cannot_adopt_another_identity(warmer):
    warmer.existing = SimpleNamespace(get_tags=AsyncMethod(lambda: {"flash_run": "foreign"}))
    with pytest.raises(UnreconciledCreateError, match="another run"):
        prepare()
    assert not warmer.creates


@pytest.mark.parametrize(
    "changes", [{"algorithm": "grpo"}, {"model_revision": "main"}, {"model": "org/private"}]
)
def test_only_pinned_catalog_sft_is_staged(warmer, changes):
    assert prepare(**changes) == {}
    assert not warmer.creates


def test_bad_prefetch_meter_is_rejected():
    with pytest.raises(ValueError, match="prefetch"):
        ModalJobHandle.from_dict(
            remote(
                cpu_prefetch={
                    "sandbox_id": "sb-test",
                    "wall_seconds": float("nan"),
                    "cost_upper_estimate_usd": 1,
                }
            )
        )


@pytest.mark.parametrize("failure", ["lookup", "tags"])
def test_lost_create_and_failed_reconciliation_remain_ambiguous(warmer, failure):
    warmer.fail_create = True
    warmer.fail_reconcile = failure
    with pytest.raises(UnreconciledCreateError, match="could not be reconciled") as error:
        prepare()
    assert "private transport detail" not in str(error.value)
    assert len(warmer.creates) == 1
    assert not warmer.events


@pytest.mark.parametrize("phase", ["lookup", "create", "reconcile"])
def test_cpu_create_timeout_preserves_launch_uncertainty(warmer, monkeypatch, phase):
    # Exercise the real api.call deadline conversion; the fake transport never completes
    # at the selected stage, so this asserts classification rather than elapsed wall time.
    module = warmer.modal
    original_find = module.Sandbox.from_name.fn
    original_create = module.Sandbox.create.fn

    async def find(*args, **kwargs):
        if phase == "lookup" or (phase == "reconcile" and warmer.creates):
            await asyncio.Future()
        return original_find(*args, **kwargs)

    async def create(*args, **kwargs):
        sandbox = original_create(*args, **kwargs)
        await asyncio.Future()
        return sandbox

    async def close(*args):
        return None

    if phase == "reconcile":
        warmer.fail_create = True
    module.Sandbox.from_name = SimpleNamespace(aio=find)
    if phase == "create":
        module.Sandbox.create = SimpleNamespace(aio=create)
    module.Client = SimpleNamespace(
        from_credentials=AsyncMethod(lambda *args: SimpleNamespace(__aexit__=close))
    )
    monkeypatch.setitem(sys.modules, "modal", module)
    monkeypatch.setattr(api, "credentials", lambda: ("test-id", "test-secret"))
    monkeypatch.setattr(
        api,
        "call",
        lambda operation, **kwargs: modal_call(operation, timeout=0.01),
    )
    expected = api.ModalApiError if phase == "lookup" else UnreconciledCreateError
    with pytest.raises(expected):
        prepare()
    assert len(warmer.creates) == int(phase != "lookup")
