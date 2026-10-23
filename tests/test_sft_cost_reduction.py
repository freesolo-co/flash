"""Cost optimizations preserve model identity, bounded resources, and safe memory fallback."""

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flash.core.spec import JobSpec
from flash.engine.worker.perf.memory import grad_checkpointing_on
from flash.providers.core.base import UnreconciledCreateError, get_gpu_info
from flash.providers.modal import api, prefetch
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


def dense_gate(**overrides):
    options = {
        "model_id": "Qwen/Qwen3.5-9B",
        "max_length": 496,
        "revision": PIN,
        "allow_disable": True,
        "allow_short_dense": True,
        "card_vram_gb": 47.695921152,
        "capability": (8, 9),
        "hidden": 4096,
        "num_layers": 32,
        "fused_ce": True,
        "per_device_bs": 1,
        "lora_rank": 8,
    }
    return grad_checkpointing_on(**(options | overrides))


def test_qualified_short_dense_workload_avoids_recompute():
    assert dense_gate() is False


@pytest.mark.parametrize(
    "override",
    [
        {"max_length": 513},
        {"max_length": 0},
        {"lora_rank": 16},
        {"per_device_bs": 2},
        {"per_device_bs": 0},
        {"card_vram_gb": 24},
        {"fused_ce": False},
        {"revision": "f" * 40},
        {"hidden": 0},
        {"num_layers": 24},
        {"capability": (9, 0)},
        {"allow_disable": False},
        {"allow_short_dense": False},
        {"active_params_b": 3.0},
    ],
)
def test_unqualified_shapes_keep_recompute(override):
    assert dense_gate(**override) is True


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

    state = SimpleNamespace(existing=None, creates=[], events=[], fail_create=False, exit=0)
    sandbox = SimpleNamespace(
        object_id="sb-weights",
        wait=AsyncMethod(lambda: state.events.append("wait")),
        poll=AsyncMethod(lambda: state.exit),
        terminate=AsyncMethod(lambda **kw: state.events.append("committed-and-stopped")),
    )

    def find(*args, **kwargs):
        if state.existing is None:
            raise Missing
        return state.existing

    def create(*args, **kwargs):
        state.creates.append((args, kwargs))
        sandbox.get_tags = AsyncMethod(lambda: kwargs["tags"])
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
