"""Checks for the exact native scheduler interface loaded inside EngineCore."""

import pytest

native = pytest.importorskip("vllm.v1.core.sched.async_scheduler")
if not hasattr(native, "AsyncScheduler"):
    pytest.skip("requires native vLLM", allow_module_level=True)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_scheduler_uses_resolved_native_async_setting(monkeypatch, asynchronous):
    from types import SimpleNamespace

    from flash.serve.runtime import cache_scheduler

    expected = object()
    selected = "AsyncCacheScheduler" if asynchronous else "CacheScheduler"
    monkeypatch.setattr(cache_scheduler, selected, lambda **kwargs: expected)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=asynchronous))
    assert cache_scheduler.QwenCacheScheduler(vllm_config=config) is expected


@pytest.mark.parametrize("name", ["CacheScheduler", "AsyncCacheScheduler"])
def test_unqualified_native_version_fails_before_initialization(monkeypatch, name):
    from flash.serve.runtime import cache_scheduler

    monkeypatch.setattr(cache_scheduler, "version", lambda package: "0.99.0")
    with pytest.raises(RuntimeError, match="qualified vLLM"):
        getattr(cache_scheduler, name)()
