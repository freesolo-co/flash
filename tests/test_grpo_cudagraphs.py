from __future__ import annotations

import sys
import types

import pytest


@pytest.mark.parametrize("cc", [(8, 0), (9, 0), (10, 3), (12, 0), (12, 1)])
def test_colocate_rollout_uses_decode_only_graphs_on_validated_arches(monkeypatch, cc):
    from flash.engine.worker import rl

    calls = []
    monkeypatch.setattr(
        rl._w, "patch_trl_colocate_llm_kwargs", lambda **kwargs: calls.append(kwargs)
    )

    selected = rl._patch_colocate_rollout_compilation(cc)

    expected = {
        "enforce_eager": False,
        "compilation_config": {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
        },
    }
    assert selected == expected
    assert calls == [expected]


def test_colocate_rollout_keeps_vllm_default_on_b200(monkeypatch):
    from flash.engine.worker import rl

    calls = []
    monkeypatch.setattr(
        rl._w, "patch_trl_colocate_llm_kwargs", lambda **kwargs: calls.append(kwargs)
    )

    selected = rl._patch_colocate_rollout_compilation((10, 0))

    assert selected is None
    assert calls == []


@pytest.mark.parametrize("cc", [(0, 0), (8, 6), (8, 9), (11, 0)])
def test_colocate_rollout_keeps_eager_fallback_on_unvalidated_arches(monkeypatch, cc):
    from flash.engine.worker import rl

    calls = []
    monkeypatch.setattr(
        rl._w, "patch_trl_colocate_llm_kwargs", lambda **kwargs: calls.append(kwargs)
    )

    selected = rl._patch_colocate_rollout_compilation(cc)

    assert selected == {"enforce_eager": True}
    assert calls == [{"enforce_eager": True}]


def test_colocate_rollout_retries_eager_after_graph_capture_failure(monkeypatch):
    from flash.engine.worker.gpu_setup import patch_trl_colocate_llm_kwargs

    calls = []

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            calls.append(dict(kwargs))
            if not kwargs.get("enforce_eager", False):
                raise RuntimeError("cuda graph capture failed while warming up model")
            self.eager = True

    module = types.ModuleType("trl.generation.vllm_generation")
    module.LLM = FakeLLM
    for name in ("trl", "trl.generation"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "trl.generation.vllm_generation", module)

    config = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    assert patch_trl_colocate_llm_kwargs(enforce_eager=False, compilation_config=config) is True

    engine = module.LLM(model="m")

    assert engine.eager is True
    assert calls == [
        {"model": "m", "enforce_eager": False, "compilation_config": config},
        {"model": "m", "enforce_eager": True},
    ]
    assert module._flash_llm_overrides == {"enforce_eager": True}
