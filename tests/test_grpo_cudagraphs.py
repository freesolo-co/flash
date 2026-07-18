from __future__ import annotations

import pytest


@pytest.mark.parametrize("cc", [(8, 0), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)])
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
