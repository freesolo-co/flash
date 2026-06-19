"""Cost estimator: lenient parsing of the LLM reply (``llm._extract``).

No network. The parser must degrade gracefully on untrusted model output -- in
particular it must never hand a non-string ``gpu`` to the GPU-match downstream.
"""

from __future__ import annotations

from flash.cost.experiment import _gpu_match
from flash.cost.llm import _extract


def test_extract_reads_cost_and_gpu():
    cost, gpu, _ = _extract('{"cost_usd": 1.2, "gpu": "RTX 5090", "reasoning": "x"}')
    assert cost == 1.2
    assert gpu == "RTX 5090"


def test_extract_coerces_numeric_gpu_to_string():
    # A model that returns a numeric/structured gpu must not crash the sweep downstream
    # (_gpu_match calls est_gpu.lower()); coerce it to a string instead.
    cost, gpu, _ = _extract('{"cost_usd": 1.2, "gpu": 5090}')
    assert cost == 1.2
    assert isinstance(gpu, str)
    # The coerced value flows safely through the GPU-match without raising.
    assert _gpu_match(gpu, "RTX 5090") is False  # "5090" != the full "RTX 5090" class


def test_extract_drops_empty_gpu_to_none():
    cost, gpu, _ = _extract('{"cost_usd": 0.5, "gpu": ""}')
    assert cost == 0.5
    assert gpu is None


def test_extract_drops_numeric_zero_gpu_to_none():
    # A numeric 0 (or the string "0") means "no GPU" -- it must coerce to None, not a
    # falsy-but-present "0" that the GPU-pick metric would then grade against the truth.
    for reply in ('{"cost_usd": 0.5, "gpu": 0}', '{"cost_usd": 0.5, "gpu": "0"}'):
        cost, gpu, _ = _extract(reply)
        assert cost == 0.5
        assert gpu is None


def test_extract_returns_none_when_no_cost():
    cost, gpu, _reasoning = _extract("the model said nothing useful")
    assert cost is None
    assert gpu is None
