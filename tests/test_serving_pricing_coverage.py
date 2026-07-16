"""Direct lookup coverage for public serving price selection."""

from __future__ import annotations

import pytest

import flash.serve.pricing as pricing


def test_serving_price_returns_the_registered_model_record() -> None:
    """Known model lookup must return the exact immutable catalog price object."""
    price = pricing.serving_price("Qwen/Qwen3.5-4B")

    assert price is pricing.SERVING_PRICES["Qwen/Qwen3.5-4B"]
    assert price.model_id == "Qwen/Qwen3.5-4B"


def test_serving_price_translates_unknown_models_to_value_error() -> None:
    """Unknown model ids must raise the public validation error rather than leaking a KeyError."""
    with pytest.raises(ValueError, match="unknown serving model 'missing/model'"):
        pricing.serving_price("missing/model")
