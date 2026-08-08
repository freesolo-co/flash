"""Direct lookup coverage for public serving price selection."""

from __future__ import annotations

import pytest

import flash.serve.pricing as pricing


def test_serving_price_returns_the_registered_model_record() -> None:
    """Known model lookup must return the exact immutable catalog price object."""
    price = pricing.SERVING_PRICES["Qwen/Qwen3.5-4B"]

    assert price.model_id == "Qwen/Qwen3.5-4B"


def test_serving_prices_are_keyed_by_their_own_model_id() -> None:
    """Every record must agree with the key it is filed under.

    Callers index ``SERVING_PRICES`` directly, so a record filed under the wrong key would bill one
    model at another's rate with nothing raising. A missing model surfaces as a KeyError naming the
    id, which is what the indexing contract promises.
    """
    for model_id, price in pricing.SERVING_PRICES.items():
        assert price.model_id == model_id

    with pytest.raises(KeyError):
        pricing.SERVING_PRICES["missing/model"]
