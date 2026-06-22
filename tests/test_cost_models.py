"""Cost estimator: model-size facts (catalog-only). No network.

The cost model supports the curated catalog only — five dense text models, no MoE and no
open-model/unlisted sizing. The numeric size is the catalog's ``params_b`` stat, read directly
(no parsing of the ``params`` display string).
"""

from __future__ import annotations

import pytest

from flash.catalog import MODELS
from flash.cost.facts import download_weight_gb, model_quant, total_params_b


@pytest.mark.parametrize("model_id", list(MODELS))
def test_total_params_is_the_catalog_stat(model_id):
    assert total_params_b(model_id) == pytest.approx(MODELS[model_id].params_b)


def test_quant_lookup():
    assert model_quant("Qwen/Qwen3.5-9B") == "bf16"
    assert model_quant("Qwen/Qwen3.5-4B") == "bf16"


def test_download_weight_gb_is_total_params_bf16():
    # Download is always the full bf16 checkpoint (2 bytes/param).
    nine = "Qwen/Qwen3.5-9B"
    assert download_weight_gb(nine) == pytest.approx(total_params_b(nine) * 2.0)


def test_unknown_model_is_rejected():
    # Catalog-only: an unknown model raises (ValueError, like catalog.get_model) rather than
    # guessing a size.
    with pytest.raises(ValueError, match="catalog models only"):
        total_params_b("nobody/never-heard-of-it")
