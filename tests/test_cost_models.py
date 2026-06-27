"""Cost estimator: model-size facts (catalog-only). No network.

The cost model supports the curated catalog only — five dense text models plus one MoE, no
open-model/unlisted sizing. The numeric size is the catalog's ``params_b`` stat, read directly
(no parsing of the ``params`` display string). For the MoE the per-token FLOPs/step-time term reads
the smaller ``active_params_b`` while memory/size terms (VRAM, disk, download) keep total ``params_b``.
"""

from __future__ import annotations

import pytest

from flash.catalog import MODELS
from flash.cost.facts import active_params_b, download_weight_gb, model_quant, total_params_b


@pytest.mark.parametrize("model_id", list(MODELS))
def test_total_params_is_the_catalog_stat(model_id):
    assert total_params_b(model_id) == pytest.approx(MODELS[model_id].params_b)


@pytest.mark.parametrize("model_id", list(MODELS))
def test_active_params_defaults_to_total_for_dense(model_id):
    # A dense model (active_params_b unset) bills FLOPs against its full param count.
    info = MODELS[model_id]
    if info.active_params_b:
        pytest.skip(f"{model_id} is an MoE (active_params_b set)")
    assert active_params_b(model_id) == pytest.approx(info.params_b)


def test_active_params_uses_the_active_count_for_the_moe():
    # The MoE's per-token FLOPs size is the ~3B active count, far below its 35B total — so cost/step
    # time isn't ~10x overstated. Memory/size terms still use the 35B total (asserted just below).
    moe = "Qwen/Qwen3.6-35B-A3B"
    assert active_params_b(moe) == pytest.approx(3.0)
    assert active_params_b(moe) < total_params_b(moe)


def test_moe_memory_terms_still_use_total_params():
    # Download (and VRAM/disk, which read total_params_b too) size the FULL checkpoint — all experts
    # are materialized on the GPU — so they must NOT shrink to the active count.
    moe = "Qwen/Qwen3.6-35B-A3B"
    assert download_weight_gb(moe) == pytest.approx(total_params_b(moe) * 2.0)
    assert download_weight_gb(moe) == pytest.approx(70.0)


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
    with pytest.raises(ValueError, match="catalog models only"):
        active_params_b("nobody/never-heard-of-it")
