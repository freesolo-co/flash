"""Cost estimator: model-size facts (total params, MoE active params, quant, download).

No network. Catalog-backed values must agree with ``flash.catalog``.
"""

from __future__ import annotations

import pytest

from flash.catalog import MODELS
from flash.cost.facts import (
    DEFAULT_PARAMS_B,
    active_params_b,
    download_weight_gb,
    model_quant,
    total_params_b,
)
from flash.engine.vram import params_b_from_str


@pytest.mark.parametrize("model_id", list(MODELS))
def test_total_params_match_catalog(model_id):
    expected = params_b_from_str(MODELS[model_id].params)
    assert total_params_b(model_id) == pytest.approx(expected)


@pytest.mark.parametrize("model_id", list(MODELS))
def test_dense_catalog_models_have_active_equals_total(model_id):
    # The curated catalog is all dense -> every parameter is active.
    assert active_params_b(model_id) == pytest.approx(total_params_b(model_id))


def test_moe_active_params_parsed_from_id_or_string():
    # MoE ids carry the active count after the "A": 35B total, 3B active.
    assert active_params_b("Qwen/Qwen3.6-35B-A3B") == pytest.approx(3.0)
    assert total_params_b("Qwen/Qwen3.6-35B-A3B") == pytest.approx(35.0)
    # Also parsed from a catalog-style params string.
    assert active_params_b("some/model", params_str="80B-A13B MoE") == pytest.approx(13.0)


def test_quant_lookup():
    assert model_quant("Qwen/Qwen3.5-9B") == "4bit-qlora"
    assert model_quant("Qwen/Qwen3.5-4B") == "bf16"
    assert model_quant("unlisted/model") == "bf16"  # default


def test_download_weight_gb_is_total_params_bf16():
    # Download is always the full bf16 checkpoint (2 bytes/param), even for QLoRA.
    assert download_weight_gb("Qwen/Qwen3.5-9B") == pytest.approx(
        total_params_b("Qwen/Qwen3.5-9B") * 2.0
    )


def test_unknown_model_falls_back_to_default():
    assert total_params_b("nobody/never-heard-of-it") == DEFAULT_PARAMS_B
    assert active_params_b("nobody/never-heard-of-it") == DEFAULT_PARAMS_B


def test_lowercase_size_suffix_in_id_is_parsed():
    # A lowercase "b" suffix in an unlisted id is read (9B), not dropped to the 4B
    # default -- which would underestimate compute, download, and VRAM.
    assert total_params_b("Qwen/Qwen3.5-9b-instruct") == pytest.approx(9.0)
    assert total_params_b("vendor/foo-13b") == pytest.approx(13.0)
    # Uppercase still parses too.
    assert total_params_b("vendor/foo-7B") == pytest.approx(7.0)


def test_million_suffix_in_id_is_parsed():
    # A sub-1B checkpoint with an "M" (million) suffix is read as a fraction of a billion,
    # not quoted as the 4B default (which would massively over-size its GPU/download).
    assert total_params_b("google/gemma-3-270m") == pytest.approx(0.27)
    assert total_params_b("vendor/foo-750M") == pytest.approx(0.75)
    # Download footprint follows the parsed size (0.27B * 2 bytes/param = 0.54 GB).
    assert download_weight_gb("google/gemma-3-270m") == pytest.approx(0.54)


def test_expert_count_moe_suffix_is_expanded_to_total():
    # An "NxM B" expert-count id is the TOTAL param count (8 experts x 7B = 56B), not a
    # single 7B expert -- otherwise a large MoE is sized/downloaded as a small dense model.
    assert total_params_b("mistralai/Mixtral-8x7B-Instruct") == pytest.approx(56.0)
    # No "A<active>B" hint in an expert-count id -> active defaults to total (dense-equivalent).
    assert active_params_b("mistralai/Mixtral-8x7B-Instruct") == pytest.approx(56.0)
