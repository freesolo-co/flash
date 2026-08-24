"""Owned prequantized FP8 checkpoint mappings and serve-model resolution."""

from __future__ import annotations

import pytest

from flash.serving.src.model_config import base_models
from flash.serving.src.prequant_config import (
    FP8_SERVE_MODEL_REPOS,
    OFFICIAL_FP8_MODEL_REPOS,
    OWNED_FP8_MODEL_REPOS,
    fp8_serve_model_for,
)

OWNED_DENSE_MODELS = (
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-27B",
)
DENSE_27B = "Qwen/Qwen3.6-27B"
MOE_35B = "Qwen/Qwen3.6-35B-A3B"


def test_owned_fp8_covers_the_dense_models_only() -> None:
    assert set(OWNED_FP8_MODEL_REPOS) == set(OWNED_DENSE_MODELS)
    for repo in OWNED_FP8_MODEL_REPOS.values():
        assert repo.startswith("Freesolo-Co/")
        assert repo.endswith("-FP8")
    # the 27B dense model is owned; only the 35B MoE uses the official Qwen FP8 checkpoint.
    assert OWNED_FP8_MODEL_REPOS[DENSE_27B] == "Freesolo-Co/Qwen3.6-27B-FP8"
    assert MOE_35B not in OWNED_FP8_MODEL_REPOS


def test_fp8_serve_map_uses_official_fp8_for_35b() -> None:
    assert OFFICIAL_FP8_MODEL_REPOS[MOE_35B] == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert set(OFFICIAL_FP8_MODEL_REPOS) == {MOE_35B}
    for base in OWNED_DENSE_MODELS:
        assert FP8_SERVE_MODEL_REPOS[base] == OWNED_FP8_MODEL_REPOS[base]
    assert set(FP8_SERVE_MODEL_REPOS) == set(base_models())


def test_fp8_serve_model_for_resolves_fp8_for_every_base() -> None:
    for base in OWNED_DENSE_MODELS:
        assert fp8_serve_model_for(base).endswith("-FP8")
    assert fp8_serve_model_for(DENSE_27B) == "Freesolo-Co/Qwen3.6-27B-FP8"
    # the 35B serves the official Qwen FP8 (VL-preserving).
    assert fp8_serve_model_for(MOE_35B) == "Qwen/Qwen3.6-35B-A3B-FP8"


def test_fp8_serve_model_for_rejects_unknown_base() -> None:
    with pytest.raises(ValueError, match="Allowed models"):
        fp8_serve_model_for("some/unlisted-model")
