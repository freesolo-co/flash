"""Prequantized FP8 checkpoint ownership and serve-model resolution."""

from __future__ import annotations

import pytest

from flash.serving.src.engine.model_config import base_models
from flash.serving.src.engine.prequant_config import (
    FP8_SERVE_MODEL_REPOS,
    OFFICIAL_FP8_MODEL_REPOS,
    OWNED_FP8_MODEL_REPOS,
    fp8_serve_model_for,
)

OWNED_9B = "Qwen/Qwen3.5-9B"
DENSE_27B = "Qwen/Qwen3.8-27B"
MOE_35B = "Qwen/Qwen3.6-35B-A3B"


def test_owned_fp8_contains_only_the_freesolo_9b_checkpoint() -> None:
    assert OWNED_FP8_MODEL_REPOS == {OWNED_9B: "Freesolo-Co/Qwen3.5-9B-FP8"}


def test_official_fp8_contains_the_active_27b_and_35b_moe_defaults() -> None:
    assert OFFICIAL_FP8_MODEL_REPOS == {
        DENSE_27B: "Qwen/Qwen3.8-27B-FP8",
        MOE_35B: "Qwen/Qwen3.6-35B-A3B-FP8",
    }
    assert set(FP8_SERVE_MODEL_REPOS) == set(base_models())


def test_fp8_serve_model_for_resolves_every_active_base_without_translation() -> None:
    assert fp8_serve_model_for(OWNED_9B) == "Freesolo-Co/Qwen3.5-9B-FP8"
    assert fp8_serve_model_for(DENSE_27B) == "Qwen/Qwen3.8-27B-FP8"
    assert fp8_serve_model_for(MOE_35B) == "Qwen/Qwen3.6-35B-A3B-FP8"
    # the retired 3.6-27B is a DIFFERENT model from the active 3.8-27B and stays unresolvable.
    with pytest.raises(ValueError, match="Allowed models"):
        fp8_serve_model_for("Qwen/Qwen3.6-27B")


def test_fp8_serve_model_for_rejects_unknown_base() -> None:
    with pytest.raises(ValueError, match="Allowed models"):
        fp8_serve_model_for("some/unlisted-model")
