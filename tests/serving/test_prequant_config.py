"""Prequantized FP8 checkpoint ownership and serve-model resolution."""

from __future__ import annotations

import pytest

from flash.serving.src.model_config import base_models
from flash.serving.src.prequant_config import (
    FP8_SERVE_MODEL_REPOS,
    OFFICIAL_FP8_MODEL_REPOS,
    OWNED_FP8_MODEL_REPOS,
    fp8_serve_model_for,
)

OWNED_9B = "Qwen/Qwen3.5-9B"
PENDING_27B = "Qwen/Qwen3.8-27B"
MOE_35B = "Qwen/Qwen3.6-35B-A3B"


def test_owned_fp8_contains_only_the_freesolo_9b_checkpoint() -> None:
    assert OWNED_FP8_MODEL_REPOS == {OWNED_9B: "Freesolo-Co/Qwen3.5-9B-FP8"}


def test_official_fp8_contains_only_the_active_35b_moe_default() -> None:
    assert OFFICIAL_FP8_MODEL_REPOS == {
        MOE_35B: "Qwen/Qwen3.6-35B-A3B-FP8",
    }
    assert PENDING_27B not in FP8_SERVE_MODEL_REPOS
    assert set(FP8_SERVE_MODEL_REPOS) == set(base_models())


def test_fp8_serve_model_for_resolves_every_active_base_without_translation() -> None:
    assert fp8_serve_model_for(OWNED_9B) == "Freesolo-Co/Qwen3.5-9B-FP8"
    assert fp8_serve_model_for(MOE_35B) == "Qwen/Qwen3.6-35B-A3B-FP8"
    for inactive in (PENDING_27B, "Qwen/Qwen3.6-27B"):
        with pytest.raises(ValueError, match="Allowed models"):
            fp8_serve_model_for(inactive)


def test_fp8_serve_model_for_rejects_unknown_base() -> None:
    with pytest.raises(ValueError, match="Allowed models"):
        fp8_serve_model_for("some/unlisted-model")
