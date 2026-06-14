"""Guardrails on the model catalog and the package-wide default model.

These keep the curated catalog internally consistent and make sure the out-of-the-box
default an average developer hits is a real, supported, non-experimental model.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoslm.catalog import DEFAULT_MODEL, MODELS, get_model
from autoslm.flash.gpus import SUPPORTED, canonical_gpu


def test_recommended_gpu_is_supported():
    # Every catalog entry must recommend a GPU AutoSLM actually manages (RTX 4090 / 5090).
    for model_id, info in MODELS.items():
        assert canonical_gpu(info.recommended_gpu) in SUPPORTED, (
            f"{model_id} recommends unsupported GPU {info.recommended_gpu!r}"
        )


def test_default_model_is_supported_and_non_experimental():
    info = get_model(DEFAULT_MODEL)
    assert not info.experimental, "default model must not be experimental"
    # The default is GRPO+SFT capable so both algorithms work out of the box.
    assert "grpo" in info.algos
    assert "sft" in info.algos


def test_recipe_and_jobspec_defaults_match_catalog_default():
    from autoslm.engine.recipe import RECIPE
    from autoslm.worker_spec import JobSpec

    # When BENCH_HF_MODEL isn't overriding it, the recipe + JobSpec default to the catalog default.
    if not os.environ.get("BENCH_HF_MODEL"):
        assert RECIPE.hf_model_id == DEFAULT_MODEL
    assert JobSpec().model == DEFAULT_MODEL
    assert not get_model(JobSpec().model).experimental


def test_thinking_capability_values_are_valid():
    # The config validator branches on these exact values; "unknown" is reserved for
    # open-model-policy entries and must not appear in the curated catalog.
    for model_id, info in MODELS.items():
        assert info.thinking in ("none", "hybrid", "always"), (model_id, info.thinking)


def test_default_model_is_non_thinking():
    # Default runs must never require the thinking flag (out-of-the-box behavior is
    # byte-identical to the pre-thinking recipe).
    assert get_model(DEFAULT_MODEL).thinking == "none"


def test_default_model_is_a_dense_text_model():
    # Guard against regressing the default back to a multimodal / novel-arch model: the
    # default should be the proven dense instruction model.
    assert DEFAULT_MODEL == "Qwen/Qwen3-4B-Instruct-2507"


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
            passed += 1
        except Exception:
            print("FAIL", fn.__name__)
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
