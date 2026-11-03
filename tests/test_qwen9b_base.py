"""Keep pretrained 9B weights and adapter identities separate from post-trained 9B."""

import pytest

from flash.core.catalog import DEFAULT_MODEL, get_model, supports_image_training
from flash.engine.plan.vram import sft_chunked_nll_enabled
from flash.schema import spec_from_dict
from flash.serve.deployment.profiles import get_profile
from flash.serving.src.engine.model_config import (
    engine_overrides_for,
    gpu_for,
    serve_model_for,
    tokenizer_model_for,
    tool_parser_for,
)

BASE = "Qwen/Qwen3.5-9B-Base"
POST_TRAINED = "Qwen/Qwen3.5-9B"
REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
FP8_REVISION = "b1a2cf5f5e7bb4ad786e206f834569be3a3d8aa9"


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_pretrained_model_is_admitted_without_aliasing(algorithm):
    spec = spec_from_dict(
        {"model": BASE, "algorithm": algorithm, "environment": {"id": "org/project/env"}},
        run_id="base-9b",
    )
    assert spec.model == BASE
    info = get_model(spec.model)
    assert info.managed_revision == REVISION
    assert info.quant == "bf16"
    assert info.lora_language_prefix == "model.language_model"
    assert sft_chunked_nll_enabled(BASE)
    assert supports_image_training(BASE)
    assert DEFAULT_MODEL == POST_TRAINED


def test_pretrained_serving_uses_its_own_weights_tokenizer_and_revision():
    overrides = engine_overrides_for(BASE)
    assert serve_model_for(BASE) == overrides["serve_model_id"] == "Freesolo-Co/Qwen3.5-9B-Base-FP8"
    assert tokenizer_model_for(BASE) == BASE
    assert overrides["model_revision"] == FP8_REVISION
    for key in ("tokenizer_revision", "processor_revision"):
        assert overrides[key] == REVISION
    assert overrides["quantization"] is None
    assert get_model(BASE).serving.serve_model_id == BASE
    profile = get_profile(BASE)
    assert profile.served_model == profile.tokenizer_model == BASE
    assert profile.served_model_revision == REVISION
    assert profile.modal_gpu_request == "H100!"
    assert profile.max_model_len == 262144
    assert profile.max_num_seqs == get_model(BASE).serving.max_num_seqs == 32
    # Hosted FP8 serving uses L40S; the separate customer-owned BF16 profile stays on H100.
    assert gpu_for(BASE) == gpu_for(POST_TRAINED) == "L40S"
    assert overrides["max_num_seqs"] == 64
    assert "max_num_batched_tokens" not in overrides
    assert tool_parser_for(BASE) == "qwen3_coder"
    # Existing instruction-tuned serving still uses its original FP8 checkpoint.
    assert serve_model_for(POST_TRAINED) == "Freesolo-Co/Qwen3.5-9B-FP8"
