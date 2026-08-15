"""the torch-free image estimate must match the real processor token boundary."""

from __future__ import annotations

import io
import json

import pytest

from flash.content.multimodal import normalize_image_source
from flash.engine.profiling.image_tokens import geometry_from_preprocessor_config
from flash.engine.profiling.sft_image_rows import (
    estimate_sft_image_row,
    process_sft_image_row,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"

pytestmark = pytest.mark.wallclock


@pytest.fixture(scope="module")
def image_stack():
    pytest.importorskip("torch", reason="the VL processor cannot load without torch")
    pytest.importorskip("torchvision", reason="the VL image processor requires torchvision")
    transformers = pytest.importorskip("transformers")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    try:
        processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
        path = huggingface_hub.hf_hub_download(MODEL_ID, "preprocessor_config.json")
        with open(path, encoding="utf-8") as handle:
            geometry = geometry_from_preprocessor_config(json.load(handle))
    except Exception as exc:  # pragma: no cover - network/hub availability
        pytest.skip(f"could not load {MODEL_ID} image stack: {exc}")
    return processor, tokenizer, geometry


def _descriptor(width: int, height: int) -> str:
    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (width, height), (17, 99, 200)).save(buffer, format="PNG")
    return normalize_image_source(buffer.getvalue(), None)


@pytest.mark.parametrize(
    ("sizes", "question"),
    [
        pytest.param([(56, 56)], "tiny", id="minimum-budget-single-image"),
        pytest.param([(300, 200), (112, 84)], "compare these", id="mixed-size-multi-image"),
        pytest.param([(4000, 60)], "wide", id="extreme-valid-aspect-ratio"),
    ],
)
def test_estimator_matches_the_processor_at_the_production_seam(image_stack, sizes, question):
    processor, tokenizer, geometry = image_stack
    descriptors = [_descriptor(width, height) for width, height in sizes]
    prompt = [
        {
            "role": "user",
            "content": [
                *({"type": "image"} for _ in descriptors),
                {"type": "text", "text": question},
            ],
        }
    ]
    completion = [{"role": "assistant", "content": "the answer"}]
    common = {"package_root": None, "max_length": 8192, "thinking": False}

    estimated = estimate_sft_image_row(
        tokenizer,
        prompt,
        completion,
        descriptors,
        geometry=geometry,
        validation_cache={},
        **common,
    )
    processed = process_sft_image_row(
        processor,
        prompt,
        completion,
        descriptors,
        **common,
    )

    assert estimated[:2] == processed[:2]
    assert estimated[3] == processed[3]
    assert estimated[2] == b""
    assert processed[2]
    assert sum(estimated[1]) > 0
