"""The torch-free image sft estimate must equal what the VL processor produces.

The control plane quotes image sft without torch by expanding rendered `<|image_pad|>` placeholders
arithmetically. The GPU worker tokenizes the same rows through the real processor. If those two
disagree, the user is billed for one workload and trained on another -- and the disagreement would
be invisible, because the worker only WARNS when its recomputed profile differs
(`sft_train_runner`: "environment processing changed the packaged-dataset token estimate").

So this compares the two directly, on the same rows, and requires the full token sequence to match
rather than just the count: the completion mask is derived from where the prompt ends, so equal
totals with a shifted boundary would still train on the wrong span.

Requires torch + torchvision (the VL processor cannot import without them), which the control plane
deliberately does not have. Skipped where they are absent; the point is that it runs where the real
processor is available.
"""

from __future__ import annotations

import io

import pytest

from flash.engine.profiling.image_tokens import (
    geometry_from_preprocessor_config,
    image_pad_tokens,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"

pytestmark = pytest.mark.wallclock


@pytest.fixture(scope="module")
def processor():
    pytest.importorskip("torch", reason="the VL processor cannot load without torch")
    pytest.importorskip("torchvision", reason="the VL image processor requires torchvision")
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - network/hub availability
        pytest.skip(f"could not load {MODEL_ID} processor: {exc}")


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        # exactly what the control plane uses: no remote code, no processor.
        return transformers.AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    except Exception as exc:  # pragma: no cover - network/hub availability
        pytest.skip(f"could not load {MODEL_ID} tokenizer: {exc}")


@pytest.fixture(scope="module")
def geometry():
    huggingface_hub = pytest.importorskip("huggingface_hub")
    import json

    try:
        path = huggingface_hub.hf_hub_download(MODEL_ID, "preprocessor_config.json")
    except Exception as exc:  # pragma: no cover - network/hub availability
        pytest.skip(f"could not read {MODEL_ID} preprocessor config: {exc}")
    with open(path, encoding="utf-8") as handle:
        return geometry_from_preprocessor_config(json.load(handle))


def _image(width: int, height: int):
    image_module = pytest.importorskip("PIL.Image")
    return image_module.new("RGB", (width, height), (17, 99, 200))


def _ids(value) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


CASES = [
    pytest.param([(640, 480)], "describe this", id="one-image"),
    pytest.param([(56, 56)], "tiny", id="smallest-after-resize"),
    pytest.param([(300, 200), (112, 84)], "compare these two", id="two-images"),
    pytest.param([(64, 64), (1024, 768), (97, 53)], "rank them", id="three-mixed-sizes"),
    pytest.param([(4000, 60)], "wide", id="extreme-aspect-ratio"),
]


@pytest.mark.parametrize(
    ("sizes", "question"), [(p.values[0], p.values[1]) for p in CASES], ids=[p.id for p in CASES]
)
def test_estimated_ids_equal_the_processors_ids(processor, tokenizer, geometry, sizes, question):
    from flash.content.multimodal import IMAGE_PAD_TOKEN
    from flash.engine.profiling.image_tokens import expand_image_pad_runs

    images = [_image(width, height) for width, height in sizes]
    completion = [{"role": "assistant", "content": "the answer"}]

    # what the GPU worker computes: the real processor, with pixels.
    processor_content = [{"type": "image", "image": image} for image in images]
    processor_content.append({"type": "text", "text": question})
    processor_prompt = [{"role": "user", "content": processor_content}]
    expected_full = _ids(
        dict(
            processor.apply_chat_template(
                [*processor_prompt, *completion],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )["input_ids"]
    )

    # what the control plane computes: plain tokenizer plus arithmetic, no pixels decoded.
    pad_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    pad_counts = [image_pad_tokens(width, height, geometry) for width, height in sizes]
    plain_content = [{"type": "image"} for _ in sizes]
    plain_content.append({"type": "text", "text": question})
    rendered = dict(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": plain_content}, *completion],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )["input_ids"]
    estimated_full = expand_image_pad_runs(_ids(rendered), pad_token_id, pad_counts)

    assert estimated_full == expected_full, (
        "the torch-free estimate diverged from the processor: quote and training would disagree"
    )


def test_estimated_completion_mask_matches_the_processors(processor, tokenizer, geometry):
    """Equal totals are not enough: the supervised span has to land on the same tokens."""
    from flash.content.multimodal import IMAGE_PAD_TOKEN
    from flash.engine.profiling.image_tokens import expand_image_pad_runs
    from flash.engine.worker.model.packing import completion_mask_from_ids

    sizes = [(300, 200), (64, 64)]
    images = [_image(width, height) for width, height in sizes]
    question = "what is in these"
    completion = [{"role": "assistant", "content": "two shapes"}]

    def processor_ids(messages, add_generation_prompt):
        return _ids(
            dict(
                processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=False,
                )
            )["input_ids"]
        )

    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    prompt = [{"role": "user", "content": content}]
    expected_mask = completion_mask_from_ids(
        processor_ids(prompt, True),
        processor_ids([*prompt, *completion], False),
    )

    pad_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    pad_counts = [image_pad_tokens(width, height, geometry) for width, height in sizes]
    plain_content = [{"type": "image"} for _ in sizes]
    plain_content.append({"type": "text", "text": question})
    plain_prompt = [{"role": "user", "content": plain_content}]

    def estimated_ids(messages, add_generation_prompt):
        rendered = dict(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        )["input_ids"]
        return expand_image_pad_runs(_ids(rendered), pad_token_id, pad_counts)

    estimated_mask = completion_mask_from_ids(
        estimated_ids(plain_prompt, True),
        estimated_ids([*plain_prompt, *completion], False),
    )

    assert estimated_mask == expected_mask
    assert sum(estimated_mask) > 0, "the supervised span cannot be empty"


def test_pad_run_length_equals_the_processors_grid(processor, geometry):
    """The arithmetic must reproduce the processor's own image_grid_thw, not merely a plausible count."""
    for width, height in [(640, 480), (56, 56), (1024, 768), (97, 53), (4000, 60)]:
        out = dict(
            processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": _image(width, height)},
                            {"type": "text", "text": "q"},
                        ],
                    }
                ],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        grid = out["image_grid_thw"].tolist()[0]
        expected_pads = (grid[1] * grid[2]) // (geometry.merge_size**2)
        assert image_pad_tokens(width, height, geometry) == expected_pads, (
            f"pad run diverged at {width}x{height}"
        )


def test_a_packaged_png_is_measured_without_decoding_its_pixels(geometry, monkeypatch):
    """Dimensions come from the header; a full decode would make large images expensive to quote."""
    from flash.content.multimodal import normalize_image_source
    from flash.engine.profiling.image_tokens import descriptor_pad_tokens

    image_module = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (1024, 768), (5, 5, 5)).save(buffer, format="PNG")
    descriptor = normalize_image_source(buffer.getvalue(), None)

    real_load = image_module.Image.load

    def fail_on_load(self, *args, **kwargs):
        raise AssertionError("pixels were decoded while counting tokens")

    monkeypatch.setattr(image_module.Image, "load", fail_on_load)
    try:
        assert descriptor_pad_tokens([descriptor], None, geometry) == [768]
    finally:
        monkeypatch.setattr(image_module.Image, "load", real_load)
