from __future__ import annotations

import base64
import io
import json
import urllib.parse
from types import SimpleNamespace

import pytest

from flash import multimodal as mm
from tests._helpers.profile import attach_sft_profile

# one text row and one image row: the shape a ceiling-based quote cannot describe.
_MIXED_RECORDS = [
    {"input": "text", "output": "answer"},
    {"input": "image", "output": "red", "image": "dataset/red.png"},
]


def _png_bytes(color=(255, 0, 0), size=(2, 2)) -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", size, color)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _data_uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _package(tmp_path):
    root = tmp_path / "env"
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    image = dataset / "red.png"
    image.write_bytes(_png_bytes())
    return root, image


def test_normalizes_openai_blocks_and_top_level_images_without_mutating_record(tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    record = {
        "input": "describe the colors",
        "image": "dataset/red.png",
        "images": [_data_uri(data)],
        "reward_metadata": {"expected": "red"},
    }
    original = dict(record)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image", "image": data},
                {"type": "image_url", "image_url": {"url": _data_uri(data), "detail": "low"}},
                {"type": "input_image", "image_url": _data_uri(data)},
                {"type": "image"},
                {"type": "image"},
            ],
        }
    ]

    normalized = mm.normalize_prompt_images(record, messages, root)

    assert record == original
    assert len(normalized.descriptors) == 5
    assert [block["type"] for block in normalized.messages[0]["content"]] == [
        "text",
        "image",
        "image",
        "image",
        "image",
        "image",
    ]
    assert all(isinstance(value, str) for value in normalized.descriptors)
    assert record["reward_metadata"] == {"expected": "red"}


def test_mixed_placeholders_keep_source_order_and_reject_extra_top_level_images(tmp_path):
    root, _image = _package(tmp_path)
    red = _png_bytes((255, 0, 0))
    green = _png_bytes((0, 255, 0))
    blue = _png_bytes((0, 0, 255))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image_url", "image_url": _data_uri(green)},
                {"type": "image"},
            ],
        }
    ]

    normalized = mm.normalize_prompt_images({"images": [red, blue]}, messages, root)
    decoded = mm.decode_image_descriptors(normalized.descriptors, root)
    assert [image.getpixel((0, 0)) for image in decoded] == [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]

    extra_messages = [{"role": "user", "content": [{"type": "image"}]}]
    with pytest.raises(ValueError, match="extra top-level image"):
        mm.normalize_prompt_images({"images": [red, blue]}, extra_messages, root)


def test_supported_source_forms_decode_from_arrow_safe_descriptors(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    root, _image = _package(tmp_path)
    data = _png_bytes()
    pil = image_module.open(io.BytesIO(data))
    sources = [data, bytearray(data), pil, _data_uri(data), "dataset/red.png"]

    descriptors = [mm.normalize_image_source(source, root) for source in sources]

    assert all(isinstance(value, str) for value in descriptors)
    assert all(set(json.loads(value)) == {"kind", "value"} for value in descriptors)
    decoded = mm.decode_image_descriptors(descriptors, root)
    assert [image.size for image in decoded] == [(2, 2)] * len(sources)


def test_image_descriptors_convert_to_base64_data_uris_without_paths(tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    descriptors = [
        mm.normalize_image_source(data, root),
        mm.normalize_image_source("dataset/red.png", root),
    ]

    uris = mm.image_descriptors_to_data_uris(descriptors, root)

    assert len(uris) == 2
    assert all(uri.startswith("data:image/png;base64,") for uri in uris)
    assert all("dataset/red.png" not in uri for uri in uris)
    assert [base64.b64decode(uri.split(",", 1)[1], validate=True) for uri in uris] == [
        data,
        data,
    ]


def test_data_uri_preserves_base64_and_percent_encoded_images(tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    percent_uri = "data:image/png," + urllib.parse.quote_from_bytes(data)

    descriptors = [
        mm.normalize_image_source(_data_uri(data), root),
        mm.normalize_image_source(percent_uri, root),
    ]

    assert [image.size for image in mm.decode_image_descriptors(descriptors, root)] == [(2, 2)] * 2


def test_data_uri_rejects_oversized_header_before_descriptor_storage(tmp_path):
    root, _image = _package(tmp_path)
    header = "data:image/png;name=" + ("x" * mm.MAX_DATA_URI_HEADER_BYTES) + ";base64"
    uri = header + "," + base64.b64encode(_png_bytes()).decode("ascii")

    with pytest.raises(ValueError, match="header exceeds"):
        mm.normalize_image_source(uri, root)


def test_normalization_bounds_aggregate_encoded_descriptors(monkeypatch, tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    descriptor_size = len(mm.normalize_image_source(_data_uri(data), root).encode("utf-8"))
    monkeypatch.setattr(mm, "MAX_TOTAL_IMAGE_DESCRIPTOR_BYTES", descriptor_size * 2 - 1)

    with pytest.raises(ValueError, match="encoded image descriptors total"):
        mm.normalize_prompt_images(
            {"images": [_data_uri(data), _data_uri(data)]},
            [{"role": "user", "content": "compare"}],
            root,
        )


def test_relative_paths_are_confined_to_packaged_dataset_directory(tmp_path):
    root, _image = _package(tmp_path)
    outside = root / "outside.png"
    outside.write_bytes(_png_bytes())
    escaped = root.parent / "escaped.png"
    escaped.write_bytes(_png_bytes())
    (root / "dataset" / "escape-link.png").symlink_to(escaped)

    with pytest.raises(ValueError, match="dataset/"):
        mm.normalize_image_source("outside.png", root)
    with pytest.raises(ValueError, match="dataset/"):
        mm.normalize_image_source("../escaped.png", root)
    with pytest.raises(ValueError, match="dataset/"):
        mm.normalize_image_source("dataset/escape-link.png", root)
    with pytest.raises(ValueError, match="relative"):
        mm.normalize_image_source(str(outside.resolve()), root)
    with pytest.raises(ValueError, match="file://"):
        mm.normalize_image_source(outside.resolve().as_uri(), root)


@pytest.mark.parametrize(
    "url",
    ["http://images.example/red.png", "https://images.example/red.png"],
)
def test_remote_image_urls_are_always_rejected(monkeypatch, url):
    """Flash never fetches a user-supplied URL server-side, and nothing can re-enable it.

    The rejection is unconditional by construction: there is no env flag, no argument, and no
    module attribute left to flip, so a dataset carrying a remote URL fails at normalization
    rather than turning the trainer into an SSRF vector.
    """
    with pytest.raises(ValueError, match="remote image URLs are not supported"):
        mm.normalize_image_source(url, None)

    # the fetch machinery itself is gone, not merely gated
    for removed in ("_read_remote", "_remote_enabled", "_validate_remote_url", "REMOTE_IMAGE_ENV"):
        assert not hasattr(mm, removed)


def test_malformed_blocks_fail_clearly(tmp_path):
    root, _image = _package(tmp_path)
    with pytest.raises(ValueError, match="expected an object"):
        mm.normalize_prompt_images({}, [{"role": "user", "content": ["bad"]}], root)
    with pytest.raises(ValueError, match="missing text"):
        mm.normalize_prompt_images({}, [{"role": "user", "content": [{"type": "text"}]}], root)
    with pytest.raises(ValueError, match="missing image source"):
        mm.normalize_prompt_images({}, [{"role": "user", "content": [{"type": "image_url"}]}], root)
    with pytest.raises(ValueError, match="unsupported content block"):
        mm.normalize_prompt_images(
            {}, [{"role": "user", "content": [{"type": "audio", "audio": "x"}]}], root
        )


def test_count_source_byte_pixel_and_decoded_byte_limits(monkeypatch, tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    messages = [{"role": "user", "content": "what color?"}]

    monkeypatch.setattr(mm, "MAX_IMAGES_PER_EXAMPLE", 1)
    with pytest.raises(ValueError, match="image limit"):
        mm.normalize_prompt_images({"images": [data, data]}, messages, root)

    monkeypatch.setattr(mm, "MAX_IMAGE_SOURCE_BYTES", len(data) - 1)
    with pytest.raises(ValueError, match="source exceeds"):
        mm.normalize_image_source(data, root)

    monkeypatch.setattr(mm, "MAX_IMAGE_SOURCE_BYTES", 3)
    with pytest.raises(ValueError, match="source exceeds"):
        mm.normalize_image_source(_data_uri(b"four"), root)

    monkeypatch.setattr(mm, "MAX_IMAGE_SOURCE_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr(mm, "MAX_IMAGES_PER_EXAMPLE", 8)
    monkeypatch.setattr(mm, "MAX_TOTAL_IMAGE_SOURCE_BYTES", len(data))
    with pytest.raises(ValueError, match="image sources total"):
        mm.normalize_prompt_images({"images": [data, data]}, messages, root)

    monkeypatch.setattr(mm, "MAX_TOTAL_IMAGE_SOURCE_BYTES", 32 * 1024 * 1024)
    monkeypatch.setattr(mm, "MAX_IMAGE_WIDTH", 1)
    with pytest.raises(ValueError, match="dimensions"):
        mm.normalize_image_source(data, root)

    monkeypatch.setattr(mm, "MAX_IMAGE_WIDTH", 8192)
    monkeypatch.setattr(mm, "MAX_IMAGE_PIXELS", 3)
    with pytest.raises(ValueError, match="pixel limit"):
        mm.normalize_image_source(data, root)

    monkeypatch.setattr(mm, "MAX_IMAGE_PIXELS", 40_000_000)
    monkeypatch.setattr(mm, "MAX_TOTAL_DECODED_BYTES", 1)
    descriptor = mm.normalize_image_source(data, root)
    with pytest.raises(ValueError, match="decoded images"):
        mm.decode_image_descriptors([descriptor], root)


def test_total_decoded_budget_is_checked_before_any_image_load_or_conversion(monkeypatch, tmp_path):
    root, _image = _package(tmp_path)
    data = _png_bytes()
    descriptors = [mm.normalize_image_source(data, root), mm.normalize_image_source(data, root)]
    monkeypatch.setattr(mm, "MAX_TOTAL_DECODED_BYTES", 12)
    decode_calls = []

    def fail_decode(_data):
        decode_calls.append(True)
        raise AssertionError("image load and conversion must not start before aggregate preflight")

    monkeypatch.setattr(mm, "_decode_image_bytes", fail_decode)

    with pytest.raises(ValueError, match="decoded images"):
        mm.decode_image_descriptors(descriptors, root)
    assert decode_calls == []


def test_a_row_whose_completion_truncated_away_is_dropped_not_trained_on():
    """A row that keeps no unmasked, non-special target token teaches nothing and must be dropped.

    `sft_max_len` truncates from the right, so a long prompt can leave a row whose completion is
    gone entirely: every position is either prompt (loss_mask 0) or a structural special token.
    Training on it is not merely wasted -- the row still contributes its prompt to the batch, so the
    reported mask ratio and token counts describe a dataset the model never learned from, and a
    dataset that truncates away *every* completion must abort rather than run to completion having
    learned nothing.

    This was `filter_vlm_sft_rows`, which measured the same thing off the vision collator's labels.
    verl pre-tokenizes to `input_ids`/`loss_mask` instead, so the check moved to `has_real_target`
    -- same invariant, different representation, and nothing covered it after the move.
    """
    from flash.engine.worker.sft import has_real_target

    eos = 2
    special = {eos}
    # a real target: one unmasked token that is not a special.
    assert has_real_target([9, 9, 7, eos], [0, 0, 1, 1], special)
    # completion truncated away: every unmasked position is prompt.
    assert not has_real_target([9, 9, 7, eos], [0, 0, 0, 0], special)
    # content-free completion: unmasked, but only the structural end token.
    assert not has_real_target([9, eos], [0, 1], special)
    # the mask -- not the token -- decides: the same ids masked as prompt are not a target.
    assert not has_real_target([7, 7], [0, 0], special)
    # and a special id that is NOT registered special still counts as real content.
    assert has_real_target([9, eos], [0, 1], set())


def test_grpo_rows_retain_arrow_safe_images_and_reward_examples(tmp_path):
    pytest.importorskip("datasets")
    from datasets import Dataset

    from flash.engine.worker.grpo import build_grpo_prompt_dataset

    root, _image = _package(tmp_path)
    descriptor = mm.normalize_image_source("dataset/red.png", root)
    text_example = {"input": "text?", "metadata": {"mixed": "text"}}
    image_example = {"input": "color?", "metadata": {"mixed": 7}}
    prompts = [
        {
            "prompt": [{"role": "user", "content": [{"type": "text", "text": "text?"}]}],
            "images": [],
            "example": text_example,
        },
        {
            "prompt": [{"role": "user", "content": [{"type": "image"}]}],
            "images": [descriptor],
            "example": image_example,
        },
    ]
    rows, examples = build_grpo_prompt_dataset(prompts)
    dataset = Dataset.from_list(rows)

    assert dataset[0]["images"] == []
    assert dataset[1]["images"] == [descriptor]
    assert examples == [text_example, image_example]
    assert examples[rows[1]["example_idx"]] is image_example


def test_single_turn_conversational_completion_keeps_text_reward_semantics():
    completion = [{"role": "assistant", "content": [{"type": "text", "text": "red"}]}]
    assert mm.assistant_completion_text(completion) == "red"


def test_sft_rejects_image_completion_when_prompt_is_text_only():
    from flash.engine.worker.sft import _reject_image_completion

    completion = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "answer"},
                {"type": "image_url", "image_url": {"url": "https://images.example/x.png"}},
            ],
        }
    ]

    with pytest.raises(ValueError, match="image-bearing SFT completions are not supported"):
        _reject_image_completion(completion)


def test_sft_mixed_text_completion_shapes_are_arrow_safe():
    import inspect

    pytest.importorskip("datasets")
    from datasets import Dataset

    from flash.engine import sft_workload

    completions = [
        [{"role": "assistant", "content": "red"}],
        [{"role": "assistant", "content": [{"type": "text", "text": "blue"}]}],
    ]
    rows = [
        {
            "prompt": [{"role": "user", "content": [{"type": "text", "text": "color?"}]}],
            "completion": mm.text_only_prompt_messages(completion),
            "images": [],
        }
        for completion in completions
    ]

    dataset = Dataset.from_list(rows)

    assert [row["completion"][0]["content"] for row in dataset] == ["red", "blue"]
    # the normalizer must actually be applied on the training path, not merely importable: a mixed
    # str/list `content` column makes Arrow infer a struct type and drops one shape at write time.
    assert (
        "completion_messages = text_only_prompt_messages(completion_messages)"
        in inspect.getsource(sft_workload)
    )


def test_image_teacher_prompt_uses_one_media_pad_per_descriptor_in_order():
    messages = [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "name": "viewer",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "middle"},
                {"type": "image"},
                {"type": "text", "text": "after"},
            ],
        },
    ]

    rendered = mm.image_teacher_prompt_messages(messages, 2)

    assert rendered == [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "name": "viewer",
            "content": "before<|media_pad|>middle<|media_pad|>after",
        },
    ]
    assert rendered[1]["content"].count(mm.IMAGE_TEACHER_PLACEHOLDER) == 2
    with pytest.raises(ValueError, match="expected 1 normalized image descriptor"):
        mm.image_teacher_prompt_messages(messages, 1)


def test_text_only_prompt_messages_drops_images_and_preserves_text_order():
    image_module = pytest.importorskip("PIL.Image")
    pil = image_module.new("RGB", (1, 1), "red")
    messages = [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "name": "viewer",
            "content": [
                {"type": "text", "text": "first "},
                {"type": "image", "image": pil},
                {"type": "image_url", "image_url": {"url": "dataset/red.png"}},
                {"type": "text", "text": "second"},
                {"type": "input_image", "image_url": "dataset/blue.png"},
            ],
        },
        {"role": "user", "content": [{"type": "image", "image": "dataset/only.png"}]},
    ]

    stripped = mm.text_only_prompt_messages(messages)

    assert stripped == [
        {"role": "system", "content": "rules"},
        {"role": "user", "name": "viewer", "content": "first second"},
        {"role": "user", "content": ""},
    ]
    assert messages[1]["content"][1]["image"] is pil


def test_multimodal_algorithm_validation_rejects_all_image_opd_after_model_validation():
    mm.validate_multimodal_training("Qwen/Qwen3.5-4B", "sft")
    mm.validate_multimodal_training("Qwen/Qwen3.5-4B", "grpo")
    with pytest.raises(ValueError, match="image-bearing opd is not supported"):
        mm.validate_multimodal_training("Qwen/Qwen3.5-4B", "opd")
    with pytest.raises(ValueError, match="does not support"):
        mm.validate_multimodal_training("meta-llama/Llama-3.2-1B", "opd")


def test_native_single_turn_image_grpo_suppresses_image_pad_generation():
    """An image run must not be able to generate the image-pad token itself. verl generates in its
    own subprocess, so the ban is injected as a rollout shim rather than a generate kwarg."""
    import inspect

    from flash.engine.worker import rl_train

    # the shim's own rendering is covered in test_rl_train.py; what belongs here is the multimodal
    # wiring -- the pad id comes from the PROCESSOR (a text run resolves none) and reaches the shim.
    resolver = inspect.getsource(rl_train._resolve_grpo_inputs)
    assert "image_pad_token_id = resolve_image_pad_token_id(processor, tok)" in resolver

    entry = inspect.getsource(rl_train.run_rl_train)
    assert 'render_image_pad_ban_shim(inp["image_pad_token_id"])' in entry


def test_image_opd_preflight_rejects_packaged_dataset_before_allocation(tmp_path):
    root, _image = _package(tmp_path)
    env_file = root / "environment.py"
    env_file.write_text("def load_environment(**kwargs):\n    return None\n")
    (root / "dataset" / "train.jsonl").write_text(
        json.dumps({"input": "color?", "output": "red", "image": "dataset/red.png"}) + "\n"
    )
    environment = SimpleNamespace(id=str(env_file), resolved_sha="", params={})
    supported = SimpleNamespace(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        environment=environment,
        train=SimpleNamespace(teacher_model="kimi-k3"),
    )
    with pytest.raises(ValueError, match="image-bearing opd is not supported"):
        mm.preflight_validate_image_opd(supported)

    unsupported = SimpleNamespace(
        model="meta-llama/Llama-3.2-1B",
        algorithm="opd",
        environment=environment,
        train=SimpleNamespace(teacher_model="kimi-k3"),
    )
    with pytest.raises(ValueError, match="does not support image-bearing"):
        mm.preflight_validate_image_opd(unsupported)


@pytest.mark.parametrize("record_source", ["inline", "packaged"])
def test_image_opd_preflight_limits_scan_to_max_examples(tmp_path, record_source):
    records = [
        {"input": "text only", "output": "answer"},
        {"input": "image", "output": "red", "image": "dataset/red.png"},
    ]
    if record_source == "inline":
        environment = SimpleNamespace(
            id="local",
            resolved_sha="",
            params={"records": records},
        )
    else:
        root = tmp_path / "env"
        (root / "dataset").mkdir(parents=True)
        env_file = root / "environment.py"
        env_file.write_text("def load_environment(**kwargs):\n    return None\n")
        (root / "dataset" / "train.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
        environment = SimpleNamespace(id=str(env_file), resolved_sha="", params={})
    spec = SimpleNamespace(
        model="meta-llama/Llama-3.2-1B",
        algorithm="opd",
        environment=environment,
        train=SimpleNamespace(max_examples=1),
    )

    mm.preflight_validate_image_opd(spec)


@pytest.mark.parametrize("background", [False, True])
def test_image_opd_submit_preflight_rejects_supported_single_turn_records(
    monkeypatch, tmp_path, background
):
    from flash import runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    def fail(*args, **kwargs):
        raise AssertionError("rejected submit must not mutate warm-start state or reach providers")

    monkeypatch.setattr(runner, "_mark_warmstart_source", fail)
    monkeypatch.setattr(runner, "_run_job", fail)
    monkeypatch.setattr(runner, "_run_job_background", fail)
    monkeypatch.setattr(runner.threading, "Thread", fail)

    spec = JobSpec.from_dict(
        {
            "run_id": f"image-opd-{'background' if background else 'sync'}",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "opd",
            "environment": {
                "id": "local",
                "params": {
                    "records": [{"input": "color?", "output": "red", "image": "dataset/red.png"}]
                },
            },
            "train": {"epochs": 1, "max_examples": 1, "teacher_model": "kimi-k3"},
        }
    )

    with pytest.raises(ValueError, match="image-bearing opd is not supported"):
        runner.submit_job(spec, background=background)
    with pytest.raises(FileNotFoundError):
        runner.get_status(spec.run_id)


def test_image_opd_submit_preflight_preserves_unsupported_model_precedence(monkeypatch, tmp_path):
    from flash import runner
    from flash.spec import JobSpec

    algorithm = "opd"
    model = "meta-llama/Llama-3.2-1B"
    message = "does not support image-bearing"

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))

    def fail(*args, **kwargs):
        raise AssertionError("rejected submit must not mutate warm-start state or reach providers")

    monkeypatch.setattr(runner, "_mark_warmstart_source", fail)
    monkeypatch.setattr(runner, "_run_job", fail)
    monkeypatch.setattr(runner, "_run_job_background", fail)
    monkeypatch.setattr(runner.threading, "Thread", fail)
    params = {"records": [{"input": "color?", "output": "red", "image": "dataset/red.png"}]}
    spec = JobSpec.from_dict(
        {
            "run_id": f"image-{algorithm}-reject-{model.rsplit('/', 1)[-1]}",
            "model": model,
            "algorithm": algorithm,
            "environment": {"id": "local", "params": params},
            "train": {"epochs": 1, "max_examples": 1, "teacher_model": "kimi-k3"},
        }
    )
    prepared = runner.PreparedJob(public_spec=spec, worker_spec=spec, estimated_cost_usd=0.0)

    with pytest.raises(ValueError, match=message):
        runner.submit_job(spec, prepared_job=prepared)
    with pytest.raises(FileNotFoundError):
        runner.get_status(spec.run_id)


def test_grpo_prices_the_full_context_budget_for_image_and_mixed_rows():
    """An image prompt occupies its context budget, so grpo prices the budget, not the text length.

    sft used to be asserted here against the same ceiling. It no longer prices from the ceiling at
    all: the workload profile measures the tokens the rows actually produce, which for a mixed
    dataset is the whole point of measuring. The companion test below holds the multimodal half of
    that -- an image sft run cannot be quoted from an assumed context.
    """
    from flash.cost.spec import runconfig_from_spec
    from flash.spec import JobSpec

    grpo_spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "environment": {"id": "local", "params": {"records": _MIXED_RECORDS}},
            "train": {
                "epochs": 1,
                "max_examples": 2,
                "max_context_tokens": 2048,
                "max_completion_tokens": 64,
            },
        }
    )

    assert runconfig_from_spec(grpo_spec).seq_len == 2048


def test_image_sft_cannot_be_priced_from_an_assumed_context():
    """The failure mode this replaces: quoting a mixed dataset off max_context_tokens.

    Image rows and text rows produce wildly different token counts, so a ceiling-based quote for a
    mixed dataset is a guess wearing an exact number. Pricing now requires the profile that
    tokenized these exact rows, and without one the quote fails rather than defaulting.
    """
    from flash.cost.spec import runconfig_from_spec
    from flash.spec import JobSpec
    from flash.workload_profile import WorkloadProfileMismatch

    sft_spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "environment": {"id": "local", "params": {"records": _MIXED_RECORDS}},
            "train": {"epochs": 1, "max_examples": 2, "max_context_tokens": 1536},
        }
    )

    with pytest.raises(WorkloadProfileMismatch):
        runconfig_from_spec(sft_spec)

    # with the measurement attached, the priced length is the profile's, never the config ceiling.
    priced = runconfig_from_spec(attach_sft_profile(sft_spec))
    assert priced.seq_len == attach_sft_profile(sft_spec).workload_profile["max_length"]


def test_catalog_image_capability_does_not_change_public_rows():
    from flash.catalog import public_model_rows, supports_image_training

    assert supports_image_training("Qwen/Qwen3.5-4B")
    assert supports_image_training("Qwen/Qwen3.6-27B")
    assert not supports_image_training("meta-llama/Llama-3.2-1B")
    forbidden = {"modalities", "multimodal", "supports_images", "image_training"}
    assert all(not (forbidden & set(row)) for row in public_model_rows())
