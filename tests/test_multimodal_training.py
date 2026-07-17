from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
import urllib.parse
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from flash import multimodal as mm


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


def _addrinfo(*addresses):
    return [
        (
            mm.socket.AF_INET6 if ":" in address else mm.socket.AF_INET,
            mm.socket.SOCK_STREAM,
            6,
            "",
            (address, 80, 0, 0) if ":" in address else (address, 80),
        )
        for address in addresses
    ]


def _install_http_response(monkeypatch, *, status=200, data=b"", headers=None):
    class _Response:
        def __init__(self):
            self.status = status
            self.headers = dict(headers or {})

        def read(self, size):
            assert size == mm.MAX_IMAGE_SOURCE_BYTES + 1
            return data

        def close(self):
            return None

    class _Connection:
        def __init__(self, address, port, timeout):
            assert address == "93.184.216.34"
            assert port == 80
            assert timeout == mm._REMOTE_TIMEOUT_SECONDS

        def request(self, method, target, headers):
            assert method == "GET"
            assert target == "/red.png"
            assert headers["Host"] == "images.example"

        def getresponse(self):
            return _Response()

        def close(self):
            return None

    monkeypatch.setattr(mm.http.client, "HTTPConnection", _Connection)


def test_remote_images_are_opt_in_and_bounded(monkeypatch, tmp_path):
    root, _image = _package(tmp_path)
    url = "http://images.example/red.png"
    with pytest.raises(ValueError, match="disabled"):
        mm.normalize_image_source(url, root)

    monkeypatch.setenv(mm.REMOTE_IMAGE_ENV, "1")
    monkeypatch.setattr(
        mm.socket, "getaddrinfo", lambda *args, **kwargs: _addrinfo("93.184.216.34")
    )
    descriptor = mm.normalize_image_source(url, root)
    data = _png_bytes()

    _install_http_response(
        monkeypatch,
        headers={"Content-Length": str(mm.MAX_IMAGE_SOURCE_BYTES + 1)},
    )
    with pytest.raises(ValueError, match="source exceeds"):
        mm.decode_image_descriptor(descriptor, root)

    _install_http_response(
        monkeypatch,
        data=data,
        headers={"Content-Length": str(len(data))},
    )
    image, _encoded, _decoded = mm.decode_image_descriptor(descriptor, root)
    assert image.size == (2, 2)


@pytest.mark.parametrize(
    "addresses",
    [
        ("169.254.169.254",),
        ("127.0.0.1",),
        ("10.0.0.7",),
        ("::1",),
        ("fe80::1",),
        ("93.184.216.34", "10.0.0.7"),
    ],
)
def test_remote_images_reject_any_non_global_dns_answer(monkeypatch, addresses):
    monkeypatch.setenv(mm.REMOTE_IMAGE_ENV, "1")
    monkeypatch.setattr(mm.socket, "getaddrinfo", lambda *args, **kwargs: _addrinfo(*addresses))

    with pytest.raises(ValueError, match="globally routable"):
        mm.normalize_image_source("http://images.example/red.png", None)


def test_remote_images_do_not_follow_redirects_to_private_hosts(monkeypatch):
    monkeypatch.setenv(mm.REMOTE_IMAGE_ENV, "1")
    monkeypatch.setattr(
        mm.socket, "getaddrinfo", lambda *args, **kwargs: _addrinfo("93.184.216.34")
    )
    descriptor = mm.normalize_image_source("http://images.example/red.png", None)
    _install_http_response(
        monkeypatch,
        status=302,
        headers={"Location": "http://169.254.169.254/latest/meta-data/"},
    )

    with pytest.raises(ValueError, match="redirects are not allowed"):
        mm.decode_image_descriptor(descriptor, None)


def test_malformed_blocks_fail_clearly(tmp_path):
    root, _image = _package(tmp_path)
    with pytest.raises(ValueError, match="expected an object"):
        mm.normalize_prompt_images({}, [{"role": "user", "content": ["bad"]}], root)
    with pytest.raises(ValueError, match="missing text"):
        mm.normalize_prompt_images(
            {}, [{"role": "user", "content": [{"type": "text"}]}], root
        )
    with pytest.raises(ValueError, match="missing image source"):
        mm.normalize_prompt_images(
            {}, [{"role": "user", "content": [{"type": "image_url"}]}], root
        )
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


def test_total_decoded_budget_is_checked_before_any_image_load_or_conversion(
    monkeypatch, tmp_path
):
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


def test_sft_collator_decodes_arrow_safe_images_at_batch_time(tmp_path):
    root, _image = _package(tmp_path)
    descriptor = mm.normalize_image_source("dataset/red.png", root)
    observed = {}

    def delegate(rows):
        observed["rows"] = rows
        return {"labels": [[-100, 7]]}

    collator = mm.ArrowSafeVisionCollator(None, 32, root, delegate=delegate)
    result = collator([{"prompt": [], "completion": [], "images": [descriptor]}])

    assert result == {"labels": [[-100, 7]]}
    assert observed["rows"][0]["images"][0].size == (2, 2)


def test_dataset_transform_decodes_only_when_rows_are_requested(monkeypatch):
    calls = []

    def fake_decode(values, package_root):
        calls.append((list(values), package_root))
        return ["decoded"]

    monkeypatch.setattr(mm, "decode_image_descriptors", fake_decode)
    transform = mm.lazy_image_dataset_transform("/package")
    batch = {"prompt": [[{"role": "user", "content": "x"}]], "images": [["descriptor"]]}

    assert calls == []
    transformed = transform(batch)
    assert transformed["images"] == [["decoded"]]
    assert calls == [(["descriptor"], "/package")]


def test_vlm_sft_filter_preserves_completion_mask_and_max_context_behavior():
    rows = [{"id": "kept"}, {"id": "empty"}, {"id": "truncated"}]

    def collator(batch):
        row = batch[0]
        if row["id"] == "kept":
            return {"labels": [[-100, -100, 7, 2]], "attention_mask": [[1, 1, 1, 1]]}
        if row["id"] == "empty":
            return {"labels": [[-100, 2]], "attention_mask": [[1, 1]]}
        return {"labels": [[-100, -100]], "attention_mask": [[1, 1]]}

    kept, dropped, masked, total = mm.filter_vlm_sft_rows(rows, collator, {2})
    assert kept == [{"id": "kept"}]
    assert dropped == 2
    assert masked == 2
    assert total == 4


def test_processor_prompt_count_uses_expanded_vision_tokens(monkeypatch, tmp_path):
    root, _image = _package(tmp_path)
    descriptor = mm.normalize_image_source("dataset/red.png", root)
    observed = {}

    trl_module = ModuleType("trl")
    data_utils_module = ModuleType("trl.data_utils")
    data_utils_module.prepare_multimodal_messages = lambda messages, images: [
        {"role": "user", "content": [{"type": "image", "image": images[0]}]}
    ]
    trl_module.data_utils = data_utils_module
    monkeypatch.setitem(sys.modules, "trl", trl_module)
    monkeypatch.setitem(sys.modules, "trl.data_utils", data_utils_module)

    class _Processor:
        def apply_chat_template(self, **kwargs):
            observed.update(kwargs)
            return {"input_ids": [[1] * 257]}

    count = mm.processor_prompt_token_count(
        _Processor(),
        [{"role": "user", "content": [{"type": "image"}]}],
        [descriptor],
        root,
    )
    assert count == 257
    assert observed["tokenize"] is True
    assert observed["return_dict"] is True


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
            "prompt": [
                {"role": "user", "content": [{"type": "text", "text": "text?"}]}
            ],
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
    completion = [
        {"role": "assistant", "content": [{"type": "text", "text": "red"}]}
    ]
    assert mm.assistant_completion_text(completion) == "red"


def test_multimodal_algorithm_validation_rejects_unsupported_modes():
    mm.validate_multimodal_training("Qwen/Qwen3.5-4B", "sft")
    mm.validate_multimodal_training("Qwen/Qwen3.5-4B", "grpo", multi_turn=False)
    with pytest.raises(ValueError, match="multi-turn"):
        mm.validate_multimodal_training("Qwen/Qwen3.5-4B", "grpo", multi_turn=True)
    with pytest.raises(ValueError, match="opd"):
        mm.validate_multimodal_training("Qwen/Qwen3.5-4B", "opd")
    with pytest.raises(ValueError, match="does not support"):
        mm.validate_multimodal_training("openbmb/MiniCPM5-1B", "sft")


def test_image_opd_preflight_rejects_packaged_dataset_before_allocation(tmp_path):
    root, _image = _package(tmp_path)
    env_file = root / "environment.py"
    env_file.write_text("def load_environment(**kwargs):\n    return None\n")
    (root / "dataset" / "train.jsonl").write_text(
        json.dumps({"input": "color?", "output": "red", "image": "dataset/red.png"}) + "\n"
    )
    spec = SimpleNamespace(
        algorithm="opd",
        environment=SimpleNamespace(id=str(env_file), resolved_sha="", params={}),
    )

    with pytest.raises(ValueError, match="opd does not support image-bearing"):
        mm.preflight_reject_image_opd(spec)

    from flash.runner.lifecycle import _run_job

    with pytest.raises(ValueError, match="opd does not support image-bearing"):
        _run_job(spec)


@pytest.mark.parametrize("background", [False, True])
def test_image_opd_submit_preflight_rejects_before_status_or_provider_paths(
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
                    "records": [
                        {"input": "color?", "output": "red", "image": "dataset/red.png"}
                    ]
                },
            },
            "train": {"epochs": 1, "max_examples": 1},
        }
    )

    with pytest.raises(ValueError, match="opd does not support image-bearing"):
        runner.submit_job(spec, background=background)
    with pytest.raises(FileNotFoundError):
        runner.get_status(spec.run_id)


def test_image_color_example_reward_accepts_only_exact_lowercase_red():
    env_path = Path(__file__).parents[1] / "examples" / "image-color" / "environment.py"
    spec = importlib.util.spec_from_file_location("image_color_environment", env_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    reward = module.ImageColorEnvironment.score_response
    assert reward(None, None, " red ").score == 1.0
    assert reward(None, None, "RED").score == 0.0


def test_cost_specs_price_the_full_context_budget_for_image_and_mixed_rows():
    from flash.cost.spec import runconfig_from_spec
    from flash.spec import JobSpec

    mixed_records = [
        {"input": "text", "output": "answer"},
        {"input": "image", "output": "red", "image": "dataset/red.png"},
    ]
    sft_spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "environment": {"id": "local", "params": {"records": mixed_records}},
            "train": {"epochs": 1, "max_examples": 2, "max_context_tokens": 1536},
        }
    )
    grpo_spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "environment": {"id": "local", "params": {"records": mixed_records}},
            "train": {
                "epochs": 1,
                "max_examples": 2,
                "max_context_tokens": 2048,
                "max_completion_tokens": 64,
            },
        }
    )

    assert runconfig_from_spec(sft_spec).seq_len == 1536
    assert runconfig_from_spec(grpo_spec).seq_len == 2048


def test_catalog_image_capability_does_not_change_public_rows():
    from flash.catalog import public_model_rows, supports_image_training

    assert supports_image_training("Qwen/Qwen3.5-4B")
    assert not supports_image_training("openbmb/MiniCPM5-1B")
    forbidden = {"modalities", "multimodal", "supports_images", "image_training"}
    assert all(not (forbidden & set(row)) for row in public_model_rows())
