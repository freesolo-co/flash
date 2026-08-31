from __future__ import annotations

import asyncio
import base64
import io
import types
from collections import OrderedDict
from typing import Any
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serving.src.engine.lora_engine import _LoraEngineImpl
from flash.serving.src.engine.support import _num_prompt_tokens
from flash.serving.src.http.router import AdapterRouter
from flash.serving.src.http.router import build_offline_serving_app as build_serving_app
from flash.serving.src.http.router import build_serving_app as build_durable_serving_app
from flash.serving.src.io import multimodal
from flash.serving.src.io.multimodal import (
    MultimodalRequestError,
    has_image_blocks,
    normalize_chat_messages,
    prepare_multimodal_request,
)
from flash.serving.src.io.schemas import AdapterRecord, GenerateRequest, internal_adapter_payload
from flash.serving.src.store.registry import AdapterRegistry
from tests.serving.conftest import RecordingUsageStore

QWEN = "Qwen/Qwen3.5-9B"
QWEN_35B = "Qwen/Qwen3.6-35B-A3B"
RUN_ID = "flash-1234567890-abcdef12"
SHA = "a" * 40
REVISION_ID = f"{RUN_ID}/step-20"


def _data_uri(
    image_format: str = "PNG",
    *,
    size: tuple[int, int] = (2, 2),
    color: tuple[int, int, int] = (10, 20, 30),
    frames: int = 1,
) -> str:
    buffer = io.BytesIO()
    image = Image.new("RGB", size, color)
    mime = {"JPEG": "jpeg", "PNG": "png", "WEBP": "webp"}[image_format]
    if frames == 1:
        image.save(buffer, format=image_format)
    else:
        second = Image.new("RGB", size, (30, 20, 10))
        image.save(
            buffer,
            format=image_format,
            save_all=True,
            append_images=[second],
            duration=10,
            loop=0,
        )
    return f"data:image/{mime};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _messages(source: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": source or _data_uri()}},
                {"type": "input_text", "text": "after"},
            ],
        }
    ]


def _tool_history_messages(*, with_image: bool) -> list[dict[str, Any]]:
    user_content: Any = "calculate what is shown"
    if with_image:
        user_content = [
            {"type": "image_url", "image_url": {"url": _data_uri()}},
            {"type": "text", "text": "calculate what is shown"},
        ]
    return [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "i should call the calculator",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "4",
            "tool_call_id": "call_1",
            "name": "calculator",
        },
    ]


def _validate(messages: Any, *, supports_images: bool = True, image_limit: int | None = 4) -> None:
    normalize_chat_messages(
        messages,
        supports_images=supports_images,
        image_limit=image_limit,
    )


def test_detector_is_cheap_and_specific() -> None:
    assert has_image_blocks(_messages()) is True
    assert has_image_blocks([{"role": "user", "content": "plain"}]) is False
    assert has_image_blocks("not messages") is False


def test_prepare_preserves_content_order_and_materializes_rgb_pil() -> None:
    template_messages, images = prepare_multimodal_request(_messages(), image_limit=4)
    try:
        assert template_messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image"},
                    {"type": "text", "text": "after"},
                ],
            }
        ]
        assert len(images) == 1
        assert images[0].mode == "RGB"
        assert images[0].size == (2, 2)
        assert images[0].getpixel((0, 0)) == (10, 20, 30)
    finally:
        for image in images:
            image.close()


def test_prepare_applies_jpeg_exif_orientation_before_rgb_conversion() -> None:
    source = Image.new("RGB", (3, 2))
    source.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
    )
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", quality=100, subsampling=0, exif=exif)
    data = buffer.getvalue()
    uri = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"

    with Image.open(io.BytesIO(data)) as encoded:
        expected = ImageOps.exif_transpose(encoded).convert("RGB")
    _, images = prepare_multimodal_request(_messages(uri), image_limit=4)
    try:
        assert images[0].size == (2, 3)
        assert images[0].tobytes() == expected.tobytes()
    finally:
        expected.close()
        source.close()
        for image in images:
            image.close()


def test_transformers_image_url_field_is_accepted() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "url": _data_uri()}],
        }
    ]
    template_messages, images = prepare_multimodal_request(messages, image_limit=4)
    try:
        assert template_messages == [{"role": "user", "content": [{"type": "image"}]}]
        assert len(images) == 1
        assert images[0].size == (2, 2)
    finally:
        for image in images:
            image.close()


def test_developer_role_is_normalized_to_system_for_image_chats() -> None:
    messages = [
        {"role": "developer", "content": "be terse"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_uri()}}]},
    ]
    template_messages, images = prepare_multimodal_request(messages, image_limit=4)
    try:
        assert template_messages[0] == {"role": "system", "content": "be terse"}
        assert template_messages[1] == {"role": "user", "content": [{"type": "image"}]}
        assert len(images) == 1
    finally:
        for image in images:
            image.close()


def test_prepare_preserves_tool_history_fields() -> None:
    messages = _tool_history_messages(with_image=True)
    template_messages, images = prepare_multimodal_request(messages, image_limit=4)
    try:
        assert template_messages[0] == {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "calculate what is shown"},
            ],
        }
        assert template_messages[1] == messages[1]
        assert template_messages[2] == messages[2]
    finally:
        for image in images:
            image.close()


def test_real_qwen_processor_renders_preserved_tool_history() -> None:
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(QWEN, local_files_only=True)
    except OSError:
        pytest.skip("Qwen processor is not cached locally")
    except ImportError as exc:
        # The Qwen3-VL processor pulls in a video processor that requires torch/torchvision. The
        # flash offline test env installs neither (torch lives in the `gpu` extra), so without this
        # the test FAILS there rather than skipping -- a red required check for a missing optional
        # dependency, not a defect in the code under test.
        pytest.skip(f"Qwen processor needs torch/torchvision: {exc}")
    template_messages, images = prepare_multimodal_request(
        _tool_history_messages(with_image=True),
        image_limit=4,
    )
    try:
        rendered = processor.apply_chat_template(
            template_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "<think>\ni should call the calculator\n</think>" in rendered
        assert "<tool_call>\n<function=calculator>" in rendered
        assert "<parameter=expression>\n2+2\n</parameter>" in rendered
        assert "<tool_response>\n4\n</tool_response>" in rendered
    finally:
        for image in images:
            image.close()


@pytest.mark.parametrize("role", ["system", "assistant", "tool"])
def test_images_are_user_only(role: str) -> None:
    messages = _messages()
    messages[0]["role"] = role
    with pytest.raises(MultimodalRequestError, match="only in user"):
        _validate(messages)


@pytest.mark.parametrize(
    "source",
    [
        "http://example.com/a.png",
        "https://example.com/a.png",
        "file:///tmp/a.png",
        "ftp://example.com/a.png",
        "s3://bucket/a.png",
    ],
)
def test_non_data_sources_are_rejected_without_fetch(source: str) -> None:
    with pytest.raises(MultimodalRequestError, match="data URI"):
        _validate(_messages(source))


def test_transformers_image_url_field_rejects_remote_fetch() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "url": "https://example.com/x.png"}],
        }
    ]
    with pytest.raises(MultimodalRequestError, match="data URI"):
        _validate(messages)


def test_non_base64_data_uri_is_rejected() -> None:
    with pytest.raises(MultimodalRequestError, match="base64"):
        _validate(_messages("data:image/png,not-base64"))


@pytest.mark.parametrize("mime", ["gif", "svg+xml", "bmp"])
def test_unsupported_mime_is_rejected(mime: str) -> None:
    encoded = base64.b64encode(b"not an image").decode()
    with pytest.raises(MultimodalRequestError, match="unsupported MIME"):
        _validate(_messages(f"data:image/{mime};base64,{encoded}"))


def test_mime_and_actual_format_must_match() -> None:
    source = _data_uri("JPEG").replace("data:image/jpeg", "data:image/png", 1)
    with pytest.raises(MultimodalRequestError, match="does not match"):
        _validate(_messages(source))


def test_invalid_base64_is_rejected_strictly() -> None:
    with pytest.raises(MultimodalRequestError, match="invalid base64"):
        _validate(_messages("data:image/png;base64,%%%%"))


def test_animated_image_is_rejected() -> None:
    source = _data_uri("WEBP", frames=2)
    with pytest.raises(MultimodalRequestError, match="single-frame"):
        _validate(_messages(source))


def test_truncated_image_is_rejected() -> None:
    source = _data_uri("PNG")
    prefix, encoded = source.split(",", 1)
    raw = base64.b64decode(encoded)
    truncated = f"{prefix},{base64.b64encode(raw[:-8]).decode()}"
    with pytest.raises(MultimodalRequestError, match="invalid or truncated"):
        _validate(_messages(truncated))


def test_decompression_bomb_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(MultimodalRequestError, match="decompression bomb"):
        _validate(_messages(_data_uri(size=(2, 2))))


def test_per_image_source_string_limit_is_checked_before_decode(monkeypatch) -> None:
    monkeypatch.setattr(multimodal, "_MAX_SOURCE_CHARS", 20)
    with pytest.raises(MultimodalRequestError, match="encoded-size"):
        _validate(_messages())


def test_per_image_compressed_byte_limit(monkeypatch) -> None:
    monkeypatch.setattr(multimodal, "_MAX_COMPRESSED_BYTES", 1)
    with pytest.raises(MultimodalRequestError, match="compressed-byte"):
        _validate(_messages())


def test_total_compressed_byte_limit(monkeypatch) -> None:
    monkeypatch.setattr(multimodal, "_MAX_TOTAL_COMPRESSED_BYTES", 1)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": _data_uri()},
                {"type": "input_image", "input_image": _data_uri()},
            ],
        }
    ]
    with pytest.raises(MultimodalRequestError, match="total compressed"):
        _validate(messages)


def test_dimension_limit(monkeypatch) -> None:
    monkeypatch.setattr(multimodal, "_MAX_DIMENSION", 1)
    with pytest.raises(MultimodalRequestError, match="dimension"):
        _validate(_messages())


def test_per_image_pixel_limit(monkeypatch) -> None:
    monkeypatch.setattr(multimodal, "_MAX_PIXELS", 3)
    with pytest.raises(MultimodalRequestError, match="pixel limit"):
        _validate(_messages())


def test_no_cumulative_pixel_cap_exists() -> None:
    # a mode-blind sum of pixel counts cannot see decoded mode or ordering, so any cumulative
    # pixel total is wrong in one direction. the cumulative decoded-memory guard bounds it exactly.
    assert not hasattr(multimodal, "_MAX_TOTAL_PIXELS")


def test_total_decoded_memory_limit(monkeypatch) -> None:
    monkeypatch.setattr(multimodal, "_MAX_TOTAL_DECODED_BYTES", 11)
    with pytest.raises(MultimodalRequestError, match="decoded-memory"):
        _validate(_messages())


def test_multi_image_decoded_memory_counts_resident_not_summed_peaks(monkeypatch) -> None:
    # two 2x2 rgb images: earlier images are resident only as their 3-byte rgb copies, so the
    # worst-case peak is 3*4 + 9*4 = 48 (resident copies + the current image's mode+6 peak), not
    # the summed per-image peaks (9*4 + 9*4 = 72). a 50-byte budget must accept them; the
    # previous summed-peak accounting would have falsely rejected.
    monkeypatch.setattr(multimodal, "_MAX_TOTAL_DECODED_BYTES", 50)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_uri()}},
                {"type": "image_url", "image_url": {"url": _data_uri()}},
            ],
        }
    ]
    _template_messages, images = prepare_multimodal_request(messages, image_limit=4)
    try:
        assert len(images) == 2
    finally:
        for image in images:
            image.close()


def test_more_than_four_images_are_rejected() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": _data_uri()} for _ in range(5)],
        }
    ]
    with pytest.raises(MultimodalRequestError, match="at most 4"):
        _validate(messages)


@pytest.mark.parametrize("block_type", ["video", "audio", "file", "gif", "svg", None])
def test_unknown_content_blocks_are_rejected(block_type: str | None) -> None:
    messages = [{"role": "user", "content": [{"type": block_type, "text": "x"}]}]
    with pytest.raises(MultimodalRequestError, match="unsupported type"):
        _validate(messages)


@pytest.mark.parametrize(
    ("messages", "detail"),
    [
        (["not an object"], "message 0"),
        ([{"role": "user", "content": ["not an object"]}], "content block 0"),
        ([{"role": "user", "content": None}], "content must"),
        ([{"role": "unknown", "content": "x"}], "role must"),
        ([{"role": "user", "content": [{"type": "text", "text": 1}]}], "text must"),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": _data_uri(),
                            "image_url": {"url": _data_uri()},
                        }
                    ],
                }
            ],
            "exactly one",
        ),
    ],
)
def test_malformed_messages_and_blocks_are_rejected(messages: Any, detail: str) -> None:
    with pytest.raises(MultimodalRequestError, match=detail):
        _validate(messages)


def test_text_only_model_rejects_images_before_decode() -> None:
    with pytest.raises(MultimodalRequestError, match="does not support"):
        _validate(
            _messages("data:image/png;base64,not-valid"), supports_images=False, image_limit=None
        )


class _Processor:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append((messages, kwargs))
        return "rendered multimodal prompt"


class _Engine:
    def __init__(self) -> None:
        self.prompt_inputs: list[dict[str, Any]] = []

    async def generate(self, prompt_input: dict[str, Any], *_args: Any, **_kwargs: Any):
        image_data = prompt_input["multi_modal_data"]["image"]
        assert isinstance(image_data, Image.Image)
        assert image_data.mode == "RGB"
        assert image_data.getpixel((0, 0)) == (10, 20, 30)
        self.prompt_inputs.append(prompt_input)
        yield types.SimpleNamespace(
            outputs=[
                types.SimpleNamespace(index=0, text="ok", finish_reason="stop", token_ids=[7, 8])
            ],
            prompt_token_ids=list(range(17)),
            num_cached_tokens=0,
        )


def _base_record() -> AdapterRecord:
    return AdapterRecord(
        adapter_id=QWEN,
        repo_id=QWEN,
        base_model=QWEN,
        serve_base_model=True,
        thinking=False,
        status="ready",
    )


def _engine_impl(*, processor: Any = None) -> _LoraEngineImpl:
    engine = object.__new__(_LoraEngineImpl)
    engine.base_model = QWEN
    engine.processor = processor
    engine.tokenizer = types.SimpleNamespace()
    engine.reasoning_parser = "qwen3"
    engine.registry = AdapterRegistry()
    engine.registry.hydrate([_base_record()])
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()
    engine._prompt_token_cache = OrderedDict()
    engine._prompt_cache_size = 8
    engine.engine = _Engine()
    engine._self_heal_if_dead = lambda _reason: None
    return engine


def test_prompt_token_accounting_uses_vllm_expanded_prompt_token_ids() -> None:
    from vllm.outputs import RequestOutput

    output = RequestOutput(
        request_id="request-1",
        prompt=None,
        prompt_token_ids=list(range(19)),
        prompt_logprobs=None,
        outputs=[],
        finished=True,
        num_cached_tokens=0,
    )
    assert not hasattr(output, "num_prompt_tokens")
    assert _num_prompt_tokens(output) == 19


def test_prompt_token_accounting_keeps_forward_compat_fallback() -> None:
    output = types.SimpleNamespace(prompt_token_ids=None, num_prompt_tokens=19)
    assert _num_prompt_tokens(output) == 19
    with pytest.raises(RuntimeError, match="did not report"):
        _num_prompt_tokens(types.SimpleNamespace())


def test_engine_processor_render_and_vllm_multimodal_input_shape() -> None:
    processor = _Processor()
    engine = _engine_impl(processor=processor)
    result = asyncio.run(
        engine._generate(
            {
                "adapter_id": QWEN,
                "messages": _messages(),
                "chat_template_kwargs": {"enable_thinking": True, "custom": "value"},
            },
            internal_adapter_payload(_base_record()),
        )
    )
    assert result["prompt_tokens"] == 17
    assert result["completion_tokens"] == 2
    assert len(processor.calls) == 1
    template_messages, kwargs = processor.calls[0]
    assert template_messages[0]["content"][1] == {"type": "image"}
    assert kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "custom": "value",
        "enable_thinking": True,
        "preserve_thinking": False,
    }
    prompt_input = engine.engine.prompt_inputs[0]
    assert prompt_input["prompt"] == "rendered multimodal prompt"
    assert set(prompt_input) == {"prompt", "multi_modal_data"}
    assert "prompt_token_ids" not in prompt_input
    assert engine._prompt_token_cache == OrderedDict()


def test_image_requests_bypass_text_prompt_cache() -> None:
    engine = _engine_impl(processor=_Processor())

    async def _fail(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise AssertionError("text prompt path must be bypassed")

    engine._prompt_input = _fail
    payload = GenerateRequest(adapter_id=QWEN, messages=_messages())
    prompt_input = asyncio.run(engine._prepare_prompt_input(payload, thinking_default=False))
    try:
        assert prompt_input["prompt"] == "rendered multimodal prompt"
    finally:
        engine._close_prompt_images(prompt_input)


def test_image_capable_engine_without_processor_rejects_request() -> None:
    engine = _engine_impl(processor=None)
    payload = GenerateRequest(adapter_id=QWEN, messages=_messages())
    with pytest.raises(MultimodalRequestError, match="initialized processor"):
        asyncio.run(engine._prepare_prompt_input(payload, thinking_default=False))


def test_image_prompt_failure_closes_decoded_images() -> None:
    """Every raise after the images are decoded must still close them, or the request leaks the
    decoded RGB buffers. Covers both the chat-template-kwargs resolve and the render itself."""

    class _RaisingProcessor:
        def apply_chat_template(self, *_a: Any, **_k: Any) -> str:
            raise RuntimeError("render exploded")

    class _TrackedImage:
        """Records close() so the test asserts closure directly rather than probing PIL internals."""

        def __init__(self, image: Image.Image) -> None:
            self._image = image
            self.closed = False

        def close(self) -> None:
            self.closed = True
            self._image.close()

    tracked: list[_TrackedImage] = []
    real_prepare = multimodal.prepare_multimodal_request

    def _tracking_prepare(*args: Any, **kwargs: Any):
        template_messages, images = real_prepare(*args, **kwargs)
        wrapped = [_TrackedImage(image) for image in images]
        tracked.extend(wrapped)
        return template_messages, wrapped

    payload = GenerateRequest(adapter_id=QWEN, messages=_messages())
    # _prepare_prompt_input imports prepare_multimodal_request lazily FROM src.multimodal on every
    # call, so patching the source module is what the engine actually resolves.
    with mock.patch.object(multimodal, "prepare_multimodal_request", _tracking_prepare):
        # the render itself raises
        engine = _engine_impl(processor=_RaisingProcessor())
        with pytest.raises(RuntimeError, match="render exploded"):
            asyncio.run(engine._prepare_prompt_input(payload, thinking_default=False))
        # resolving the effective chat_template_kwargs raises (no adapter thinking default)
        engine = _engine_impl(processor=_Processor())
        with pytest.raises(ValueError, match="thinking default"):
            asyncio.run(engine._prepare_prompt_input(payload, thinking_default=None))

    assert len(tracked) == 2, f"expected one decoded image per attempt, got {len(tracked)}"
    assert all(image.closed for image in tracked)


class _OverflowEngine(_Engine):
    async def generate(self, *_args: Any, **_kwargs: Any):
        raise ValueError("context length exceeded")
        yield None


def test_stream_validation_and_first_engine_error_happen_before_ready() -> None:
    engine = _engine_impl(processor=_Processor())
    engine.engine = _OverflowEngine()

    async def first_event() -> Any:
        return await anext(
            engine._stream_generate(
                {"adapter_id": QWEN, "messages": _messages()},
                internal_adapter_payload(_base_record()),
            )
        )

    with pytest.raises(ValueError, match="context length"):
        asyncio.run(first_event())


def _revision(base_model: str) -> AdapterRecord:
    values = {
        "adapter_id": REVISION_ID,
        "repo_id": "org/run",
        "org_id": "org-1",
        "base_model": base_model,
        "subfolder": "checkpoints/step-20",
        "checkpoint": REVISION_ID,
        "status": "ready",
        "thinking": False,
        "run_id": RUN_ID,
        "checkpoint_step": 20,
        "artifact_revision": SHA,
        "artifact_digest": "b" * 64,
        "lora_rank": 16,
    }
    values["artifact_fingerprint"] = immutable_binding_fingerprint(values)
    return AdapterRecord.model_validate(values)


class _Pool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.payloads: list[Any] = []

    async def generate(
        self,
        base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        del expected_checkpoint
        self.calls.append((base_model, payload.adapter_id, record.adapter_id))
        self.payloads.append(payload)
        return {
            "adapter_id": payload.adapter_id,
            # a real engine attests the adapter it actually resolved, which for a revision is the
            # resolved record rather than whatever id the caller asked with.
            **({"lora_request_adapter": record.adapter_id} if record.is_checkpoint else {}),
            "text": "ok",
            "finish_reason": "stop",
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "checkpoint": record.checkpoint,
        }

    async def stream_generate(
        self,
        base_model: str,
        payload: Any,
        record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ):
        del expected_checkpoint
        self.calls.append((base_model, payload.adapter_id, record.adapter_id))
        self.payloads.append(payload)
        yield {"type": "ready", "checkpoint": record.checkpoint}
        yield {
            "type": "final",
            "finish_reason": "stop",
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "checkpoint": record.checkpoint,
        }

    async def register(self, _base_model: str, _record: AdapterRecord) -> None:
        return None

    async def unregister(
        self,
        _base_model: str,
        _adapter_id: str,
        _expected_generation: str | None = None,
    ) -> None:
        return None


async def _allow(_token: str, _adapter_id: str, _scope: dict | None = None) -> str:
    return "org-1"


def _client(base_model: str) -> tuple[TestClient, _Pool]:
    revision = _revision(base_model)
    pool = _Pool()
    app = build_serving_app(
        pool,
        AdapterRouter([revision]),
        chat_authorizer=_allow,
    )
    return TestClient(app, headers={"Authorization": "Bearer test"}), pool


def test_text_only_tool_history_skips_image_decoding(monkeypatch) -> None:
    """Text-only requests still get shape/role validation; only IMAGE work is skipped.

    They used to bypass validation entirely, which let a malformed `{"role": "user"}` reach the
    chat template and bill an empty completion. The cheap shared checks are now universal; what
    a text-only request must still avoid is decoding images it does not have.
    """

    def unexpected_image_work(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("text-only requests must not decode images")

    monkeypatch.setattr(multimodal, "_decode_images", unexpected_image_work)
    client, pool = _client(QWEN_35B)
    response = client.post(
        "/v1/chat/completions",
        json={"model": REVISION_ID, "messages": _tool_history_messages(with_image=False)},
    )
    assert response.status_code == 200
    assert response.json()["model"] == REVISION_ID
    assert pool.calls == [(QWEN_35B, REVISION_ID, REVISION_ID)]


def test_chat_permanent_checkpoint_echoes_public_checkpoint_identity() -> None:
    client, pool = _client(QWEN)
    response = client.post(
        "/v1/chat/completions",
        json={"model": REVISION_ID, "messages": _messages()},
    )
    assert response.status_code == 200
    assert response.json()["model"] == REVISION_ID
    assert response.json()["freesolo"] == {"checkpoint_id": REVISION_ID}
    assert response.headers["X-Freesolo-Checkpoint"] == REVISION_ID
    assert "X-Freesolo-Adapter-Revision" not in response.headers
    assert "X-Freesolo-HF-Revision" not in response.headers
    assert pool.calls == [(QWEN, REVISION_ID, REVISION_ID)]


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/generate", {"adapter_id": REVISION_ID, "messages": _messages()}),
        (f"/adapters/{REVISION_ID}/generate", {"messages": _messages()}),
        ("/v1/chat/completions", {"model": REVISION_ID, "messages": _messages()}),
    ],
)
def test_all_message_entry_points_validate_against_resolved_model(
    path: str, body: dict[str, Any], monkeypatch
) -> None:
    # every image entry point must route through multimodal validation, resolving image
    # capabilities from the resolved target model (not the requested id) before dispatch.
    validated: list[dict[str, Any]] = []
    supports_args: list[str] = []
    limit_args: list[str] = []
    real_validate = multimodal.normalize_chat_messages
    real_supports = multimodal.supports_image_input
    real_limit = multimodal.image_limit_for

    def spy_validate(messages: Any, **kwargs: Any) -> None:
        validated.append(kwargs)
        return real_validate(messages, **kwargs)

    def spy_supports(base_model: str) -> bool:
        supports_args.append(base_model)
        return real_supports(base_model)

    def spy_limit(base_model: str) -> int | None:
        limit_args.append(base_model)
        return real_limit(base_model)

    monkeypatch.setattr(multimodal, "normalize_chat_messages", spy_validate)
    monkeypatch.setattr(multimodal, "supports_image_input", spy_supports)
    monkeypatch.setattr(multimodal, "image_limit_for", spy_limit)
    client, pool = _client(QWEN_35B)
    response = client.post(path, json=body)

    assert response.status_code == 200
    # image capabilities were resolved from the resolved target model, not the requested id.
    assert supports_args == [QWEN_35B]
    assert limit_args == [QWEN_35B]
    # and multimodal validation was invoked exactly once with those resolved capabilities.
    assert validated == [
        {"supports_images": real_supports(QWEN_35B), "image_limit": real_limit(QWEN_35B)}
    ]
    assert len(pool.calls) == 1
    assert pool.calls[0][0] == QWEN_35B


def test_bad_streaming_image_is_json_400_before_sse() -> None:
    client, pool = _client(QWEN)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": REVISION_ID,
            "stream": True,
            "messages": _messages("https://example.com/image.png"),
        },
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert pool.calls == []


def test_context_overflow_is_json_400_before_sse() -> None:
    client, pool = _client(QWEN)

    async def overflow(*_args: Any, **_kwargs: Any):
        raise ValueError("context length exceeded")
        yield None

    pool.stream_generate = overflow  # type: ignore[method-assign]
    response = client.post(
        "/v1/chat/completions",
        json={"model": REVISION_ID, "stream": True, "messages": _messages()},
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert "context length exceeded" in response.json()["detail"]


def test_missing_attestation_on_a_revision_is_a_bad_gateway(monkeypatch) -> None:
    # sabotage: an engine that serves a revision without naming the adapter it resolved cannot
    # prove the weights came from the requested revision, so the router must refuse the response
    # rather than bill for provenance it never established.
    original = _Pool.generate

    async def unattested(self, base_model, payload, record, *, expected_checkpoint=None):
        result = await original(
            self, base_model, payload, record, expected_checkpoint=expected_checkpoint
        )
        result.pop("lora_request_adapter", None)
        return result

    monkeypatch.setattr(_Pool, "generate", unattested)
    client, _pool = _client(QWEN)
    response = client.post(
        "/v1/chat/completions",
        json={"model": REVISION_ID, "messages": _messages()},
    )
    assert response.status_code == 502
    assert "attest" in response.json()["detail"]


def _unattesting_pool(monkeypatch) -> None:
    """make the engine answer without attesting the adapter it served."""

    original = _Pool.generate

    async def unattested(self, base_model, payload, record, *, expected_checkpoint=None):
        result = await original(
            self, base_model, payload, record, expected_checkpoint=expected_checkpoint
        )
        result.pop("lora_request_adapter", None)
        return result

    monkeypatch.setattr(_Pool, "generate", unattested)


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/generate", {"adapter_id": REVISION_ID, "prompt": "hi"}),
        (f"/adapters/{REVISION_ID}/generate", {"prompt": "hi"}),
    ],
)
def test_unattested_revision_is_refused_on_the_plain_generate_routes(
    monkeypatch, path: str, body: dict[str, Any]
) -> None:
    # the attestation check originally lived only in the OpenAI handler, so these two routes
    # served an unattested adapter with no check at all -- the same request 502s as
    # /v1/chat/completions but succeeded here.
    _unattesting_pool(monkeypatch)
    client, _pool = _client(QWEN)

    response = client.post(path, json=body)

    assert response.status_code == 502
    assert "attest" in response.json()["detail"]


def test_a_refused_attestation_is_never_metered(monkeypatch) -> None:
    # usage is scheduled inside generate_once, so attesting after it returned meant the caller had
    # already been billed for a generation we then rejected with a 502. the check has to run
    # before metering, not after it.
    _unattesting_pool(monkeypatch)
    store = RecordingUsageStore()
    revision = _revision(QWEN).model_copy(update={"thinking": False})
    app = build_durable_serving_app(
        _Pool(),
        AdapterRouter([revision]),
        chat_authorizer=_allow,
        usage_store=store,
    )
    client = TestClient(app, headers={"Authorization": "Bearer test"})

    response = client.post("/generate", json={"adapter_id": REVISION_ID, "prompt": "hi"})

    assert response.status_code == 502
    assert store.captured == []
    assert store.finalized == []
    assert store.failed == []


def test_mismatched_checkpoint_attestation_is_a_bad_gateway(monkeypatch) -> None:
    original = _Pool.generate

    async def wrong_checkpoint(self, base_model, payload, record, *, expected_checkpoint=None):
        result = await original(
            self, base_model, payload, record, expected_checkpoint=expected_checkpoint
        )
        result["checkpoint"] = "flash-9999999999-99999999/step-1"
        return result

    monkeypatch.setattr(_Pool, "generate", wrong_checkpoint)
    client, _pool = _client(QWEN)
    response = client.post(
        "/v1/chat/completions",
        json={"model": REVISION_ID, "messages": _messages()},
    )
    assert response.status_code == 502
    assert "immutable checkpoint" in response.json()["detail"]
    assert "X-Freesolo-Checkpoint" not in response.headers


def test_mismatched_attestation_on_a_revision_is_a_bad_gateway(monkeypatch) -> None:
    # sabotage: attesting a DIFFERENT adapter is the failure the header exists to catch - the
    # engine served weights that are not the ones the caller pinned.
    original = _Pool.generate

    async def wrong_adapter(self, base_model, payload, record, *, expected_checkpoint=None):
        result = await original(
            self, base_model, payload, record, expected_checkpoint=expected_checkpoint
        )
        result["lora_request_adapter"] = "flash-9999999999-99999999/step-1"
        return result

    monkeypatch.setattr(_Pool, "generate", wrong_adapter)
    client, _pool = _client(QWEN)
    response = client.post(
        "/v1/chat/completions",
        json={"model": REVISION_ID, "messages": _messages()},
    )
    assert response.status_code == 502


def test_revision_response_carries_the_attestation_header_and_hides_the_field() -> None:
    client, _pool = _client(QWEN)
    response = client.post(
        "/v1/chat/completions",
        json={"model": REVISION_ID, "messages": _messages()},
    )
    assert response.status_code == 200
    assert response.headers["X-Freesolo-LoRA-Request-Adapter"] == REVISION_ID
    # the attestation is engine-internal plumbing; it must not leak into the caller-facing body.
    assert "lora_request_adapter" not in response.text


@pytest.mark.parametrize(
    ("messages", "expect"),
    [
        ([{"role": "user"}], "content must be a string"),
        ([{"role": "user", "content": None}], "content must be a string"),
        ([{"role": "nope", "content": "hi"}], "role must be"),
    ],
)
def test_malformed_text_only_messages_are_rejected_before_gpu_dispatch(
    messages: list[Any], expect: str
) -> None:
    """A text-only body used to skip validation entirely and bill an empty completion.

    `GenerateRequest` only checks "list of dicts", and the engine hands the list straight to the
    chat template, so `{"role": "user"}` rendered as an empty prompt and still charged the caller.
    """
    client, pool = _client(QWEN_35B)
    response = client.post(
        "/v1/chat/completions", json={"model": REVISION_ID, "messages": messages}
    )
    assert response.status_code == 400
    assert expect in response.text
    assert pool.calls == [], "a malformed request must never reach the gpu"


def test_developer_role_is_rewritten_to_system_for_text_only_requests() -> None:
    """`developer` is OpenAI's successor to `system`; these models have no such role.

    It was only rewritten when a request also carried list-form content, so a plain text chat
    reached the tokenizer with a role the template may reject -- after gpu dispatch.
    """
    client, pool = _client(QWEN_35B)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": REVISION_ID,
            "messages": [
                {"role": "developer", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    assert response.status_code == 200
    assert pool.calls == [(QWEN_35B, REVISION_ID, REVISION_ID)]
    sent = pool.payloads[-1].messages
    assert [m["role"] for m in sent] == ["system", "user"], (
        "the engine must template the normalized roles, not the original spelling"
    )


def test_image_request_reaches_the_engine_with_its_sources_intact() -> None:
    """The text-only write-back must not strip image sources on the way to the engine.

    `_normalize_messages` lifts each data uri into a separate `sources` channel and leaves a bare
    `{"type": "image"}` behind. That stripped block still satisfies `has_image_blocks`, so writing
    it back would route to the image path and then fail there with "must contain exactly one image
    source" -- after the request had already been accepted.
    """
    client, pool = _client(QWEN_35B)
    response = client.post(
        "/v1/chat/completions", json={"model": REVISION_ID, "messages": _messages()}
    )
    assert response.status_code == 200
    assert pool.calls == [(QWEN_35B, REVISION_ID, REVISION_ID)]
    sent = pool.payloads[-1].messages
    blocks = sent[0]["content"]
    image_blocks = [b for b in blocks if b.get("type") in {"image_url", "input_image", "image"}]
    assert image_blocks, "the engine must still receive an image block"
    # the engine re-normalizes from the original, so the source must survive the boundary
    assert prepare_multimodal_request(sent, image_limit=4)[1], (
        "the engine must be able to decode the images it was handed"
    )


@pytest.mark.parametrize(
    "tool_calls", [1, "x", [], ["not-an-object"], [{"function": {"name": "f"}}, 2]]
)
def test_malformed_tool_calls_are_rejected_before_the_chat_template(tool_calls: Any) -> None:
    """`tool_calls` replaces content here, so an unusable value renders nothing.

    The branch only checked that the key was present, so `tool_calls: 1` reached the chat template
    and raised a jinja error from outside the rejection handler -- answered 503, which tells the
    caller to retry a request that must fail identically. `flash/serve/runtime/multimodal.py`
    already rejected these; both serving paths must answer the same way.
    """
    with pytest.raises(MultimodalRequestError):
        normalize_chat_messages(
            [{"role": "assistant", "content": None, "tool_calls": tool_calls}],
            supports_images=False,
            image_limit=4,
        )


def test_a_usable_tool_call_history_still_passes() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    assert normalize_chat_messages(messages, supports_images=False, image_limit=4) == messages
