from __future__ import annotations

import sys
import types
from itertools import pairwise
from types import SimpleNamespace

import pytest

from flash.engine.multiturn_rollout import build_rollout_func
from flash.multimodal import (
    collapse_image_pad_runs,
    image_content_key,
    multimodal_prompt_key,
    resolve_image_pad_token_id,
)

_IMAGE_PAD_ID = 99


class _Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        text = "".join(f"<{message['role']}>{message.get('content', '')}" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])


class _Processor:
    image_token_id = _IMAGE_PAD_ID

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((multimodal_prompt_key(messages), kwargs))
        image_count = sum(
            part.get("type") == "image"
            for message in messages
            for part in message.get("content", [])
            if isinstance(part, dict)
        )
        image_ids = [token for _ in range(image_count) for token in [_IMAGE_PAD_ID] * 3 + [14]]
        return {"input_ids": [[11, *image_ids, 13]]}


class _TwoTurnEnv:
    multi_turn = True

    def new_rollout_state(self, example):
        prompt = example.get("prompt") or [{"role": "user", "content": "reconstructed"}]
        return {"prompt": prompt, "messages": list(prompt), "completion": []}

    def record_model_turn(self, state, content):
        state["completion"].append({"role": "assistant", "content": content})

    def rollout_done(self, state, max_turns):
        return sum(message["role"] == "assistant" for message in state["completion"]) >= 2

    def env_reply(self, messages, state):
        reply = {"role": "user", "content": "next"}
        state["completion"].append(reply)
        return [reply]

    def reward(self, completion, example, state=None):
        return 1.0


def test_collapse_image_pad_runs_collapses_one_expanded_run():
    assert collapse_image_pad_runs([1, 9, 9, 9, 2], 9, 1) == [1, 9, 2]


def test_collapse_image_pad_runs_collapses_separated_runs_and_preserves_other_tokens():
    ids = [1, 9, 9, 2, 3, 9, 9, 9, 4]
    assert collapse_image_pad_runs(ids, 9, 2) == [1, 9, 2, 3, 9, 4]


def test_collapse_image_pad_runs_rejects_image_count_mismatch():
    with pytest.raises(ValueError, match=r"found 1 image-pad run.*expected 2"):
        collapse_image_pad_runs([1, 9, 9, 2], 9, 2)


class _Engine:
    def __init__(self):
        self.requests = []
        self.sampling = []
        self.pending = []
        self.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(get_vocab_size=lambda: 10000),
            add_request=self.add_request,
            abort_request=self.abort_request,
            step=self.step,
            has_unfinished_requests=lambda: bool(self.pending),
        )

    def add_request(self, request_id, prompt, sampling_params):
        self.requests.append(prompt)
        self.sampling.append(sampling_params)
        self.pending.append(request_id)

    def abort_request(self, request_ids):
        dropped = set(request_ids)
        self.pending = [request_id for request_id in self.pending if request_id not in dropped]

    def step(self):
        pending, self.pending = self.pending, []
        return [
            SimpleNamespace(
                request_id=request_id,
                finished=True,
                outputs=[SimpleNamespace(token_ids=[20], logprobs=None, text="ok")],
            )
            for request_id in pending
        ]


@pytest.fixture
def _stub_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    sampling = types.ModuleType("vllm.sampling_params")
    sampling.RequestOutputKind = SimpleNamespace(FINAL_ONLY="final_only")
    vllm.sampling_params = sampling
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)


def _trainer(engine):
    return SimpleNamespace(
        vllm_generation=SimpleNamespace(llm=engine),
        args=SimpleNamespace(vllm_enable_sleep_mode=False),
    )


def _rollout_func(*, examples_by_key, processor=None, multimodal=False):
    return build_rollout_func(
        active_env=_TwoTurnEnv(),
        tok=_Tokenizer(),
        examples_by_key=examples_by_key,
        processor=processor,
        multimodal=multimodal,
        max_completion=8,
        max_turns=2,
        temperature=0.7,
        top_p=1.0,
        stop=None,
        thinking=False,
        engine_max_len=128,
    )


@pytest.mark.usefixtures("_stub_vllm")
@pytest.mark.parametrize("image_count", [1, 2])
def test_multiturn_image_rollout_attaches_images_to_every_request(image_count):
    image_module = pytest.importorskip("PIL.Image")
    images = [
        image_module.new("RGB", (2, 2), (index * 80, 0, 255 - index * 80))
        for index in range(image_count)
    ]
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "compare"},
                *[{"type": "image", "image": image} for image in images],
            ],
        }
    ]
    example = {"id": "image-example"}
    processor = _Processor()
    engine = _Engine()
    rollout = _rollout_func(
        examples_by_key={multimodal_prompt_key(prompt): example},
        processor=processor,
        multimodal=True,
    )

    result = rollout([prompt], _trainer(engine))

    assert result["reward"] == [1.0]
    assert result["prompt_ids"][0].count(_IMAGE_PAD_ID) == 3 * image_count
    assert len(engine.requests) == 2
    assert len(processor.calls) == 1
    rendered_key, render_kwargs = processor.calls[0]
    assert rendered_key == multimodal_prompt_key(prompt)
    assert render_kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "enable_thinking": False,
    }
    for request in engine.requests:
        assert set(request) == {"prompt_token_ids", "multi_modal_data"}
        request_ids = request["prompt_token_ids"]
        assert request_ids.count(_IMAGE_PAD_ID) == image_count
        assert all(
            current != _IMAGE_PAD_ID or previous != _IMAGE_PAD_ID
            for previous, current in pairwise(request_ids)
        )
        attached = request["multi_modal_data"]["image"]
        attached_images = attached if isinstance(attached, list) else [attached]
        assert [image_content_key(image) for image in attached_images] == [
            image_content_key(image) for image in images
        ]


def test_real_qwen_processor_image_pad_id_and_expansion():
    image_module = pytest.importorskip("PIL.Image")
    auto_processor = pytest.importorskip("transformers").AutoProcessor
    try:
        processor = auto_processor.from_pretrained(
            "Qwen/Qwen3.5-0.8B",
            trust_remote_code=True,
            local_files_only=True,
        )
    except (ImportError, OSError, ValueError) as exc:
        pytest.skip(f"Qwen3.5 processor is unavailable offline: {exc}")

    image_pad_id = resolve_image_pad_token_id(processor, processor.tokenizer)
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_module.new("RGB", (2, 2), "red")},
                {"type": "text", "text": "what color?"},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        enable_thinking=False,
    )
    ids = rendered["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]

    collapsed = collapse_image_pad_runs(list(ids), image_pad_id, 1)
    assert image_pad_id >= 0
    assert ids.count(image_pad_id) > 1
    assert collapsed.count(image_pad_id) == 1
    assert len(collapsed) == len(ids) - ids.count(image_pad_id) + 1


@pytest.mark.usefixtures("_stub_vllm")
def test_multiturn_image_rollout_suppresses_image_pad_token():
    # a stray image-pad token emitted into an assistant turn would corrupt the next turn's
    # collapse_image_pad_runs count and misalign vision data; the image branch must bias the
    # pad token out of the sampler on every generated turn (mirrors the single-turn opd guard).
    image_module = pytest.importorskip("PIL.Image")
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "image": image_module.new("RGB", (2, 2), "red")},
            ],
        }
    ]
    engine = _Engine()
    rollout = _rollout_func(
        examples_by_key={multimodal_prompt_key(prompt): {"id": "img"}},
        processor=_Processor(),
        multimodal=True,
    )

    rollout([prompt], _trainer(engine))

    assert len(engine.sampling) == 2
    for sampling_params in engine.sampling:
        assert getattr(sampling_params, "logit_bias", None) == {_IMAGE_PAD_ID: -100.0}


@pytest.mark.usefixtures("_stub_vllm")
def test_text_only_multiturn_rollout_payload_is_unchanged():
    prompt = [{"role": "user", "content": "hi"}]
    engine = _Engine()
    rollout = _rollout_func(examples_by_key={})

    result = rollout([prompt], _trainer(engine))

    assert result["reward"] == [1.0]
    assert len(engine.requests) == 2
    assert all(set(request) == {"prompt_token_ids"} for request in engine.requests)
    # no pad-suppression bias leaks into text-only turns
    assert all(
        getattr(sampling_params, "logit_bias", None) is None for sampling_params in engine.sampling
    )
