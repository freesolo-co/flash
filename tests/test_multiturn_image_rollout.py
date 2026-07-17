from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from flash.engine.multiturn_rollout import build_rollout_func
from flash.multimodal import image_content_key, multimodal_prompt_key


class _Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        text = "".join(f"<{message['role']}>{message.get('content', '')}" for message in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])


class _Processor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((multimodal_prompt_key(messages), kwargs))
        return {"input_ids": [[11, 12, 13]]}


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


class _Engine:
    def __init__(self):
        self.requests = []
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
        attached = request["multi_modal_data"]["image"]
        attached_images = attached if isinstance(attached, list) else [attached]
        assert [image_content_key(image) for image in attached_images] == [
            image_content_key(image) for image in images
        ]


@pytest.mark.usefixtures("_stub_vllm")
def test_text_only_multiturn_rollout_payload_is_unchanged():
    prompt = [{"role": "user", "content": "hi"}]
    engine = _Engine()
    rollout = _rollout_func(examples_by_key={})

    result = rollout([prompt], _trainer(engine))

    assert result["reward"] == [1.0]
    assert len(engine.requests) == 2
    assert all(set(request) == {"prompt_token_ids"} for request in engine.requests)
