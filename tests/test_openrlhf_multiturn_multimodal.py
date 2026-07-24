"""CPU parity for OpenRLHF multi-turn *multimodal* GRPO (parity plan #10).

Single-turn multimodal GRPO already exists on the OpenRLHF path; the multi-turn executor
(:mod:`flash.engine.worker.openrlhf_multiturn`) was text-only. The one multi-turn-specific piece TRL
has and OpenRLHF lacked is the per-request payload construction: collapse the processor-expanded
image-pad runs in the re-submitted prefix, attach ``multi_modal_data``, and bias the image-pad token
out of the sampler. :func:`build_multiturn_multimodal_request` provides it, reusing the same
``flash.multimodal.collapse_image_pad_runs`` primitive as TRL. These tests prove it is byte-identical
to :func:`flash.engine.multiturn_rollout.build_rollout_func`'s live ``submit`` and that the existing
token-exact assembly stays correct once image-pad tokens are present. Live colocated generation over
the payload is GPU-deferred behind the multi-turn gate, so only the pure construction is exercised
here.
"""

from __future__ import annotations

import sys
import types
from itertools import pairwise
from types import SimpleNamespace

import pytest

from flash.engine.worker.openrlhf_multiturn import (
    assemble_multiturn_rollout,
    build_multiturn_multimodal_request,
)
from flash.multimodal import image_content_key, multimodal_prompt_key

_IMAGE_PAD_ID = 99


# --- pure builder: shape + collapse + bias parity -------------------------------------------


def _expanded_prompt(image_count: int) -> list[int]:
    """The exact prefix the multimodal test processor renders: [11, (pad,pad,pad,14)*n, 13]."""
    image_ids = [token for _ in range(image_count) for token in [_IMAGE_PAD_ID] * 3 + [14]]
    return [11, *image_ids, 13]


@pytest.mark.parametrize("image_count", [1, 2, 3])
def test_multimodal_request_collapses_pads_attaches_images_and_biases(image_count):
    images = [f"image-{index}" for index in range(image_count)]
    prefix = _expanded_prompt(image_count)

    prompt, logit_bias = build_multiturn_multimodal_request(prefix, images, _IMAGE_PAD_ID)

    assert set(prompt) == {"prompt_token_ids", "multi_modal_data"}
    ids = prompt["prompt_token_ids"]
    # one pad per image after the collapse, and never two pads in a row (vLLM re-expands from mm data)
    assert ids.count(_IMAGE_PAD_ID) == image_count
    assert all(
        current != _IMAGE_PAD_ID or previous != _IMAGE_PAD_ID for previous, current in pairwise(ids)
    )
    # non-pad tokens are preserved in order
    assert [t for t in ids if t != _IMAGE_PAD_ID] == [t for t in prefix if t != _IMAGE_PAD_ID]
    # >1 image -> list; exactly 1 image -> the bare image (matches TRL submit)
    attached = prompt["multi_modal_data"]["image"]
    assert attached == (images if image_count > 1 else images[0])
    assert logit_bias == {_IMAGE_PAD_ID: -100.0}


def test_text_only_request_is_bare_prompt_with_no_bias():
    prompt, logit_bias = build_multiturn_multimodal_request([5, 6, 7], None, _IMAGE_PAD_ID)
    assert prompt == {"prompt_token_ids": [5, 6, 7]}
    assert logit_bias is None
    # empty image list is text-only too
    assert build_multiturn_multimodal_request([5, 6], [], _IMAGE_PAD_ID) == (
        {"prompt_token_ids": [5, 6]},
        None,
    )


def test_missing_image_pad_id_with_images_raises():
    with pytest.raises(ValueError, match="image-pad token id"):
        build_multiturn_multimodal_request(_expanded_prompt(1), ["image-0"], None)


def test_empty_prefix_raises():
    with pytest.raises(ValueError, match="empty prompt"):
        build_multiturn_multimodal_request([], None, _IMAGE_PAD_ID)


def test_image_count_mismatch_propagates_collapse_error():
    # two declared images but only one pad run in the prefix -> the shared primitive rejects it,
    # exactly as TRL's submit would, rather than silently misaligning vision features.
    with pytest.raises(ValueError, match=r"image-pad run"):
        build_multiturn_multimodal_request([11, _IMAGE_PAD_ID, 13], ["a", "b"], _IMAGE_PAD_ID)


# --- assembly stays correct once image-pad tokens live in the prompt region -------------------


def test_assemble_multiturn_rollout_keeps_image_pad_tokens_masked():
    # expanded multimodal prompt (image-pad tokens) followed by two assistant turns + one glue seam.
    prompt_ids = _expanded_prompt(1)  # [11, 99,99,99,14, 13]
    turns = [
        {"action_ids": [21, 22], "glue_ids": [31]},
        {"action_ids": [23, 24, 25], "glue_ids": []},
    ]
    out = assemble_multiturn_rollout(prompt_ids, turns)

    # the training sequence uses the EXPANDED prompt (the forward pass fills pads with vision
    # features); generation collapses separately via build_multiturn_multimodal_request.
    assert out["token_ids"] == [*prompt_ids, 21, 22, 31, 23, 24, 25]
    # every prompt-region token (image-pad included) is masked 0; only assistant spans are trained.
    assert out["response_mask"] == [0] * len(prompt_ids) + [1, 1, 0, 1, 1, 1]
    # no image-pad token falls inside a trained action span
    for start, end in out["action_ranges"]:
        assert _IMAGE_PAD_ID not in out["token_ids"][start:end]


# --- direct byte-parity against TRL's live rollout submit -------------------------------------


class _Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        text = "".join(f"<{m['role']}>{m.get('content', '')}" for m in messages)
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
        return {"input_ids": [_expanded_prompt(image_count)]}


class _Engine:
    def __init__(self):
        self.requests = []
        self.sampling = []
        self.pending = []
        self.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(get_vocab_size=lambda: 10000),
            add_request=self._add_request,
            abort_request=self._abort_request,
            step=self._step,
            has_unfinished_requests=lambda: bool(self.pending),
        )

    def _add_request(self, request_id, prompt, sampling_params):
        self.requests.append(prompt)
        self.sampling.append(sampling_params)
        self.pending.append(request_id)

    def _abort_request(self, request_ids):
        dropped = set(request_ids)
        self.pending = [r for r in self.pending if r not in dropped]

    def _step(self):
        pending, self.pending = self.pending, []
        return [
            SimpleNamespace(
                request_id=r,
                finished=True,
                outputs=[SimpleNamespace(token_ids=[20], logprobs=None, text="ok")],
            )
            for r in pending
        ]


class _TwoTurnEnv:
    multi_turn = True

    def new_rollout_state(self, example):
        prompt = example.get("prompt") or [{"role": "user", "content": "reconstructed"}]
        return {"prompt": prompt, "messages": list(prompt), "completion": []}

    def record_model_turn(self, state, content):
        state["completion"].append({"role": "assistant", "content": content})

    def rollout_done(self, state, max_turns):
        return sum(m["role"] == "assistant" for m in state["completion"]) >= 2

    def env_reply(self, messages, state):
        reply = {"role": "user", "content": "next"}
        state["completion"].append(reply)
        return [reply]

    def reward(self, completion, example, state=None):
        return 1.0


@pytest.fixture
def _stub_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    sampling = types.ModuleType("vllm.sampling_params")
    sampling.RequestOutputKind = SimpleNamespace(FINAL_ONLY="final_only")
    vllm.sampling_params = sampling
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)


def _mm_images(prompt_payload: dict) -> list:
    attached = prompt_payload["multi_modal_data"]["image"]
    return attached if isinstance(attached, list) else [attached]


@pytest.mark.usefixtures("_stub_vllm")
@pytest.mark.parametrize("image_count", [1, 2])
def test_builder_reproduces_trl_live_submit_byte_for_byte(image_count):
    # drive the REAL TRL multimodal rollout and confirm our OpenRLHF builder reproduces the exact
    # engine request + logit_bias it emits for the initial (turn-0) prompt.
    image_module = pytest.importorskip("PIL.Image")
    from flash.engine.multiturn_rollout import build_rollout_func

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
    engine = _Engine()
    rollout = build_rollout_func(
        active_env=_TwoTurnEnv(),
        tok=_Tokenizer(),
        examples_by_key={multimodal_prompt_key(prompt): {"id": "x"}},
        processor=_Processor(),
        multimodal=True,
        max_completion=8,
        max_turns=2,
        temperature=0.7,
        top_p=1.0,
        stop=None,
        thinking=False,
        engine_max_len=128,
    )
    rollout([prompt], SimpleNamespace(
        vllm_generation=SimpleNamespace(llm=engine),
        args=SimpleNamespace(vllm_enable_sleep_mode=False),
    ))

    trl_turn0_prompt = engine.requests[0]
    trl_turn0_bias = getattr(engine.sampling[0], "logit_bias", None)

    # our builder, fed the same turn-0 expanded prefix + images, must produce the identical payload:
    # byte-identical collapsed prompt_token_ids and logit_bias, and the same multi_modal_data shape
    # + content (PIL images have no __eq__, so compare by the same content key the TRL test uses).
    ours_prompt, ours_bias = build_multiturn_multimodal_request(
        _expanded_prompt(image_count), images, _IMAGE_PAD_ID
    )
    assert ours_prompt["prompt_token_ids"] == trl_turn0_prompt["prompt_token_ids"]
    assert ours_bias == trl_turn0_bias
    assert isinstance(ours_prompt["multi_modal_data"]["image"], list) == isinstance(
        trl_turn0_prompt["multi_modal_data"]["image"], list
    )
    assert [image_content_key(i) for i in _mm_images(ours_prompt)] == [
        image_content_key(i) for i in _mm_images(trl_turn0_prompt)
    ]
