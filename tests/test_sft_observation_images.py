"""multi-turn sft rows whose completion shows the model a screenshot between its actions.

a browser-agent trajectory is system, user[text + screenshot], assistant action, user[text +
screenshot], assistant action, ... the observation images are model input: they must reach the
processor in conversation order with pad runs the processor sizes, while the loss stays on the
assistant turns only.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from freesolo.environments import EnvironmentSingleTurn, RewardResult
from PIL import Image

from flash.content import multimodal as mm
from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
from flash.engine.profiling import image_tokens
from flash.engine.profiling.dataset_profile import _RawRecordEnvironment
from flash.engine.profiling.sft_workload import prepare_sft_workload
from flash.engine.profiling.workload_profile import sft_profile_input_digest
from flash.envs.loading.adapter import FreesoloEnvironment

IM_START = 0x110000
IM_END = 0x110001
IMAGE_PAD = 0x110002
VISION_START = 0x110003
VISION_END = 0x110004
_SPECIALS = {
    "<|im_start|>": IM_START,
    "<|im_end|>": IM_END,
    mm.IMAGE_PAD_TOKEN: IMAGE_PAD,
    "<|vision_start|>": VISION_START,
    "<|vision_end|>": VISION_END,
}
_GEOMETRY = image_tokens.ImageGeometry(
    patch_size=16, merge_size=2, min_pixels=1024, max_pixels=1 << 20
)
_PLACEHOLDER = f"<|vision_start|>{mm.IMAGE_PAD_TOKEN}<|vision_end|>"


def _image_uri(size: int, color: str) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _pads(size: int) -> int:
    return image_tokens.image_pad_tokens(size, size, _GEOMETRY)


def _render(messages, *, add_generation_prompt: bool) -> str:
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            body = "".join(
                _PLACEHOLDER if block.get("type") == "image" else str(block.get("text") or "")
                for block in content
                if isinstance(block, dict)
            )
        else:
            body = str(content or "")
        parts.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


class _ChatMlTokenizer:
    """character tokenizer with the qwen chatml and vision control tokens registered once."""

    eos_token = "!"
    eos_token_id = ord("!")
    pad_token = None
    unk_token_id = -1
    all_special_ids: ClassVar[list[int]] = [0, ord("!"), *_SPECIALS.values()]
    chat_template = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    )

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, return_dict=False, **_kwargs
    ):
        text = _render(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        assert return_dict is True
        return {"input_ids": self._encode(text)}

    def __call__(self, texts, *, truncation=False, max_length=None):
        rows = [self._encode(text) for text in ([texts] if isinstance(texts, str) else texts)]
        if truncation:
            rows = [row[:max_length] for row in rows]
        return {"input_ids": rows}

    def convert_tokens_to_ids(self, token):
        return _SPECIALS.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return {value: key for key, value in _SPECIALS.items()}.get(token_id)

    def get_vocab(self):
        return dict(_SPECIALS)

    def decode(self, ids):
        names = {value: key for key, value in _SPECIALS.items()}
        return "".join(names[token_id] if token_id in names else chr(token_id) for token_id in ids)

    @staticmethod
    def _encode(text: str) -> list[int]:
        ids = []
        index = 0
        while index < len(text):
            special = next((name for name in _SPECIALS if text.startswith(name, index)), None)
            if special is not None:
                ids.append(_SPECIALS[special])
                index += len(special)
            else:
                ids.append(ord(text[index]))
                index += 1
        return ids


class _ExpandingProcessor:
    """a processor that expands each bound image into the pad run its geometry gives it."""

    def __init__(self):
        self.tokenizer = _ChatMlTokenizer()
        self.chat_template = self.tokenizer.chat_template
        self.rendered_image_sizes: list[list[int]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
        return_tensors=None,
        **_kwargs,
    ):
        text = _render(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        assert return_dict is True
        assert return_tensors == "pt"
        images = [
            block["image"]
            for message in messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "image"
        ]
        assert all(isinstance(image, Image.Image) for image in images)
        sizes = [image.width for image in images]
        self.rendered_image_sizes.append(sizes)
        runs = iter(sizes)
        ids = []
        for token_id in self.tokenizer._encode(text):
            ids.extend([IMAGE_PAD] * _pads(next(runs)) if token_id == IMAGE_PAD else [token_id])
        grid = np.array(
            [[1, size // _GEOMETRY.patch_size, size // _GEOMETRY.patch_size] for size in sizes],
            dtype=np.int64,
        ).reshape(-1, 3)
        return {"input_ids": [ids], "attention_mask": [[1] * len(ids)], "image_grid_thw": grid}


class _TrajectoryEnvironment:
    multi_turn = True
    package_root = None

    def __init__(self, prompt, completion):
        self._prompt = prompt
        self._completion = completion

    def dataset(self):
        return [{"prompt": "unused", "output": "unused"}]

    def prompt_messages(self, _row):
        return [dict(message) for message in self._prompt]

    def sft_completion(self, _row):
        return [dict(message) for message in self._completion]


def _observation(text: str | None, size: int, color: str) -> dict:
    blocks = [] if text is None else [{"type": "text", "text": text}]
    return {
        "role": "user",
        "content": [*blocks, {"type": "image_url", "image_url": {"url": _image_uri(size, color)}}],
    }


PROMPT = [
    {"role": "system", "content": "you drive a browser"},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "click ok"},
            {"type": "image_url", "image_url": {"url": _image_uri(32, "red")}},
        ],
    },
]
COMPLETION = [
    {"role": "assistant", "content": "ACT1"},
    _observation("page changed", 64, "green"),
    {"role": "assistant", "content": "ACT2"},
    # a screenshot-only observation still sits between the two assistant turns
    _observation(None, 48, "blue"),
    {"role": "assistant", "content": "ACT3"},
]
SIZES = [32, 64, 48]


def _spec(max_context_tokens: int = 4096) -> JobSpec:
    spec = JobSpec(
        model="Qwen/Qwen3.5-9B",
        model_revision="a" * 40,
        algorithm="sft",
        environment=EnvironmentSpec(id="team/browser", resolved_sha="b" * 40),
        train=TrainSpec(
            epochs=1, batch_size=1, max_context_tokens=max_context_tokens, max_examples=1
        ),
        seed=7,
    )
    return replace(
        spec,
        workload_profile_input_digest=sft_profile_input_digest(
            spec, tokenizer_revision=spec.model_revision, producer_version="1.2.3"
        ),
        workload_profile_producer_version="1.2.3",
    )


def _prepare(
    completion=COMPLETION,
    *,
    prompt=PROMPT,
    processor=None,
    estimate=False,
    image_dir=None,
    max_context_tokens=4096,
    monkeypatch=None,
):
    common = {
        "tokenizer_loader": lambda _model, _revision: _ChatMlTokenizer(),
        "producer_version": "1.2.3",
        "packing_support": lambda _model, _revision: ("multimodal", False),
    }
    if estimate:
        assert monkeypatch is not None
        monkeypatch.setattr(image_tokens, "load_image_geometry", lambda *_args: _GEOMETRY)
        return prepare_sft_workload(
            _spec(max_context_tokens),
            _TrajectoryEnvironment(prompt, completion),
            require_processor=False,
            **common,
        )
    processor = processor or _ExpandingProcessor()
    return prepare_sft_workload(
        _spec(max_context_tokens),
        _TrajectoryEnvironment(prompt, completion),
        processor_loader=lambda _model, _revision: processor,
        image_dir=image_dir,
        **common,
    )


def _pad_runs(ids: list[int]) -> list[int]:
    runs: list[int] = []
    previous = None
    for token_id in ids:
        if token_id == IMAGE_PAD:
            if previous == IMAGE_PAD:
                runs[-1] += 1
            else:
                runs.append(1)
        previous = token_id
    return runs


def _supervised_text(row) -> str:
    tokenizer = _ChatMlTokenizer()
    return tokenizer.decode(
        [token for token, mask in zip(row["input_ids"], row["loss_mask"], strict=True) if mask]
    )


def test_observation_images_train_in_order_with_loss_on_assistant_turns_only(tmp_path):
    processor = _ExpandingProcessor()
    prepared = _prepare(processor=processor, image_dir=str(tmp_path))

    row = prepared.rows[0]
    assert prepared.multimodal is True
    assert prepared.profile.architecture_mode == "multimodal"
    # every screenshot reached the processor, in conversation order, with its own pad run
    assert processor.rendered_image_sizes[0] == SIZES
    assert _pad_runs(row["input_ids"]) == [_pads(size) for size in SIZES]
    # the rows verl loads keep one saved image per pad run, in the same order
    assert len(row["images"]) == 3
    with np.load(io.BytesIO(row["multimodal_inputs"])) as arrays:
        assert arrays["image_grid_thw"].shape == (3, 3)
    saved = []
    for uri in row["images"]:
        with Image.open(uri.removeprefix("file://")) as image:
            saved.append(image.width)
    assert saved == SIZES
    # loss only on the authored assistant bodies and their closers: never an observation or pad
    assert _supervised_text(row) == "ACT1<|im_end|>ACT2<|im_end|>ACT3<|im_end|>"
    assert not any(
        mask
        for token, mask in zip(row["input_ids"], row["loss_mask"], strict=True)
        if token == IMAGE_PAD
    )
    assert prepared.role_aware_multiturn_targets == 1
    assert prepared.fallback_multiturn_targets == 0


def test_estimate_matches_the_processor_rows_it_quotes(monkeypatch):
    trained = _prepare()
    quoted = _prepare(estimate=True, monkeypatch=monkeypatch)

    assert quoted.rows[0]["input_ids"] == trained.rows[0]["input_ids"]
    assert quoted.rows[0]["loss_mask"] == trained.rows[0]["loss_mask"]
    assert quoted.profile.real_tokens_per_epoch == trained.profile.real_tokens_per_epoch
    assert quoted.profile.supervised_tokens_per_epoch == trained.profile.supervised_tokens_per_epoch


def test_prompt_only_image_rows_tokenize_as_before():
    prepared = _prepare([{"role": "assistant", "content": "ACT1"}])

    row = prepared.rows[0]
    assert _pad_runs(row["input_ids"]) == [_pads(32)]
    assert _supervised_text(row) == "ACT1<|im_end|>"


@pytest.mark.parametrize("estimate", [False, True], ids=["processor", "estimate"])
def test_an_overlong_trajectory_is_trimmed_to_whole_turns_with_its_images(monkeypatch, estimate):
    whole = _prepare().rows[0]
    act2_end = _ChatMlTokenizer().decode(whole["input_ids"]).index("ACT2<|im_end|>\n")
    # a cap that ends inside the last observation's screenshot: slicing would cut its pad run
    cap = len(_ChatMlTokenizer()._encode(_ChatMlTokenizer().decode(whole["input_ids"])[:act2_end]))
    cap += len("ACT2<|im_end|>\n") + 10
    assert cap < len(whole["input_ids"])

    prepared = _prepare(max_context_tokens=cap, estimate=estimate, monkeypatch=monkeypatch)

    row = prepared.rows[0]
    assert len(row["input_ids"]) <= cap
    assert _pad_runs(row["input_ids"]) == [_pads(32), _pads(64)]
    assert _supervised_text(row) == "ACT1<|im_end|>ACT2<|im_end|>"
    # the cap still reports the whole trajectory, so the truncation warning describes it
    assert prepared.profile.truncated_examples == 1
    assert prepared.profile.untruncated_max_length == len(whole["input_ids"])


def test_an_observation_that_cannot_fit_whole_is_refused():
    completion = [_observation("start", 64, "green"), {"role": "assistant", "content": "ACT1"}]

    with pytest.raises(ValueError, match="would cut an observation image"):
        _prepare(completion, max_context_tokens=40)


def test_an_image_in_an_assistant_message_is_refused():
    completion = [
        {"role": "assistant", "content": "ACT1"},
        _observation("page", 32, "green"),
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "here"},
                {"type": "image_url", "image_url": {"url": _image_uri(32, "red")}},
            ],
        },
    ]

    with pytest.raises(ValueError, match="not supported in assistant messages"):
        _prepare(completion)


def test_an_output_image_is_refused_even_in_a_user_message():
    completion = [
        {"role": "assistant", "content": "ACT1"},
        {"role": "user", "content": [{"type": "output_image", "image_url": _image_uri(32, "red")}]},
        {"role": "assistant", "content": "ACT2"},
    ]

    with pytest.raises(ValueError, match="not supported in assistant messages"):
        _prepare(completion)


def test_a_tool_message_cannot_carry_an_observation_image():
    image = _observation("result", 32, "green")
    completion = [
        {"role": "assistant", "content": "ACT1"},
        {**image, "role": "tool"},
        {"role": "assistant", "content": "ACT2"},
    ]

    with pytest.raises(ValueError, match="only in user observation messages"):
        _prepare(completion)


def test_the_image_count_limit_bounds_the_whole_conversation(monkeypatch):
    monkeypatch.setattr(mm, "MAX_IMAGES_PER_EXAMPLE", 2)

    with pytest.raises(ValueError, match="3 images, exceeding the 2-image limit"):
        _prepare()


def test_the_decoded_memory_limit_bounds_the_whole_conversation(monkeypatch):
    # each image fits alone, but the prompt's and both observations' decoded bytes together do not
    per_image = 64 * 64 * 4
    monkeypatch.setattr(mm, "MAX_TOTAL_DECODED_BYTES", per_image * 2)

    with pytest.raises(ValueError, match="decoded"):
        _prepare()


def test_the_longest_fitting_prefix_wins_even_when_lengths_are_not_monotonic():
    from flash.engine.profiling.sft_image_rows import fit_observation_conversation

    # a template that drops earlier reasoning can render a longer prefix SHORTER than a shorter one
    lengths = {1: 30, 3: 1060, 5: 110, 7: 2000}
    # the fitter sees normalized messages, whose images are {"type": "image"} blocks
    seen_image = {"role": "user", "content": [{"type": "text", "text": "obs"}, {"type": "image"}]}
    completion = [
        {"role": "assistant", "content": "ACT1"},
        seen_image,
        {"role": "assistant", "content": "ACT2"},
        seen_image,
        {"role": "assistant", "content": "ACT3"},
        seen_image,
        {"role": "assistant", "content": "ACT4"},
    ]
    seen = []

    def tokenize(prompt, part, used, *, package_root):
        seen.append(len(part))
        return [0], [1], b"", lengths[len(part)], True

    fitted = fit_observation_conversation(
        tokenize,
        [{"role": "user", "content": "go"}],
        completion,
        ["a", "b", "c"],
        package_root=None,
        max_length=300,
    )
    assert len(fitted.completion_messages) == 5
    assert fitted.descriptors == ["a", "b"]
    assert fitted.untruncated_length == 2000
    # the oversized middle prefix never pruned the longer one that fits
    assert seen == [7, 5]


# ---- preview/worker parity for an output authored as data, read by the sdk's default hook ----
#
# the cases above hand prepare_sft_workload a stand-in environment with its own sft_completion.
# real environments that do not override the hook get the sdk default, whose target_messages
# refused content-block lists: the control-plane quote (preview and submit) accepted these rows,
# and the worker then failed on the gpu with "Expected output messages[1].content must be a
# string or null". both paths must read the same row the same way.


class _DefaultHookTrajectories(EnvironmentSingleTurn):
    """the miniwob sft shape: trajectories in `output`, no sft_completion override."""

    def build_prompt_messages(self, example, prompt_text):
        return [{"role": "user", "content": example.input}]

    def score_response(self, example, response_text):
        return RewardResult(score=0.0, threshold=1.0)


def _worker_env(rows):
    return FreesoloEnvironment(_DefaultHookTrajectories(), "team/browser", source=rows)


def _quote_env(rows):
    return _RawRecordEnvironment(rows=[dict(row) for row in rows], package_root=Path.cwd())


def _prepare_both(rows, monkeypatch):
    """the worker (adapter + processor) and the control-plane quote (raw records + estimate)."""
    monkeypatch.setattr(image_tokens, "load_image_geometry", lambda *_args: _GEOMETRY)
    common = {
        "tokenizer_loader": lambda _model, _revision: _ChatMlTokenizer(),
        "producer_version": "1.2.3",
        "packing_support": lambda _model, _revision: ("multimodal", False),
    }
    worker = prepare_sft_workload(
        _spec(),
        _worker_env(rows),
        processor_loader=lambda _model, _revision: _ExpandingProcessor(),
        **common,
    )
    quote = prepare_sft_workload(_spec(), _quote_env(rows), require_processor=False, **common)
    return worker, quote


@pytest.mark.parametrize(
    "output",
    [{"messages": COMPLETION}, COMPLETION],
    ids=["messages-container", "bare-message-list"],
)
def test_default_hook_trains_the_observation_rows_the_quote_accepted(monkeypatch, output):
    rows = [{"input": "click ok", "output": output}]

    worker, quote = _prepare_both(rows, monkeypatch)

    assert worker.rows[0]["input_ids"] == quote.rows[0]["input_ids"]
    assert worker.rows[0]["loss_mask"] == quote.rows[0]["loss_mask"]
    assert _pad_runs(worker.rows[0]["input_ids"]) == [_pads(64), _pads(48)]
    assert _supervised_text(worker.rows[0]) == "ACT1<|im_end|>ACT2<|im_end|>ACT3<|im_end|>"
    assert worker.profile.real_tokens_per_epoch == quote.profile.real_tokens_per_epoch


@pytest.mark.parametrize(
    ("completion", "match"),
    [
        (
            [{"role": "assistant", "content": "ACT1"}, {"role": "user", "content": 5}],
            r"messages\[1\]\.content must be a string, a content-block list, or null",
        ),
        (
            [{"role": "assistant", "content": "ACT1"}, {"content": "page"}],
            r"messages\[1\]\.role must be a non-empty string",
        ),
        (
            [{"role": "assistant", "content": None}],
            r"messages\[0\]\.content must be a string or a content-block list",
        ),
        (
            [{"role": "assistant", "content": "ACT1"}, {"role": "user", "content": ["page"]}],
            r"messages\[1\]\.content list must be a non-empty list of content-block objects",
        ),
        (
            [
                {"role": "assistant", "content": "ACT1"},
                _observation("page", 32, "green"),
                {
                    "role": "assistant",
                    "content": [{"type": "image_url", "image_url": {"url": _image_uri(32, "red")}}],
                },
            ],
            "not supported in assistant messages",
        ),
    ],
    ids=[
        "scalar-content",
        "missing-role",
        "null-without-tool-calls",
        "non-block-list",
        "assistant-image",
    ],
)
def test_quote_refuses_exactly_what_the_worker_refuses(monkeypatch, completion, match):
    rows = [{"input": "click ok", "output": {"messages": completion}}]
    monkeypatch.setattr(image_tokens, "load_image_geometry", lambda *_args: _GEOMETRY)
    common = {
        "tokenizer_loader": lambda _model, _revision: _ChatMlTokenizer(),
        "producer_version": "1.2.3",
        "packing_support": lambda _model, _revision: ("multimodal", False),
    }

    with pytest.raises(ValueError, match=match):
        prepare_sft_workload(_spec(), _quote_env(rows), require_processor=False, **common)
    with pytest.raises(ValueError, match=match):
        prepare_sft_workload(
            _spec(),
            _worker_env(rows),
            processor_loader=lambda _model, _revision: _ExpandingProcessor(),
            **common,
        )


def test_an_authored_sft_completion_override_still_owns_the_target():
    class _Override(_DefaultHookTrajectories):
        def sft_completion(self, example):
            return [{"role": "assistant", "content": "from the hook"}]

    env = FreesoloEnvironment(
        _Override(), "team/browser", source=[{"input": "q", "output": {"messages": COMPLETION}}]
    )

    assert env.sft_completion(env.dataset()[0]) == [
        {"role": "assistant", "content": "from the hook"}
    ]


def test_scalar_outputs_still_use_the_sdk_default_hook():
    env = _worker_env([{"input": "q", "output": {"answer": 1}}])

    # the sdk serializes a structured scalar as json (not a python repr); its hook keeps that case
    (message,) = env.sft_completion(env.dataset()[0])
    assert message["role"] == "assistant"
    assert json.loads(message["content"]) == {"answer": 1}


# ---- reproduction: the miniwob-fable sft package (flash-1795014931/-14940/-15157, 2026-11-18) ----
#
# environment.py is the researcher's MiniwobSFT verbatim minus its later workaround override; the
# row is the shape its build_sft.py wrote (system + first observation in prompt_messages, the rest
# of the episode in output.messages). before the fix the real preview profiler quoted this package
# and the real worker data path then failed on the gpu with
# "ValueError: Expected output messages[1].content must be a string or null".

_MINIWOB_ENVIRONMENT = '''"""MiniWoB browser-action SFT environment: rows are complete authored transcripts."""

from __future__ import annotations

import json
from pathlib import Path

from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult

DEFAULT_DATASET_PATH = Path(__file__).parent / "dataset" / "train.jsonl"


def load_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class MiniwobSFT(EnvironmentSingleTurn):
    dataset = load_jsonl(DEFAULT_DATASET_PATH)

    def build_prompt_messages(self, example: TaskExample, prompt_text: str):
        return [{"role": "user", "content": example.input}]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        return RewardResult(score=0.0, threshold=1.0)


def load_environment(dataset_path: str | None = None, **kwargs) -> MiniwobSFT:
    env = MiniwobSFT()
    if dataset_path:
        env.dataset = load_jsonl(dataset_path)
    return env
'''


def _miniwob_user(obs: str, size: int, color: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": obs},
            {"type": "image_url", "image_url": {"url": _image_uri(size, color), "detail": "high"}},
        ],
    }


def _miniwob_row() -> dict:
    return {
        "input": "click-button-sequence-0",
        "prompt_messages": [
            {"role": "system", "content": "You control a browser. Reply with one action."},
            _miniwob_user("Task: click ONE, then TWO.", 32, "red"),
        ],
        "output": {
            "messages": [
                {"role": "assistant", "content": 'click(ref="ONE")'},
                _miniwob_user("Clicked ONE.", 64, "green"),
                {"role": "assistant", "content": 'click(ref="TWO")'},
            ]
        },
        "metadata": {"task": "click-button-sequence", "seed": 0, "source": "demo"},
    }


def _miniwob_package(tmp_path) -> Path:
    package = tmp_path / "miniwob-sft"
    (package / "dataset").mkdir(parents=True)
    (package / "environment.py").write_text(_MINIWOB_ENVIRONMENT, encoding="utf-8")
    (package / "dataset" / "train.jsonl").write_text(
        json.dumps(_miniwob_row()) + "\n", encoding="utf-8"
    )
    return package / "environment.py"


def test_miniwob_fable_package_previews_and_trains_identically(tmp_path, monkeypatch):
    from flash.engine.profiling.dataset_profile import profile_packaged_sft_dataset
    from flash.envs.loading.base import load_environment

    monkeypatch.setattr(image_tokens, "load_image_geometry", lambda *_args: _GEOMETRY)
    entrypoint = _miniwob_package(tmp_path)
    spec = replace(
        _spec(),
        environment=EnvironmentSpec(id=str(entrypoint), resolved_sha="b" * 40),
    )
    spec = replace(
        spec,
        workload_profile_input_digest=sft_profile_input_digest(
            spec, tokenizer_revision=spec.model_revision, producer_version="1.2.3"
        ),
    )
    tokenizer_loader = lambda _model, _revision: _ChatMlTokenizer()  # noqa: E731
    packing_support = lambda _model, _revision: ("multimodal", False)  # noqa: E731

    # preview / submit: the control-plane quote (runner/lifecycle/preparation.py)
    quoted = profile_packaged_sft_dataset(
        spec,
        producer_version="1.2.3",
        tokenizer_loader=tokenizer_loader,
        packing_support=packing_support,
    )
    assert quoted.retained_examples == 1
    assert quoted.architecture_mode == "multimodal"

    # gpu worker: the environment executed, then the same preparation the sft runner calls
    env = load_environment(spec.environment.id, spec.environment.params, "b" * 40)
    trained = prepare_sft_workload(
        spec,
        env,
        tokenizer_loader=tokenizer_loader,
        processor_loader=lambda _model, _revision: _ExpandingProcessor(),
        producer_version="1.2.3",
        image_dir=str(tmp_path / "images"),
        packing_support=packing_support,
    )

    row = trained.rows[0]
    assert _pad_runs(row["input_ids"]) == [_pads(32), _pads(64)]
    assert _supervised_text(row) == 'click(ref="ONE")<|im_end|>click(ref="TWO")<|im_end|>'
    assert trained.profile.real_tokens_per_epoch == quoted.real_tokens_per_epoch
    assert trained.profile.supervised_tokens_per_epoch == quoted.supervised_tokens_per_epoch
