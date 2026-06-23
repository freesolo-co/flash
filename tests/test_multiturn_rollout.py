"""CPU tests for the multi-turn rollout core (no GPU/tokenizer needed).

Exercises :func:`rollout_one` with a fake chat tokenizer that models role headers + an
end-of-turn token, so the prefix-preserving token alignment + env_mask construction are
verified the same way a real template (Qwen-style <|im_start|>/<|im_end|>) would behave.
"""

from __future__ import annotations

import sys
import types

import pytest

from flash.engine.multiturn_rollout import (
    build_examples_index,
    index_collisions,
    rollout_one,
)

# Fake vocab: role headers, an end-of-turn token, and one token per message "content" key.
HDR = {"user": 100, "assistant": 101, "system": 102}
END = 199
CONTENT = {"u1": 1, "u2": 2, "a1": 90, "GOOD": 91}


def render(messages, add_generation_prompt):
    """Model a prefix-preserving chat template: [header, content, end] per message, and a
    trailing assistant header when priming generation."""
    ids = []
    for m in messages:
        ids.append(HDR[m["role"]])
        ids.append(CONTENT[m["content"]])
        ids.append(END)
    if add_generation_prompt:
        ids.append(HDR["assistant"])
    return ids


class FakeEnv:
    """Minimal MultiTurnEnv-shaped adapter: one env (user) turn, stops after 2 model turns."""

    multi_turn = True

    def new_rollout_state(self, example):
        return {
            "prompt": [{"role": "user", "content": "u1"}],
            "completion": [],
            "answer": example.get("answer"),
        }

    def record_model_turn(self, state, content):
        state["completion"].append({"role": "assistant", "content": content})

    def rollout_done(self, state, max_turns):
        n_model = sum(1 for m in state["completion"] if m["role"] == "assistant")
        return n_model >= 2

    def env_reply(self, messages, state):
        msg = {"role": "user", "content": "u2"}
        state["completion"].append(msg)
        return [msg]

    def reward(self, completion, example, state=None):
        # State-preserving scoring: read the transcript off the rollout state.
        msgs = (state or {}).get("completion") or []
        return 1.0 if any(m.get("content") == "GOOD" for m in msgs) else 0.0


def _generator(turn_texts):
    """generate() that yields one assistant turn per call: content token + END."""
    seq = iter(turn_texts)

    def generate(prefix_ids, max_tokens):
        text = next(seq)
        token_ids = [CONTENT[text], END]
        logprobs = [-0.1, -0.2]
        return token_ids, logprobs, text

    return generate


def test_rollout_one_interleaves_and_masks_env_tokens():
    out = rollout_one(
        example={"answer": "GOOD"},
        active_env=FakeEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        max_turns=10,
        per_turn_max_tokens=64,
    )
    # prompt = [user u1] + assistant header
    assert out["prompt_ids"] == [HDR["user"], CONTENT["u1"], END, HDR["assistant"]]
    # completion = asst1(a1,END) + env(user header,u2,END,asst header) + asst2(GOOD,END)
    assert out["completion_ids"] == [
        CONTENT["a1"],
        END,
        HDR["user"],
        CONTENT["u2"],
        END,
        HDR["assistant"],
        CONTENT["GOOD"],
        END,
    ]
    # env tokens masked 0, model tokens masked 1
    assert out["env_mask"] == [1, 1, 0, 0, 0, 0, 1, 1]
    # all three token-aligned arrays share length
    assert len(out["completion_ids"]) == len(out["logprobs"]) == len(out["env_mask"])
    # only model tokens carry real logprobs (env positions are 0.0 placeholders)
    model_lp = [lp for lp, m in zip(out["logprobs"], out["env_mask"], strict=True) if m == 1]
    assert all(lp != 0.0 for lp in model_lp)
    assert all(out["logprobs"][i] == 0.0 for i, m in enumerate(out["env_mask"]) if m == 0)
    # reward came from the transcript rubric
    assert out["reward"] == 1.0


def test_rollout_one_respects_max_turns():
    # max_turns=1 stops after the first model turn (env never gets to reply).
    out = rollout_one(
        example={"answer": "x"},
        active_env=FakeEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        max_turns=1,
        per_turn_max_tokens=64,
    )
    assert out["env_mask"] == [1, 1]
    assert out["completion_ids"] == [CONTENT["a1"], END]


def test_rollout_one_stops_when_env_has_no_reply():
    class NoReplyEnv(FakeEnv):
        def rollout_done(self, state, max_turns):
            return False  # never done by env

        def env_reply(self, messages, state):
            return []  # ...but nothing to add -> rollout must stop

    out = rollout_one(
        example={"answer": "x"},
        active_env=NoReplyEnv(),
        render=render,
        generate=_generator(["a1", "a1"]),
        max_turns=10,
        per_turn_max_tokens=64,
    )
    assert out["env_mask"] == [1, 1]


def test_rollout_one_accepts_messages_only_initial_state():
    class MessagesOnlyEnv(FakeEnv):
        def new_rollout_state(self, example):
            return {
                "messages": [{"role": "user", "content": "u1"}],
                "completion": [],
                "answer": example.get("answer"),
            }

    out = rollout_one(
        example={"answer": "x"},
        active_env=MessagesOnlyEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        max_turns=1,
        per_turn_max_tokens=64,
    )
    assert out["prompt_ids"] == [HDR["user"], CONTENT["u1"], END, HDR["assistant"]]
    assert out["env_mask"] == [1, 1]


def test_non_prefix_template_raises():
    # A non-prefix-preserving template would mis-mask model vs env tokens, so the rollout must
    # fail loudly rather than silently mis-train.
    warnings: list[str] = []

    def bad_render(messages, add_generation_prompt):
        ids = render(messages, add_generation_prompt)
        return list(reversed(ids))  # earlier tokens shift -> prefix invariant broken

    with pytest.raises(ValueError, match="prefix-preserving"):
        rollout_one(
            example={"answer": "GOOD"},
            active_env=FakeEnv(),
            render=bad_render,
            generate=_generator(["a1", "GOOD"]),
            max_turns=10,
            per_turn_max_tokens=64,
            on_warn=warnings.append,
        )
    assert warnings


def test_engine_max_len_caps_total_completion():
    # With a tight engine budget, the rollout stops before the env turn so completion stays
    # within prompt + budget (no overflow of the next generate()).
    out = rollout_one(
        example={"answer": "GOOD"},
        active_env=FakeEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        max_turns=10,
        per_turn_max_tokens=64,
        engine_max_len=len(render([{"role": "user", "content": "u1"}], True)) + 10,
    )
    # prompt + only the first assistant turn (2 tokens), then the budget halts it
    assert out["env_mask"] == [1, 1]


# ---------------------------------------------------------------------------
# build_rollout_func: colocate-engine wake/sleep + prompt-id guard (issue #162)
# ---------------------------------------------------------------------------


class _FakeTok:
    """Minimal tokenizer: render to text then map each char to its ordinal id."""

    def __init__(self, force_ids=None):
        self._force_ids = force_ids

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids = self._force_ids if self._force_ids is not None else [ord(c) for c in text]
        return types.SimpleNamespace(input_ids=ids)


class _OneTurnEnv:
    """multi-turn-shaped env that stops after the first model turn (no env reply)."""

    multi_turn = True

    def new_rollout_state(self, example):
        return {"prompt": [{"role": "user", "content": "hi"}], "completion": []}

    def record_model_turn(self, state, content):
        state["completion"].append({"role": "assistant", "content": content})

    def rollout_done(self, state, max_turns):
        return True

    def env_reply(self, messages, state):
        return []

    def reward(self, completion, example, state=None):
        return 0.5


class _FakeEngine:
    def __init__(self, vocab=1000):
        self.events = []
        self.llm_engine = types.SimpleNamespace(
            model_config=types.SimpleNamespace(get_vocab_size=lambda: vocab)
        )

    def wake_up(self, tags=None):
        self.events.append(("wake", tuple(tags or [])))

    def sleep(self, level=None):
        self.events.append(("sleep", level))

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        self.events.append(("generate", tuple(prompts[0]["prompt_token_ids"])))
        comp = types.SimpleNamespace(token_ids=[5, 6], logprobs=None, text="ok")
        return [types.SimpleNamespace(outputs=[comp])]


def _fake_trainer(engine, *, sleep_mode):
    return types.SimpleNamespace(
        vllm_generation=types.SimpleNamespace(llm=engine),
        num_generations=1,
        args=types.SimpleNamespace(vllm_enable_sleep_mode=sleep_mode),
    )


@pytest.fixture
def _stub_vllm():
    """Stub the GPU-only ``vllm.SamplingParams`` so build_rollout_func imports on CPU."""
    prev = sys.modules.get("vllm")
    mod = types.ModuleType("vllm")
    mod.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    sys.modules["vllm"] = mod
    try:
        yield
    finally:
        if prev is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = prev


def _build(tok):
    from flash.engine.multiturn_rollout import build_rollout_func

    return build_rollout_func(
        active_env=_OneTurnEnv(),
        tok=tok,
        examples_by_key={},
        max_completion=8,
        max_turns=4,
        temperature=0.7,
        top_p=1.0,
        stop=None,
        thinking=False,
        engine_max_len=None,
    )


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_wakes_kv_cache_around_generation_when_sleep_mode():
    # The bug (#162): TRL's rollout_func path wakes only the weights, never the KV cache, so the
    # first engine.generate() faults. The fix wakes tags=["kv_cache"] BEFORE any generate and
    # re-sleeps AFTER the batch — assert that exact ordering.
    engine = _FakeEngine()
    rf = _build(_FakeTok())
    out = rf([[{"role": "user", "content": "hi"}]], _fake_trainer(engine, sleep_mode=True))
    kinds = [e[0] for e in engine.events]
    assert kinds == ["wake", "generate", "sleep"]
    assert engine.events[0] == ("wake", ("kv_cache",))
    assert engine.events[-1] == ("sleep", 2)
    assert out["reward"] == [0.5]


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_no_wakesleep_when_sleep_mode_off():
    engine = _FakeEngine()
    rf = _build(_FakeTok())
    rf([[{"role": "user", "content": "hi"}]], _fake_trainer(engine, sleep_mode=False))
    assert [e[0] for e in engine.events] == ["generate"]


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_re_sleeps_even_if_generation_raises():
    # finally: the engine must return to the offloaded state even when a rollout throws, or the
    # next step inherits a half-woken engine.
    engine = _FakeEngine(vocab=50)  # ord('h')=104 >= 50 -> guard fires inside generate()
    rf = _build(_FakeTok())
    with pytest.raises(ValueError, match="out-of-range token id"):
        rf([[{"role": "user", "content": "hi"}]], _fake_trainer(engine, sleep_mode=True))
    assert engine.events[0] == ("wake", ("kv_cache",))
    assert engine.events[-1] == ("sleep", 2)


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_guards_empty_prompt():
    engine = _FakeEngine()
    rf = _build(_FakeTok(force_ids=[]))  # tokenizer yields no ids -> empty prompt
    with pytest.raises(ValueError, match="empty prompt"):
        rf([[{"role": "user", "content": "hi"}]], _fake_trainer(engine, sleep_mode=False))


def test_examples_index_and_collisions():
    rows = [
        {"prompt": [{"role": "user", "content": "a"}], "answer": "1"},
        {"prompt": [{"role": "user", "content": "b"}], "answer": "2"},
        {"prompt": [{"role": "user", "content": "a"}], "answer": "3"},  # collides with row 0
    ]
    idx = build_examples_index(rows, lambda r: r["prompt"])
    assert len(idx) == 2  # collision collapsed
    assert index_collisions(rows, lambda r: r["prompt"]) == 1
    # last write wins on collision
    from flash.engine.multiturn_rollout import _prompt_key

    assert idx[_prompt_key([{"role": "user", "content": "a"}])]["answer"] == "3"
