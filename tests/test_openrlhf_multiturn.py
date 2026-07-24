"""CPU parity/contract tests for pure multi-turn text GRPO on OpenRLHF (parity plan #8).

No torch/OpenRLHF/vLLM: these prove the parent-side env->bridge session driver produces byte-exact
TRL inter-turn glue, seam de-duplication, per-token environment masks, episode reward, and the
OpenRLHF ``action_ranges`` span math. The live vLLM multi-turn ``AgentExecutor`` + response-mask on a
real rollout is the preregistered GPU smoke follow-up (parity ladder step 4).
"""

import re

import pytest

from flash.engine.worker.openrlhf_multiturn import (
    FlashEnvSessionDriver,
    build_multiturn_action_ranges,
)

_SPECIALS = ["<|im_start|>", "<|im_end|>", "\n"]
_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in _SPECIALS) + r"|\s+)")


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids  # HF returns a flat list for a single string


class FakeTokenizer:
    """Special-token-aware fake: ``<|im_end|>`` is exactly one token so seam de-dup behaves like Qwen."""

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        return self._vocab.setdefault(tok, len(self._vocab) + 10)

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, enable_thinking):
        assert tokenize is False
        out = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)

    def __call__(self, text, *, add_special_tokens=False, return_tensors=None):
        return _Enc([self._id(t) for t in _SPLIT.split(text) if t])


class FakeEnv:
    """Deterministic multi-turn env; scores by assistant turn count."""

    multi_turn = True

    def __init__(self, replies=2):
        self._replies = replies
        self.closed = False

    def new_rollout_state(self, example):
        return {"prompt": [{"role": "user", "content": example["q"]}], "n": 0}

    def record_model_turn(self, state, text):
        state["n"] += 1

    def rollout_done(self, state, max_turns):
        return False

    def env_reply(self, messages, state):
        if state["n"] > self._replies:
            return []
        return [{"role": "user", "content": f"more {state['n']}"}]

    def reward_from_messages(self, messages, example):
        return float(sum(1 for m in messages if m["role"] == "assistant"))

    def close(self):
        self.closed = True


def _driver(env, tok, **over):
    kw = {"tokenizer": tok, "thinking": False, "max_turns": 8, "completion_budget": None}
    kw.update(over)
    return FlashEnvSessionDriver(env, {"q": "hi"}, **kw)


def test_glue_mask_and_seam_dedup():
    tok = FakeTokenizer()
    imend = tok._id("<|im_end|>")
    d = _driver(FakeEnv(replies=2), tok)
    prompt_ids = d.initial_observation()["token_ids"]
    assert prompt_ids  # first generation is conditioned on the rendered prompt

    obs0, done0 = d.step({"text": "a1", "token_ids": [1, 2, imend]})
    glue0 = obs0["token_ids"]
    assert done0 is False
    assert glue0, "inter-turn glue must be non-empty"
    assert glue0[0] != imend, "seam terminator was not de-duplicated"
    assert obs0["env_mask"] == [0] * len(glue0)  # every glue token is masked out of the loss

    _obs1, done1 = d.step({"text": "a2", "token_ids": [3, imend]})
    assert done1 is False
    _obs2, done2 = d.step({"text": "a3", "token_ids": [4, imend]})
    assert done2 is True  # env_reply returned [] after replies exhausted


def test_action_ranges_span_math():
    # prompt=5; turns of 3 and 2 assistant tokens; a 4-token glue between, none after the final turn.
    assert build_multiturn_action_ranges(5, [3, 2], [4]) == [(5, 8), (12, 14)]
    assert build_multiturn_action_ranges(10, [7], []) == [(10, 17)]  # single-turn -> one span
    with pytest.raises(ValueError, match="glue segment count"):
        build_multiturn_action_ranges(0, [1, 1], [1, 1])  # more glue segments than allowed


def test_reward_is_episode_scalar_and_finite():
    tok = FakeTokenizer()
    d = _driver(FakeEnv(replies=1), tok)
    d.initial_observation()
    for text, ids in [("a1", [1]), ("a2", [2]), ("a3", [3])]:
        _o, done = d.step({"text": text, "token_ids": ids})
        if done:
            break
    r = d.score()
    assert isinstance(r, float)
    assert r == pytest.approx(2.0)  # replies=1 -> exactly 2 assistant turns, one point each


def test_completion_budget_terminates_before_overflow():
    tok = FakeTokenizer()
    d = _driver(FakeEnv(replies=9), tok, max_turns=99, completion_budget=2)
    d.initial_observation()
    _o, done = d.step({"text": "a1", "token_ids": [1, 2]})
    assert done is True  # two assistant tokens already meet the budget -> stop, no glue overflow


def test_env_secret_never_enters_observations():
    """Only ids/mask cross the bridge wire; the env credential stays parent-side in the driver."""
    tok = FakeTokenizer()
    env = FakeEnv(replies=1)
    env._secret = "do-not-leak"  # stand-in for a real env credential kept parent-side
    d = FlashEnvSessionDriver(
        env, {"q": "hi"}, tokenizer=tok, thinking=False, max_turns=8, completion_budget=None
    )
    payloads = [d.initial_observation()]
    for text, ids in [("a1", [1]), ("a2", [2])]:
        obs, done = d.step({"text": text, "token_ids": ids})
        payloads.append(obs)
        if done:
            break
    for p in payloads:
        assert set(p).issubset({"token_ids", "env_mask"})
        assert "do-not-leak" not in repr(p)
    d.close()
    assert env.closed is True
