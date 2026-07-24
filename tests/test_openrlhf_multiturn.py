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


# --- parity plan #8b: child executor response-mask (CPU-proved byte-identical to TRL) -----------

from flash.engine.worker.openrlhf_multiturn import (
    FlashBridgeAgentInstance,
    assemble_multiturn_rollout,
    build_response_mask,
    flash_multiturn_agent_instance_cls,
)


def _drive_and_collect(env, tok, actions, **over):
    """Run the #677 parent driver over ``actions``; collect prompt ids, the per-turn
    ``{action_ids, glue_ids}`` the child executor would assemble, and the TRL-exact per-token env
    mask (0 prompt, 1 assistant, 0 glue) taken straight from the driver's own outputs. Because the
    driver reproduces TRL glue/masking byte-for-byte (proved above), this env mask *is* the frozen
    TRL fixture the response mask must match.
    """
    d = _driver(env, tok, **over)
    prompt_ids = d.initial_observation()["token_ids"]
    turns: list[dict] = []
    trl_mask = [0] * len(prompt_ids)
    for act in actions:
        obs, done = d.step(act)
        glue = obs["token_ids"]
        assert obs["env_mask"] == [0] * len(glue)  # driver invariant: glue is never trainable
        trl_mask.extend([1] * len(act["token_ids"]))
        if done:
            turns.append({"action_ids": act["token_ids"], "glue_ids": []})
            break
        trl_mask.extend([0] * len(glue))
        turns.append({"action_ids": act["token_ids"], "glue_ids": glue})
    return prompt_ids, turns, trl_mask, d


def test_build_response_mask_is_one_inside_action_ranges_only():
    # ranges are half-open [start, end); everything outside stays masked out of the loss.
    assert build_response_mask(6, [(1, 3), (4, 6)]) == [0, 1, 1, 0, 1, 1]
    assert build_response_mask(4, []) == [0, 0, 0, 0]
    # clamps to bounds without raising, so a truncated final range can never index past the stream.
    assert build_response_mask(3, [(1, 99)]) == [0, 1, 1]
    with pytest.raises(ValueError, match="non-negative"):
        build_response_mask(-1, [])


def test_response_mask_is_byte_identical_to_trl_env_mask_multiturn():
    tok = FakeTokenizer()
    imend = tok._id("<|im_end|>")
    prompt_ids, turns, trl_mask, _ = _drive_and_collect(
        FakeEnv(replies=2),
        tok,
        [
            {"text": "a1", "token_ids": [1, 2, imend]},
            {"text": "a2", "token_ids": [3, imend]},
            {"text": "a3", "token_ids": [4, imend]},
        ],
    )
    out = assemble_multiturn_rollout(prompt_ids, turns)
    # the assembled stream is prompt || (action||glue)* and the mask is TRL's env mask exactly.
    assert len(out["token_ids"]) == len(trl_mask)
    assert out["response_mask"] == trl_mask
    # and the trained tokens are precisely the assistant spans, nothing before or between them.
    assert sum(out["response_mask"]) == sum(len(t["action_ids"]) for t in turns)
    for start, end in out["action_ranges"]:
        assert all(out["response_mask"][i] == 1 for i in range(start, end))


def test_response_mask_single_turn_degenerate_matches_trl():
    tok = FakeTokenizer()
    imend = tok._id("<|im_end|>")
    # replies=0 -> env_reply returns [] after the first turn, so this collapses to one span.
    prompt_ids, turns, trl_mask, _ = _drive_and_collect(
        FakeEnv(replies=0), tok, [{"text": "only", "token_ids": [7, 8, 9, imend]}]
    )
    assert len(turns) == 1
    assert turns[0]["glue_ids"] == []
    out = assemble_multiturn_rollout(prompt_ids, turns)
    assert out["response_mask"] == trl_mask
    assert out["action_ranges"] == [(len(prompt_ids), len(prompt_ids) + 4)]


def test_response_mask_budget_truncated_final_turn_matches_trl():
    tok = FakeTokenizer()
    # completion_budget stops the episode on the turn that meets it, with no trailing glue.
    prompt_ids, turns, trl_mask, _ = _drive_and_collect(
        FakeEnv(replies=9),
        tok,
        [{"text": "a1", "token_ids": [1, 2]}, {"text": "a2", "token_ids": [3, 4]}],
        max_turns=99,
        completion_budget=2,
    )
    assert turns[-1]["glue_ids"] == []  # truncation leaves the final turn glue-free
    out = assemble_multiturn_rollout(prompt_ids, turns)
    assert out["response_mask"] == trl_mask


def test_assemble_rejects_trailing_glue_on_final_turn():
    with pytest.raises(ValueError, match="final multi-turn turn must not carry inter-turn glue"):
        assemble_multiturn_rollout([0, 1], [{"action_ids": [2], "glue_ids": [3]}])


def test_bridge_agent_instance_is_token_exact_and_leaks_no_secret():
    import asyncio

    tok = FakeTokenizer()
    imend = tok._id("<|im_end|>")
    env = FakeEnv(replies=1)
    env._secret = "do-not-leak"  # env credential must stay parent-side, never cross the bridge
    driver = _driver(env, tok)

    class _FakeBridgeClient:
        """In-process stand-in for the authenticated localhost bridge client (no HTTP, no Ray)."""

        def reset(self, states):
            obs = driver.initial_observation()
            return {"session_id": "s1", "lease": "L1", "token_ids": obs["token_ids"]}

        def step(self, session_id, lease, action):
            assert (session_id, lease) == ("s1", "L1")
            obs, done = driver.step(action)
            return obs, done, (driver.score() if done else 0.0)

    inst = FlashBridgeAgentInstance(_FakeBridgeClient())
    opened = asyncio.run(inst.reset({"q": "hi"}))
    assert opened["token_ids"], "reset must return the prompt tokens the first generation conditions on"

    fb0 = asyncio.run(inst.step({"action": {"text": "a1", "token_ids": [1, imend]}}))
    assert set(fb0) == {"environment_feedback", "done", "rewards", "scores"}
    ef0 = fb0["environment_feedback"]
    assert ef0["env_mask"] == [0] * len(ef0["token_ids"])  # bridge glue is never trainable
    assert fb0["done"] is False

    fb1 = asyncio.run(inst.step({"action": {"text": "a2", "token_ids": [2, imend]}}))
    assert fb1["done"] is True
    assert fb1["rewards"] == pytest.approx(2.0)  # replies=1 -> exactly two assistant turns
    for payload in (opened, fb0, fb1):
        assert "do-not-leak" not in repr(payload)


def test_agent_instance_cls_is_child_only():
    # OpenRLHF lives only in the cu13 child image; the factory must fail loudly off-child, so the
    # module still imports (and every mask test above runs) on the CPU parent with no OpenRLHF.
    with pytest.raises(ImportError):
        flash_multiturn_agent_instance_cls()
