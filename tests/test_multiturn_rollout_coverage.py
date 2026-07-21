"""Additional CPU coverage for the multi-turn rollout core.

Targets the branches the primary suite (``tests/test_multiturn_rollout.py``) leaves uncovered:
the budget/engine-headroom early-exits and error paths in :func:`rollout_one`,
:func:`rollout_one_records` and :func:`rollout_async` (plus their shared ``_advance_after_turn`` /
``_turn_budget`` helpers), the pure-function guards (``_prompt_key`` fallback, ``_LRUCache``
constructor, ``_engine_vocab_size``), and the pure rollout helpers. No torch, vLLM, or GPU code is executed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.engine.multiturn_rollout import (
    _engine_vocab_size,
    _LRUCache,
    _prompt_key,
    rollout_async,
    rollout_one,
    rollout_one_records,
)

# --- Fake prefix-preserving chat scheme (mirrors the sibling test's vocab) ----------------------
HDR = {"user": 100, "assistant": 101, "system": 102}
END = 199
CONTENT = {"u1": 1, "u2": 2, "a1": 90, "GOOD": 91}


def render(messages, add_generation_prompt):
    ids = []
    for m in messages:
        ids.append(HDR[m["role"]])
        ids.append(CONTENT[m["content"]])
        ids.append(END)
    if add_generation_prompt:
        ids.append(HDR["assistant"])
    return ids


def env_glue(env_messages):
    return render(env_messages, True)


def realistic_env_glue(env_messages):
    """Glue that LEADS with the turn terminator (END), like build_rollout_func's real derivation."""
    return [END, *render(env_messages, True)]


PROMPT_LEN = len(render([{"role": "user", "content": "u1"}], True))  # == 4


def _generator(turn_texts):
    """rollout_one generate(): one (ids, logprobs, text) turn per call."""
    seq = iter(turn_texts)

    def generate(prefix_ids, max_tokens):
        text = next(seq)
        return [CONTENT[text], END], [-0.1, -0.2], text

    return generate


def _det_generate(prefix_ids, max_tokens):
    return [CONTENT["a1"], END], [-0.1, -0.2], "a1"


def _gen_obj(completion_ids, completion_text, *, truncated=False, skip=False):
    """OPD _GenResult stand-in: the four attributes rollout_one_records reads."""
    return SimpleNamespace(
        completion_ids=completion_ids,
        completion_text=completion_text,
        truncated=truncated,
        skip=skip,
    )


def _records_generator(turns):
    seq = iter(turns)

    def generate(prefix_ids, max_new):
        return next(seq)

    return generate


class FakeEnv:
    """One env (user) turn; stops after 2 assistant turns."""

    multi_turn = True

    def new_rollout_state(self, example):
        return {"prompt": [{"role": "user", "content": "u1"}], "completion": []}

    def record_model_turn(self, state, content):
        state["completion"].append({"role": "assistant", "content": content})

    def rollout_done(self, state, max_turns):
        return sum(1 for m in state["completion"] if m["role"] == "assistant") >= 2

    def env_reply(self, messages, state):
        msg = {"role": "user", "content": "u2"}
        state["completion"].append(msg)
        return [msg]

    def reward(self, completion, example, state=None):
        return 3.0


class _VarTurnEnv:
    """Stops after a per-example number of model turns; reward == #model turns."""

    multi_turn = True

    def new_rollout_state(self, example):
        return {
            "prompt": [{"role": "user", "content": "u1"}],
            "completion": [],
            "max_model": example["max_model"],
        }

    def record_model_turn(self, state, content):
        state["completion"].append({"role": "assistant", "content": content})

    def rollout_done(self, state, max_turns):
        return sum(1 for m in state["completion"] if m["role"] == "assistant") >= state["max_model"]

    def env_reply(self, messages, state):
        msg = {"role": "user", "content": "u2"}
        state["completion"].append(msg)
        return [msg]

    def reward(self, completion, example, state=None):
        return float(
            sum(1 for m in (state or {}).get("completion", []) if m["role"] == "assistant")
        )


class _NoPromptEnv:
    """new_rollout_state omits both prompt and messages -> the drivers must raise KeyError."""

    multi_turn = True

    def new_rollout_state(self, example):
        return {"completion": []}

    def record_model_turn(self, state, content):  # pragma: no cover - never reached
        pass

    def rollout_done(self, state, max_turns):  # pragma: no cover
        return True

    def env_reply(self, messages, state):  # pragma: no cover
        return []

    def reward(self, completion, example, state=None):  # pragma: no cover
        return 0.0


class _DoneAfterReplyEnv:
    """rollout_done flips True only AFTER env_reply runs (exercises the post-reply break)."""

    multi_turn = True

    def new_rollout_state(self, example):
        return {"prompt": [{"role": "user", "content": "u1"}], "completion": [], "phase": 0}

    def record_model_turn(self, state, content):
        state["completion"].append({"role": "assistant", "content": content})

    def rollout_done(self, state, max_turns):
        return state["phase"] >= 1

    def env_reply(self, messages, state):
        state["phase"] = 1
        return [{"role": "user", "content": "u2"}]

    def reward(self, completion, example, state=None):
        return 5.0


def _fake_async_engine(gen):
    """submit/poll/busy over an in-memory queue; poll finishes ALL pending per call."""
    pending = []

    def submit(req_id, prefix_ids, max_tokens, initial, images):
        pending.append((req_id, list(prefix_ids), max_tokens))

    def poll():
        if not pending:
            return []
        batch = pending[:]
        pending.clear()
        events = []
        for rid, ids, mt in batch:
            completion_ids, logprobs, text = gen(ids, mt)
            events.append((rid, completion_ids, logprobs, text))
        return events

    def busy():
        return bool(pending)

    return submit, poll, busy


def _run_async(examples, active_env, **kw):
    submit, poll, busy = _fake_async_engine(_det_generate)
    return rollout_async(
        examples=examples,
        active_env=active_env,
        render=render,
        submit=submit,
        poll=poll,
        busy=busy,
        abort=lambda ids: None,
        env_glue=env_glue,
        max_turns=kw.get("max_turns", 8),
        per_turn_max_tokens=kw.get("per_turn_max_tokens", 8),
        engine_max_len=kw.get("engine_max_len"),
    )


# --- pure-function guards -----------------------------------------------------------------------


def test_prompt_key_falls_back_to_str_on_unserializable_prompt():
    # A dict with mixed-type keys can't be json-sorted (sort_keys compares int vs str -> TypeError);
    # _prompt_key must swallow it and fall back to str(), never propagate the encoder error.
    weird = {1: "a", "b": "c"}
    assert _prompt_key(weird) == str(weird)


def test_lru_cache_rejects_nonpositive_maxsize():
    with pytest.raises(ValueError, match="maxsize must be positive"):
        _LRUCache(0)
    with pytest.raises(ValueError, match="maxsize must be positive"):
        _LRUCache(-3)


def test_engine_vocab_size_variants():
    # engine with no llm_engine -> AttributeError swallowed -> None
    assert _engine_vocab_size(SimpleNamespace()) is None

    # get_vocab_size present and working -> used directly
    mc_ok = SimpleNamespace(get_vocab_size=lambda: 999)
    assert (
        _engine_vocab_size(SimpleNamespace(llm_engine=SimpleNamespace(model_config=mc_ok))) == 999
    )

    # get_vocab_size raises -> fall through to hf_text_config.vocab_size
    def _boom():
        raise RuntimeError("vocab probe blew up")

    mc_hf = SimpleNamespace(get_vocab_size=_boom, hf_text_config=SimpleNamespace(vocab_size=1234))
    assert (
        _engine_vocab_size(SimpleNamespace(llm_engine=SimpleNamespace(model_config=mc_hf))) == 1234
    )

    # no getters and no hf_text_config -> None (never raises)
    mc_bare = SimpleNamespace()
    assert (
        _engine_vocab_size(SimpleNamespace(llm_engine=SimpleNamespace(model_config=mc_bare)))
        is None
    )


# --- rollout_one edge branches ------------------------------------------------------------------


def test_rollout_one_raises_without_prompt_or_messages():
    with pytest.raises(KeyError, match="must include prompt or messages"):
        rollout_one(
            example={},
            active_env=_NoPromptEnv(),
            render=render,
            generate=_generator(["a1"]),
            env_glue=env_glue,
            max_turns=4,
            per_turn_max_tokens=8,
        )


def test_rollout_one_budget_edge_breaks():
    # (a) engine budget already exhausted by the prompt -> loop breaks before any generation.
    out_zero = rollout_one(
        example={},
        active_env=FakeEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=8,
        engine_max_len=PROMPT_LEN + 8,  # token_budget == 0 -> remaining <= 0 on entry
    )
    assert out_zero["completion_ids"] == []
    assert out_zero["env_mask"] == []
    assert out_zero["reward"] == 3.0

    # (b) first turn fits but the inter-turn glue would overflow the budget -> stop before the glue.
    out_glue = rollout_one(
        example={},
        active_env=FakeEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=8,
        engine_max_len=PROMPT_LEN + 8 + 4,  # token_budget == 4: fits the 2-token turn, not +4 glue
    )
    # only the first assistant turn survives (a1, END); no env/glue tokens appended.
    assert out_glue["completion_ids"] == [CONTENT["a1"], END]
    assert out_glue["env_mask"] == [1, 1]


def test_rollout_one_stops_when_env_finishes_after_reply():
    # rollout_done is False before env_reply (so the env DOES reply) but True right after it -> the
    # loop must stop without appending glue for a next model turn that will never happen.
    out = rollout_one(
        example={},
        active_env=_DoneAfterReplyEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        env_glue=realistic_env_glue,
        max_turns=8,
        per_turn_max_tokens=8,
    )
    assert out["completion_ids"] == [CONTENT["a1"], END]  # no glue appended after the env finished
    assert out["env_mask"] == [1, 1]
    assert out["reward"] == 5.0


# --- rollout_one_records edge branches ----------------------------------------------------------


def test_rollout_one_records_raises_without_prompt_or_messages():
    with pytest.raises(KeyError, match="must include prompt or messages"):
        rollout_one_records(
            example={},
            active_env=_NoPromptEnv(),
            render=render,
            generate=_records_generator([_gen_obj([1], "x")]),
            env_glue=env_glue,
            max_turns=4,
            per_turn_max_tokens=8,
        )


def test_rollout_one_records_budget_and_progress():
    # (a) on_turn_generated fires once per generated turn, and a normal (budgeted) run records both.
    ticks = {"n": 0}

    def _tick():
        ticks["n"] += 1

    recs = rollout_one_records(
        example={},
        active_env=FakeEnv(),
        render=render,
        generate=_records_generator(
            [_gen_obj([CONTENT["a1"], END], "a1"), _gen_obj([CONTENT["GOOD"], END], "GOOD")]
        ),
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=16,
        engine_max_len=100,  # budget non-None but generous -> exercises the min(max_new, remaining) arm
        on_turn_generated=_tick,
    )
    assert len(recs) == 2
    assert ticks["n"] == 2  # one progress ping per generated turn

    # (b) budget already spent by the prompt -> not a single turn is generated / recorded.
    recs_zero = rollout_one_records(
        example={},
        active_env=FakeEnv(),
        render=render,
        generate=_records_generator([_gen_obj([CONTENT["a1"], END], "a1")]),
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=16,
        engine_max_len=PROMPT_LEN + 8,  # token_budget == 0
    )
    assert recs_zero == []

    # (c) the first turn exactly fills the budget -> recorded, then the episode halts at that turn.
    recs_fill = rollout_one_records(
        example={},
        active_env=FakeEnv(),
        render=render,
        generate=_records_generator(
            [_gen_obj([CONTENT["a1"], END], "a1"), _gen_obj([CONTENT["GOOD"], END], "GOOD")]
        ),
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=16,
        engine_max_len=PROMPT_LEN + 8 + 2,  # token_budget == 2 == one 2-token turn
    )
    assert len(recs_fill) == 1


def test_rollout_one_records_advance_branch_terminations():
    # (a) an empty env reply ends the episode after the recorded turn.
    class _EmptyReplyEnv(FakeEnv):
        def rollout_done(self, state, max_turns):
            return False  # never done by the env itself

        def env_reply(self, messages, state):
            return []

    recs_empty = rollout_one_records(
        example={},
        active_env=_EmptyReplyEnv(),
        render=render,
        generate=_records_generator([_gen_obj([CONTENT["a1"], END], "a1")]),
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=16,
    )
    assert len(recs_empty) == 1

    # (b) rollout_done flips True only after env_reply -> stop without emitting the next turn's glue.
    recs_done = rollout_one_records(
        example={},
        active_env=_DoneAfterReplyEnv(),
        render=render,
        generate=_records_generator(
            [_gen_obj([CONTENT["a1"], END], "a1"), _gen_obj([CONTENT["GOOD"], END], "GOOD")]
        ),
        env_glue=realistic_env_glue,
        max_turns=8,
        per_turn_max_tokens=16,
    )
    assert len(recs_done) == 1

    # (c) the env replies but the glue would overflow the engine budget -> stop before the next turn.
    recs_glue = rollout_one_records(
        example={},
        active_env=FakeEnv(),
        render=render,
        generate=_records_generator(
            [_gen_obj([CONTENT["a1"], END], "a1"), _gen_obj([CONTENT["GOOD"], END], "GOOD")]
        ),
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=16,
        engine_max_len=PROMPT_LEN + 8 + 4,  # token_budget == 4: fits the turn (2) but not +4 glue
    )
    assert len(recs_glue) == 1


# --- rollout_async edge branches ----------------------------------------------------------------


def test_rollout_async_raises_without_prompt_or_messages():
    submit, poll, busy = _fake_async_engine(_det_generate)
    with pytest.raises(KeyError, match="must include prompt or messages"):
        rollout_async(
            examples=[{}],
            active_env=_NoPromptEnv(),
            render=render,
            submit=submit,
            poll=poll,
            busy=busy,
            abort=lambda ids: None,
            env_glue=env_glue,
            max_turns=4,
            per_turn_max_tokens=8,
        )


def test_rollout_async_reward_many_wrong_count_raises():
    class _BadRewardManyEnv(_VarTurnEnv):
        def reward_many(self, items):
            return [1.0]  # deliberately the wrong length for a 2-rollout batch

    with pytest.raises(RuntimeError, match="wrong number of rewards"):
        _run_async([{"max_model": 1}, {"max_model": 1}], _BadRewardManyEnv())


def test_rollout_async_budget_paths():
    # (a) the budget is filled by the first turn -> the rollout stops with exactly that turn.
    out_fill = _run_async([{"max_model": 3}], _VarTurnEnv(), engine_max_len=PROMPT_LEN + 8 + 2)
    assert out_fill[0]["completion_ids"] == [CONTENT["a1"], END]
    assert out_fill[0]["env_mask"] == [1, 1]
    assert out_fill[0]["reward"] == 1.0

    # (b) the prompt already exhausts the budget -> the rollout is skipped in the initial submit loop
    # (no turn generated, empty completion, zero reward) without hanging the worker.
    out_zero = _run_async([{"max_model": 3}], _VarTurnEnv(), engine_max_len=PROMPT_LEN + 8)
    assert out_zero[0]["completion_ids"] == []
    assert out_zero[0]["env_mask"] == []
    assert out_zero[0]["reward"] == 0.0


def test_rollout_async_advance_env_branches():
    # (a) empty env reply terminates the rollout after the first turn.
    class _EmptyReplyEnv(_VarTurnEnv):
        def env_reply(self, messages, state):
            return []

    out_empty = _run_async([{"max_model": 3}], _EmptyReplyEnv())
    assert out_empty[0]["completion_ids"] == [CONTENT["a1"], END]
    assert out_empty[0]["reward"] == 1.0

    # (b) rollout_done flips True after env_reply -> stop with no trailing glue.
    out_done = _run_async([{}], _DoneAfterReplyEnv())
    assert out_done[0]["completion_ids"] == [CONTENT["a1"], END]
    assert out_done[0]["reward"] == 5.0

    # (c) the env replies but the glue would overflow the engine budget -> stop before the next turn.
    out_glue = _run_async([{"max_model": 3}], _VarTurnEnv(), engine_max_len=PROMPT_LEN + 8 + 4)
    assert out_glue[0]["completion_ids"] == [CONTENT["a1"], END]
    assert out_glue[0]["reward"] == 1.0
