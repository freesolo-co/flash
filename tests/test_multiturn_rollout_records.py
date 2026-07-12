"""focused tests for per-turn rollout records."""

from types import SimpleNamespace

import pytest

from flash.engine.multiturn_rollout import make_env_glue, rollout_one_records

# fake vocab: role headers, an end-of-turn token, and one token per message "content" key.
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


def env_glue(env_messages):
    """Inter-turn glue for the fake scheme: the env turn + next generation prompt. (In this
    fake the model's generated tokens already carry the end-of-turn token, so the glue does not
    add a separate close — mirroring how build_rollout_func derives the real glue.)"""
    return render(env_messages, True)


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
        # state-preserving scoring: read the transcript off the rollout state.
        msgs = (state or {}).get("completion") or []
        return 1.0 if any(m.get("content") == "GOOD" for m in msgs) else 0.0


def realistic_env_glue(env_messages):
    """Mirror build_rollout_func's REAL glue: it is the rendered text AFTER the probe assistant
    CONTENT, so it LEADS with the assistant turn's terminator (END). The simpler fake env_glue above
    omits that leading close; this one reproduces the double-terminator condition the dedup must fix."""
    return [END, *render(env_messages, True)]


def _generator(turn_texts):
    """generate() that yields one assistant turn per call: content token + END."""
    seq = iter(turn_texts)

    def generate(prefix_ids, max_tokens):
        text = next(seq)
        token_ids = [CONTENT[text], END]
        logprobs = [-0.1, -0.2]
        return token_ids, logprobs, text

    return generate


class _VarTurnEnv:
    """Multi-turn env that stops after a PER-EXAMPLE number of model turns, so a batch contains
    rollouts of different depths (exercises rollouts finishing at different turns)."""

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
        n_model = sum(1 for m in state["completion"] if m["role"] == "assistant")
        return n_model >= state["max_model"]

    def env_reply(self, messages, state):
        msg = {"role": "user", "content": "u2"}
        state["completion"].append(msg)
        return [msg]

    def reward(self, completion, example, state=None):
        # distinct per rollout -> proves per-rollout scoring survives batching
        return float(
            sum(1 for m in (state or {}).get("completion", []) if m["role"] == "assistant")
        )


def _det_generate(prefix_ids, max_tokens):
    """Deterministic single-turn generation: a pure function of nothing but the call, so running
    rollout_one per example and rollout_async over all examples see byte-identical turns."""
    return [CONTENT["a1"], END], [-0.1, -0.2], "a1"


# --- rollout_one_records: the per-turn record driver opd distils each turn from -----------------


def _gen_obj(completion_ids, completion_text, *, truncated=False, skip=False):
    """A minimal stand-in for OPD's _GenResult: the four attributes rollout_one_records reads."""
    return SimpleNamespace(
        completion_ids=completion_ids,
        completion_text=completion_text,
        gen_tokens=len(completion_ids or []),
        truncated=truncated,
        skip=skip,
    )


def _records_generator(turns):
    """generate(prefix_ids, max_new) -> a _GenResult-shaped object, one per turn (list order)."""
    seq = iter(turns)

    def generate(prefix_ids, max_new):
        return next(seq)

    return generate


def test_rollout_one_records_grows_prefix_with_verbatim_sampled_tokens():
    """The multi-turn OPD prefix MUST be the accumulated on-policy TOKEN STREAM, not a re-render of the
    message history — a re-render would strip prior-turn reasoning on non-round-tripping templates and
    desync the student's loss prefix from what it actually sampled. Assert each turn's prefix_ids grows
    and literally CONTAINS the previous turn's sampled ids + the deduped env glue."""
    env = FakeEnv()  # 1 env (user) turn, stops after 2 assistant turns
    gen = _records_generator(
        [_gen_obj([CONTENT["a1"], END], "a1"), _gen_obj([CONTENT["GOOD"], END], "GOOD")]
    )
    records = rollout_one_records(
        example={"answer": "GOOD"},
        active_env=env,
        render=render,
        generate=gen,
        env_glue=realistic_env_glue,  # LEADS with END -> exercises the seam dedup
        max_turns=8,
        per_turn_max_tokens=16,
    )
    assert len(records) == 2
    p0, p1 = records[0]["prefix_ids"], records[1]["prefix_ids"]
    # turn 0 prefix = the initial rendered prompt (no completion yet).
    assert p0 == render([{"role": "user", "content": "u1"}], True)
    # turn 1 prefix strictly extends turn 0 with the verbatim sampled turn-0 ids, then the env glue with
    # its duplicate leading terminator collapsed: realistic_env_glue([u2]) is [end, *render([u2],true)],
    # and the leading end dups a1's own end so it is dropped, leaving exactly render([u2], true).
    assert p1 == [*p0, CONTENT["a1"], END, *render([{"role": "user", "content": "u2"}], True)]
    assert len(p1) > len(p0)
    # context_messages is the parallel text history the teacher conditions on, growing per turn.
    assert records[0]["context_messages"] == [{"role": "user", "content": "u1"}]
    assert {"role": "assistant", "content": "a1"} in records[1]["context_messages"]


def test_rollout_one_records_truncated_turn_halts_episode():
    """A turn that did not terminate naturally (truncated) is RECORDED (so the caller counts it) but
    ENDS the episode — a student that can't end its turn shouldn't keep generating, and its broken turn
    must not pollute a later turn's context. env_reply is never reached."""
    calls = {"env_reply": 0}

    class _Env(FakeEnv):
        def env_reply(self, messages, state):
            calls["env_reply"] += 1
            return super().env_reply(messages, state)

    env = _Env()
    gen = _records_generator([_gen_obj(None, "", truncated=True)])
    records = rollout_one_records(
        example={"answer": "GOOD"},
        active_env=env,
        render=render,
        generate=gen,
        env_glue=realistic_env_glue,
        max_turns=8,
        per_turn_max_tokens=16,
    )
    assert len(records) == 1
    assert records[0]["gen"].truncated is True
    assert calls["env_reply"] == 0  # halted before the env stepped


def test_make_env_glue_rejects_non_verbatim_template():
    """make_env_glue's probe trick only works when the chat template inserts assistant content
    verbatim; a template that doesn't (so the probe can't be located) must HARD-FAIL, never silently
    produce skewed glue. This is the load-bearing guard that keeps a non-round-tripping model from
    training/serving skew."""

    class _BadTok:
        def apply_chat_template(self, messages, **kw):
            return "template-that-drops-assistant-content"

    glue = make_env_glue(_BadTok(), thinking=False)
    with pytest.raises(ValueError, match="could not uniquely locate its probe"):
        glue([{"role": "user", "content": "obs"}])
