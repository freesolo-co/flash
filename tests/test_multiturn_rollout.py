"""CPU tests for the multi-turn rollout core (no GPU/tokenizer needed).

Exercises :func:`rollout_one` with a fake chat tokenizer that models role headers + an
end-of-turn token, so the prefix-preserving token alignment + env_mask construction are
verified the same way a real template (Qwen-style <|im_start|>/<|im_end|>) would behave.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from flash.engine.multiturn_rollout import (
    RolloutRequestExhaustedError,
    _LRUCache,
    _prompt_key,
    build_examples_index,
    index_collisions,
    make_env_glue,
    resolve_rollout_request_timeout_seconds,
    rollout_async,
    rollout_one,
    rollout_one_records,
)


def test_prompt_key_is_insensitive_to_arrow_null_injection():
    """Dataset.from_list unifies a list<struct> schema and injects key:null on rows that lacked a key
    another row added. The rollout_func example lookup builds the index from the RAW row but looks up
    the Arrow-materialized prompt; the key must match across that null-injection or every example
    falls through to a stub (wrong/zero reward, or a step-0 crash)."""
    raw = [{"role": "user", "content": "u1"}]
    arrow_materialized = [{"role": "user", "content": "u1", "name": None}]  # null injected by Arrow
    assert _prompt_key(raw) == _prompt_key(arrow_materialized)

    index = build_examples_index([{"prompt": raw, "answer": "GOOD"}], lambda r: r["prompt"])
    assert index.get(_prompt_key(arrow_materialized)) == {"prompt": raw, "answer": "GOOD"}


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
        # State-preserving scoring: read the transcript off the rollout state.
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


def test_rollout_one_interleaves_and_masks_env_tokens():
    out = rollout_one(
        example={"answer": "GOOD"},
        active_env=FakeEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        env_glue=env_glue,
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


def test_rollout_dedups_duplicate_turn_terminator_at_glue_seam():
    # With a real-shaped glue (leads with END) and a model turn that also ends in END, the seam must
    # keep EXACTLY ONE terminator (the assistant's own, env_mask=1), not a <END><END> pair that would
    # push every later turn off-distribution and mismatch the single-terminator SFT transcripts.
    out = rollout_one(
        example={"answer": "GOOD"},
        active_env=FakeEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        env_glue=realistic_env_glue,
        max_turns=10,
        per_turn_max_tokens=64,
    )
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
    assert out["env_mask"] == [1, 1, 0, 0, 0, 0, 1, 1]
    ids = out["completion_ids"]
    assert not any(ids[i] == END and ids[i + 1] == END for i in range(len(ids) - 1))
    assert len(out["completion_ids"]) == len(out["logprobs"]) == len(out["env_mask"])


def test_rollout_one_respects_max_turns():
    # max_turns=1 stops after the first model turn (env never gets to reply).
    out = rollout_one(
        example={"answer": "x"},
        active_env=FakeEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        env_glue=env_glue,
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
        env_glue=env_glue,
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
        env_glue=env_glue,
        max_turns=1,
        per_turn_max_tokens=64,
    )
    assert out["prompt_ids"] == [HDR["user"], CONTENT["u1"], END, HDR["assistant"]]
    assert out["env_mask"] == [1, 1]


def test_rollout_one_builds_from_env_glue_without_rerender():
    # The sequence is built from generate() ids + env_glue ids only — never a re-render of the
    # full conversation — so a template that does not round-trip history (e.g. Qwen3's <think>
    # block) stays aligned. Inject an opaque glue and assert it lands verbatim, masked 0,
    # between the model turns. (`render` here is used ONLY for the initial prompt.)
    GLUE = [501, 502, 503]
    out = rollout_one(
        example={"answer": "GOOD"},
        active_env=FakeEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        env_glue=lambda env_msgs: list(GLUE),
        max_turns=10,
        per_turn_max_tokens=64,
    )
    # completion = asst1(a1,END) + GLUE(masked 0) + asst2(GOOD,END)
    assert out["completion_ids"] == [CONTENT["a1"], END, *GLUE, CONTENT["GOOD"], END]
    assert out["env_mask"] == [1, 1, 0, 0, 0, 1, 1]
    assert out["logprobs"][2:5] == [0.0, 0.0, 0.0]  # glue carries placeholder logprobs


def test_engine_max_len_caps_total_completion():
    # With a tight engine budget, the rollout stops before the env turn so completion stays
    # within prompt + budget (no overflow of the next generate()).
    out = rollout_one(
        example={"answer": "GOOD"},
        active_env=FakeEnv(),
        render=render,
        generate=_generator(["a1", "GOOD"]),
        env_glue=env_glue,
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


def _mk_logprobs(token_ids, lps):
    """vLLM's per-position [{token_id: Logprob}] structure (or None when lps is None)."""
    if lps is None:
        return None
    return [
        {tid: types.SimpleNamespace(logprob=lp)} for tid, lp in zip(token_ids, lps, strict=True)
    ]


class _FakeEngine:
    """Step-able fake colocate engine mirroring vLLM's V1 manual loop: ``llm_engine.add_request``
    enqueues a turn, ``step()`` finishes ALL pending requests at once (one decode round) returning a
    RequestOutput per finished request, ``has_unfinished_requests`` reports the queue. Records
    ('wake'|'add'|'step'|'sleep', ...) events in order so the wake/sleep ordering can be asserted."""

    def __init__(self, vocab=1000, gen=None, finish="all"):
        self.events = []
        self._pending = []  # (req_id, prompt_ids, sampling_params)
        self._finish = (
            finish  # "all" -> finish every pending request per step; "one" -> one per step
        )
        self.aborted = []
        # default generation: two tokens, no logprobs, text 'ok' (matches the prior fake)
        self._gen = gen or (lambda ids, sp: ([5, 6], None, "ok"))
        self.llm_engine = types.SimpleNamespace(
            model_config=types.SimpleNamespace(get_vocab_size=lambda: vocab),
            add_request=self._add_request,
            step=self._step,
            has_unfinished_requests=lambda: bool(self._pending),
            abort_request=self._abort_request,
        )

    def _add_request(self, req_id, prompt, sampling_params):
        self.events.append(("add", req_id))
        self._pending.append((req_id, list(prompt["prompt_token_ids"]), sampling_params))

    def _abort_request(self, ids):
        self.aborted.extend(ids)
        drop = set(ids)
        self._pending = [p for p in self._pending if p[0] not in drop]

    def _step(self):
        self.events.append(("step", len(self._pending)))
        if self._finish == "one":
            batch, self._pending = self._pending[:1], self._pending[1:]
        else:
            batch, self._pending = self._pending, []
        outs = []
        for req_id, ids, sp in batch:
            token_ids, lps, text = self._gen(ids, sp)
            comp = types.SimpleNamespace(
                token_ids=token_ids, logprobs=_mk_logprobs(token_ids, lps), text=text
            )
            outs.append(types.SimpleNamespace(request_id=req_id, finished=True, outputs=[comp]))
        return outs

    def wake_up(self, tags=None):
        self.events.append(("wake", tuple(tags or [])))

    def sleep(self, level=None):
        self.events.append(("sleep", level))


def _fake_trainer(engine, *, sleep_mode):
    return types.SimpleNamespace(
        vllm_generation=types.SimpleNamespace(llm=engine),
        num_generations=1,
        args=types.SimpleNamespace(vllm_enable_sleep_mode=sleep_mode),
    )


class _StubStructuredOutputsParams:
    """CPU stand-in for vllm.sampling_params.StructuredOutputsParams; records its kwargs."""

    def __init__(self, **kw):
        self.kwargs = dict(kw)


@pytest.fixture
def _stub_vllm():
    """Stub the GPU-only ``vllm.SamplingParams`` so build_rollout_func imports on CPU."""
    prev = sys.modules.get("vllm")
    prev_sp = sys.modules.get("vllm.sampling_params")
    mod = types.ModuleType("vllm")
    mod.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    sp_mod = types.ModuleType("vllm.sampling_params")
    sp_mod.StructuredOutputsParams = _StubStructuredOutputsParams
    mod.sampling_params = sp_mod
    sys.modules["vllm"] = mod
    sys.modules["vllm.sampling_params"] = sp_mod
    try:
        yield
    finally:
        for name, prev_mod in (("vllm", prev), ("vllm.sampling_params", prev_sp)):
            if prev_mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev_mod


def _build(tok, active_env=None, structured_outputs=None, **kwargs):
    from flash.engine.multiturn_rollout import build_rollout_func

    return build_rollout_func(
        active_env=active_env or _OneTurnEnv(),
        tok=tok,
        examples_by_key={},
        max_completion=8,
        max_turns=4,
        temperature=0.7,
        top_p=1.0,
        stop=None,
        thinking=False,
        engine_max_len=kwargs.pop("engine_max_len", None),
        structured_outputs=structured_outputs,
        **kwargs,
    )


class _TwoTurnEnv(_OneTurnEnv):
    """Replies once (so the rollout must derive env_glue) before stopping."""

    def rollout_done(self, state, max_turns):
        # not done after the first model turn -> env_reply + env_glue run
        return sum(1 for m in state["completion"] if m["role"] == "assistant") >= 2

    def env_reply(self, messages, state):
        msg = {"role": "user", "content": "result"}
        state["completion"].append(msg)
        return [msg]


class _ProbeDropTok(_FakeTok):
    """A chat template that does NOT insert assistant content verbatim (drops the probe)."""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        return "<no-probe-in-this-template>"


@pytest.mark.usefixtures("_stub_vllm")
def test_env_glue_fails_loud_when_template_drops_probe():
    # If a template doesn't round-trip the probe, env_glue must raise a CLEAR error (not a bare
    # "substring not found") so the failure points at glue derivation / template behavior.
    rf = _build(_ProbeDropTok(), active_env=_TwoTurnEnv())
    with pytest.raises(ValueError, match="could not uniquely locate its probe"):
        rf([[{"role": "user", "content": "hi"}]], _fake_trainer(_FakeEngine(), sleep_mode=False))


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_wakes_kv_cache_around_generation_when_sleep_mode():
    # The bug (#162): TRL's rollout_func path wakes only the weights, never the KV cache, so the
    # first decode faults. The fix wakes tags=["kv_cache"] BEFORE any add_request/step and
    # re-sleeps AFTER the whole batch — assert that exact ordering around the continuous-batch loop.
    engine = _FakeEngine()
    rf = _build(_FakeTok())
    out = rf([[{"role": "user", "content": "hi"}]], _fake_trainer(engine, sleep_mode=True))
    kinds = [e[0] for e in engine.events]
    assert kinds == ["wake", "add", "step", "sleep"]
    assert engine.events[0] == ("wake", ("kv_cache",))
    assert engine.events[-1] == ("sleep", 2)
    assert out["reward"] == [0.5]


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_returns_one_completion_per_prompt():
    # TRL's RepeatSampler already repeats each unique prompt num_generations times before calling
    # rollout_func, and expects exactly len(prompts) completions back. Producing num_generations
    # PER prompt would over-generate and trip a CUDA device-side assert in shuffle_sequence_dict.
    engine = _FakeEngine()
    rf = _build(_FakeTok())
    prompts = [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
        [{"role": "user", "content": "c"}],
    ]
    trainer = _fake_trainer(engine, sleep_mode=False)
    trainer.num_generations = 4  # must be IGNORED by rollout_func (the sampler already repeated)
    out = rf(prompts, trainer)
    assert len(out["completion_ids"]) == len(prompts) == 3
    assert len(out["reward"]) == 3
    assert len(out["prompt_ids"]) == 3


def _fake_async_engine(gen, *, one_at_a_time=False, lifo=False):
    """submit/poll/busy over an in-memory queue, for CPU-testing rollout_async. By default poll()
    finishes ALL pending requests (one decode round); ``one_at_a_time`` finishes a single request
    per poll (``lifo`` -> most-recently submitted) to exercise an arbitrary, non-FIFO finish order —
    the real engine finishes requests in completion-length order, not submission order."""
    pending = []  # (req_id, prefix_ids, max_tokens)

    def submit(req_id, prefix_ids, max_tokens, initial):
        pending.append((req_id, list(prefix_ids), max_tokens))

    def poll():
        if not pending:
            return []
        if one_at_a_time:
            batch = [pending.pop(-1 if lifo else 0)]
        else:
            batch = pending[:]
            pending.clear()
        events = []
        for rid, ids, mt in batch:
            completion_ids, logprobs, text = gen(ids, mt)
            events.append(
                {
                    "request_id": rid,
                    "finished": True,
                    "cumulative_tokens": len(completion_ids),
                    "completion_ids": completion_ids,
                    "logprobs": logprobs,
                    "text": text,
                }
            )
        return events

    def busy():
        return bool(pending)

    return submit, poll, busy


def test_rollout_async_equals_rollout_one():
    """rollout_async (continuous-batched, no turn barrier) returns byte-identical rollouts to one
    rollout_one per example — same token alignment, env_mask, logprobs, per-rollout reward and input
    order. Only the SCHEDULING differs from the pure single-rollout reference."""
    examples = [{"max_model": 1}, {"max_model": 3}, {"max_model": 2}]
    ones = [
        rollout_one(
            example=e,
            active_env=_VarTurnEnv(),
            render=render,
            generate=_det_generate,
            env_glue=env_glue,
            max_turns=8,
            per_turn_max_tokens=8,
        )
        for e in examples
    ]
    submit, poll, busy = _fake_async_engine(_det_generate)
    out = rollout_async(
        examples=examples,
        active_env=_VarTurnEnv(),
        render=render,
        submit=submit,
        poll=poll,
        busy=busy,
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=8,
    )
    assert out == ones
    assert [r["reward"] for r in out] == [1.0, 3.0, 2.0]


def test_rollout_async_robust_to_arbitrary_finish_order():
    """Continuous batching finishes requests in completion order, NOT submission order. Even when
    turns finish one-at-a-time in LIFO order, rollout_async still produces input-order, byte-identical
    results: each rollout has at most one in-flight turn, so cross-rollout finish order can't perturb
    any single rollout's transcript — and a finished rollout's slot is free for others' next turns."""
    examples = [{"max_model": 1}, {"max_model": 3}, {"max_model": 2}, {"max_model": 1}]
    ones = [
        rollout_one(
            example=e,
            active_env=_VarTurnEnv(),
            render=render,
            generate=_det_generate,
            env_glue=env_glue,
            max_turns=8,
            per_turn_max_tokens=8,
        )
        for e in examples
    ]
    submit, poll, busy = _fake_async_engine(_det_generate, one_at_a_time=True, lifo=True)
    out = rollout_async(
        examples=examples,
        active_env=_VarTurnEnv(),
        render=render,
        submit=submit,
        poll=poll,
        busy=busy,
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=8,
    )
    assert out == ones  # input order + byte-identical despite LIFO one-at-a-time finishing
    assert [r["reward"] for r in out] == [1.0, 3.0, 2.0, 1.0]


class _BatchRewardEnv(_VarTurnEnv):
    """_VarTurnEnv that also exposes reward_many (scores every rollout in ONE batched call) and
    counts which scoring path the rollout took."""

    def __init__(self):
        self.reward_many_calls = 0
        self.per_rollout_reward_calls = 0

    def reward(self, completion, example, state=None):
        self.per_rollout_reward_calls += 1
        return super().reward(completion, example, state)

    def reward_many(self, items):
        self.reward_many_calls += 1
        return [super(_BatchRewardEnv, self).reward("", ex, st) for ex, st in items]


def test_reward_many_batches_scoring():
    """When the env exposes reward_many, rollout_async scores every rollout in ONE batched call
    (env scores them concurrently) instead of a blocking reward() per rollout — the judge/expensive-
    reward win — at the same per-rollout values + input order as one rollout_one per example."""
    examples = [{"max_model": 1}, {"max_model": 3}, {"max_model": 2}]
    ones = [
        rollout_one(
            example=e,
            active_env=_VarTurnEnv(),
            render=render,
            generate=_det_generate,
            env_glue=env_glue,
            max_turns=8,
            per_turn_max_tokens=8,
        )
        for e in examples
    ]
    env = _BatchRewardEnv()
    submit, poll, busy = _fake_async_engine(_det_generate)
    out = rollout_async(
        examples=examples,
        active_env=env,
        render=render,
        submit=submit,
        poll=poll,
        busy=busy,
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=8,
    )
    assert env.reward_many_calls == 1  # ONE batched scoring call...
    assert env.per_rollout_reward_calls == 0  # ...not one reward() per rollout
    assert out == ones  # byte-identical to the single-rollout reference (reward_many only batches)
    assert [r["reward"] for r in out] == [1.0, 3.0, 2.0]


def _run_async(examples, active_env):
    submit, poll, busy = _fake_async_engine(_det_generate)
    return rollout_async(
        examples=examples,
        active_env=active_env,
        render=render,
        submit=submit,
        poll=poll,
        busy=busy,
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=8,
    )


# _score_rollouts thread-pool fallback (PR #224): when the env has NO reward_many, the per-rollout
# reward — often an IO-bound judge/tool round-trip — is scored concurrently, in input order, with a
# reward_thread_safe opt-out. Exercised via rollout_async (the one shipped path through _score_rollouts).
def test_async_scores_rewards_concurrently():
    """No reward_many -> the fallback scores the batch CONCURRENTLY: correct + in INPUT ORDER, and N
    slow rewards take ~1x (not Nx) the per-call latency."""
    import time

    class _SlowRewardEnv(_VarTurnEnv):
        def reward(self, completion, example, state=None):
            time.sleep(0.2)  # stand in for an IO-bound judge/tool round-trip (releases the GIL)
            return float(example["rid"])  # per-rollout id -> proves order survives the pool

    examples = [{"max_model": 1, "rid": i} for i in range(8)]
    t0 = time.perf_counter()
    out = _run_async(examples, _SlowRewardEnv())
    elapsed = time.perf_counter() - t0
    assert [r["reward"] for r in out] == [float(i) for i in range(8)]  # correct + in input order
    # 8 x 0.2s = 1.6s if serial; concurrent (<=16 workers) is ~0.2s. Generous bound for CI jitter.
    assert elapsed < 1.0, f"reward scoring did not run concurrently ({elapsed:.2f}s for 8x0.2s)"


def test_async_serial_when_reward_not_thread_safe():
    """An env that declares ``reward_thread_safe = False`` is scored SERIALLY — a scorer with mutable
    state or a thread-bound client is never raced (it worked serially and must keep working)."""
    import time

    class _UnsafeSlowEnv(_VarTurnEnv):
        reward_thread_safe = False  # opt out of concurrent scoring

        def reward(self, completion, example, state=None):
            time.sleep(0.15)
            return float(example["rid"])

    examples = [{"max_model": 1, "rid": i} for i in range(6)]
    t0 = time.perf_counter()
    out = _run_async(examples, _UnsafeSlowEnv())
    elapsed = time.perf_counter() - t0
    assert [r["reward"] for r in out] == [float(i) for i in range(6)]  # correct + in order
    # 6 x 0.15s = 0.9s serial vs ~0.15s concurrent -> a >=0.6s floor proves it did NOT parallelize.
    assert elapsed >= 0.6, f"reward_thread_safe=False env was parallelized ({elapsed:.2f}s)"


def test_async_reward_failure_drains_not_backgrounds():
    """On a reward failure: scorers that haven't started are cancelled (no new calls launched) and the
    few already in flight are DRAINED before we raise — none keep scoring in the background after
    rollout_async returns (no spent calls / mutated scorer state bleeding into the next step)."""
    import threading
    import time

    started, finished = [], []
    lock = threading.Lock()

    class _FailEnv(_VarTurnEnv):
        def reward(self, completion, example, state=None):
            with lock:
                started.append(example["rid"])
            if example["rid"] == 0:
                raise RuntimeError("judge 500")  # fails right away
            time.sleep(0.25)  # the in-flight peers are slow
            with lock:
                finished.append(example["rid"])  # records ONLY if the call ran to completion
            return 1.0

    examples = [
        {"max_model": 1, "rid": i} for i in range(8)
    ]  # 8 <= max_workers(=8): all start at once
    with pytest.raises(RuntimeError, match="judge 500"):
        _run_async(examples, _FailEnv())
    # Every scorer that STARTED has also FINISHED by the time we get here (drained, not backgrounded);
    # nothing is still running. With wait=False the slow peers would still be sleeping -> finished < started.
    assert sorted(finished) == sorted(s for s in started if s != 0), (
        "in-flight rewards were not drained"
    )


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_no_wakesleep_when_sleep_mode_off():
    engine = _FakeEngine()
    rf = _build(_FakeTok())
    rf([[{"role": "user", "content": "hi"}]], _fake_trainer(engine, sleep_mode=False))
    assert [e[0] for e in engine.events] == ["add", "step"]


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_re_sleeps_even_if_generation_raises():
    # finally: the engine must return to the offloaded state even when a rollout throws, or the
    # next step inherits a half-woken engine.
    engine = _FakeEngine(vocab=50)  # ord('h')=104 >= 50 -> guard fires inside submit()
    rf = _build(_FakeTok())
    with pytest.raises(ValueError, match="out-of-range token id"):
        rf([[{"role": "user", "content": "hi"}]], _fake_trainer(engine, sleep_mode=True))
    assert engine.events[0] == ("wake", ("kv_cache",))
    assert engine.events[-1] == ("sleep", 2)


class _RaiseEnvReplyEnv(_TwoTurnEnv):
    """Raises during the env reply (after the first model turn), to exercise a mid-rollout failure
    while OTHER rollouts still have in-flight engine requests."""

    def env_reply(self, messages, state):
        raise RuntimeError("boom in env_reply")


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_no_inflight_leak_on_error():
    # A rollout whose env reply raises mid-flight must propagate the error from the worker thread
    # (not hang) and leave NO in-flight engine request behind for the next GRPO step — whether the
    # leftover requests were aborted in the finally or had already finished.
    engine = _FakeEngine(finish="one")
    rf = _build(_FakeTok(), active_env=_RaiseEnvReplyEnv())
    prompts = [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
        [{"role": "user", "content": "c"}],
    ]
    with pytest.raises(RuntimeError, match="boom in env_reply"):
        rf(prompts, _fake_trainer(engine, sleep_mode=False))
    assert not engine.llm_engine.has_unfinished_requests()  # nothing leaked into the engine


class _CountingTok(_FakeTok):
    """_FakeTok that counts how many times env_glue rendered (apply_chat_template with the probe)."""

    def __init__(self):
        super().__init__()
        self.glue_renders = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        if any("flash-env-glue-probe" in str(m.get("content", "")) for m in messages):
            self.glue_renders += 1
        return super().apply_chat_template(
            messages, add_generation_prompt, tokenize, enable_thinking
        )


@pytest.mark.usefixtures("_stub_vllm")
def test_env_glue_render_is_cached_across_repeated_env_messages():
    # Every rollout in the group gets the SAME env reply ("result") each turn, so the inter-turn
    # glue is byte-identical — apply_chat_template must render it ONCE (cached), not once per rollout.
    tok = _CountingTok()
    engine = _FakeEngine()
    rf = _build(tok, active_env=_TwoTurnEnv())
    prompts = [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
        [{"role": "user", "content": "c"}],
    ]
    rf(prompts, _fake_trainer(engine, sleep_mode=False))
    assert tok.glue_renders == 1  # 3 rollouts' identical env-glue rendered once, not 3x


def test_lru_cache_evicts_oldest_when_full_and_still_caches_new():
    # Regression: the render/glue caches used to FREEZE when full (no new key admitted past the cap),
    # so later-repeated diverse prompts never cached -> perf regressed over a long run. The LRU cache
    # must instead EVICT the least-recently-used entry and keep admitting new keys.
    c = _LRUCache(2)
    c.put("a", [1])
    c.put("b", [2])
    assert c.get("a") == [1]  # both fit at capacity
    assert c.get("b") == [2]
    c.put("c", [3])  # over capacity -> evict the least-recently-used ("a")
    assert len(c) == 2
    assert c.get("a") is None  # oldest evicted, NOT frozen-out of the cache
    assert c.get("b") == [2]  # survivor kept
    assert c.get("c") == [3]  # new entry cached


def test_lru_cache_hit_refreshes_recency():
    # A get() must mark its key most-recently-used so an actively-reused key isn't evicted while a
    # stale one lingers (the whole point of LRU over a freeze/FIFO cache).
    c = _LRUCache(2)
    c.put("a", [1])
    c.put("b", [2])
    assert c.get("a") == [1]  # touch "a" -> "b" is now the least-recently-used
    c.put("c", [3])  # evicts "b", keeps the recently-used "a"
    assert c.get("b") is None
    assert c.get("a") == [1]  # recently-used survivor kept
    assert c.get("c") == [3]


def test_lru_cache_put_existing_key_updates_value_without_growing():
    c = _LRUCache(2)
    c.put("a", [1])
    c.put("a", [9])  # refresh same key -> overwrite, no size growth
    assert len(c) == 1
    assert c.get("a") == [9]


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_guards_empty_prompt():
    engine = _FakeEngine()
    rf = _build(_FakeTok(force_ids=[]))  # tokenizer yields no ids -> empty prompt
    with pytest.raises(ValueError, match="empty prompt"):
        rf([[{"role": "user", "content": "hi"}]], _fake_trainer(engine, sleep_mode=False))


class _EchoPromptEnv(_OneTurnEnv):
    """Renders each example's OWN prompt (so a test can vary token ids per prompt in a batch)."""

    def new_rollout_state(self, example):
        return {"prompt": example["prompt"], "completion": []}


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_bounds_checks_every_prompt_not_just_the_first():
    # Regression: the bounds guard is re-armed per prompt. A LATER prompt in the batch carrying an
    # out-of-range id must still be caught — a batch-shared "already checked" flag would validate
    # only prompt #0 and let prompt #1's bad ids reach vLLM (CUDA illegal-access).
    engine = _FakeEngine(vocab=200)  # template wrapper chars (~117) in range; '￿' (65535) not
    rf = _build(_FakeTok(), active_env=_EchoPromptEnv())
    prompts = [
        [{"role": "user", "content": "a"}],  # all ids < 200 -> generates fine
        [{"role": "user", "content": "￿"}],  # id 65535 >= 200 -> must raise on the 2nd prompt
    ]
    with pytest.raises(ValueError, match="out-of-range token id"):
        rf(prompts, _fake_trainer(engine, sleep_mode=False))


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


# --- rollout_one_records: the per-turn record driver OPD distils each turn from -----------------


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
    # turn 1 prefix strictly extends turn 0 with the VERBATIM sampled turn-0 ids, then the env glue with
    # its duplicate LEADING terminator collapsed: realistic_env_glue([u2]) is [END, *render([u2],True)],
    # and the leading END dups a1's own END so it is dropped, leaving exactly render([u2], True).
    assert p1 == [*p0, CONTENT["a1"], END, *render([{"role": "user", "content": "u2"}], True)]
    assert len(p1) > len(p0)
    # context_messages is the parallel TEXT history the teacher conditions on, growing per turn.
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


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_constrains_every_turn_with_structured_outputs():
    """[train] structured_outputs must reach SamplingParams on EVERY assistant turn (mid-rollout
    turns included), as a FRESH StructuredOutputsParams per request — vLLM's processor stamps a
    per-request backend on the instance, so sharing one across requests is unsafe."""
    captured = []

    def _gen(ids, sp):
        captured.append(sp)
        return ([5, 6], None, "ok")

    engine = _FakeEngine(gen=_gen)
    spec = {"json": {"type": "object"}, "disable_any_whitespace": True}
    rf = _build(_FakeTok(), active_env=_TwoTurnEnv(), structured_outputs=spec)
    rf([[{"role": "user", "content": "hi"}]], _fake_trainer(engine, sleep_mode=False))
    assert len(captured) == 2  # both model turns of the two-turn rollout
    for sp in captured:
        assert sp.structured_outputs.kwargs == spec
    assert captured[0].structured_outputs is not captured[1].structured_outputs


@pytest.mark.usefixtures("_stub_vllm")
def test_rollout_func_unconstrained_without_structured_outputs():
    # No [train] structured_outputs -> the sampler kwargs must not carry the field at all.
    captured = []

    def _gen(ids, sp):
        captured.append(sp)
        return ([5, 6], None, "ok")

    engine = _FakeEngine(gen=_gen)
    rf = _build(_FakeTok())
    rf([[{"role": "user", "content": "hi"}]], _fake_trainer(engine, sleep_mode=False))
    assert captured
    assert all(not hasattr(sp, "structured_outputs") for sp in captured)


class _FakeMonotonic:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


def _finished_event(request_id, token=5, text="ok"):
    return {
        "request_id": request_id,
        "finished": True,
        "cumulative_tokens": 1,
        "completion_ids": [token],
        "logprobs": [-0.1],
        "text": text,
    }


class _CountingOneTurnEnv(FakeEnv):
    def __init__(self):
        self.record_calls = []

    def record_model_turn(self, state, content):
        self.record_calls.append(content)
        super().record_model_turn(state, content)

    def rollout_done(self, state, max_turns):
        return bool(self.record_calls)


def test_rollout_request_timeout_default_resolution():
    assert resolve_rollout_request_timeout_seconds(None, 1024) == 600.0
    assert resolve_rollout_request_timeout_seconds(None, 4096) == 2048.0
    assert resolve_rollout_request_timeout_seconds(37.5, 4096) == 37.5


def test_first_physical_attempt_succeeds_without_abort():
    pending = []
    submitted = []
    aborted = []

    def submit(req_id, prefix, max_tokens, initial):
        submitted.append((req_id, list(prefix), max_tokens, initial))
        pending.append(req_id)

    def poll():
        return [_finished_event(pending.pop())]

    env = _CountingOneTurnEnv()
    ids = iter(["attempt-1"])
    out = rollout_async(
        examples=[{}],
        active_env=env,
        render=render,
        submit=submit,
        poll=poll,
        busy=lambda: bool(pending),
        abort=lambda request_ids: aborted.extend(request_ids),
        env_glue=env_glue,
        max_turns=1,
        per_turn_max_tokens=8,
        request_timeout_seconds=5.0,
        request_id_factory=lambda: next(ids),
    )

    assert submitted == [("attempt-1", render([{"role": "user", "content": "u1"}], True), 8, True)]
    assert aborted == []
    assert env.record_calls == ["ok"]
    assert out[0]["completion_ids"] == [5]


def test_timeout_aborts_then_retries_identical_request_and_ignores_stale_output():
    clock = _FakeMonotonic()
    pending = {}
    submitted = []
    aborted = []
    poll_count = 0

    def submit(req_id, prefix, max_tokens, initial):
        submitted.append((req_id, list(prefix), max_tokens, initial))
        pending[req_id] = True

    def abort(request_ids):
        aborted.extend(request_ids)
        for req_id in request_ids:
            pending.pop(req_id, None)

    def poll():
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            clock.advance(6.0)
            return []
        clock.advance(0.1)
        pending.pop("attempt-2", None)
        return [
            _finished_event("attempt-1", token=99, text="stale"),
            _finished_event("attempt-2", token=7, text="accepted"),
        ]

    env = _CountingOneTurnEnv()
    ids = iter(["attempt-1", "attempt-2"])
    out = rollout_async(
        examples=[{}],
        active_env=env,
        render=render,
        submit=submit,
        poll=poll,
        busy=lambda: bool(pending),
        abort=abort,
        env_glue=env_glue,
        max_turns=1,
        per_turn_max_tokens=8,
        request_timeout_seconds=5.0,
        request_max_attempts=2,
        monotonic=clock,
        request_id_factory=lambda: next(ids),
    )

    assert aborted == ["attempt-1"]
    assert [row[0] for row in submitted] == ["attempt-1", "attempt-2"]
    assert submitted[0][1:] == submitted[1][1:]
    assert env.record_calls == ["accepted"]
    assert out[0]["completion_ids"] == [7]


def test_exhausted_physical_attempts_raise_dedicated_error():
    clock = _FakeMonotonic()
    pending = set()
    aborted = []
    ids = iter(["attempt-1", "attempt-2"])

    def submit(req_id, prefix, max_tokens, initial):
        pending.add(req_id)

    def abort(request_ids):
        aborted.extend(request_ids)
        pending.difference_update(request_ids)

    def poll():
        clock.advance(6.0)
        return []

    with pytest.raises(RolloutRequestExhaustedError, match="exhausted 2 physical attempt"):
        rollout_async(
            examples=[{}],
            active_env=_CountingOneTurnEnv(),
            render=render,
            submit=submit,
            poll=poll,
            busy=lambda: bool(pending),
            abort=abort,
            env_glue=env_glue,
            max_turns=1,
            per_turn_max_tokens=8,
            request_timeout_seconds=5.0,
            request_max_attempts=2,
            monotonic=clock,
            request_id_factory=lambda: next(ids),
        )

    assert aborted == ["attempt-1", "attempt-2"]


def test_abort_precedes_retry_submission_in_shared_op_order():
    clock = _FakeMonotonic()
    ops = []
    pending = set()
    poll_count = 0
    ids = iter(["attempt-1", "attempt-2"])

    def submit(req_id, prefix, max_tokens, initial):
        ops.append(("submit", req_id))
        pending.add(req_id)

    def abort(request_ids):
        for req_id in request_ids:
            ops.append(("abort", req_id))
            pending.discard(req_id)

    def poll():
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            clock.advance(6.0)
            return []
        clock.advance(0.1)
        current = next(iter(pending))
        pending.discard(current)
        return [_finished_event(current)]

    out = rollout_async(
        examples=[{}],
        active_env=_CountingOneTurnEnv(),
        render=render,
        submit=submit,
        poll=poll,
        busy=lambda: bool(pending),
        abort=abort,
        env_glue=env_glue,
        max_turns=1,
        per_turn_max_tokens=8,
        request_timeout_seconds=5.0,
        request_max_attempts=2,
        monotonic=clock,
        request_id_factory=lambda: next(ids),
    )

    assert ops == [("submit", "attempt-1"), ("abort", "attempt-1"), ("submit", "attempt-2")]
    assert out[0]["completion_ids"] == [5]


def test_token_progress_never_extends_the_absolute_request_deadline():
    clock = _FakeMonotonic()
    pending = set()
    aborted = []
    poll_count = 0
    ids = iter(["attempt-1", "attempt-2"])

    def submit(req_id, prefix, max_tokens, initial):
        pending.add(req_id)

    def abort(request_ids):
        aborted.extend(request_ids)
        pending.difference_update(request_ids)

    def poll():
        nonlocal poll_count
        poll_count += 1
        current = next(iter(pending))
        if poll_count <= 3:
            # tokens grow on every poll, so the stall guard stays quiet the whole time.
            clock.advance(2.0)
            return [{"request_id": current, "finished": False, "cumulative_tokens": poll_count}]
        clock.advance(0.1)
        pending.discard(current)
        return [_finished_event(current)]

    out = rollout_async(
        examples=[{}],
        active_env=_CountingOneTurnEnv(),
        render=render,
        submit=submit,
        poll=poll,
        busy=lambda: bool(pending),
        abort=abort,
        env_glue=env_glue,
        max_turns=1,
        per_turn_max_tokens=8,
        request_timeout_seconds=5.0,
        request_max_attempts=2,
        stall_timeout_seconds=100.0,
        monotonic=clock,
        request_id_factory=lambda: next(ids),
    )

    assert aborted == ["attempt-1"]
    assert out[0]["completion_ids"] == [5]


def test_tombstone_eviction_is_bounded_and_still_suppresses_stale_output(monkeypatch):
    import flash.engine.multiturn_rollout as mtr

    monkeypatch.setattr(mtr, "_TOMBSTONE_LIMIT", 1)
    clock = _FakeMonotonic()
    pending = set()
    aborted = []
    poll_count = 0
    ids = iter(["attempt-1", "attempt-2", "attempt-3"])

    def submit(req_id, prefix, max_tokens, initial):
        pending.add(req_id)

    def abort(request_ids):
        aborted.extend(request_ids)
        pending.difference_update(request_ids)

    def poll():
        nonlocal poll_count
        poll_count += 1
        if poll_count <= 2:
            clock.advance(6.0)
            return []
        clock.advance(0.1)
        pending.discard("attempt-3")
        # attempt-1's tombstone was evicted by the limit; attempt-2's is retained. neither
        # stale completion may reach the environment, only the live third attempt.
        return [
            _finished_event("attempt-1", token=98, text="stale-evicted"),
            _finished_event("attempt-2", token=99, text="stale-tombstoned"),
            _finished_event("attempt-3", token=7, text="accepted"),
        ]

    env = _CountingOneTurnEnv()
    out = rollout_async(
        examples=[{}],
        active_env=env,
        render=render,
        submit=submit,
        poll=poll,
        busy=lambda: bool(pending),
        abort=abort,
        env_glue=env_glue,
        max_turns=1,
        per_turn_max_tokens=8,
        request_timeout_seconds=5.0,
        request_max_attempts=3,
        monotonic=clock,
        request_id_factory=lambda: next(ids),
    )

    assert aborted == ["attempt-1", "attempt-2"]
    assert env.record_calls == ["accepted"]
    assert out[0]["completion_ids"] == [7]


def test_stall_timeout_tracks_only_cumulative_token_growth():
    clock = _FakeMonotonic()
    pending = set()
    aborted = []
    poll_count = 0
    ids = iter(["attempt-1", "attempt-2"])

    def submit(req_id, prefix, max_tokens, initial):
        pending.add(req_id)

    def abort(request_ids):
        aborted.extend(request_ids)
        pending.difference_update(request_ids)

    def poll():
        nonlocal poll_count
        poll_count += 1
        current = next(iter(pending))
        if poll_count == 1:
            clock.advance(4.0)
            return [{"request_id": current, "finished": False, "cumulative_tokens": 1}]
        if poll_count == 2:
            clock.advance(4.0)
            return [{"request_id": current, "finished": False, "cumulative_tokens": 1}]
        if poll_count == 3:
            clock.advance(2.0)
            return [{"request_id": current, "finished": False, "cumulative_tokens": 1}]
        clock.advance(1.0)
        pending.remove(current)
        return [_finished_event(current)]

    out = rollout_async(
        examples=[{}],
        active_env=_CountingOneTurnEnv(),
        render=render,
        submit=submit,
        poll=poll,
        busy=lambda: bool(pending),
        abort=abort,
        env_glue=env_glue,
        max_turns=1,
        per_turn_max_tokens=8,
        request_timeout_seconds=100.0,
        request_max_attempts=2,
        stall_timeout_seconds=5.0,
        monotonic=clock,
        request_id_factory=lambda: next(ids),
    )

    assert aborted == ["attempt-1"]
    assert out[0]["completion_ids"] == [5]


def test_request_deadlines_do_not_create_episode_wall_clock_cutoff():
    clock = _FakeMonotonic()
    pending = []

    def submit(req_id, prefix, max_tokens, initial):
        pending.append(req_id)

    def poll():
        clock.advance(4.0)
        return [_finished_event(pending.pop(0))]

    out = rollout_async(
        examples=[{}],
        active_env=FakeEnv(),
        render=render,
        submit=submit,
        poll=poll,
        busy=lambda: bool(pending),
        abort=lambda ids: None,
        env_glue=env_glue,
        max_turns=2,
        per_turn_max_tokens=8,
        request_timeout_seconds=5.0,
        monotonic=clock,
    )

    assert clock.value == 8.0
    assert len(out[0]["completion_ids"]) > 1


def test_rollout_exception_aborts_all_active_physical_requests():
    pending = set()
    aborted = []

    def submit(req_id, prefix, max_tokens, initial):
        pending.add(req_id)

    def abort(request_ids):
        aborted.extend(request_ids)
        pending.difference_update(request_ids)

    with pytest.raises(RuntimeError, match="engine step failed"):
        rollout_async(
            examples=[{}, {}],
            active_env=_CountingOneTurnEnv(),
            render=render,
            submit=submit,
            poll=lambda: (_ for _ in ()).throw(RuntimeError("engine step failed")),
            busy=lambda: bool(pending),
            abort=abort,
            env_glue=env_glue,
            max_turns=1,
            per_turn_max_tokens=8,
        )

    assert len(aborted) == 2
    assert pending == set()


def test_submit_exception_aborts_the_attempt_id():
    aborted = []

    def submit(req_id, prefix, max_tokens, initial):
        raise RuntimeError("submit failed")

    with pytest.raises(RuntimeError, match="submit failed"):
        rollout_async(
            examples=[{}],
            active_env=_CountingOneTurnEnv(),
            render=render,
            submit=submit,
            poll=lambda: [],
            busy=lambda: False,
            abort=lambda request_ids: aborted.extend(request_ids),
            env_glue=env_glue,
            max_turns=1,
            per_turn_max_tokens=8,
            request_id_factory=lambda: "submit-failure-id",
        )

    assert aborted == ["submit-failure-id"]


def test_concurrent_rollout_invocations_use_disjoint_physical_ids():
    import threading

    barrier = threading.Barrier(2)
    all_ids = [[], []]
    results = [None, None]
    errors = []

    def run(index):
        pending = []

        def submit(req_id, prefix, max_tokens, initial):
            all_ids[index].append(req_id)
            pending.append(req_id)
            barrier.wait(timeout=2.0)

        try:
            results[index] = rollout_async(
                examples=[{}],
                active_env=_CountingOneTurnEnv(),
                render=render,
                submit=submit,
                poll=lambda: [_finished_event(pending.pop())],
                busy=lambda: bool(pending),
                env_glue=env_glue,
                max_turns=1,
                per_turn_max_tokens=8,
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert results[0] is not None
    assert results[1] is not None
    assert set(all_ids[0]).isdisjoint(all_ids[1])


@pytest.mark.usefixtures("_stub_vllm")
def test_retry_builds_fresh_sampling_and_structured_output_params():
    clock = _FakeMonotonic()
    submissions = []
    pending = {}
    aborted = []
    first_step = True

    def add_request(req_id, prompt, sampling_params):
        submissions.append((req_id, list(prompt["prompt_token_ids"]), sampling_params))
        pending[req_id] = sampling_params

    def step():
        nonlocal first_step
        if first_step:
            first_step = False
            clock.advance(6.0)
            return []
        clock.advance(0.1)
        req_id = next(iter(pending))
        pending.pop(req_id)
        comp = SimpleNamespace(token_ids=[5], logprobs=None, text="ok")
        return [SimpleNamespace(request_id=req_id, finished=True, outputs=[comp])]

    def abort_request(request_ids):
        aborted.extend(request_ids)
        for req_id in request_ids:
            pending.pop(req_id, None)

    llm_engine = SimpleNamespace(
        model_config=SimpleNamespace(get_vocab_size=lambda: 1000),
        add_request=add_request,
        step=step,
        has_unfinished_requests=lambda: bool(pending),
        abort_request=abort_request,
    )
    engine = SimpleNamespace(llm_engine=llm_engine)
    trainer = _fake_trainer(engine, sleep_mode=False)
    ids = iter(["attempt-1", "attempt-2"])
    spec = {"json": {"type": "object"}}
    rf = _build(
        _FakeTok(),
        structured_outputs=spec,
        engine_max_len=100,
        request_timeout_seconds=5.0,
        request_max_attempts=2,
        monotonic=clock,
        request_id_factory=lambda: next(ids),
    )
    rf([[{"role": "user", "content": "hi"}]], trainer)

    assert aborted == ["attempt-1"]
    assert [row[0] for row in submissions] == ["attempt-1", "attempt-2"]
    assert submissions[0][1] == submissions[1][1]
    first_params, second_params = submissions[0][2], submissions[1][2]
    assert first_params is not second_params
    assert first_params.max_tokens == second_params.max_tokens
    assert first_params.temperature == second_params.temperature
    assert first_params.top_p == second_params.top_p
    assert first_params.structured_outputs is not second_params.structured_outputs
    assert first_params.structured_outputs.kwargs == second_params.structured_outputs.kwargs == spec
