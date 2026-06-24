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
    rollout_async,
    rollout_batch,
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
    rollouts of different lengths (exercises rollout_batch's lockstep drop-out)."""

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
        return float(sum(1 for m in (state or {}).get("completion", []) if m["role"] == "assistant"))


def _det_generate(prefix_ids, max_tokens):
    """Deterministic single-turn generation: a pure function of nothing but the call, so running
    rollout_one per example and rollout_batch over all examples see byte-identical turns."""
    return [CONTENT["a1"], END], [-0.1, -0.2], "a1"


def test_rollout_batch_equals_rollout_one_per_example():
    """rollout_batch is exactly N independent rollout_one() calls — same token alignment, env_mask,
    logprobs and per-rollout reward, in input order — only the generation is batched. Different
    per-example turn counts force the batch to shrink as rollouts finish at different turns."""
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

    def batched(prefixes, max_tokens_list):
        return [_det_generate(p, m) for p, m in zip(prefixes, max_tokens_list, strict=True)]

    batch = rollout_batch(
        examples=examples,
        active_env=_VarTurnEnv(),
        render=render,
        batched_generate=batched,
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=8,
    )
    assert batch == ones
    # rewards differ per rollout (per-rollout scoring survived batching)
    assert [r["reward"] for r in batch] == [1.0, 3.0, 2.0]
    # the multi-turn rollouts carry masked (0) env-glue tokens; the single-turn one does not
    assert set(batch[0]["env_mask"]) == {1}  # max_model=1: one turn, no env tokens
    three_turn = batch[1]  # max_model=3: env replies -> masked glue tokens present
    assert 0 in three_turn["env_mask"]  # masked env-glue tokens
    assert 1 in three_turn["env_mask"]  # trained model tokens


def test_rollout_batch_preserves_input_order_and_count():
    """One result per example, in input order (so TRL's GRPO group stays aligned)."""

    def batched(prefixes, max_tokens_list):
        return [_det_generate(p, m) for p, m in zip(prefixes, max_tokens_list, strict=True)]

    examples = [{"max_model": 2}, {"max_model": 1}, {"max_model": 1}, {"max_model": 3}]
    out = rollout_batch(
        examples=examples,
        active_env=_VarTurnEnv(),
        render=render,
        batched_generate=batched,
        env_glue=env_glue,
        max_turns=8,
        per_turn_max_tokens=8,
    )
    assert len(out) == 4
    assert [r["reward"] for r in out] == [2.0, 1.0, 1.0, 3.0]


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
    return [{tid: types.SimpleNamespace(logprob=lp)} for tid, lp in zip(token_ids, lps, strict=True)]


class _FakeEngine:
    """Step-able fake colocate engine mirroring vLLM's V1 manual loop: ``llm_engine.add_request``
    enqueues a turn, ``step()`` finishes ALL pending requests at once (one decode round) returning a
    RequestOutput per finished request, ``has_unfinished_requests`` reports the queue. Records
    ('wake'|'add'|'step'|'sleep', ...) events in order so the wake/sleep ordering can be asserted."""

    def __init__(self, vocab=1000, gen=None, finish="all"):
        self.events = []
        self._pending = []  # (req_id, prompt_ids, sampling_params)
        self._finish = finish  # "all" -> finish every pending request per step; "one" -> one per step
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


def _build(tok, active_env=None):
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
        engine_max_len=None,
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
    prompts = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}],
               [{"role": "user", "content": "c"}]]
    trainer = _fake_trainer(engine, sleep_mode=False)
    trainer.num_generations = 4  # must be IGNORED by rollout_func (the sampler already repeated)
    out = rf(prompts, trainer)
    assert len(out["completion_ids"]) == len(prompts) == 3
    assert len(out["reward"]) == 3
    assert len(out["prompt_ids"]) == 3


def test_batched_makes_far_fewer_engine_calls_than_sequential():
    # The CPU-measurable heart of the optimization: rollout_batch issues ONE batched generate per
    # TURN (every active rollout together) vs one generate per prompt per turn for per-example
    # rollout_one. For P prompts over T model turns the batched path makes ~T dispatches, the
    # sequential path ~P*T — far fewer, far larger GPU calls (vLLM batches the decode), at identical
    # results. Measured at the core-function level (no env toggle — batching is always on).
    examples = [{"max_model": 2} for _ in range(6)]  # 6 prompts, _VarTurnEnv -> 2 model turns each

    seq_calls = 0

    def counting_gen(prefix, mt):
        nonlocal seq_calls
        seq_calls += 1
        return _det_generate(prefix, mt)

    seq = [
        rollout_one(
            example=e, active_env=_VarTurnEnv(), render=render, generate=counting_gen,
            env_glue=env_glue, max_turns=8, per_turn_max_tokens=8,
        )
        for e in examples
    ]

    batch_calls = 0

    def counting_batched(prefixes, max_tokens_list):
        nonlocal batch_calls
        batch_calls += 1
        return [_det_generate(p, m) for p, m in zip(prefixes, max_tokens_list, strict=True)]

    batch = rollout_batch(
        examples=examples, active_env=_VarTurnEnv(), render=render, batched_generate=counting_batched,
        env_glue=env_glue, max_turns=8, per_turn_max_tokens=8,
    )

    assert batch == seq  # byte-identical rollouts
    assert seq_calls == 6 * 2  # sequential: one engine call per prompt per turn
    assert batch_calls == 2  # batched: one engine call per turn (all 6 rollouts in each)
    assert batch_calls < seq_calls


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
        return [(rid, *gen(ids, mt)) for rid, ids, mt in batch]

    def busy():
        return bool(pending)

    return submit, poll, busy


def test_rollout_async_equals_rollout_batch_and_one():
    """rollout_async (continuous-batched, no turn barrier) returns byte-identical rollouts to one
    rollout_one per example — same token alignment, env_mask, logprobs, per-rollout reward and input
    order. Only the SCHEDULING differs from the synchronized rollout_batch."""
    examples = [{"max_model": 1}, {"max_model": 3}, {"max_model": 2}]
    ones = [
        rollout_one(example=e, active_env=_VarTurnEnv(), render=render, generate=_det_generate,
                    env_glue=env_glue, max_turns=8, per_turn_max_tokens=8)
        for e in examples
    ]
    submit, poll, busy = _fake_async_engine(_det_generate)
    out = rollout_async(examples=examples, active_env=_VarTurnEnv(), render=render, submit=submit,
                        poll=poll, busy=busy, env_glue=env_glue, max_turns=8, per_turn_max_tokens=8)
    assert out == ones
    assert [r["reward"] for r in out] == [1.0, 3.0, 2.0]


def test_rollout_async_robust_to_arbitrary_finish_order():
    """Continuous batching finishes requests in completion order, NOT submission order. Even when
    turns finish one-at-a-time in LIFO order, rollout_async still produces input-order, byte-identical
    results: each rollout has at most one in-flight turn, so cross-rollout finish order can't perturb
    any single rollout's transcript — and a finished rollout's slot is free for others' next turns."""
    examples = [{"max_model": 1}, {"max_model": 3}, {"max_model": 2}, {"max_model": 1}]
    ones = [
        rollout_one(example=e, active_env=_VarTurnEnv(), render=render, generate=_det_generate,
                    env_glue=env_glue, max_turns=8, per_turn_max_tokens=8)
        for e in examples
    ]
    submit, poll, busy = _fake_async_engine(_det_generate, one_at_a_time=True, lifo=True)
    out = rollout_async(examples=examples, active_env=_VarTurnEnv(), render=render, submit=submit,
                        poll=poll, busy=busy, env_glue=env_glue, max_turns=8, per_turn_max_tokens=8)
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


def test_reward_many_batches_scoring_in_both_paths():
    """When the env exposes reward_many, BOTH rollout paths score every rollout in ONE batched call
    (env scores them concurrently) instead of a blocking reward() per rollout — the judge/expensive-
    reward win — at identical per-rollout values + order, and the batched result stays byte-identical
    across sync and async."""
    examples = [{"max_model": 1}, {"max_model": 3}, {"max_model": 2}]

    def batched(prefixes, mt):
        return [_det_generate(p, m) for p, m in zip(prefixes, mt, strict=True)]

    env_b = _BatchRewardEnv()
    out_b = rollout_batch(examples=examples, active_env=env_b, render=render, batched_generate=batched,
                          env_glue=env_glue, max_turns=8, per_turn_max_tokens=8)
    env_a = _BatchRewardEnv()
    submit, poll, busy = _fake_async_engine(_det_generate)
    out_a = rollout_async(examples=examples, active_env=env_a, render=render, submit=submit, poll=poll,
                          busy=busy, env_glue=env_glue, max_turns=8, per_turn_max_tokens=8)
    for env, out in ((env_b, out_b), (env_a, out_a)):
        assert env.reward_many_calls == 1  # ONE batched scoring call...
        assert env.per_rollout_reward_calls == 0  # ...not one reward() per rollout
        assert [r["reward"] for r in out] == [1.0, 3.0, 2.0]
    assert out_a == out_b  # batched-reward path is byte-identical across sync/async


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
    prompts = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}],
               [{"role": "user", "content": "c"}]]
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
        return super().apply_chat_template(messages, add_generation_prompt, tokenize, enable_thinking)


@pytest.mark.usefixtures("_stub_vllm")
def test_env_glue_render_is_cached_across_repeated_env_messages():
    # Every rollout in the group gets the SAME env reply ("result") each turn, so the inter-turn
    # glue is byte-identical — apply_chat_template must render it ONCE (cached), not once per rollout.
    tok = _CountingTok()
    engine = _FakeEngine()
    rf = _build(tok, active_env=_TwoTurnEnv())
    prompts = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}],
               [{"role": "user", "content": "c"}]]
    rf(prompts, _fake_trainer(engine, sleep_mode=False))
    assert tok.glue_renders == 1  # 3 rollouts' identical env-glue rendered once, not 3x


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
