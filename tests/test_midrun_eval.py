"""CPU tests for periodic mid-run GRPO eval (no GPU / vLLM / tokenizer needed).

The production core is pure with render/generate/engine INJECTED (mirrors
``test_multiturn_rollout``), so cadence gating, engine binding, skip handling, heartbeat
emission, and both scorers are exercised end-to-end here. The single vLLM-touching helper
(``build_hf_greedy_generate``) is tested against a fake ``torch`` + fake model/tok. Evaluation
distinct from the reward (the env's eval-metric rubric metrics) is verified via the adapter in
``test_verifiers.py`` and surfaced through ``EvalRecord.metrics`` here.
"""

from __future__ import annotations

import sys
import types

import pytest

from flash.engine.midrun_eval import (
    EvalRecord,
    EvalSummary,
    PeriodicEval,
    build_hf_greedy_generate,
    eval_config,
    evaluate_policy,
    make_periodic_eval_callback,
    multi_turn_scorer,
    single_turn_scorer,
    summarize,
)

# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #


def test_summarize_empty_is_all_zero():
    s = summarize([], step=5)
    assert s == EvalSummary(0, 0.0, 0.0, 0.0, 0.0, {}, [], 5)


def test_summarize_mean_min_max_and_rewards_preserved():
    s = summarize(
        [EvalRecord(0.0), EvalRecord(1.0), EvalRecord(0.5)],
        step=3,
    )
    assert s.n == 3
    assert s.mean_reward == pytest.approx(0.5)
    assert s.min_reward == 0.0
    assert s.max_reward == 1.0
    assert s.rewards == [0.0, 1.0, 0.5]
    assert s.step == 3


def test_summarize_averages_env_metrics_independently():
    """Each eval-metric env metric is averaged across the examples that reported it."""
    s = summarize(
        [
            EvalRecord(0.3, {"accuracy": 1.0, "length": 10.0}),
            EvalRecord(0.1, {"accuracy": 0.0, "length": 20.0}),
        ]
    )
    assert s.metric_means["accuracy"] == pytest.approx(0.5)
    assert s.metric_means["length"] == pytest.approx(15.0)
    # the eval metric is independent of the (shaped) training reward
    assert s.mean_reward == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("rewards", "threshold", "expected_pass_rate"),
    [
        ([1.0, 1.0, 0.0, 0.0], 0.5, 0.5),
        ([0.5, 0.5], 0.5, 1.0),  # boundary: >= threshold passes
        ([0.49, 0.5], 0.5, 0.5),
        ([0.0, 0.0], 0.5, 0.0),
    ],
)
def test_summarize_pass_rate_threshold(rewards, threshold, expected_pass_rate):
    recs = [EvalRecord(r) for r in rewards]
    assert summarize(recs, pass_threshold=threshold).pass_rate == pytest.approx(expected_pass_rate)


def test_summary_heartbeat_fields_include_metrics():
    s = summarize([EvalRecord(0.0, {"accuracy": 1.0}), EvalRecord(1.0, {"accuracy": 0.0})])
    fields = s.as_heartbeat_fields()
    assert fields["eval_n"] == 2
    assert fields["eval_reward"] == pytest.approx(0.5)
    assert fields["eval_metric_accuracy"] == pytest.approx(0.5)  # env metric streamed separately
    assert "eval_pass_rate" in fields


# --------------------------------------------------------------------------- #
# evaluate_policy
# --------------------------------------------------------------------------- #


def test_evaluate_policy_aggregates_scores():
    examples = [{"x": 0.2}, {"x": 0.8}]
    s = evaluate_policy(examples, lambda ex: EvalRecord(ex["x"]), step=7, pass_threshold=0.5)
    assert s.n == 2
    assert s.mean_reward == pytest.approx(0.5)
    assert s.pass_rate == pytest.approx(0.5)
    assert s.step == 7


def test_evaluate_policy_swallows_per_example_error_as_zero():
    warnings = []

    def score(ex):
        if ex["bad"]:
            raise RuntimeError("boom")
        return EvalRecord(1.0)

    s = evaluate_policy(
        [{"bad": False}, {"bad": True}, {"bad": False}],
        score,
        on_warn=warnings.append,
    )
    assert s.rewards == [1.0, 0.0, 1.0]
    assert s.mean_reward == pytest.approx(2 / 3)
    assert len(warnings) == 1
    assert "boom" in warnings[0]


def test_evaluate_policy_raise_mode_propagates():
    def score(ex):
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        evaluate_policy([{"a": 1}], score, on_error="raise")


def test_evaluate_policy_systemic_failure_raises_for_skip():
    """If EVERY example errors (e.g. generation OOMs each call), raise rather than report a
    misleading flat eval_reward=0 — the caller turns the raise into eval_skipped."""

    def always_fails(ex):
        raise RuntimeError("OOM")

    with pytest.raises(RuntimeError, match="all 3 examples failed"):
        evaluate_policy([{"a": 1}, {"a": 2}, {"a": 3}], always_fails)


def test_evaluate_policy_partial_failure_does_not_raise():
    def score(ex):
        if ex["a"] == 2:
            raise RuntimeError("one bad")
        return EvalRecord(1.0)

    s = evaluate_policy([{"a": 1}, {"a": 2}], score)  # one ok, one bad -> not systemic
    assert s.rewards == [1.0, 0.0]


# --------------------------------------------------------------------------- #
# single_turn_scorer (uses env.evaluate -> reward + metrics)
# --------------------------------------------------------------------------- #


class _SingleTurnEnv:
    """Minimal env exposing evaluate(): a shaped reward + a eval-metric 'accuracy' metric."""

    def evaluate(self, completion, example):
        hit = completion == example["answer"]
        return {"reward": 0.3 if hit else 0.0, "metrics": {"accuracy": 1.0 if hit else 0.0}}


def test_single_turn_scorer_returns_reward_and_metrics():
    rendered = []

    def render_prompt_ids(example):
        rendered.append(example["question"])
        return [1, 2, 3]

    def generate(prefix_ids, max_tokens):
        assert prefix_ids == [1, 2, 3]
        assert max_tokens == 64
        return [9], [0.0], "42"

    score = single_turn_scorer(_SingleTurnEnv(), render_prompt_ids, generate, 64)
    rec = score({"question": "6*7?", "answer": "42"})
    assert rec.reward == pytest.approx(0.3)  # shaped reward
    assert rec.metrics == {"accuracy": 1.0}  # eval metric distinct from reward
    miss = score({"question": "1+1?", "answer": "2"})
    assert miss.reward == 0.0
    assert miss.metrics == {"accuracy": 0.0}
    assert rendered == ["6*7?", "1+1?"]


def test_single_turn_scorer_applies_graded_text():
    """graded_text strips <think> before the env scores (thinking-run parity)."""

    def render_prompt_ids(example):
        return [1]

    def generate(prefix_ids, max_tokens):
        return [9], [0.0], "<think>scratch work 5</think>42"

    def strip_think(text):
        return text.rsplit("</think>", 1)[1] if text and "</think>" in text else text

    score = single_turn_scorer(
        _SingleTurnEnv(), render_prompt_ids, generate, 64, graded_text=strip_think
    )
    assert score({"question": "q", "answer": "42"}).metrics == {"accuracy": 1.0}


# --------------------------------------------------------------------------- #
# multi_turn_scorer (reuses rollout_one; metrics empty in v1)
# --------------------------------------------------------------------------- #

_HDR = {"user": 100, "assistant": 101}
_END = 199
_CONTENT = {"u1": 1, "u2": 2, "a1": 90, "GOOD": 91}


def _render(messages, add_generation_prompt):
    ids = []
    for m in messages:
        ids.append(_HDR[m["role"]])
        ids.append(_CONTENT[m["content"]])
        ids.append(_END)
    if add_generation_prompt:
        ids.append(_HDR["assistant"])
    return ids


class _MultiTurnEnv:
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
        msgs = (state or {}).get("completion") or []
        return 1.0 if any(m.get("content") == "GOOD" for m in msgs) else 0.0


def _seq_generate(turn_texts):
    seq = iter(turn_texts)

    def generate(prefix_ids, max_tokens):
        text = next(seq)
        return [_CONTENT[text], _END], [0.0, 0.0], text

    return generate


def test_multi_turn_scorer_drives_turn_loop_and_scores():
    score = multi_turn_scorer(
        _MultiTurnEnv(), _render, _seq_generate(["a1", "GOOD"]), max_turns=10, max_new_tokens=64
    )
    rec = score({"answer": "GOOD"})
    assert rec.reward == 1.0
    assert rec.metrics == {}


def test_multi_turn_scorer_zero_when_target_missing():
    score = multi_turn_scorer(
        _MultiTurnEnv(), _render, _seq_generate(["a1", "a1"]), max_turns=10, max_new_tokens=64
    )
    assert score({"answer": "GOOD"}).reward == 0.0


# --------------------------------------------------------------------------- #
# PeriodicEval: cadence
# --------------------------------------------------------------------------- #


def _periodic(**over):
    kw = {
        "examples": [{"a": 1}],
        "score_one_builder": lambda engine: lambda ex: EvalRecord(1.0),
        "every_steps": 10,
        "heartbeat_fn": lambda *a, **k: None,
        "model_getter": lambda: object(),
        "pass_threshold": 0.5,
    }
    kw.update(over)
    return PeriodicEval(**kw)


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, False), (5, False), (10, True), (20, True), (15, False)],
)
def test_should_run_cadence(step, expected):
    assert _periodic(every_steps=10).should_run(step) is expected


def test_should_run_disabled_when_every_zero():
    assert _periodic(every_steps=0).should_run(10) is False


def test_should_run_false_when_no_examples():
    assert _periodic(examples=[]).should_run(10) is False


# --------------------------------------------------------------------------- #
# PeriodicEval: run_eval / maybe_run
# --------------------------------------------------------------------------- #


class _HBSink:
    def __init__(self):
        self.calls = []

    def __call__(self, stage, **kw):
        self.calls.append((stage, kw))


def test_run_eval_happy_path_heartbeats_metrics():
    hb = _HBSink()
    pe = _periodic(
        examples=[{"a": 1}, {"a": 1}],
        score_one_builder=lambda engine: lambda ex: EvalRecord(1.0, {"accuracy": 1.0}),
        heartbeat_fn=hb,
    )
    summary = pe.run_eval(10)
    assert summary is not None
    assert summary.mean_reward == 1.0
    assert len(hb.calls) == 1
    stage, kw = hb.calls[0]
    assert stage == "rl_eval"
    assert kw["step"] == 10
    assert kw["eval_reward"] == 1.0
    assert kw["eval_metric_accuracy"] == 1.0
    assert kw["eval_n"] == 2
    assert "eval_skipped" not in kw


def test_run_eval_no_model_skips_this_cadence_without_permanent_disable():
    hb = _HBSink()
    pe = _periodic(model_getter=lambda: None, heartbeat_fn=hb)
    assert pe.run_eval(10) is None
    stage, kw = hb.calls[0]
    assert stage == "rl_eval"
    assert kw["eval_skipped"] is True
    assert "no model" in kw["eval_reason"]
    # a transient None must NOT permanently disable eval — the next cadence still runs
    assert pe.should_run(20) is True


def test_run_eval_getter_raising_is_swallowed():
    hb = _HBSink()

    def boom_getter():
        raise RuntimeError("trainer torn down")

    pe = _periodic(model_getter=boom_getter, heartbeat_fn=hb)
    assert pe.run_eval(10) is None  # must not propagate (run_eval "never raises")
    assert hb.calls[0][1]["eval_skipped"] is True
    assert "model getter failed" in hb.calls[0][1]["eval_reason"]


def test_run_eval_builder_failure_is_swallowed_with_reason():
    hb = _HBSink()

    def boom_builder(engine):
        raise RuntimeError("engine asleep")

    pe = _periodic(score_one_builder=boom_builder, heartbeat_fn=hb)
    assert pe.run_eval(10) is None
    _stage, kw = hb.calls[0]
    assert kw["eval_skipped"] is True
    assert "engine asleep" in kw["eval_reason"]


def test_run_eval_per_example_error_counts_zero_not_crash():
    hb = _HBSink()

    def builder(engine):
        def score(ex):
            if ex["a"] == 2:
                raise ValueError("bad rollout")
            return EvalRecord(1.0)

        return score

    pe = _periodic(examples=[{"a": 1}, {"a": 2}], score_one_builder=builder, heartbeat_fn=hb)
    summary = pe.run_eval(10)
    assert summary.rewards == [1.0, 0.0]
    assert hb.calls[0][1]["eval_reward"] == pytest.approx(0.5)


def test_periodic_eval_accumulates_history_for_metrics_json():
    """Each eval is recorded into .history so the worker can persist the curve into metrics.json
    (what the agent reads to judge the run on held-out eval, not just the training reward)."""
    pe = _periodic(
        examples=[{"a": 1}, {"a": 1}],
        score_one_builder=lambda engine: lambda ex: EvalRecord(0.5, {"accuracy": 1.0}),
        every_steps=3,
    )
    pe.run_eval(3)
    pe.run_eval(6)
    records = pe.history_records()
    assert [r["step"] for r in records] == [3, 6]
    assert records[0]["eval_reward"] == 0.5
    assert records[0]["eval_metrics"] == {"accuracy": 1.0}
    assert records[0]["eval_n"] == 2
    # a skipped eval (no model) is NOT recorded in the curve
    pe2 = _periodic(model_getter=lambda: None, heartbeat_fn=lambda *a, **k: None)
    pe2.run_eval(9)
    assert pe2.history_records() == []


def test_eval_summary_as_record_shape():
    s = summarize([EvalRecord(0.0, {"accuracy": 1.0}), EvalRecord(1.0, {"accuracy": 0.0})], step=5)
    assert s.as_record() == {
        "step": 5,
        "eval_reward": 0.5,
        "eval_pass_rate": 0.5,
        "eval_n": 2,
        "eval_metrics": {"accuracy": 0.5},
    }


def test_maybe_run_respects_cadence():
    hb = _HBSink()
    pe = _periodic(every_steps=10, heartbeat_fn=hb)
    assert pe.maybe_run(7) is None
    assert hb.calls == []
    assert pe.maybe_run(10) is not None
    assert len(hb.calls) == 1


def test_run_final_evaluates_off_cadence_step():
    # Run length 25 is not a multiple of every_steps=10: the final step was never evaluated,
    # so run_final adds a final eval entry on the saved policy.
    hb = _HBSink()
    pe = _periodic(every_steps=10, heartbeat_fn=hb)
    pe.maybe_run(10)
    pe.maybe_run(20)
    assert [s.step for s in pe.history] == [10, 20]
    assert pe.run_final(25) is not None
    assert [s.step for s in pe.history] == [10, 20, 25]


def test_run_final_no_double_eval_when_step_already_evaluated():
    # Run length 20 IS a multiple: the final step already has a cadence eval, so run_final no-ops.
    pe = _periodic(every_steps=10)
    pe.maybe_run(10)
    pe.maybe_run(20)
    assert pe.run_final(20) is None
    assert [s.step for s in pe.history] == [10, 20]


def test_run_final_disabled_when_eval_off():
    pe = _periodic(every_steps=0)
    assert pe.run_final(25) is None
    assert pe.history == []


def test_bind_model_getter_overrides():
    pe = _periodic(model_getter=None)
    sentinel = object()
    pe.bind_model_getter(lambda: sentinel)
    captured = {}

    def builder(engine):
        captured["engine"] = engine
        return lambda ex: EvalRecord(1.0)

    pe.score_one_builder = builder
    pe.run_eval(10)
    assert captured["engine"] is sentinel


# --------------------------------------------------------------------------- #
# eval_config: cadence is the only knob (TOML [train] eval_every_steps + env override)
# --------------------------------------------------------------------------- #


def test_eval_config_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FLASH_EVAL_EVERY_STEPS", raising=False)
    monkeypatch.delenv("FLASH_EVAL_NUM", raising=False)
    cfg = eval_config(default_max_new=256)
    # nothing set -> disabled; max_new == the run's normal completion budget; cap defaults to 64
    assert cfg == {"every_steps": 0, "num_examples": 64, "max_new_tokens": 256}


def test_eval_config_cadence_from_toml(monkeypatch):
    """The cadence comes from the run's [train] eval_every_steps (no env var needed)."""
    monkeypatch.delenv("FLASH_EVAL_EVERY_STEPS", raising=False)
    cfg = eval_config(default_max_new=256, spec_every=10)
    assert cfg["every_steps"] == 10
    assert cfg["max_new_tokens"] == 256  # always the completion budget, never a separate knob


def test_eval_config_ignores_env_cadence_override(monkeypatch):
    """The cadence comes ONLY from the run spec ([train] eval_every_steps); the old
    FLASH_EVAL_EVERY_STEPS env override no longer exists."""
    monkeypatch.setenv("FLASH_EVAL_EVERY_STEPS", "25")
    assert eval_config(256, spec_every=10)["every_steps"] == 10  # spec wins, env ignored


def test_eval_config_num_examples_is_fixed_safety_cap(monkeypatch):
    """num_examples is a FIXED safety cap (64), not a knob — the old FLASH_EVAL_NUM env override no
    longer applies."""
    monkeypatch.setenv("FLASH_EVAL_NUM", "8")
    assert eval_config(256, spec_every=3)["num_examples"] == 64  # fixed cap, env ignored


# --------------------------------------------------------------------------- #
# make_periodic_eval_callback (transformers)
# --------------------------------------------------------------------------- #


def test_callback_fires_maybe_run_with_global_step():
    pytest.importorskip("transformers")
    seen = []
    pe = _periodic()
    pe.maybe_run = lambda step: seen.append(step)  # type: ignore[method-assign]
    cb = make_periodic_eval_callback(pe)
    state = types.SimpleNamespace(global_step=30)
    cb.on_step_end(args=None, state=state, control=None)
    assert seen == [30]


# --------------------------------------------------------------------------- #
# build_hf_greedy_generate (fake torch + fake model/tok)
# --------------------------------------------------------------------------- #


class _FakeTensor:
    """Minimal stand-in: holds rows of ints, supports .shape / out[0][n:].tolist()."""

    def __init__(self, rows):
        self.rows = rows

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, idx):
        return _FakeRow(self.rows[idx])


class _FakeRow:
    def __init__(self, ids):
        self.ids = list(ids)

    def __getitem__(self, sl):
        return _FakeRow(self.ids[sl])

    def tolist(self):
        return self.ids


def _install_fake_torch(monkeypatch):
    import contextlib

    ft = types.ModuleType("torch")
    ft.long = "long"
    ft.tensor = lambda data, dtype=None, device=None: _FakeTensor([list(r) for r in data])
    ft.ones_like = lambda t: t
    ft.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", ft)


class _FakeModel:
    def __init__(self, new_ids):
        self.new_ids = new_ids
        self.training = True
        self.device = "cpu"
        self.gen_kwargs = None

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def generate(self, input_ids, **kwargs):
        self.gen_kwargs = kwargs
        return _FakeTensor([input_ids.rows[0] + list(self.new_ids)])


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self, text):
        self._text = text

    def decode(self, ids, skip_special_tokens=True):
        return self._text


def test_build_hf_greedy_generate_returns_text_and_restores_train(monkeypatch):
    _install_fake_torch(monkeypatch)
    model = _FakeModel(new_ids=[5, 6, 7])
    generate = build_hf_greedy_generate(model, _FakeTok("the answer"), stop=["</s>"])
    ids, lps, text = generate([1, 2, 3], 64)

    assert ids == [5, 6, 7]
    assert lps == [0.0, 0.0, 0.0]  # eval doesn't need real logprobs
    assert text == "the answer"
    assert model.gen_kwargs["do_sample"] is False  # greedy
    assert model.gen_kwargs["use_cache"] is True  # forced (works under grad checkpointing)
    assert model.gen_kwargs["max_new_tokens"] == 64
    assert model.training is True  # restored to train mode after generate
    # stop handled by stop_strings (generation stops at the stop) so ids/text stay aligned
    assert model.gen_kwargs["stop_strings"] == ["</s>"]
    assert "tokenizer" in model.gen_kwargs


def test_build_hf_greedy_generate_clamps_and_no_stop_kwargs_when_unset(monkeypatch):
    _install_fake_torch(monkeypatch)
    tok = _FakeTok("decoded text")
    model = _FakeModel(new_ids=[9])
    _ids, _lps, text = build_hf_greedy_generate(model, tok, stop=None)([1], 0)
    assert model.gen_kwargs["max_new_tokens"] == 1  # clamped up from 0
    assert "stop_strings" not in model.gen_kwargs  # no stop -> no stop kwargs
    assert text == "decoded text"  # text is exactly decode(new_ids); never post-hoc truncated


def test_build_hf_greedy_generate_keeps_eval_mode_if_model_was_eval(monkeypatch):
    _install_fake_torch(monkeypatch)
    model = _FakeModel(new_ids=[1])
    model.eval()  # model already in eval mode
    build_hf_greedy_generate(model, _FakeTok("x"), stop=None)([1], 4)
    assert model.training is False  # not flipped to train since it wasn't training
