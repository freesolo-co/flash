"""Tests for the verifiers/Hub adapter + install manifest (no network/GPU)."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile

import pytest


class _FakeRubric:
    def __init__(self, funcs, weights):
        self.funcs = funcs
        self.weights = weights


class _FakeVfEnv:
    """Duck-typed stand-in for a verifiers SingleTurnEnv."""

    def __init__(self):
        self.system_prompt = "be brief"
        self.parser = None
        self.pass_threshold = 0.5
        self.dataset = [{"prompt": [{"role": "user", "content": "2+2?"}], "answer": "4"}]
        self.eval_dataset = [{"prompt": [{"role": "user", "content": "3+3?"}], "answer": "6"}]

        async def correct(completion, answer):
            return 1.0 if answer in completion[-1]["content"] else 0.0

        def fmt(**kwargs):  # **kwargs-style reward func
            return 0.0

        self.rubric = _FakeRubric([correct, fmt], [1.0, 0.0])


def test_verifiers_adapter_mapping():
    from autoslm.envs.adapter import VerifiersEnvironment

    env = VerifiersEnvironment(_FakeVfEnv(), "fake/env")
    assert env.id == "fake/env"

    train = env.dataset("train")
    assert train[0]["answer"] == "4"

    msgs = env.prompt_messages(train[0])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "2+2?"

    # async reward func handled; correct answer -> 1.0
    assert env.reward("the answer is 4", train[0]) == 1.0
    assert env.grade("the answer is 4", train[0]) is True
    assert env.reward("nope", train[0]) == 0.0
    assert env.grade("nope", train[0]) is False

    assert env.sft_target({"answer": "4"}) == "4"


def test_eval_split_uses_eval_dataset_not_train():
    """A populated eval_dataset is returned for the eval split (not the train split)."""
    from autoslm.envs.adapter import VerifiersEnvironment

    env = VerifiersEnvironment(_FakeVfEnv(), "fake/env")
    eval_rows = env.dataset("eval")
    assert [r["answer"] for r in eval_rows] == ["6"]  # the eval_dataset, not the train "4"


def test_empty_eval_dataset_does_not_fall_back_to_train():
    """An empty-but-configured eval split ([]) is honored as an empty eval set; it must NOT
    fall back to the TRAIN split (the `or` -> `is None` fix). Falling back would evaluate on
    training data."""
    from autoslm.envs.adapter import VerifiersEnvironment

    class _Env(_FakeVfEnv):
        def __init__(self):
            super().__init__()
            self.eval_dataset = []  # configured but empty -> falsy, must NOT fall through

    env = VerifiersEnvironment(_Env(), "fake/env")
    assert env.dataset("eval") == []  # honored as empty, NOT the train rows
    assert env.dataset("train")[0]["answer"] == "4"  # train split still works


def test_eval_get_eval_dataset_empty_does_not_fall_back():
    """An empty list from get_eval_dataset() (not the attr) is likewise honored as empty and
    must not fall through to get_dataset/train."""
    from autoslm.envs.adapter import VerifiersEnvironment

    class _Env:
        rubric = None
        parser = None

        def __init__(self):
            self.dataset = [{"prompt": [{"role": "user", "content": "q"}], "answer": "train"}]

        def get_eval_dataset(self, n=-1, seed=0):
            return []  # explicit empty eval set

    env = VerifiersEnvironment(_Env(), "fake/env")
    assert env.dataset("eval") == []  # honored, not the "train" row


def test_missing_eval_dataset_falls_back_to_train():
    """When NO eval split is configured at all (genuinely absent / None), the eval split falls
    back to the env's train split — the only legitimate fallback case."""
    from autoslm.envs.adapter import VerifiersEnvironment

    class _Env:
        rubric = None
        parser = None
        eval_dataset = None  # genuinely absent

        def __init__(self):
            self.dataset = [{"prompt": [{"role": "user", "content": "q"}], "answer": "train"}]

    env = VerifiersEnvironment(_Env(), "fake/env")
    assert [r["answer"] for r in env.dataset("eval")] == ["train"]


def test_install_manifest_and_worker_deps():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["AUTOSLM_ENVS_MANIFEST"] = os.path.join(tmp, "envs.json")
        import autoslm.envs.registry as registry

        importlib.reload(registry)
        # The worker only pip-installs `verifiers`; the env itself is installed via the
        # authenticated `prime env install` (see worker_hub_env_ids), not pip.
        assert registry.worker_pip_for_env("owner/env") == ["verifiers"]
        assert registry.worker_hub_env_ids("owner/env") == ["owner/env"]
        # A separate eval Hub env is also prime-installed.
        assert registry.worker_hub_env_ids("owner/train", {"eval_env_id": "owner/eval"}) == [
            "owner/train",
            "owner/eval",
        ]
        # The train env doubling as the eval env is installed once.
        assert registry.worker_hub_env_ids("owner/x", {"eval_env": "owner/x"}) == ["owner/x"]

        os.environ.pop("AUTOSLM_ENVS_MANIFEST", None)
        importlib.reload(registry)


# ---------------------------------------------------------------------------
# Regression tests for the Prime Intellect Hub interop defects fixed in this PR.
# ---------------------------------------------------------------------------
def test_vf_load_id_strips_owner_slug():
    """DEFECT: the adapter passed the full ``owner/name`` slug to verifiers, which only
    resolves the bare env id."""
    from autoslm.envs.adapter import vf_load_id

    assert vf_load_id("primeintellect/hendrycks-math") == "hendrycks-math"
    assert vf_load_id("math500") == "math500"  # already bare -> unchanged


def test_load_verifiers_environment_uses_bare_ids(monkeypatch):
    """DEFECT: load must hand verifiers the BARE id (owner stripped)."""
    import types as _types

    from autoslm.envs import adapter as va

    seen = []

    class _Env:
        rubric = None
        parser = None

    fake_vf = _types.ModuleType("verifiers")
    fake_vf.load_environment = lambda env_id, **kw: (seen.append(env_id), _Env())[1]
    monkeypatch.setitem(sys.modules, "verifiers", fake_vf)

    va.load_verifiers_environment("primeintellect/hendrycks-math")
    assert seen == ["hendrycks-math"]  # owner stripped


def test_rubric_group_is_flattened_and_zero_weight_skipped():
    """DEFECT: a RubricGroup's top-level funcs is empty, so reward was always 0; and a
    zero-weight monitor func (e.g. num_turns) crashed on missing state."""
    from autoslm.envs.adapter import VerifiersEnvironment, _flatten_rubric

    def correct(completion, answer):
        return 1.0 if str(answer) in completion[-1]["content"] else 0.0

    def monitor(**kwargs):
        raise KeyError("trajectory")  # weight 0 -> must be skipped/guarded

    class _Sub:
        def __init__(self, funcs, weights):
            self.funcs, self.weights = funcs, weights

    class _Group:  # RubricGroup: empty top-level funcs
        funcs, weights = (), ()
        rubrics = (_Sub([correct], [1.0]), _Sub([monitor], [0.0]))

    class _Env:
        rubric = _Group()
        parser = None

    pairs = _flatten_rubric(_Group())
    assert len(pairs) == 2  # recursed into nested rubrics

    env = VerifiersEnvironment(_Env(), "owner/x")
    assert env.reward("the answer is 4", {"answer": "4"}) == 1.0
    assert env.reward("nope", {"answer": "4"}) == 0.0


def test_reward_from_weighted_func_propagates_exception():
    """A WEIGHTED reward func that raises must PROPAGATE (fail loudly), not be swallowed as
    0.0 — a raise in a weighted func is a real reward failure (e.g. judge API error) that would
    otherwise train/score on an all-zero signal. (Zero-weight funcs run with guarded exceptions
    instead — see test_reward_zero_weight_crashing_func_runs_with_guarded_exception.)"""
    from autoslm.envs.adapter import VerifiersEnvironment

    def boom(**kwargs):
        raise RuntimeError("kaboom")

    class _Rubric:
        funcs = (boom,)
        weights = (1.0,)

    class _Env:
        rubric = _Rubric()
        parser = None

    env = VerifiersEnvironment(_Env(), "owner/x")
    with pytest.raises(RuntimeError, match="kaboom"):
        env.reward("anything", {"answer": "4"})


def test_reward_zero_weight_crashing_func_runs_with_guarded_exception():
    """Per verifiers semantics a zero-weight monitor func RUNS (it may mutate shared state /
    be logged), but its exceptions are GUARDED (swallowed) — a thrown monitor must not fail the
    run — and it contributes 0 to the reward."""
    from autoslm.envs.adapter import VerifiersEnvironment

    calls = []

    def good(completion, answer, **kwargs):
        return 1.0 if str(answer) in completion[-1]["content"] else 0.0

    def boom(**kwargs):
        calls.append("boom")  # observed -> it actually ran
        raise RuntimeError("kaboom")

    class _Rubric:
        funcs = (good, boom)
        weights = (1.0, 0.0)  # boom is zero-weight -> runs, guarded, contributes 0

    class _Env:
        rubric = _Rubric()
        parser = None

    env = VerifiersEnvironment(_Env(), "owner/x")
    # The crashing zero-weight func runs (its exception is swallowed) and contributes nothing.
    assert env.reward("the answer is 4", {"answer": "4"}) == 1.0
    assert calls == ["boom"]


def test_zero_weight_func_runs_and_can_mutate_shared_state():
    """A zero-weight func runs BEFORE a later weighted func and may set shared ``state`` the
    weighted func then reads. It still contributes 0 itself and is absent from the breakdown."""
    from autoslm.envs.adapter import VerifiersEnvironment

    def monitor(state, **kwargs):  # zero-weight: seeds shared state, returns nothing useful
        state["seen"] = True
        return 0.0

    def scored(state, **kwargs):  # weighted: rewards iff the monitor ran first
        return 1.0 if state.get("seen") else 0.0

    class _Rubric:
        funcs = (monitor, scored)  # monitor first so it can prepare state
        weights = (0.0, 1.0)

    class _Env:
        rubric = _Rubric()
        parser = None

    env = VerifiersEnvironment(_Env(), "owner/x")
    state = {}
    breakdown = env.scores_breakdown("anything", {"answer": "4"}, state)
    assert breakdown["total"] == 1.0  # weighted func saw the monitor's state mutation
    assert breakdown == {"scored": 1.0, "total": 1.0}  # zero-weight monitor not in breakdown
    assert state["seen"] is True


def test_scores_breakdown_collision_does_not_clobber_existing_distinct_key():
    """Colliding reward-func names get a probed exact-key suffix, never one a
    prefix/length heuristic could recompute onto an already-recorded scorer.

    The failing case for the old ``len([k for k in breakdown if k.startswith(name)])``
    rule: a func named ``score`` followed by one named ``score_detail``, then a second
    ``score``. After the first two keys (``score``, ``score_detail``) the heuristic counts
    BOTH (both start with ``score``) and forms ``score_2`` for the third — fine here, but
    a distinct-but-prefixed name in the wrong order makes it recompute an occupied key and
    silently drop a scorer. Exact-key probing keeps every scorer."""
    from autoslm.envs.adapter import VerifiersEnvironment

    def make(name):
        def f(**kwargs):
            return 1.0

        f.__name__ = name
        return f

    funcs = (make("score"), make("score_1"), make("score"), make("score"))

    class _Rubric:
        funcs = ()
        weights = ()

    _Rubric.funcs = funcs
    _Rubric.weights = (1.0, 1.0, 1.0, 1.0)

    class _Env:
        rubric = _Rubric()
        parser = None

    env = VerifiersEnvironment(_Env(), "owner/x")
    breakdown = env.scores_breakdown("anything", {"answer": "x"}, {})
    # Every one of the 4 weighted scorers survives as a distinct key (+ total); none is
    # overwritten. Probing yields: score, score_1, then score_2, score_3 for the dupes.
    assert breakdown["total"] == 4.0
    assert set(breakdown) == {"score", "score_1", "score_2", "score_3", "total"}


def test_reward_available_uses_state_transcript_in_multi_turn():
    """In multi-turn mode, `completion`/`prompt` passed to reward funcs come from the
    accumulated `state` transcript (full message list), not the scalar completion wrapped as a
    lone assistant message. Single-turn keeps the scalar-wrapping behavior."""
    from autoslm.envs.adapter import VerifiersEnvironment

    class _Env:
        rubric = None
        parser = None

    env = VerifiersEnvironment(_Env(), "owner/x")
    example = {"prompt": [{"role": "user", "content": "hi"}], "answer": "a"}

    # Single-turn: scalar completion wrapped as one assistant message.
    single = env._reward_available("the reply", example, None)
    assert single["completion"] == [{"role": "assistant", "content": "the reply"}]
    assert single["prompt"] == example["prompt"]

    # Multi-turn: full transcript from state is used.
    env.multi_turn = True
    state = {
        "prompt": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "completion": [
            {"role": "assistant", "content": "call tool"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "final"},
        ],
    }
    multi = env._reward_available("final", example, state)
    assert multi["completion"] == state["completion"]
    assert multi["completion"] is not state["completion"]  # copied, not aliased
    assert multi["prompt"] == state["prompt"]

    # Multi-turn with no transcript yet falls back to scalar wrapping.
    fallback = env._reward_available("solo", example, {"turn": 0})
    assert fallback["completion"] == [{"role": "assistant", "content": "solo"}]


def test_group_reward_func_is_rejected_at_construction():
    """A weighted group/batch reward func (plural required arg the single-turn worker
    can't supply) must fail fast, not silently score 0.0 on a paid run."""
    from autoslm.envs.adapter import VerifiersEnvironment

    def group_reward(completions, answers):  # plural batch args — unsupported
        return [1.0 for _ in completions]

    class _Rubric:
        funcs = (group_reward,)
        weights = (1.0,)

    class _Env:
        rubric = _Rubric()
        parser = None

    with pytest.raises(ValueError, match="completions"):
        VerifiersEnvironment(_Env(), "owner/x")

    # A zero-weight group func is skipped (never invoked), so it must NOT block load.
    class _RubricZero:
        funcs = (group_reward,)
        weights = (0.0,)

    class _EnvZero:
        rubric = _RubricZero()
        parser = None

    VerifiersEnvironment(_EnvZero(), "owner/x")  # does not raise


def test_reward_parses_json_string_info():
    """A Hub row may store `info` as a JSON string; reward funcs that index it must
    receive a dict, not raise TypeError (swallowed as 0.0)."""
    from autoslm.envs.adapter import VerifiersEnvironment

    def uses_info(completion, info):
        return 1.0 if info.get("want") == "yes" else 0.0

    class _Rubric:
        funcs = (uses_info,)
        weights = (1.0,)

    class _Env:
        rubric = _Rubric()
        parser = None

    env = VerifiersEnvironment(_Env(), "owner/x")
    # info as a JSON string is parsed -> dict; the func indexes it and scores 1.0.
    assert env.reward("x", {"answer": "a", "info": '{"want": "yes"}'}) == 1.0
    # a dict info still works
    assert env.reward("x", {"answer": "a", "info": {"want": "yes"}}) == 1.0
    # non-JSON string falls back to {} (no crash -> 0.0, not a swallowed TypeError)
    assert env.reward("x", {"answer": "a", "info": "not json"}) == 0.0


def test_dataset_getter_requiring_args_is_called_correctly():
    """A Hub env exposing get_dataset(n, seed) without defaults must be called with
    those args, not no-arg (which raised TypeError -> swallowed -> empty train set)."""
    from autoslm.envs.adapter import VerifiersEnvironment

    rows = [{"prompt": [{"role": "user", "content": "q"}], "answer": "a"}]

    class _Env:
        rubric = None
        parser = None

        def get_dataset(self, n, seed):  # required args, no defaults
            assert n == -1
            assert seed == 0
            return rows

    env = VerifiersEnvironment(_Env(), "owner/x")
    train = env.dataset("train")
    assert train, "getter with required args must yield rows (not swallowed to empty)"
    assert train[0]["answer"] == "a"


def test_worker_installs_env_via_prime():
    """The Flash worker pip-installs only verifiers; the Hub env installs via `prime`."""
    import autoslm.envs.registry as registry

    assert registry.worker_pip_for_env("primeintellect/hendrycks-math") == ["verifiers"]
    # The env id is handed to the worker to `prime env install` (authenticated, public+private).
    assert registry.worker_hub_env_ids("primeintellect/hendrycks-math") == [
        "primeintellect/hendrycks-math"
    ]


def _multi_turn_env(vf_env):
    """A VerifiersEnvironment forced into multi-turn mode (so env_reply/rollout_done run)."""
    from autoslm.envs.adapter import VerifiersEnvironment

    env = VerifiersEnvironment(vf_env, "owner/x")
    env.multi_turn = True
    return env


def test_env_reply_propagates_real_env_response_error(capsys):
    """A genuine bug in a MultiTurnEnv's `env_response` must NOT be silently swallowed as
    [] (which would collapse every multi-turn rollout to a single turn and train a paid GRPO
    run on degenerate transcripts). Mirroring `_invoke_reward`, env_reply logs context and
    re-raises so the run fails fast. A NotImplementedError stays the benign "no env turn"."""

    class _Env:
        rubric = None
        parser = None

        async def env_response(self, messages, state):
            raise RuntimeError("env boom")

    env = _multi_turn_env(_Env())
    with pytest.raises(RuntimeError, match="env boom"):
        env.env_reply([{"role": "assistant", "content": "hi"}], {"turn": 1})
    # the failure is surfaced, not vanished
    assert "env_response failed" in capsys.readouterr().out

    class _NotImpl:
        rubric = None
        parser = None

        async def env_response(self, messages, state):
            raise NotImplementedError

    # NotImplementedError is the legitimate "no env turn" signal -> [] (does not raise).
    assert _multi_turn_env(_NotImpl()).env_reply([], {"turn": 0}) == []


def test_rollout_done_propagates_real_is_completed_error(capsys):
    """A genuine bug in `is_completed` must NOT be silently treated as done=True (which would
    truncate every rollout). It logs context and re-raises; a NotImplementedError remains the
    benign "no completion check -> rely on the turn cap" signal."""

    class _Env:
        rubric = None
        parser = None

        async def is_completed(self, state):
            raise RuntimeError("done boom")

    env = _multi_turn_env(_Env())
    with pytest.raises(RuntimeError, match="done boom"):
        env.rollout_done({"turn": 1})
    assert "is_completed failed" in capsys.readouterr().out

    class _NotImpl:
        rubric = None
        parser = None

        async def is_completed(self, state):
            raise NotImplementedError

    # NotImplementedError -> treat as done (turn-cap-only), does not raise.
    assert _multi_turn_env(_NotImpl()).rollout_done({"turn": 0}) is True
