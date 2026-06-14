"""Tests for the verifiers/Hub adapter + install manifest (no network/GPU)."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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
    from autoslm.envs.verifiers_adapter import VerifiersEnvironment

    env = VerifiersEnvironment(_FakeVfEnv(), "fake/env")
    assert env.id == "fake/env"

    train = env.dataset("train")
    assert train[0]["answer"] == "4"
    ev = env.dataset("eval")
    assert ev[0]["answer"] == "6"

    msgs = env.prompt_messages(train[0])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "2+2?"

    # async reward func handled; correct answer -> 1.0
    assert env.reward("the answer is 4", train[0]) == 1.0
    assert env.grade("the answer is 4", train[0]) is True
    assert env.reward("nope", train[0]) == 0.0
    assert env.grade("nope", train[0]) is False

    assert env.sft_target({"answer": "4"}) == "4"


def test_install_manifest_and_worker_deps():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["AUTOSLM_ENVS_MANIFEST"] = os.path.join(tmp, "envs.json")
        import autoslm.envs.registry as registry

        importlib.reload(registry)
        assert registry.worker_pip_for_env("gsm8k") == []  # builtin
        assert registry.list_installed_verifiers_envs() == []

        registry.record_installed_env("owner/math", package="vf-math")
        assert "owner/math" in registry.list_installed_verifiers_envs()
        assert registry.worker_pip_for_env("owner/math") == ["verifiers", "vf-math"]

        os.environ.pop("AUTOSLM_ENVS_MANIFEST", None)
        importlib.reload(registry)


# ---------------------------------------------------------------------------
# Regression tests for the Prime Intellect Hub interop defects fixed in this PR.
# ---------------------------------------------------------------------------
def test_vf_load_id_strips_owner_slug():
    """DEFECT: the adapter passed the full ``owner/name`` slug to verifiers, which only
    resolves the bare env id."""
    from autoslm.envs.verifiers_adapter import vf_load_id

    assert vf_load_id("primeintellect/hendrycks-math") == "hendrycks-math"
    assert vf_load_id("math500") == "math500"  # already bare -> unchanged


def test_load_verifiers_environment_uses_bare_ids(monkeypatch):
    """DEFECT: load must hand verifiers the BARE id (train + optional eval env)."""
    import sys
    import types as _types

    from autoslm.envs import verifiers_adapter as va

    seen = []

    class _Env:
        rubric = None
        parser = None

    fake_vf = _types.ModuleType("verifiers")
    fake_vf.load_environment = lambda env_id, **kw: (seen.append(env_id), _Env())[1]
    monkeypatch.setitem(sys.modules, "verifiers", fake_vf)

    va.load_verifiers_environment(
        "primeintellect/hendrycks-math", eval_env_id="primeintellect/math500"
    )
    assert seen == ["hendrycks-math", "math500"]  # owner stripped for both


def test_rubric_group_is_flattened_and_zero_weight_skipped():
    """DEFECT: a RubricGroup's top-level funcs is empty, so reward was always 0; and a
    zero-weight monitor func (e.g. num_turns) crashed on missing state."""
    from autoslm.envs.verifiers_adapter import VerifiersEnvironment, _flatten_rubric

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


def test_reward_guards_crashing_nonzero_weight_func():
    """A reward func that raises must score 0.0, not crash the whole reward."""
    from autoslm.envs.verifiers_adapter import VerifiersEnvironment

    def boom(**kwargs):
        raise RuntimeError("kaboom")

    class _Rubric:
        funcs = (boom,)
        weights = (1.0,)

    class _Env:
        rubric = _Rubric()
        parser = None

    env = VerifiersEnvironment(_Env(), "owner/x")
    assert env.reward("anything", {"answer": "4"}) == 0.0


def test_group_reward_func_is_rejected_at_construction():
    """A weighted group/batch reward func (plural required arg the single-turn worker
    can't supply) must fail fast, not silently score 0.0 on a paid run."""
    import pytest

    from autoslm.envs.verifiers_adapter import VerifiersEnvironment

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
    from autoslm.envs.verifiers_adapter import VerifiersEnvironment

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
    from autoslm.envs.verifiers_adapter import VerifiersEnvironment

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


def test_separate_eval_env_and_fixed_subset():
    """DEFECT: no way to eval on a different Hub env. eval_env_id + a fixed-size subset."""
    from autoslm.envs.verifiers_adapter import VerifiersEnvironment

    class _Train:
        rubric = None
        parser = None
        dataset = tuple(
            {"prompt": [{"role": "user", "content": f"t{i}"}], "answer": str(i)} for i in range(50)
        )

    class _Eval:
        eval_dataset = tuple(
            {"prompt": [{"role": "user", "content": f"e{i}"}], "answer": str(i)} for i in range(40)
        )

    env = VerifiersEnvironment(
        _Train(),
        "owner/train",
        eval_vf_env=_Eval(),
        eval_env_id="owner/eval",
        eval_examples=8,
        eval_seed=123,
    )
    train_rows = env.dataset("train")
    eval_rows = env.dataset("eval")
    assert len(train_rows) == 50  # train from the train env
    assert len(eval_rows) == 8  # fixed subset from the eval env
    assert all(r["prompt"][0]["content"].startswith("e") for r in eval_rows)


def test_worker_pip_installs_train_and_eval_wheels_with_index():
    """DEFECT: the Flash worker must install verifiers + train AND eval Hub wheels + the
    recorded extra index."""
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["AUTOSLM_ENVS_MANIFEST"] = os.path.join(tmp, "envs.json")
        import autoslm.envs.registry as registry

        importlib.reload(registry)
        idx = "https://hub.primeintellect.ai/primeintellect/simple/"
        registry.record_installed_env(
            "primeintellect/hendrycks-math",
            package="hendrycks-math",
            extras={"extra_index_url": idx},
        )
        registry.record_installed_env(
            "primeintellect/math500", package="math500", extras={"extra_index_url": idx}
        )

        deps = registry.worker_pip_for_env(
            "primeintellect/hendrycks-math",
            {"eval_env_id": "primeintellect/math500", "eval_examples": 500},
        )
        assert "verifiers" in deps
        assert "hendrycks-math" in deps
        assert "math500" in deps
        assert "--extra-index-url" in deps
        assert idx in deps

        os.environ.pop("AUTOSLM_ENVS_MANIFEST", None)
        importlib.reload(registry)
